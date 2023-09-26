import os
import yaml
import numpy as np
import h3
import json
import geopandas as gpd
import torch
import time
import pandas as pd

from metaflow import FlowSpec, Parameter, step

import vectorgeo.constants as c
import vectorgeo.transfer as transfer

from vectorgeo.raster import RasterPatches


class InferenceLandCoverFlow(FlowSpec):
    """
    Flow for taking a pretrained model for land cover embedding and running inference
    on the same data, uploading the results to S3.
    """

    inference_batch_size = Parameter(
        "inference_batch_size", help="Batch size for inference", default=256
    )

    h3_resolution = Parameter(
        "h3_resolution", help="H3 resolution for inference", default=7
    )

    image_size = Parameter(
        "image_size",
        help="Size of the square image to extract from the raster",
        default=32,
    )

    model_filename = Parameter(
        "model_filename",
        help="Filename of the model to load",
        default="resnet-triplet-lc.pt",
    )

    embed_dim = Parameter(
        "embed_dim", help="Dimension of the embedding space", default=16
    )
    
    max_iters = Parameter(
        "max_iters", help="Maximum number of iterations to run", default=None
    )

    device = Parameter(
        "device", help="Device to use for PyTorch operations", default="cuda"
    )

    @step
    def start(self):
        """
        Start the flow.
        """

        # If desired, we use this geometry to mask out ocean or other areas far outside national boundaries.
        world_path = os.path.join(c.TMP_DIR, "world.gpkg")
        transfer.download_file("misc/world.gpkg", world_path)
        self.world_gdf = gpd.read_file(world_path)
        self.world_geom = self.world_gdf.iloc[0].geometry.simplify(0.1)

        # Make sure we have DEM / LC raster data as well.
        dem_key = "raw/" + c.GMTED_DEM_KEY
        transfer.download_file(dem_key, c.DEM_LOCAL_PATH)
        lc_key = "raw/" + c.COPERNICUS_LC_KEY
        transfer.download_file(lc_key, c.LC_LOCAL_PATH)

        self.next(self.run_inference)

    @step
    def run_inference(self):
        """
        Runs inference on land cover patches, uploading to S3 when they are finished.
        """

        key = f"models/{self.model_filename}"
        local_model_path = os.path.join(c.TMP_DIR, self.model_filename)
        transfer.download_file(key, local_model_path)

        # Load the PyTorch model
        self.model = torch.load(local_model_path).to(self.device)
        self.model.eval()
        print(f"Loaded model from {key}")

        lc_generator = RasterPatches(
            c.LC_LOCAL_PATH, self.world_gdf, self.image_size, c.LC_RES_M, full_load=True
        )

        dem_generator = RasterPatches(
            c.DEM_LOCAL_PATH, self.world_gdf, self.image_size, c.DEM_RES_M, full_load=True
        )
        
        int_map = {x: i for i, x in enumerate(c.LC_LEGEND.keys())}
        int_map_fn = np.vectorize(int_map.get)

        state_filepath = os.path.join(c.TMP_DIR, c.H3_STATE_FILENAME)

        h3_filename = f"h3s-processed-{self.h3_resolution}.json"
        h3_filepath = os.path.join(c.TMP_DIR, h3_filename)
        h3_key = f"misc/{h3_filename}"
        transfer.download_file(h3_key, h3_filepath)

        print(f"Loading set of valid H3s for inference from {h3_key}")
        with open(h3_filepath, "r") as src:
            valid_h3s = set(json.loads(src.read()))
        valid_h3s = sorted(list(valid_h3s))

        h3_batch, xs_batch, zs_batch, upload_batch = [], [], [], []
        self.h3s_processed = set()

        # Our main inference loop runs over points and when enough valid point/image pairs
        # have been found, we run them through the embedding network and then upload
        # the results to S3.
        start_time = time.time()
        print("Starting first iteration...")
        n_cells = len(valid_h3s)
        for i, cell in enumerate(valid_h3s):
            if i % 1_000_000 == 0 and i > 0:
                iter_rate = i / (time.time() - start_time)
                print(f"Processing cell {cell} ({i}/{n_cells}) with inference rate: {iter_rate:.1f} iterations per second")
                transfer.upload_file(c.H3_STATE_KEY, state_filepath)

            if self.max_iters and i >= int(self.max_iters):
                print(f"Reached max_iters {self.max_iters}; stopping")
                break

            # When there are None elements in the patch, we get a TypeError
            # and this is the least disruptive way to handle it.
            try:
                xs_lc  = int_map_fn(lc_generator.h3_to_patch(cell))
                xs_dem = dem_generator.h3_to_patch(cell)
            except Exception as e:
                print(
                    f"Found anomalous cell {cell} with error {e}. This cell will be skipped."
                )
                continue

            xs_dem -= np.min(xs_dem)
            xs_one_hot = np.zeros((c.LC_N_CLASSES, self.image_size, self.image_size))

            for j in range(c.LC_N_CLASSES):
                xs_one_hot[j] = (xs_lc == j).squeeze().astype(int)

            h3_batch.append(cell)
            xs_batch.append(np.concatenate([xs_one_hot, xs_dem], axis=0))
    
            # NOTE: we have two different batch sizes at this stage - 
            # the first is for inference on the GPU and the second
            # is for uploading data to S3.
            if len(h3_batch) >= self.inference_batch_size:
                xs_one_hot_tensor = torch.tensor(
                    np.stack(xs_batch, axis=0), dtype=torch.float32
                ).to(self.device)
                with torch.no_grad():
                    zs_batch = (
                        self.model(xs_one_hot_tensor).cpu().numpy().squeeze().tolist()
                    )

                coords = [h3.h3_to_geo(h3_index) for h3_index in h3_batch]

                upload_batch += [
                    {"id": id, "vector": vector, "lat": lat, "lng": lng}
                    for id, vector, lng, lat in zip(h3_batch, zs_batch, *coords)
                ]

                self.h3s_processed = self.h3s_processed.union(set(h3_batch))
                h3_batch, xs_batch = [], []

            if len(upload_batch) >= c.INFERENCE_UPLOAD_BATCH_SIZE:
                print(f"Uploading batch of {len(upload_batch)} rows to S3")

                file_id = f"{upload_batch[0]['id']}-{upload_batch[-1]['id']}"
                filename = f"vector-upload-{file_id}.parquet"
                filepath = os.path.join(c.TMP_DIR, filename)

                pd.DataFrame(upload_batch).to_parquet(filepath)
                transfer.upload_file(f"vectors/{filename}", filepath)
                upload_batch = []

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    InferenceLandCoverFlow()
