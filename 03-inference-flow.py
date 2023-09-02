import os
import yaml
import numpy as np
import h3
import geopandas as gpd
import torch

from metaflow import FlowSpec, Parameter, step
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from shapely.geometry import Polygon

import vectorgeo.constants as c
import vectorgeo.transfer as transfer

from vectorgeo.h3_utils import H3GlobalIterator
from vectorgeo.landcover import LandCoverPatches

class InferenceLandCoverFlow(FlowSpec):

    wipe_qdrant = Parameter(
        'wipe_qdrant',
        help='Whether to wipe the Qdrant collection before starting inference',
        default=False)
    
    inference_batch_size = Parameter(
        'inference_batch_size',
        help='Batch size for inference',
        default=32)
    
    h3_resolution = Parameter(
        'h3_resolution',
        help='H3 resolution for inference',
        default=7)
    
    image_size = Parameter(
        'image_size',
        help='Size of the square image to extract from the raster',
        default=32)
    
    model_filename = Parameter(
        'model_filename',
        help='Filename of the model to load',
        default="resnet-triplet-lc.pt")
    
    embed_dim = Parameter(
        'embed_dim',
        help='Dimension of the embedding space',
        default=16)
    
    seed_latlng = Parameter(
        'seed_latlng',
        help='Lat/lng pair to use as seeds for inference. One job will be created for each seed',
        default = (47.475099, -122.170557))   # Seattle, WA
            #(48.5987, 37.9980),             # Bakhmut, Ukraine
            #(-19.632875, 23.466110),        # Okavango Delta, Botswana
    max_iters = Parameter(
        'max_iters',
        help='Maximum number of iterations to run',
        default=None)

    qdrant_collection = Parameter(
        'qdrant_collection',
        help='Name of the Qdrant collection to use',
        default=c.QDRANT_COLLECTION_NAME)
    
    device = Parameter(
        'device',
        help='Device to use for PyTorch operations',
        default='cuda'
    )
    
    reinit_queue = Parameter(
        'reinit_queue',
        help='Whether or not to start the iteration over H3 cells from scratch',
        default=True
    )
    
    @step
    def start(self):
        """
        Start the flow.
        """

        secrets = yaml.load(open(os.path.join(c.BASE_DIR, '.secrets.yml')), Loader=yaml.FullLoader)

        
        
        # If desired, we use this geometry to mask out ocean or other areas far outside national boundaries.
        world_path = os.path.join(c.TMP_DIR, 'world.gpkg')
        transfer.download_file('misc/world.gpkg', world_path)
        self.world_gdf = gpd.read_file(world_path)
        self.world_geom =self.world_gdf \
            .iloc[0].geometry \
            .simplify(0.1)
        
        if self.wipe_qdrant:
            print(f"Wiping Qdrant collection {self.qdrant_collection}")
            qdrant_client = QdrantClient(
                url=secrets['qdrant_url'], 
                api_key=secrets['qdrant_api_key']
            )
            qdrant_client.recreate_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.DOT),
            )
        self.next(self.run_inference)

    @step
    def run_inference(self):
        """
        Runs inference on land cover patches, uploading to Qdrant when they are finished.
        """

        secrets = yaml.load(open(os.path.join(c.BASE_DIR, '.secrets.yml')), Loader=yaml.FullLoader)
        qdrant_client = QdrantClient(
            url=secrets['qdrant_url'], 
            api_key=secrets['qdrant_api_key']
        )
        
        key = f"models/{self.model_filename}"
        local_model_path = os.path.join(c.TMP_DIR, self.model_filename)
        transfer.download_file(key, local_model_path)

        # Load the PyTorch model
        self.model = torch.load(local_model_path).to(self.device)
        self.model.eval() 
        print(f"Loaded model from {key}")
        
        lc_key = 'raw/' + c.COPERNICUS_LC_KEY
        transfer.download_file(lc_key, c.LC_LOCAL_PATH)
        lcp = LandCoverPatches(c.LC_LOCAL_PATH, self.world_gdf, self.image_size, full_load=False)

        int_map       = {x: i for i, x in enumerate(c.LC_LEGEND.keys())}
        int_map_fn    = np.vectorize(int_map.get)

        seed_lat, seed_lng = self.seed_latlng


        state_filepath = os.path.join(c.TMP_DIR, c.H3_STATE_FILENAME)
        #try:
        if not self.reinit_queue:
            print("Attempting to use existing H3 queue file...")
            transfer.download_file(c.H3_STATE_KEY,state_filepath)
        #except Exception as e:
        #    print(f"Encountered exception {e} while downloading state file")
        #    print("No state file found; starting from scratch")
            
        print("Setting up H3 execution queue at resolution", self.h3_resolution)
        iterator       = H3GlobalIterator(seed_lat, seed_lng, self.h3_resolution,
                                          state_file=None if self.reinit_queue else state_filepath)
        h3_batch       = []
        xs_batch       = []
        h3s_processed  = set()
        
        # Our main inference loop runs over points and when enough valid point/image pairs
        # have been found, we run them through the embedding network and then upload
        # the results to Qdrant.
        print("Starting inference loop for job with seed coordinates", seed_lat, seed_lng)
        for i, cell in enumerate(iterator):
            if i == 0:
                print("Starting first iteration...")
            if i % 5000 == 0:
                print(f"Processing cell {i}: {cell}")
                iterator.save_state(state_filepath)
                transfer.upload_file(c.H3_STATE_KEY, state_filepath)        

            if self.max_iters and i >= int(self.max_iters):
                print(f"Reached max_iters {self.max_iters}; stopping")
                break

            # Order of lat-lng vs lng-lat is reversed relative to what shapely expects
            poly = Polygon((x,y) for y,x in h3.h3_to_geo_boundary(cell))

            # This catches points that are in the middle of the ocean and lets
            # us bypass running inference on them.
            if not self.world_geom.intersects(poly):
                h3s_processed.add(cell)
                continue

            xs = int_map_fn(lcp.h3_to_patch(cell))

            xs_one_hot = np.zeros((c.LC_N_CLASSES, self.image_size, self.image_size))

            for i in range(c.LC_N_CLASSES):
                xs_one_hot[i] = (xs == i).squeeze().astype(int)

            h3_batch.append(cell)
            xs_batch.append(xs_one_hot)

            # Qdrant won't allow arbitrary string id fields, so 
            # we convert the H3 index to an integer which is 
            # allowed as an id field.
            if len(h3_batch) >= self.inference_batch_size:

                xs_one_hot_tensor = torch.tensor(np.stack(xs_batch,axis=0), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    zs_batch = self.model(xs_one_hot_tensor).cpu().numpy().squeeze().tolist()

                coords = [h3.h3_to_geo(h3_index) for h3_index in h3_batch]
                lats, lngs = zip(*coords)    

                _ = qdrant_client.upsert(
                    collection_name=c.QDRANT_COLLECTION_NAME,
                    wait=True,   
                    points=[PointStruct(
                        id=int("0x"+id, 0),
                        vector=vector,
                        payload={"location":{"lon": lng, "lat": lat}}
                        ) for id, vector, lng, lat in zip(h3_batch, zs_batch, lngs, lats)]
                )
                h3s_processed = h3s_processed.union(set(h3_batch))
                h3_batch = []
                xs_batch = []

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass

if __name__ == "__main__":
    InferenceLandCoverFlow()