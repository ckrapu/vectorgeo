import os
import yaml
import numpy as np
import tensorflow.keras as keras
import h3
import geopandas as gpd

from metaflow import FlowSpec, Parameter, step
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from shapely.geometry import Polygon

import constants as c
import data_utils

from vectorgeo.h3_utils import h3_global_iterator
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
        default="resnet-triplet-lc.keras")
    
    embed_dim = Parameter(
        'embed_dim',
        help='Dimension of the embedding space',
        default=16)
    
    seed_latlngs = Parameter(
        'seed_latlngs',
        help='List of lat/lng pairs to use as seeds for inference. One job will be created for each seed',
        default = [
            (47.475099, -122.170557),   # Seattle, WA
            (48.5987, 37.9980),         # Bakhumt, Ukraine
            (-19.632875, 23.466110),    # Okavango Delta, Botswana
            ]
        )

    
    @step
    def start(self):
        """
        Start the flow.
        """

        secrets = yaml.load(open(os.path.join(c.BASE_DIR, '.secrets.yml')), Loader=yaml.FullLoader)

        key = f"models/{self.model_filename}"
        local_model_path = os.path.join(c.TMP_DIR, self.model_filename)
        data_utils.download_file(key, local_model_path)

        self.model = keras.models.load_model(local_model_path)
        print(f"Loaded model from {key} with output shape {self.model.output_shape}")

        self.seed_latlngs_parallel = self.seed_latlngs
        
        # If desired, we use this geometry to mask out ocean or other areas far outside national boundaries.
        world_path = os.path.join(c.TMP_DIR, 'world.gpkg')
        data_utils.download_file('misc/world.gpkg', world_path)
        self.world_gdf = gpd.read_file(world_path)
        self.world_geom =self.world_gdf \
            .iloc[0].geometry \
            .simplify(0.1)
        
        if self.wipe_qdrant:
            print("Wiping Qdrant collection")
            qdrant_client = QdrantClient(
                url=secrets['qdrant_url'], 
                api_key=secrets['qdrant_api_key']
            )
            qdrant_client.recreate_collection(
                collection_name=c.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.DOT),
            )
        self.next(self.run_inference, foreach='seed_latlngs_parallel')

    @step
    def run_inference(self, seed_latlng):
        """
        Runs inference on land cover patches, uploading to Qdrant when they are finished.
        """

        secrets = yaml.load(open(os.path.join(c.BASE_DIR, '.secrets.yml')), Loader=yaml.FullLoader)
        qdrant_client = QdrantClient(
            url=secrets['qdrant_url'], 
            api_key=secrets['qdrant_api_key']
        )

        lcp = LandCoverPatches(c.LC_LOCAL_PATH, self.world_gdf, self.image_size)

        int_map       = {x: i for i, x in enumerate(c.LC_LEGEND.keys())}
        int_map_fn    = np.vectorize(int_map.get)

        seed_lat, seed_lng = seed_latlng

        iterator       = h3_global_iterator(seed_lat, seed_lng, self.h3_resolution)
        h3_batch       = []
        zs_batch       = []
        h3s_processed  = set()

        # Our main inference loop runs over points and when enough valid point/image pairs
        # have been found, we run them through the embedding network and then upload
        # the results to Qdrant.
        print("Starting inference loop for job with seed coordinates", seed_latlng)
        for i, cell in enumerate(tqdm(iterator)):
            if i % 1000 == 0:
                print(f"Processing cell {i}: {cell}")

            # Order of lat-lng vs lng-lat is reversed relative to what shapely expects
            poly = Polygon((x,y) for y,x in h3.h3_to_geo_boundary(cell))

            # This catches points that are in the middle of the ocean and lets
            # us bypass running inference on them.
            if not self.world_geom.intersects(poly):
                h3s_processed.add(cell)
                continue

            xs = int_map_fn(lcp.h3_to_patch(cell))

            xs_one_hot = np.zeros((1, self.image_size, self.image_size, c.LC_N_CLASSES))

            for i in range(c.LC_N_CLASSES):
                xs_one_hot[..., i] = (xs == i).squeeze().astype(int)

            zs = self.model(xs_one_hot).numpy().squeeze().tolist()

            h3_batch.append(cell)
            zs_batch.append(zs)
            
            # Qdrant won't allow arbitrary string id fields, so 
            # we convert the H3 index to an integer which is 
            # allowed as an id field.
            if len(zs_batch) >= self.inference_batch_size:

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
                zs_batch = []

        self.next(self.join)

    @step
    def join(self, _):
        """
        Join the parallel inference steps.
        """
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass