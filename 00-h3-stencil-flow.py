import geopandas as gpd
import rasterio as rio
import h3
import numpy as np
import os
import json

from metaflow import FlowSpec, step, Parameter, parallel_map
from rasterio import features
from tqdm import tqdm

from vectorgeo.h3_utils import generate_h3_indexes_at_resolution
from vectorgeo.transfer import upload_file
from vectorgeo import constants as c


class H3StencilFlow(FlowSpec):
    """
    Creates a binary raster mask of the world, where 1s represent land and 0s represent water.
    This mask is then used to stencil out H3 cells, avoiding those which are entirely water.
    """

    n_cells = Parameter(
        "n_cells",
        help="Number of cells in both x and y direction for binary mask",
        default=1000,
    )

    h3_resolution = Parameter("h3_resolution", help="H3 resolution level", default=7)

    n_batches = Parameter(
        "n_batches", help="Number of batches to split the H3 cells into", default=8
    )

    @step
    def start(self):
        """
        Prepares set of all H3 cell IDs as well as a buffered and simplified
        world geometry for masking out in the following step.
        """

        # Generate all h3 cells at a given resolution by considering
        # all hexadecimal numbers of the appropriate length with
        # f values at the end
        self.h3s = generate_h3_indexes_at_resolution(self.h3_resolution)
        print(f"After generating all h3s, there are {len(self.h3s)} cells")

        print("Reading world geometry")
        world_gdf = gpd.read_file("tmp/world.gpkg")
        world_gdf.geometry = world_gdf.buffer(0.05).simplify(0.1)

        # File should already be in geographic CRS -
        # this is just to be sure.
        self.world_gdf = world_gdf.to_crs("EPSG:4326")

        self.next(self.end)

    @step
    def end(self):
        # Create the raster
        print("Creating raster")

        # Bounds should run over all of earth
        bounds = (-180, -90, 180, 90)

        transform = rio.transform.from_bounds(*bounds, self.n_cells, self.n_cells)

        image = features.rasterize(
            ((geom, 1) for geom in self.world_gdf.geometry),
            out_shape=(self.n_cells, self.n_cells),
            transform=transform,
        )

        # For all H3 cells, check whether their centroid lands on a 1 cell;
        # if they do, add them to the set. Otherwise, discard them.
        print("Processing H3 cells")
        h3_batches = [
            x.tolist() for x in np.array_split(list(self.h3s), self.n_batches)
        ]

        def stencil_h3_batch(h3s):
            """
            Filters out any H3 cells which aren't on or close to land.
            """
            h3s_processed = set()

            for cell in tqdm(h3s):
                centroid = h3.h3_to_geo(cell)
                lat, lng = centroid
                row, col = rio.transform.rowcol(transform, lng, lat)
                val = image[row, col]

                if val == 1:
                    h3s_processed.add(cell)

            return h3s_processed

        self.h3s_processed = parallel_map(stencil_h3_batch, h3_batches)
        self.h3s_processed = list(set.union(*self.h3s_processed))

        # print fraction of cells which are added to set
        print(f"Retained {len(self.h3s_processed) / len(self.h3s)} of cells")

        filename = f"h3s-processed-{self.h3_resolution}.json"
        filepath = os.path.join(c.TMP_DIR, filename)

        # Save to JSON and upload to S3
        with open(filepath, "w") as f:
            json.dump(list(self.h3s_processed), f)

        upload_file(f"misc/{filename}", filepath)


if __name__ == "__main__":
    H3StencilFlow()
