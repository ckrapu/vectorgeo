import geopandas as gpd
import rasterio as rio
import h3
import numpy as np
import argparse
import json
from multiprocessing import Pool
from rasterio import features
from tqdm import tqdm

from vectorgeo.h3_utils import generate_h3_indexes_at_resolution


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a binary raster mask of the world."
    )
    parser.add_argument(
        "--n_cells",
        type=int,
        default=1000,
        help="Number of cells in both x and y direction for binary mask",
    )
    parser.add_argument(
        "--h3_resolution", type=int, default=7, help="H3 resolution level"
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=8,
        help="Number of batches to split the H3 cells into",
    )
    args = parser.parse_args()

    print("Generating H3 cells")
    h3s = generate_h3_indexes_at_resolution(args.h3_resolution)
    print(f"After generating all h3s, there are {len(h3s)} cells")

    crs = "EPSG:4326"
    print("Reading world geometry")
    world_gdf = gpd.read_file("tmp/world.gpkg")
    world_gdf.geometry = world_gdf.buffer(0.05).simplify(0.1)
    world_gdf = world_gdf.to_crs(crs)

    print("Creating raster")
    bounds = (-180, -90, 180, 90)
    transform = rio.transform.from_bounds(*bounds, args.n_cells, args.n_cells)

    image = features.rasterize(
        ((geom, 1) for geom in world_gdf.geometry),
        out_shape=(args.n_cells, args.n_cells),
        transform=transform,
    )

    print("Processing H3 cells")
    h3_batches = [x.tolist() for x in np.array_split(list(h3s), args.n_batches)]

    with Pool() as pool:
        h3s_processed_list = pool.map(stencil_h3_batch, h3_batches)

    h3s_processed = list(set.union(*h3s_processed_list))
    print(f"Retained {len(h3s_processed) / len(h3s)} of cells")

    with open("tmp/h3s_masked.json", "w") as f:
        json.dump(h3s_processed, f)
