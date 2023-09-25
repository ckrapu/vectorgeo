"""MetaFlow flow for identifying dominant geographical units for each user on the basis of their viewing history.

Tested resolutions include geo units and neighborhoods.
"""

from metaflow import (
    FlowSpec,
    Parameter,  # pylint: disable=no-name-in-module
    step,
)

import time
import numpy as np
import geopandas as gpd
import os
from vectorgeo import raster as lc
from vectorgeo import constants as c
from vectorgeo import transfer


class PreprocessLandCoverFlow(FlowSpec):
    """
    Flow for producing training dataset of paired anchor/neighbor land cover images
    from Copernicus LULC data. The end result is a sequence of .npy files on the targeted
    S3 bucket, each of which contains matched anchor/neighbor pairs of land use / land cover
    data encoded as integers and with the resolution specified by the `patch_size` parameter.
    """

    patch_size = Parameter(
        "patch_size",
        help="Size of the square patch to extract from the raster",
        default=32,
    )

    n_files = Parameter("n_files", help="Number of files to generate", default=100)

    samples_per_file = Parameter(
        "samples_per_file", help="Number of samples per file", default=5000
    )

    n_jobs = Parameter("n_jobs", help="Number of jobs to run in parallel", default=2)

    pair_distance_meters = Parameter(
        "pair_distance_meters",
        help="Maximum istance between centroids of pairs of anchor/neighbor images in meters",
        default=8000,
    )

    full_load = Parameter(
        "full_load",
        help="Whether or not to attempt to load the full Copernicus land cover .tif into memory",
        default=False,
    )

    @step
    def start(self):
        """
        Loads the boundary shapefile for all countries and starts the sampler.
        """

        # Shapefile with a single geometry indicating boundaries / coastlines for all countries
        world_key, world_path = "misc/world.gpkg", os.path.join(c.TMP_DIR, "world.gpkg")
        transfer.download_file(world_key, world_path)

        self.world_gdf = gpd.read_file(world_path, driver="GPKG")

        self.job_ids = range(self.n_jobs)
        self.int_map = {x: i for i, x in enumerate(c.LC_LEGEND.keys())}
        self.next(self.run_samplers, foreach="job_ids")

    @step
    def run_samplers(self):
        """
        Run the sampler in parallel across different jobs to iteratively samples of neighboring
        land cover patches and store them as Numpy arrays on S3.
        """
        self.uploaded_filekeys = []

        lc_key = "raw/" + c.COPERNICUS_LC_KEY
        transfer.download_file(lc_key, c.LC_LOCAL_PATH)

        dem_key = "raw/" + c.GMTED_DEM_KEY
        transfer.download_file(dem_key, c.DEM_LOCAL_PATH)

        print(f"Creating patch generator...")
        lulc_generator = lc.RasterPatches(
            c.LC_LOCAL_PATH, self.world_gdf, self.patch_size, full_load=self.full_load
        )

        dem_generator = lc.RasterPatches(
            c.DEM_LOCAL_PATH, self.world_gdf, self.patch_size, full_load=self.full_load
        )

        print(
            f"Generating {self.n_files} files with {self.samples_per_file} samples each..."
        )
        for _ in range(self.n_files):
            print(f"...Generating file {_ + 1} of {self.n_files}...")
            file_id = abs(hash(str(time.time()))) % (10**6)
            all_patches, all_patch_nbrs = [], []
            all_pts, all_pt_nbrs = [], []

            for (pt, patch), (pt_nbr, patch_nbr) in lulc_generator.generate_patches(
                self.samples_per_file, create_pairs=True, pair_distance_meters=8_000
            ):
                all_patches.append(patch)
                all_patch_nbrs.append(patch_nbr)
                all_pts.append(pt)
                all_pt_nbrs.append(pt_nbr)

            out_shape = len(all_patches), 1, self.patch_size, self.patch_size

            all_patches = np.array(all_patches).reshape(*out_shape)
            all_patch_nbrs = np.array(all_patch_nbrs).reshape(*out_shape)

            patches_array = np.stack([all_patches, all_patch_nbrs], axis=-1)

            # For Copernicus LC data, 255 is a "no data" value
            has_nones = np.any(
                np.isnan(patches_array) | patches_array == 255, axis=(1, 2, 3, 4)
            )

            if np.sum(has_nones) > 0:
                print(
                    f"Found {np.sum(has_nones)} patches with NaNs or 255s; removing them..."
                )

            # At this point, the shape is (N, 1, ny, nx, 2) where the 2 is for
            # the anchor-neighbor pair.
            patches_array = patches_array[~has_nones]
            patches_array = np.vectorize(self.int_map.get)(patches_array)

            # Using the point lists, use the DEM generator's extract_patch
            # method to get the DEM images for each point, check them for NaNs
            # and then concat them with the land cover data.
            dem_patches = []
            dem_patches_nbr = []

            for pt, pt_nbr in zip(all_pts, all_pt_nbrs):
                dem_patch = dem_generator.extract_patch(pt)
                dem_patch_nbr = dem_generator.extract_patch(pt_nbr)

                if dem_patch is None or dem_patch_nbr is None:
                    raise ValueError("DEM patch is None")

                if dem_patch is not None and dem_patch_nbr is not None:
                    dem_patches.append(dem_patch)
                    dem_patches_nbr.append(dem_patch_nbr)

            dem_patches = np.array(dem_patches).reshape(*out_shape)
            dem_patches_nbr = np.array(dem_patches_nbr).reshape(*out_shape)

            dem_patches_array  = np.stack([dem_patches, dem_patches_nbr], axis=-1)

            # Here, shape should be (N, 2, ny, nx, 2) where the second axis
            # runs over [landcover, DEM] and the last axis runs over [anchor, neighbor]
            patches_array = np.concatenate([patches_array, dem_patches_array], axis=1)

            filename = (
                f"lulc-patches-pairs-{self.patch_size}x{self.patch_size}-{file_id}.npy"
            )
            filepath = os.path.join(c.TMP_DIR, filename)

            np.save(filepath, patches_array)
            key = f"train/{filename}"
            transfer.upload_file(key, filepath)
            self.uploaded_filekeys.append(filename)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.uploaded_filekeys = sum([x.uploaded_filekeys for x in inputs], [])
        print(f"{len(self.uploaded_filekeys)} files have been uploaded to S3")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PreprocessLandCoverFlow()
