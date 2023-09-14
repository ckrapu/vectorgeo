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
from vectorgeo import landcover as lc
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

        print(f"Creating patch generator...")
        data_generator = lc.LandCoverPatches(
            c.LC_LOCAL_PATH, self.world_gdf, self.patch_size, full_load=self.full_load
        )

        print(
            f"Generating {self.n_files} files with {self.samples_per_file} samples each..."
        )
        for _ in range(self.n_files):
            print(f"...Generating file {_ + 1} of {self.n_files}...")
            file_id = abs(hash(str(time.time()))) % (10**6)
            all_patches, all_patch_nbrs = [], []
            all_pts, all_pt_nbrs = [], []

            for (pt, patch), (pt_nbr, patch_nbr) in data_generator.generate_patches(
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

            patches_array = patches_array[~has_nones]
            patches_array = np.vectorize(self.int_map.get)(patches_array)

            filename = (
                f"lulc-patches-pairs-{self.patch_size}x{self.patch_size}-{file_id}.npy"
            )
            filepath = os.path.join(c.TMP_DIR, filename)

            np.save(filepath, patches_array)
            key = f"landcover/{filename}"
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
