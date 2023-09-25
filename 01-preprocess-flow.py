from metaflow import (
    FlowSpec,
    Parameter,  # pylint: disable=no-name-in-module
    retry,
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
        help="Whether or not to attempt to load full raster files into memory",
        default=False,
    )

    @step
    def start(self):
        """
        Loads the boundary shapefile for all countries and starts the sampler.
        """

        n_samples_total = self.n_files * self.samples_per_file
        print(f"Generating {n_samples_total} samples total across {self.n_files} files")

        # Shapefile with a single geometry indicating boundaries / coastlines for all countries
        world_key, world_path = "misc/world.gpkg", os.path.join(c.TMP_DIR, "world.gpkg")
        transfer.download_file(world_key, world_path)

        self.world_gdf = gpd.read_file(world_path, driver="GPKG")

        self.job_ids = range(self.n_jobs)

        # Use these to convert the ungainly string-valued land cover classes to integers
        self.int_map = {x: i for i, x in enumerate(c.LC_LEGEND.keys())}

        lc_key = "raw/" + c.COPERNICUS_LC_KEY
        transfer.download_file(lc_key, c.LC_LOCAL_PATH)

        dem_key = "raw/" + c.GMTED_DEM_KEY
        transfer.download_file(dem_key, c.DEM_LOCAL_PATH)

        self.next(self.run_samplers, foreach="job_ids")

    @retry
    @step
    def run_samplers(self):
        """
        Run the sampler in parallel across different jobs to iteratively samples of neighboring
        land cover patches and store them as Numpy arrays on S3.
        """
        self.uploaded_filekeys = []
        
        # Initialize patch generators
        print("Creating patch generator...")
        lulc_generator = lc.RasterPatches(
            c.LC_LOCAL_PATH, self.world_gdf, self.patch_size, full_load=self.full_load
        )
        dem_generator = lc.RasterPatches(
            c.DEM_LOCAL_PATH, self.world_gdf, self.patch_size, full_load=self.full_load
        )

        # Generate files with samples
        print(f"Generating {self.n_files} files with {self.samples_per_file} samples each...")
        for file_num in range(self.n_files):
            print(f"...Generating file {file_num + 1} of {self.n_files}...")
            
            # Initialize variables
            file_id = abs(hash(str(time.time()))) % (10 ** 6)
            lc_patches, lc_patch_nbrs = [], []
            all_pts, all_pt_nbrs = [], []
            dem_patches, dem_patches_nbr = [], []

            # Generate patches
            for (pt, patch), (pt_nbr, patch_nbr) in lulc_generator.generate_patches(
                self.samples_per_file, create_pairs=True, pair_distance_meters=8_000
            ):
                dem_patch = dem_generator.extract_patch(pt)
                dem_patch_nbr = dem_generator.extract_patch(pt_nbr)

                # Skip if any patch is None
                if any(x is None for x in [dem_patch, dem_patch_nbr, patch, patch_nbr]):
                    continue

                lc_patches.append(patch)
                lc_patch_nbrs.append(patch_nbr)
                all_pts.append(pt)
                all_pt_nbrs.append(pt_nbr)
                dem_patches.append(dem_patch)
                dem_patches_nbr.append(dem_patch_nbr)

            # Reshape and stack arrays
            out_shape = len(lc_patches), 1, self.patch_size, self.patch_size
            lc_patches = np.array(lc_patches).reshape(*out_shape)
            lc_patch_nbrs = np.array(lc_patch_nbrs).reshape(*out_shape)
            patches_array = np.stack([lc_patches, lc_patch_nbrs], axis=-1)

            # Remove patches with NaNs or 255s
            has_nones = np.any(np.isnan(patches_array) | (patches_array == 255), axis=(1, 2, 3, 4))
            if np.sum(has_nones) > 0:
                print(f"Found {np.sum(has_nones)} patches with NaNs or 255s; removing them...")
            patches_array = patches_array[~has_nones]
            patches_array = np.vectorize(self.int_map.get)(patches_array)

            # Reshape and stack DEM arrays
            dem_patches = np.array(dem_patches).reshape(*out_shape)
            dem_patches_nbr = np.array(dem_patches_nbr).reshape(*out_shape)
            dem_patches_array = np.stack([dem_patches, dem_patches_nbr], axis=-1)

            # Concatenate landcover and DEM arrays
            patches_array = np.concatenate([patches_array, dem_patches_array], axis=1)

            # Save and upload the file
            filename = f"lc-dem-patches-pairs-{self.patch_size}x{self.patch_size}-{file_id}.npy"
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
