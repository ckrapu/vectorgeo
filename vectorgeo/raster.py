from . import constants as c

import h3
import numpy as np
import os
import logging
import rasterio

from pyproj import Transformer
from rasterstats import zonal_stats
from tqdm import trange


def normalize_dem(dem_batch):
    """
    Takes in batch of DEM data with shape ()... H, W) and normalizes it to 
    be between 0-1
    """

    # Get the indices of the last two dimensions
    idx_pair = tuple(np.arange(dem_batch.ndim - 2, dem_batch.ndim))

    minima = dem_batch.min(axis=idx_pair, keepdims=True)
    maxima = dem_batch.max(axis=idx_pair, keepdims=True)
    
    return (dem_batch - minima) / (1e-6 + maxima - minima)

def extend_negatives(xs):
    """
    Takes in arrays of anchor-neighbor pairs and extends them to include
    negative examples. The negative examples are simply randomly shuffled
    versions of the neighbors. This assumes that the shape is
    (N, C, H, W, 2) where C is the number of classes/channels and the last axis
    is for anchor-neighbor pairs.
    """

    xs_neg = xs[..., 1:2].copy()
    np.random.shuffle(xs_neg)
    xs = np.concatenate([xs, xs_neg], axis=-1)

    return xs


def unpack_array(xs):
    """
    Provides transformations to take arrays in a more memory-efficient storage format and expands them into
    the format required for model ingestion. The data starts out with integer values and only with anchor-neighbor
    pairs; we randomly shuffle the neighbors to remove any spatial correlation and add these on as the distant
    images. We then convert the integer values to one-hot encoding, and finally we swap the second and fourth axes
    to make the data compatible with Keras.

    :param xs: (N, C, H, W, K) array of integers
    """

    xs = extend_negatives(xs)

    N, C, H, W, K = xs.shape
    xs_one_hot = np.zeros((N, c.LC_N_CLASSES, H, W, K))

    for i in range(c.LC_N_CLASSES):
        xs_one_hot[
            :,
            i,
        ] = (
            (xs == i).squeeze().astype(int)
        )

    # Initial shape: (N, 3, H, W, K)
    # Final shape: (N, 3, K, H, W)
    xs_one_hot = np.transpose(xs_one_hot, (0, 4, 1, 2, 3))
    return xs_one_hot


class RasterExtractor:
    """
    Parent class for extracting data from a raster;
    use the child classes in practice.
    """

    def __init__(self, raster_path, gdf=None, full_load=True):
        """
        Arguments:
        ----------
        raster_path: str
            Path to the raster file to extract patches from
        gdf: GeoDataFrame
            Masking geometry to use for sampling patches; patches will only be sampled
            from locations that are within the masking geometry.
        full_load: bool
            Whether to load the entire raster into memory versus using a windowed reading method
        """
        
        self.raster_path = raster_path
        self.gdf = gdf

        self.gdf_proj = self.gdf.to_crs("epsg:3857")

        self.transform_to_geo = Transformer.from_crs(3857, 4326)

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Read raster metadata upon initialization
        self.bounds, self.affine, self.no_data_value = self._read_raster_metadata()

        if full_load:
            self.raster_array = self._load_raster_data()

    def _read_raster_metadata(self):
        with rasterio.open(self.raster_path) as src:
            bounds = src.bounds
            affine = src.transform
            no_data_value = src.nodatavals[0]
        self.logger.info("Raster metadata read successfully.")
        return bounds, affine, no_data_value

    def _load_raster_data(self):
        with rasterio.open(self.raster_path) as src:
            return src.read(1)

    def _random_point_on_land(self):
        while True:
            pt = self.gdf.sample_points(1).geometry.iloc[0].coords[0]
            lng, lat = pt
            if c.LC_MIN_LAT < lat < c.LC_MAX_LAT:
                yield lng, lat
            else:
                continue

    def _random_pair_on_land(self, min_radius_meters, max_radius_meters):
        """
        Returns pairs of points within radius_meters of each other
        that are both on land and which are in a geographic CRS. To make sure
        that there is no overlap between imagery extracted from the two points,
        we sample from an annular region between min_radius_meters and max_radius_meters.

        """

        # Generate a random point on land
        while True:
            pt = self.gdf_proj.sample_points(1).geometry.iloc[0].coords[0]
            x, y = pt

            # Order of outputs is backwards, see https://github.com/pyproj4/pyproj/issues/510
            lat, lng = self.transform_to_geo.transform(x, y)

            if c.LC_MIN_LAT < lat < c.LC_MAX_LAT:
                # Sample point randomly from disk within radius_meters
                # of the first point
                theta, r = np.random.uniform(0, 2 * np.pi), np.random.uniform(
                    min_radius_meters, max_radius_meters
                )
                dx, dy = r * np.cos(theta), r * np.sin(theta)

                # Convert back to lat/long
                lat_nbr, lng_nbr = self.transform_to_geo.transform(x + dx, y + dy)

                yield (lng, lat), (lng_nbr, lat_nbr)
            else:
                continue


class RasterPatches(RasterExtractor):
    """
    Class for sampling patches from raster data in accordance with triplet loss
    model requirements (anchor, proximal positive/neighbor, and distant negative examples).

    Basic example usage:

        ```python
        from vectorgeo.landcover import RasterPatches
        raster_path = "data/train/landcover.tif"
        gdf = gpd.read_file("data/train/landcover.gpkg")
        patch_size = 64
        raster_patches = RasterPatches(raster_path, gdf, patch_size)

        # Generate 1000 patches
        for (lng, lat), patch in raster_patches.generate_patches(1000):
            print(lng, lat, patch)
        ```



    """

    def __init__(
        self,
        raster_path,
        gdf,
        patch_size,
        pixel_size,
        sameness_threshold=0.95,
        full_load=True,
    ):
        '''
        Arguments:
        ----------
        raster_path: str
            Path to the raster file to extract patches from
        gdf: GeoDataFrame
            Masking geometry to use for sampling patches; patches will only be sampled
            from locations that are within the masking geometry.
        patch_size: int
            Size of the patches to extract, in pixels
        pixel_size: float
            Size of each raw pixel in meters
        sameness_threshold: float
            Threshold for determining whether a patch is too homogeneous to be useful
        full_load: bool
            Whether to load the entire raster into memory versus using a windowed reading method
        '''
        
        super().__init__(raster_path, gdf, full_load=full_load)
        self.patch_size = patch_size  # e.g., 64 for 64x64 patches
        self.pixel_size = pixel_size  # meters
        self.full_load = full_load  # Whether to load the entire raster into memory

        # If an image is too homogeneous, we don't want to use it
        self.sameness_threshold = sameness_threshold

    def extract_patch(self, lng_lat):
        """
        Extract a patch of raster centered around the given coordinates.
        Assumes self.raster_array is a 2D numpy array.
        """
        lng, lat = lng_lat
        row, col = rasterio.transform.rowcol(self.affine, lng, lat)

        half_size = self.patch_size // 2

        if self.full_load:
            patch = self.raster_array[
                row - half_size : row + half_size, col - half_size : col + half_size
            ]
        else:
            # Read from disk
            with rasterio.open(self.raster_path) as src:
                patch = src.read(
                    1,
                    window=(
                        (row - half_size, row + half_size),
                        (col - half_size, col + half_size),
                    ),
                )

        if patch is None:
            return None
        try:
            homogeneity = np.unique(patch, return_counts=True)[1].max() / patch.size
        except ValueError:
            print(f"ValueError encountered, patch is {patch}")

        if (
            patch.shape == (self.patch_size, self.patch_size)
            and homogeneity < self.sameness_threshold
        ):
            return patch
        else:
            return None  # Invalid patch, e.g., out of raster bounds

    def generate_patches(self, N, create_pairs=False, pair_distance_meters=8000):
        """
        Can generate either single (point, patch) results or pairs of neighboring
        data pairs.
        """

        if create_pairs:
            min_dist = self.pixel_size * self.patch_size
            max_dist = min_dist + pair_distance_meters
            points_generator = self._random_pair_on_land(min_dist, max_dist)
        else:
            points_generator = self._random_point_on_land()

        for _ in trange(N):
            if create_pairs:
                point, point_nbr = next(points_generator)
                patch, patch_nbr = self.extract_patch(point), self.extract_patch(
                    point_nbr
                )
                if (
                    patch is not None
                    and patch_nbr is not None
                    and np.isfinite(patch).all()
                    and np.isfinite(patch_nbr).all()
                ):
                    yield (point, patch), (point_nbr, patch_nbr)
            else:
                point = next(points_generator)
                patch = self.extract_patch(point)
                if patch is not None and np.isfinite(patch).all():
                    yield (point, patch)

    def h3_to_patch(self, h3_index):
        """
        Extract a patch of raster centered around the given H3 index.
        Assumes self.raster_array is a 2D numpy array.
        """
        lat, lng = h3.h3_to_geo(h3_index)
        return self.extract_patch((lng, lat))
    
    def extend(self, raster_path, pixel_size):
        """
        Adds another raster to the stack of rasters for which patches should be extracted.     
        """
        
