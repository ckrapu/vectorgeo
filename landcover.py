
import constants as c

import h3
import numpy as np
import os
import logging
import rasterio

from pyproj import Transformer
from rasterstats import zonal_stats
from tqdm import trange

def unpack_array(xs):
    '''
    Provides transformations to take arrays in a more memory-efficient storage format and expands them into
    the format required for model ingestion. The data starts out with integer values and only with anchor-neighbor
    pairs; we randomly shuffle the neighbors to remove any spatial correlation and add these on as the distant
    images. We then convert the integer values to one-hot encoding, and finally we swap the second and fourth axes
    to make the data compatible with Keras.

    :param xs: (N, C, H, W, K) array of integers
    '''
    xs_neg = xs[..., 1:2].copy()
    np.random.shuffle(xs_neg)
    xs = np.concatenate([xs, xs_neg], axis=-1)

    N, C, H, W, K = xs.shape
    xs_one_hot = np.zeros((N, c.LC_N_CLASSES, H, W, K))

    for i in range(c.LC_N_CLASSES):
        xs_one_hot[:, i,] = (xs == i).squeeze().astype(int)

    # Gets dimensions (N, 3, H, W, K)
    xs_one_hot = np.swapaxes(xs_one_hot, 1, 4)
    return xs_one_hot

class LandCoverExtractor:
    """
    Parent class for extracting landcover data from a raster;
    use the child classes in practice.
    """
    def __init__(self, lc_path, gdf):
        self.lc_path = lc_path
        self.gdf = gdf

        self.gdf_proj = self.gdf.to_crs('epsg:3857')

        self.transform_to_geo  = Transformer.from_crs(3857, 4326)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Read raster metadata upon initialization
        self.bounds, self.affine, self.no_data_value = self._read_raster_metadata()
        self.raster_array = self._load_raster_data()

    def _read_raster_metadata(self):
        with rasterio.open(self.lc_path) as src:
            bounds = src.bounds
            affine = src.transform
            no_data_value = src.nodatavals[0]
        self.logger.info("Raster metadata read successfully.")
        return bounds, affine, no_data_value
    
    def _load_raster_data(self):
        with rasterio.open(self.lc_path) as src:
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
        that are both on land and which are in a geographic CRS.
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
                theta = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(min_radius_meters, max_radius_meters)
                dx = r * np.cos(theta)
                dy = r * np.sin(theta)

                # Convert back to lat/long
                lat_nbr, lng_nbr = self.transform_to_geo.transform(x + dx, y + dy)

                yield (lng, lat), (lng_nbr, lat_nbr)
            else:
                continue

class LandCoverProportions(LandCoverExtractor):

    def _get_proportions(self, lng_lat, radii, resolution):
        lng, lat = lng_lat
        h3_index = h3.geo_to_h3(lat, lng, resolution=resolution)

        ring_sets = [h3.k_ring(h3_index, r) for r in radii]
        hexagons = h3.compact(set().union(*ring_sets))
        
        geometries = [
            {
                'type': 'Polygon',
                'coordinates': [[(coord[1], coord[0]) for coord in h3.h3_to_geo_boundary(hexagon)]]
            }
            for hexagon in hexagons
        ]
        
        zs = zonal_stats(geometries, self.raster_array, affine=self.affine, categorical=True, all_touched=True)
        aggregated_counts = {}
        for stat in zs:
            for key, value in stat.items():
                if key != 'nodata':
                    aggregated_counts[key] = aggregated_counts.get(key, 0) + value

        total_pixels = sum(aggregated_counts.values())
        proportions = {key: value/total_pixels for key, value in aggregated_counts.items()}
        
        self.logger.info(f"Calculated proportions for point lat={lat:.4f}, long={lng:.4f} with {len(hexagons)} cells and radius {radii}: {proportions}")
        return h3_index, proportions

    def _expand_proportions(self, prop_dict):
        return {
            f'ring_{h3_ring}_{k}': inner_dict.get(k, 0)
            for h3_ring, inner_dict in prop_dict.items()
            for k in c.LC_LEGEND.keys()
        }

    def generate_proportions(self, N, ring_tuples, resolution):
        points_generator = self._random_point_on_land()
        
        for _ in trange(N):
            point = next(points_generator)
            proportions = {
                radii[-1]: self._get_proportions(point, radii, resolution)[1]
                for radii in ring_tuples
            }
            h3_idx = self._get_proportions(point, ring_tuples[0], resolution)[0]  # Assuming the index is same for all radii
            yield h3_idx, self._expand_proportions(proportions)

class LandCoverPatches(LandCoverExtractor):

    def __init__(self, lc_path, gdf, patch_size, sameness_threshold = 0.95):
        super().__init__(lc_path, gdf)
        self.patch_size = patch_size  # e.g., 64 for 64x64 patches
        self.pixel_size = 100 # meters

        # If an image is too homogeneous, we don't want to use it
        self.sameness_threshold = sameness_threshold

    def _extract_patch(self, lng_lat):
        """
        Extract a patch of raster centered around the given coordinates.
        Assumes self.raster_array is a 2D numpy array.
        """
        lng, lat = lng_lat
        row, col = rasterio.transform.rowcol(self.affine, lng, lat)

        half_size = self.patch_size // 2
        patch = self.raster_array[row - half_size:row + half_size,
                                  col - half_size:col + half_size]
        
        homogeneity = np.unique(patch, return_counts=True)[1].max() / patch.size

        if patch.shape == (self.patch_size, self.patch_size) and homogeneity < self.sameness_threshold:
            return patch
        else:
            return None  # Invalid patch, e.g., out of raster bounds

    def generate_patches(self, N, create_pairs=False, pair_distance_meters=8000):
        if create_pairs:
            min_dist = self.pixel_size * self.patch_size
            max_dist = min_dist + pair_distance_meters 
            points_generator = self._random_pair_on_land(min_dist, max_dist)
        else:
            points_generator = self._random_point_on_land()
        
        for _ in trange(N):
            if create_pairs:
                point, point_nbr = next(points_generator)
                patch, patch_nbr = self._extract_patch(point), self._extract_patch(point_nbr)
                if patch is not None and patch_nbr is not None and np.isfinite(patch).all() and np.isfinite(patch_nbr).all():
                    yield (point, patch), (point_nbr, patch_nbr)
            else:
                point = next(points_generator)
                patch = self._extract_patch(point)
                if patch is not None and np.isfinite(patch).all():
                    yield (point, patch)

    def h3_to_patch(self, h3_index):
        """
        Extract a patch of raster centered around the given H3 index.
        Assumes self.raster_array is a 2D numpy array.
        """
        lat, lng = h3.h3_to_geo(h3_index)
        return self._extract_patch((lng, lat))

def upload_df_s3(df, filename, vpath, s3):
    print(f"Beginning to upload {len(df)} rows to {filename}...")
    
    df.to_parquet(filename)
    s3.meta.client.upload_file(filename, c.S3_BUCKET, f'{vpath}/{filename}')
    os.remove(filename)

def upload_npy_s3(arr, filename, vpath, s3):
    print(f"Beginning to upload {len(arr)} rows to {filename}...")
    
    np.save(filename, arr)
    s3.meta.client.upload_file(filename, c.S3_BUCKET, f'{vpath}/{filename}')
    os.remove(filename)
