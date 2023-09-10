import os

# General project settings
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
TMP_DIR      = os.path.join(BASE_DIR, 'tmp')
CONFIGS_DIR  = os.path.join(BASE_DIR, 'configs')
SECRETS_PATH = os.path.join(BASE_DIR, 'secrets.yml')
S3_BUCKET    = 'lql-data'
S3_REGION    = 'us-east-1'
EMBED_VPATH  = 'embeddings' 

QDRANT_COLLECTION_NAME = 'vectorgeo'

# Landcover-specific settings
LC_LEGEND = {
    0: 'Unknown',
    20: 'Shrubs',
    30: 'Herbaceous vegetation',
    40: 'Cultivated and managed vegetation / agriculture',
    50: 'Urban / built up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    100: 'Moss and lichen',
    111: 'Closed forest, evergreen needle leaf',
    112: 'Closed forest, evergreen broad leaf',
    113: 'Closed forest, deciduous needle leaf',
    114: 'Closed forest, deciduous broad leaf',
    115: 'Closed forest, mixed',
    116: 'Closed forest, not matching any of the other definitions',
    121: 'Open forest, evergreen needle leaf',
    122: 'Open forest, evergreen broad leaf',
    123: 'Open forest, deciduous needle leaf',
    124: 'Open forest, deciduous broad leaf',
    125: 'Open forest, mixed',
    126: 'Open forest, not matching any of the other definitions',
    200: 'Oceans, seas',
}
LC_MAX_LAT = 75.0
LC_MIN_LAT = -75.0
COPERNICUS_LC_KEY = 'PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif'
LC_H3_RES = 10
LC_N_CLASSES = 23
LC_LOCAL_PATH = os.path.join(TMP_DIR, COPERNICUS_LC_KEY)

H3_STATE_FILENAME = 'h3-state.json'
H3_STATE_KEY = f'misc/{H3_STATE_FILENAME}'

"""
Use code like this to make the ring sets:
ring_tuples = []

x = 0
for increment in range(1, 8):
    ring_tuples.append(list(range(x, x+increment)))
    x += increment
"""

LC_K_RING_SETS = [[0],
 [1, 2],
 [3, 4, 5],
 [6, 7, 8, 9],
 [10, 11, 12, 13, 14],
 [15, 16, 17, 18, 19, 20],
 [21, 22, 23, 24, 25, 26, 27]]


lambda_fn_url = "https://bnb7i5k7fhodm6trqwomllqdy40zagpg.lambda-url.us-east-1.on.aws/"