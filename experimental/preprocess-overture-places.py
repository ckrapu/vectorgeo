import os
import pandas as pd

from collections import Counter
from h3 import h3
from vectorgeo.constants import TMP_DIR

resolution = 7
top_n      = 1000

def calculate_h3_index(row, resolution=7):
    minx, maxx, miny, maxy = row['minx'], row['maxx'], row['miny'], row['maxy']
    latitude = (miny + maxy) / 2
    longitude = (minx + maxx) / 2
    return h3.geo_to_h3(latitude, longitude, resolution)

# Initialize the global counter for place types
global_counter = Counter()

# Initialize the counter-of-counters for H3 indices
h3_counter = Counter()

# Directory containing the parquet files
places_dir = "/home/ubuntu/vectorgeo/tmp/theme=places/type=place"

print("Starting to process parquet files...")

# Iterate over all files in the directory
for place_filename in os.listdir(places_dir)[0:2]:
    filepath = os.path.join(places_dir, place_filename)
    
    if not os.path.isfile(filepath):
        continue
    
    print(f"Processing file: {filepath}")
    
    places_df = pd.read_parquet(filepath)
    
    # Update the global counter for place types
    local_counts = places_df['categories'].apply(lambda x: x['main']).value_counts()
    global_counter.update(local_counts.to_dict())
    
    # Calculate H3 index and update the counter-of-counters
    places_df['h3_index'] = places_df['bbox'].apply(lambda x: calculate_h3_index(x, resolution=resolution))
    for h3_index, group_df in places_df.groupby('h3_index'):
        local_counts = group_df['categories'].apply(lambda x: x['main']).value_counts()
        if h3_index not in h3_counter:
            h3_counter[h3_index] = Counter()
        h3_counter[h3_index].update(local_counts.to_dict())

print("Finished processing parquet files.")

# Get the top 1000 most common place types and sort them alphabetically
top_n_places = [place for place, _ in global_counter.most_common(top_n)]
top_n_places = sorted(top_n_places)

print("Creating DataFrame with (h3_index, place_type, value_count) triplets...")

# Create a DataFrame with (h3_index, place_type, value_count) triplets
rows = []
for h3_index, inner_counter in h3_counter.items():
    for place_type, value_count in inner_counter.items():
        if place_type in top_n_places:
            rows.append({'h3_index': h3_index, 'place_type': place_type, 'value_count': value_count})

result_df = pd.DataFrame(rows)

# Save the DataFrame to disk
filepath = os.path.join(TMP_DIR, f"place-counts-h3-{resolution}.parquet")
result_df.to_parquet(filepath, index=False)

print(f"DataFrame saved to disk at {filepath}")
