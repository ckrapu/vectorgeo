import os
import boto3
import pandas as pd
import yaml
import time

from vectorgeo.transfer import download_file
from vectorgeo import constants as c
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

CHECK_DELAY = 60

# Load secrets (adjust the path as necessary)
secrets = yaml.load(open('secrets.yml'), Loader=yaml.FullLoader)

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id=secrets['aws_access_key_id'], aws_secret_access_key=secrets['aws_secret_access_key'])

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=secrets['qdrant_url'], 
    api_key=secrets['qdrant_api_key']
)

# Specify your bucket name and prefix
bucket_name = c.S3_BUCKET
prefix = 'vectors/'

# Create set of files already run
checked_keys = set()

while True:
    time.sleep(CHECK_DELAY)

    # List all Parquet files in the S3 bucket with the specified prefix

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    print(f"Found {len(objects['Contents'])} files in S3")


    # Filter out the files that have already been run
    objects['Contents'] = [
        obj for obj in objects['Contents']
        if obj['Key'] not in checked_keys
        and obj['Key'].endswith('.parquet')
    ]
    print(f"Found {len(objects['Contents'])} files to run")

    for obj in objects['Contents']:
        print(f"...Downloading {obj['Key']} from S3")
        local_path = os.path.join(c.TMP_DIR, obj['Key'])
        download_file(obj['Key'], local_path)
        
        # Load the data into a Pandas DataFrame
        df = pd.read_parquet(local_path)
        
        # Extract vectors and other necessary information
        print(f"...Uploading {obj['Key']} to Qdrant")
        points = [
            PointStruct(
                id=row['id'],
                vector=row['vector'],
                payload={"location": {"lon": row['lng'], "lat": row['lat']}}
            )
            for _, row in df.iterrows()
        ]
        
        # Batch the vectors and upload them to Qdrant
        # If we get a timeout error, back off up to T times with a delay that quadruples each time
        uploaded = False
        delay = 4
        while not uploaded:
            try:
                qdrant_client.upsert(
                    collection_name=c.QDRANT_COLLECTION_NAME,
                    wait=True,
                    points=points
                )
                uploaded = True
            except Exception as e:
                print(f"Failed to upload batch with exception {e}")
                time.sleep(delay)
                delay = delay * 4

        # Add the file to the set of files that have already been run
        checked_keys.add(obj['Key'])
