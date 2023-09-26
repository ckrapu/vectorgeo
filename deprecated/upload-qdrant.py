import boto3
import numpy as np
import os
import pandas as pd
import sys
import time
import yaml

from vectorgeo.transfer import download_file
from vectorgeo import constants as c
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

CHECK_DELAY = 60
UPLOAD_DELAY = 1.0
WIPE_QDRANT = True
EMBED_DIM = 16
MAX_VEC_UPLOADED = 3_000_000

# Load secrets (adjust the path as necessary)
secrets = yaml.load(open("secrets.yml"), Loader=yaml.FullLoader)

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=secrets["aws_access_key_id"],
    aws_secret_access_key=secrets["aws_secret_access_key"],
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=secrets["qdrant_url"], api_key=secrets["qdrant_api_key"]
)

qdrant_client.recreate_collection(
    collection_name=c.QDRANT_COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE, on_disk=True),
    on_disk_payload=True,
    hnsw_config=models.HnswConfigDiff(on_disk=True),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
        )
    ),
)

qdrant_client.create_payload_index(
    collection_name=c.QDRANT_COLLECTION_NAME, field_name="location", field_schema="geo"
)

# Specify your bucket name and prefix
bucket_name = c.S3_BUCKET
prefix = "vectors/"

# Create set of files already run
checked_keys = set()

# Disable indexing until after upload is finished
print("Disabling indexing until upload is finished")
qdrant_client.update_collection(
    collection_name=c.QDRANT_COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
)
n_rows_uploaded = 0
try:
    while True:
        time.sleep(CHECK_DELAY)

        # List all Parquet files in the S3 bucket with the specified prefix

        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        print(f"Found {len(objects['Contents'])} files in S3")

        # Filter out the files that have already been run
        objects["Contents"] = [
            obj
            for obj in objects["Contents"]
            if obj["Key"] not in checked_keys and obj["Key"].endswith(".parquet")
        ]
        print(f"Found {len(objects['Contents'])} files to run")

        for obj in objects["Contents"]:
            print(f"...Downloading {obj['Key']} from S3")
            basename = os.path.basename(obj["Key"])
            local_path = os.path.join(c.TMP_DIR, basename)
            download_file(obj["Key"], local_path)

            # Load the data into a Pandas DataFrame
            df = pd.read_parquet(local_path)

            # Extract vectors and other necessary information
            print(f"...Uploading {obj['Key']} to Qdrant")
            for df_piece in np.array_split(df, 5):
                points = [
                    PointStruct(
                        id=row["id"],
                        vector=row["vector"].tolist(),
                        payload={"location": {"lon": row["lng"], "lat": row["lat"]}},
                    )
                    for _, row in df_piece.iterrows()
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
                            points=points,
                        )
                        uploaded = True
                        n_rows_uploaded += len(points)
                    except Exception as e:
                        print(f"Failed to upload batch with exception {e}")
                        time.sleep(delay)
                        delay = delay * 4

                # To avoid wrecking the cluster, we wait a bit between batches
                print(f"Uploaded {n_rows_uploaded} rows so far")
                if n_rows_uploaded > MAX_VEC_UPLOADED:
                    print(f"Uploaded {n_rows_uploaded} vectors. Stopping upload.")
                    raise KeyboardInterrupt

                time.sleep(UPLOAD_DELAY)

            # Add the file to the set of files that have already been run
            checked_keys.add(obj["Key"])

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Re-enabling indexing and quitting...")

qdrant_client.update_collection(
    collection_name=c.QDRANT_COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
)
