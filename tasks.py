import boto3
import psycopg2
import re
import yaml

from invoke import task
from psycopg2.extras import RealDictCursor
from pyproj import Transformer
from vectorgeo.constants import (
    S3_BUCKET,
)  # Assuming this is where your S3_BUCKET is defined
from vectorgeo import transfer

@task
def s3_size(c):
    """
    Returns the total size of all objects stored in the project S3 bucket.
    """
    s3 = boto3.client("s3")
    total_size = 0

    for obj in s3.list_objects_v2(Bucket=S3_BUCKET)["Contents"]:
        total_size += obj["Size"]

    # Convert to GB
    total_size /= 1e9

    print(f"Total size of all objects in bucket {S3_BUCKET}: {total_size:.2f} GB")


@task
def s3_delete_train(c):
    """
    Deletes all files in the S3 bucket with keys matching the pattern "train/".
    """
    s3 = boto3.client("s3")
    if "Contents" not in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="train/"):
        print("No files found for deletion!")
        return

    for obj in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="train/")["Contents"]:
        s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
        print(f"    Deleted {obj['Key']}")

@task
def s3_download_vectors(c):
    """
    Downloads all files in the S3 bucket with keys matching the pattern "vector/" to the directory 'tmp/'.
    """

    prefix = "vectors/"
    s3 = boto3.client("s3")
    if "Contents" not in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix):
        print("No files found for download!")
        return

    for obj in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)["Contents"]:
        filename = obj["Key"].split("/")[-1]
        transfer.download_file(obj["Key"], f"tmp/{filename}")
        print(f"    Downloaded {obj['Key']}")



@task
def aurora_summary(c, secrets_path="secrets.yml"):
    """
    Fetch and display summary statistics for the Aurora PostgreSQL table 'vectorgeo'.
    """

    # Read the secrets file
    secrets = yaml.load(open(secrets_path), Loader=yaml.FullLoader)

    # Database connection parameters (replace these with your actual credentials)
    params = {
        "user": secrets["aurora_user"],
        "password": secrets["aurora_password"],
        "host": secrets["aurora_url"],
        "port": 5432,
    }

    # Connect to the database
    conn = psycopg2.connect(**params)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch the number of rows in the table
    cur.execute("SELECT COUNT(*) FROM vectorgeo;")
    row_count = cur.fetchone()["count"]
    print(f"Number of rows in table 'vectorgeo': {row_count}")

    # Fetch the bounding box in EPSG:3857
    cur.execute(
        """
        SELECT ST_Extent(geom) AS bbox
        FROM vectorgeo;
    """
    )
    bbox_3857 = cur.fetchone()["bbox"]
    print(f"Bounding box in EPSG:3857: {bbox_3857}")

    # Transform the bounding box to lat-long (EPSG:4326)
    transformer = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    print(bbox_3857)
    # We need to split on either the comma or the space
    # so we use regex to do that
    minx, miny, maxx, maxy = map(float, re.split(r"[,\s]+", bbox_3857[4:-1]))
    min_lon, min_lat = transformer.transform(minx, miny)
    max_lon, max_lat = transformer.transform(maxx, maxy)

    print(
        f"Bounding box in lat-long (EPSG:4326): ({min_lat}, {min_lon}, {max_lat}, {max_lon})"
    )

    # Close the database connection
    cur.close()
    conn.close()
