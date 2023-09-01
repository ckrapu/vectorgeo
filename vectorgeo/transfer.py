import yaml
from . import constants as c
import boto3
import os

with open(c.SECRETS_PATH) as f:
    secrets = yaml.safe_load(f)

def download_file(key, local_path):
    """
    Downloads a file from S3 to a local path.
    """

    if not os.path.exists(local_path):
        s3 = boto3.resource('s3', region_name=c.S3_REGION,
                            aws_access_key_id=secrets['aws_access_key_id'],
                            aws_secret_access_key=secrets['aws_secret_access_key'])
        print(f"Downloading {key} to {local_path}...")
        s3.Bucket(c.S3_BUCKET).download_file(key, local_path)
        print(f"Downloaded complete!")
    else:
        print(f"File {local_path} already exists; skipping download")

def upload_file(key, local_path):
    """
    Uploads a file from a local path to S3.
    """
    s3 = boto3.resource('s3', region_name=c.S3_REGION,
                        aws_access_key_id=secrets['aws_access_key_id'],
                        aws_secret_access_key=secrets['aws_secret_access_key'])
    
    s3.meta.client.upload_file(local_path, c.S3_BUCKET, key)
    print(f"Uploaded {local_path} to {key}")

def ls_s3(path):
    """
    Lists the contents of an S3 path.
    """
    s3 = boto3.resource('s3', region_name=c.S3_REGION,
                        aws_access_key_id=secrets['aws_access_key_id'],
                        aws_secret_access_key=secrets['aws_secret_access_key'])
    bucket = s3.Bucket(c.S3_BUCKET)
    return [obj.key for obj in bucket.objects.filter(Prefix=path)]