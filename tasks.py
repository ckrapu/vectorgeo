from invoke import task
import boto3
from vectorgeo.constants import S3_BUCKET  # Assuming this is where your S3_BUCKET is defined

@task
def s3_size(c):
    """
    Returns the total size of all objects stored in the project S3 bucket.
    """
    s3 = boto3.client('s3')
    total_size = 0

    for obj in s3.list_objects_v2(Bucket=S3_BUCKET)['Contents']:
        total_size += obj['Size']

    # Convert to GB
    total_size /= 1e9

    print(f"Total size of all objects in bucket {S3_BUCKET}: {total_size:.2f} GB")

@task
def s3_delete_train(c):
    """
    Deletes all files in the S3 bucket with keys matching the pattern "train/".
    """
    s3 = boto3.client('s3')
    if 'Contents' not in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='train/'):
        print("No files found for deletion!")
        return
    
    for obj in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='train/')['Contents']:
        s3.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
        print(f"    Deleted {obj['Key']}")

# Add more tasks as needed
