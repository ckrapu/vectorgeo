# Overview

## Installation
Clone the repository, and run the following to make a new virtualenv in `vg-venv`:

```
python3 -m venv vg-venv
source vg-venv/bin/activate
pip install -r requirements.txt
```

You will also need `s3fs` to exchange files with the S3 bucket. 
For Mac, you may need to uninstall an existing macfuse installation and reinstall it with the `--cask` option:
```
brew uninstall macfuse 
brew install --cask macfuse
brew install gromgit/fuse/s3fs-mac
```

For Debian/Ubunutu:
```
sudo apt-get install s3fs
```

For all other platforms, see `https://github.com/s3fs-fuse/s3fs-fuse`

Next, you must ensure that the appropriate AWS environment variables are set by running
```
bash setup.sh
```

Finally, mount the S3 bucket as a local directory:
```
mkdir lql-data
s3fs lql-data ./lql-data -o use_cache=/tmp -o allow_other -o uid=$(id -u) -o mp_umask=002 -o multireq_max=5 -o use_path_request_style -o url=https://s3.us-east-1.amazonaws.com
```


# Read the YAML file (.secrets.yml) and get the values for aws_access_key_id and aws_secret_access_key
AWS_ACCESS_KEY_ID=$(yq eval 'aws_access_key_id' .secrets.yml)
AWS_SECRET_ACCESS_KEY=$(yq eval 'aws_secret_access_key' .secrets.yml)

# Check if the values are not empty
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Error: aws_access_key_id or aws_secret_access_key not found in .secrets.yml"
  exit 1
fi

# Set the environment variables
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

echo "Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY have been set."
```

## Workflow

### Testing
```
python flow_preprocess.py --no-pylint run --n_files=2 --n_jobs=1
python flow_train.py --no-pylint run --epochs=1 --n_train_files=2 

```

```

## Land cover
`landcover.py` draws random samples of land cover proportion vectors for randomly sampled points within national borders and for multiple `h3` k-rings. The output from `python landcover.py` is a sequence of parquet files located on S3, each of which has rows comprising the per-ring vector of land cover proportions across all land cover classes.

