# Overview
`vectorgeo` performs the data engineering, model training, and inference routines for producing a globally-available geospatial vector embedding data product for use in machine learning for Earth & environment and geographic similarity search. Broadly, the architecture is composed of several parts:

- Data cleaning, extraction, and feature engineering is conducted in Metaflow using Copernicus Land Cover data
- Model training and inference is done using Keras in MetaFlow
- Results are served from Qdrant cloud, making use of the `geo_filter` functionality unavailable in Pinecone, Weaviate, or other vector DB providers
- A light web viewer (`vg_site/index.html`) interfaces with an AWS Lambda function to help visualize the results overlaid on a map.

## Installation
Clone the repository, and run the following to make a new virtualenv in `vg-venv`:

```
python3 -m venv vg-venv
source vg-venv/bin/activate
pip install -r requirements.txt
```

Next, you must ensure that the appropriate AWS environment variables are set by running `bash setup.sh
`.

## Files & data transfer
Using this repository requires an S3 bucket (indicated in `constants.py`) to store files. Local copies are moved in and out of `tmp/` as required for each task. The module `vectorgeo/transfer.py` handles these exchanges with the correct bucket. **Note** the logic in `transfer.py` will avoid redownloading files if they can be found locally. To force a redownload, delete the relevant files manually from `tmp/`.

## Check Qdrant health
To debug issues with high latency / dropped requests to Qdrant, you can use the following command to check the health of the vector collection using both the Python API as well as a standard GET request:

```
python3 vectorgeo/health.py check_health
```

## Playbook

### Testing
To test the end-to-end workflow for land cover extraction and embedding, run the following commands:
```
python3 01-preprocess-flow.py --no-pylint run --n_files=2 --n_jobs=3 --samples_per_file=4
python3 02-train-flow.py --no-pylint run --epochs=1 --n_train_files=2 --model_filename=test.keras
python3 03-inference-flow.py --no-pylint run --wipe_qdrant=True --qdrant_collection='vg_test' --max_iters=10 --model_filename=test.keras
```
Take care to supply the right arguments at `--qdrant_collection` to avoid overwriting the production collection `vectorgeo`.


