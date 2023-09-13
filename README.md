# Overview
`vectorgeo` performs the data engineering, model training, and inference routines for producing a globally-available geospatial vector embedding data product for use in machine learning for Earth & environment and geographic similarity search. Broadly, the architecture is composed of several parts:

- Data cleaning, extraction, and feature engineering is conducted in Metaflow using Copernicus Land Cover data
- Model training and inference is done using Keras in MetaFlow
- Results are served from Qdrant cloud, making use of the `geo_filter` functionality unavailable in Pinecone, Weaviate, or other vector DB providers
- A light web viewer (`vg_site/index.html`) interfaces with an AWS Lambda function to help visualize the results overlaid on a map.

## Installation
Clone the repository using `git clone https://ckrapu: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@github.com/ckrapu/vectorgeo.git
`, and run`bash setup.sh`


## Files & data transfer
Using this repository requires an S3 bucket (indicated in `constants.py`) to store files. Local copies are moved in and out of `tmp/` as required for each task. The module `vectorgeo/transfer.py` handles these exchanges with the correct bucket. **Note** the logic in `transfer.py` will avoid redownloading files if they can be found locally. To force a redownload, delete the relevant files manually from `tmp/`.

## Check Qdrant health
To debug issues with high latency / dropped requests to Qdrant, you can use the following command to check the health of the vector collection using both the Python API as well as a standard GET request:

```
python3 vectorgeo/health.py check_health
```

Some basic Qdrant functionality is documented [here](https://qdrant.tech/documentation/quick-start/)

## Playbook

#### Setting up the Aurora Postgres table
To recreate the table for serving vector embeddings from AWS Aurora PostgreSQL, run the script `python setup-aurora.py`. NOTE: this command will drop any existing table and will force you to start from scratch.

### Testing
To test the end-to-end workflow for land cover extraction and embedding, run the following commands:
```
python3 00-h3-stencil-flow.py --no-pylint run --h3_resolution=3
python3 01-preprocess-flow.py --no-pylint run --n_files=2 --n_jobs=3 --samples_per_file=4
python3 02-train-flow.py --no-pylint run --epochs=1 --n_train_files=2 --model_filename=test.keras
python3 03-inference-flow.py --no-pylint run --wipe_qdrant=True --qdrant_collection='vg_test' --max_iters=10 --model_filename=test.keras
```
Take care to supply the right arguments at `--qdrant_collection` to avoid overwriting the production collection `vectorgeo`.



