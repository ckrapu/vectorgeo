<img src="banner.png" width="900">

# VectorGeo
`vectorgeo` performs the data engineering, model training, and inference routines for producing a globally-available geospatial vector embedding data product for use in machine learning for Earth & environment and geographic similarity search. Broadly, the architecture is composed of several parts:

- Data cleaning, extraction, and feature engineering is conducted in Metaflow using Copernicus Land Cover data
- Model training and inference is done using Keras in MetaFlow
- Results are served from Qdrant cloud, making use of the `geo_filter` functionality unavailable in Pinecone, Weaviate, or other vector DB providers
- A light web viewer (`vg_site/index.html`) interfaces with an AWS Lambda function to help visualize the results overlaid on a map.

## Installation
Clone the repository using `git clone https://ckrapu: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@github.com/ckrapu/vectorgeo.git` and run `bash setup.sh`.
This repository was designed to run its most intensive pieces on a Lambda Labs A10 instance with 192GB of RAM and 30 cores. Individual parts of this workflow such as the preprocess and upload may work on smaller machines.


## Files & data transfer
Using this repository requires an S3 bucket (indicated in `constants.py`) to store files. Local copies are moved in and out of `tmp/` as required for each task. The module `vectorgeo/transfer.py` handles these exchanges with the correct bucket. **Note** the logic in `transfer.py` will avoid redownloading files if they can be found locally. To force a redownload, delete the relevant files manually from `tmp/`.

### Architecture
The design of this project is a basic linear progression for (1) forming a training dataset, (2) training a model, (3) applying inference to the entire world, and (4) inserting the records into a vector database for similarity search on the front end. The following diagram shows the flow of data through the system:
```mermaid
graph TD;
    A[Retrieve Global Land Cover Raster] -->|Move to| S3
    S3 -->|Start Preprocess Flow| B[Cut Pieces of Raster]
    B -->|Store as npy files| S3
    S3 -->|Start Training Flow| C[Train Keras Model]
    C -->|Store Trained Model| S3
    S3 -->|Start Inference Flow| D[Iterate Over All Patches in Raster]
    D -->|Upload Results| S3
    S3 -->|Start Upload Job| E[Insert Results into Aurora PostgreSQL]
    E -->|Data Available for| F[AWS Lambda Function]
    F -->|Vector Similarity Search| G[vg-site Front End]
    
    style S3 fill:#f9d,stroke:#333,stroke-width:2px;

```

## Playbook

#### Setting up the Aurora Postgres table
To recreate the table for serving vector embeddings from AWS Aurora PostgreSQL, run the script `python setup-aurora.py`. NOTE: this command will drop any existing table and will force you to start from scratch.

### Testing
To test the end-to-end workflow for land cover extraction and embedding, run the following commands:
```
python3 00-h3-stencil-flow.py --no-pylint run --h3_resolution=3
python3 01-preprocess-flow.py --no-pylint run --n_files=2 --n_jobs=3 --samples_per_file=4
python3 02-train-flow.py      --no-pylint run --epochs=1 --n_train_files=2 --model_filename=test.keras
python3 03-inference-flow.py  --no-pylint run --max_iters=10 --model_filename=test.keras
```

### Production workflow
Currently, the full end-to-end workflow can be executed by running the following:
```
python3 00-h3-stencil-flow.py --no-pylint run --h3_resolution=7
python3 01-preprocess-flow.py --no-pylint run --n_files=100 --n_jobs=3 --samples_per_file=5000
python3 02-train-flow.py      --no-pylint run --epochs=100 --n_train_files=100 --model_filename=resnet-triplet-lc.pt
python3 03-inference-flow.py  --no-pylint run --model_filename=resnet-triplet-lc.pt

python3 setup-aurora.py
python3 upload-aurora.py
```



