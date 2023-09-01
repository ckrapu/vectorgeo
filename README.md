# Overview

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

## Playbook

### Testing
To test the end-to-end workflow for land cover extraction and embedding, run the following commands:
```
python3 01-preprocess-flow.py --no-pylint run --n_files=2 --n_jobs=3 --samples_per_file=4
python3 02-train-flow.py --no-pylint run --epochs=1 --n_train_files=2 --model_filename=test.keras
python3 03-inference-flow.py --no-pylint run --wipe_qdrant=True --qdrant_collection='vg_test' --max_iters=10 --model_filename=test.keras
```
Take care to supply the right arguments at `--qdrant_collection` to avoid overwriting the production collection `vectorgeo`.


