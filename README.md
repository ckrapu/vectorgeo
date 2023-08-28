# Overview

## Installation
Clone the repository, and run the following to make a new virtualenv in `vg-venv`:

```
python3 -m venv vg-venv
source vg-venv/bin/activate
pip install -r requirements.txt
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

