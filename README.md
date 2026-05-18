# recsys_training

Recommender System Training Package

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mkurovski/recsys_training/master)

## Description

Hands-on Training for Recommender Systems developed for Machine Learning Essentials 2020.

## Installation

Requires Python 3.11 or 3.12.

### Option A: conda (recommended)

```
conda env create -f environment.yaml
conda activate recsys_training
pip install -e .
```

The conda-forge build of LightFM works out of the box.

### Option B: pip + venv

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If installing LightFM from PyPI fails with `AttributeError: 'dict' object has
no attribute '__LIGHTFM_SETUP__'` (an upstream packaging bug in `lightfm==1.17`),
run `scripts/install_lightfm.sh` — it downloads the sdist, applies a one-line
patch, and installs it with build isolation disabled.

### Docker

```
docker build -t recsys-training:mle -f docker/Dockerfile .
docker compose -f docker/docker-compose.yaml up
```

JupyterLab is served on http://localhost:8888/.

## Usage

There are 9 notebooks within `notebooks/` each starting with a number followed by `_e_` for exercise. Within `notebooks/solutions/`you will find all notebooks with a solution proposal implemented. It is strongly advised to go through the notebooks in numerically ascending order.

We use MovieLens 100k as example dataset for the lessons. You can find the data in `data/raw/`.

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
