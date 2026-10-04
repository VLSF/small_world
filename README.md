## Layout

```
small_world/                 # repository root
├── pyproject.toml
└── small_world/             # Python package
    ├── models/              # neural operators (FNO, FNO_normalised), equation-independent
    ├── partition/           # partitions of unity (cos, smoothstep) and the indicator cover
    ├── training/            # generic training engine (checkpoints on disk, early stopping) and config
    ├── losses/              # equation-independent math: L2 norm, projection loss, Petrov-Galerkin solve
    └── stationary_diffusion/  # equation-specific: discretizations, random fields, dataset generation,
                             # preprocessing, basis prediction, losses and evaluation
```

Dependencies are one-way: `stationary_diffusion` uses `losses` and `partition`; `models` and `losses` know nothing about the equation. Training scripts combine a model, an equation package and `losses` and live outside the package, in `experiments/`.

```
experiments/
└── stationary_diffusion/
    ├── train.py             # training script (config file / key=value options)
    └── configs/
```

Each subfolder has its own `.md` file(s) describing the mathematics and the code. To start with learning, read [`stationary_diffusion/learning.md`](small_world/stationary_diffusion/learning.md): it explains preprocessing, the three learning setups (regression, learnable basis, learnable basis on subdomains) and their evaluation, with runnable examples.

## Workflow

1. Generate a dataset (`dataset_collection`, see below).
2. Preprocess it (train-set scales, quadrature weights, linear systems): `stationary_diffusion.preprocessing`.
3. Choose a model from `models` and a setup: `losses.regression_loss`, `losses.projection_loss` or `losses.sub_projection_loss`.
4. Evaluate with the matching function of `stationary_diffusion.metrics`.
5. Train with `experiments/stationary_diffusion/train.py`, see [`training/training.md`](small_world/training/training.md):
   `python experiments/stationary_diffusion/train.py dataset_path=data/<name>.npz results_path=results setup=projection`

```python
import numpy as np
from jax import random
from small_world.models.FNO import FNO
from small_world.stationary_diffusion import preprocessing, losses, metrics

data = np.load("data/100_128_smooth_Dirichlet_isotropic_2_11998.npz")
features, targets, coords, scales = preprocessing.preprocess(data, N_train=60)
w = preprocessing.get_quadrature_weights(data)
model = FNO(4, [coords.shape[0] + features.shape[1], 32, 1], 16, 2, random.PRNGKey(0))
loss = losses.regression_loss(model, features[:10], targets[:10], coords, w)
```

The losses are plain (not jitted) functions: apply `jit`, `eqx.filter_value_and_grad` and `scan` at the highest level (the training engine does this).

## Installation

Install the package in editable mode once per Python environment, from the repository root:

```
pip install -e .
```

Edits to the code take effect immediately. A new module inside an existing subpackage is picked up automatically; a new subpackage must be added to `packages` in `pyproject.toml` and have an `__init__.py`.

## Importing

Import from anywhere (scripts, notebooks) using the full package path:

```python
from small_world.models.FNO import FNO
from small_world.models.FNO_normalised import FNO as FNO_normalised
from small_world.stationary_diffusion import dataset_generation, conductivity_models
```

Inside the package, use relative imports (`from . import d2_dirichlet`, `from .FNO import FNO`).

## Running scripts and tests

Modules use relative imports, so run them as modules (from any directory) rather than as `python file.py`. The tests live in the `__main__` block of each file:

```
python -m small_world.partition.cos_partition
python -m small_world.partition.smoothstep_partition
python -m small_world.partition.indicator_partition
python -m small_world.stationary_diffusion.d2_dirichlet
python -m small_world.stationary_diffusion.d2_neumann
python -m small_world.stationary_diffusion.dataset_generation       # residual test
python -m small_world.stationary_diffusion.time_dataset_generation  # timing
python -m small_world.training.engine                               # training engine on a toy problem
```

The losses and metrics have no `__main__` test; the "oracle model" check in `learning.md` (a model returning the true solution must give zero error) plays that role.

Dataset generation command-line script (see `--help` for all options):

```
python -m small_world.stationary_diffusion.dataset_collection -save_path data -N_samples 100 -N_grid 128
```
