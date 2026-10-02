The study of locality inductive bias. Stay tuned for a more informative description.

## Layout

```
small_world/                 # repository root
├── pyproject.toml
└── small_world/             # Python package
    ├── models/              # neural operators (FNO, FNO_normalised)
    ├── partition/           # partitions of unity (cos, smoothstep)
    └── stationary_diffusion/  # discretizations, random fields, dataset generation
```

Each subfolder has its own `.md` file(s) describing the mathematics and the code.

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
python -m small_world.stationary_diffusion.d2_dirichlet
python -m small_world.stationary_diffusion.d2_neumann
python -m small_world.stationary_diffusion.dataset_generation       # residual test
python -m small_world.stationary_diffusion.time_dataset_generation  # timing
```

Dataset generation command-line script (see `--help` for all options):

```
python -m small_world.stationary_diffusion.dataset_collection -save_path data -N_samples 100 -N_grid 128
```
