# Training

`training/engine.py` is a generic training engine (no equation, no architecture), `training/config.py` turns a dataclass into a config with json-file and command-line overrides, and `experiments/stationary_diffusion/train.py` is the training script for stationary diffusion: it combines a model from `models`, a learning setup from `stationary_diffusion/setups.py` (regression, learnable basis, learnable basis on subdomains; see [`learning.md`](../stationary_diffusion/learning.md)) and the engine.

## Running

```
python experiments/stationary_diffusion/train.py dataset_path=data/100_128_smooth_Dirichlet_isotropic_2_11998.npz results_path=results setup=projection N_train=80 N_val=10 N_test=10
python experiments/stationary_diffusion/train.py experiments/stationary_diffusion/configs/sub_projection.json learning_rate=3e-4 keep_checkpoints=true
python experiments/stationary_diffusion/train.py experiments/stationary_diffusion/configs/sub_projection.json partition=indicator   # no partition of unity: weight 1 on every subdomain
```

Options come from an optional json file (every field of `Config` in `train.py`, with defaults), `key=value` arguments override the file. `dataset_path` and `results_path` are required. `patience=none` disables early stopping.

## What happens

1. The data are preprocessed with the scales of the first `N_train` samples; samples `[N_train, N_train+N_val)` are validation, the next `N_test` are test.
2. Training runs in chunks of `stop_each` epochs. A chunk is one jitted `lax.scan` over the optimisation steps (`N_train // N_batch` per epoch, a fresh permutation of the train set every epoch). The learning rate decays exponentially, by `gamma` every `N_drop` epochs.
3. After every chunk the mean **evaluation error** on the validation set (the protocol of the setup: relative L2 error of the solution, with Petrov–Galerkin projection for the basis setups) is computed and a checkpoint (`model.eqx`, `opt_state.eqx`) is written to disk. Nothing is kept in GPU memory.
4. Training stops after `N_epoch` epochs, when the validation error is `nan`, or when it has not improved for `patience` evaluations.
5. The best checkpoint (lowest validation error) is copied to `best/` and restored. The chunk checkpoints are deleted unless `keep_checkpoints=true`.
6. Train, validation and test errors of the best model are computed per sample.

## Output

`results_path/<sha256 of the config>/`:

| File | Content |
|---|---|
| `config.json` | resolved config |
| `best/model.eqx`, `best/opt_state.eqx` | best model and optimiser state |
| `chunk_XXXX/` | all checkpoints, only with `keep_checkpoints=true` |
| `metrics.npz` | `history` (loss of every step), `val_error_history` (after every chunk), per-sample `train_errors`, `val_errors`, `test_errors`, and the `scale_*` used in preprocessing |

`results_path/results.csv` gets one row per run (config, hash, final loss, model size, training time, stop reason, best evaluation, mean train/val/test errors).

## Restoring a model

```python
from jax import random
from small_world.models.FNO_normalised import FNO
from small_world.training import engine

like = FNO(4, [2 + 2, 32, 5], 16, 2, random.PRNGKey(0))      # same architecture as in config.json
model, opt_state = engine.load_checkpoint("results/<hash>/best", like, optim)
```

`optim` must be built like in training (`engine.make_optimizer`), it only gives the structure of `opt_state`. New data must be rescaled with the saved scales (`preprocessing.apply_scales`).

## Using the engine for another problem

`engine.fit` needs `arrays` (a pytree of device arrays), `loss_fn(model, arrays, indices) -> scalar` and `error_fn(model, arrays, indices) -> (n,)`. The data are passed to the jitted chunk as arguments and are not closed over, so they are not compiled into the program. `python -m small_world.training.engine` runs checks on a toy problem (checkpoints, cleanup, early stopping, nan detection).
