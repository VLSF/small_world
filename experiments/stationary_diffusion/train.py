"""Train a neural operator on a stationary diffusion dataset.

    python experiments/stationary_diffusion/train.py [config.json] [key=value ...]

See `Config` for the options (and small_world/training/training.md). Everything is stored in
results_path/<hash of config>/: config.json, best/ (best model and optimiser state), metrics.npz
and, if keep_checkpoints is true, all checkpoints chunk_XXXX/. results_path/results.csv gets one
line per run.
"""
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import equinox as eqx
import numpy as np
import jax.numpy as jnp
from jax import random

from small_world.models.FNO import FNO
from small_world.models.FNO_normalised import FNO as FNO_normalised
from small_world.stationary_diffusion import preprocessing, setups
from small_world.training import engine
from small_world.training.config import load_config


@dataclass
class Config:
    dataset_path: str
    results_path: str
    setup: str = "regression"        # regression, projection or sub_projection
    # model
    N_layers: int = 4
    N_processor: int = 32            # width of the processor
    N_modes: int = 16                # Fourier modes (for sub_projection: <= n//2 + 1, n = smallest subdomain side)
    N_basis: int = 5                 # basis vectors (per subdomain for sub_projection), unused for regression
    s1: float = 1e-2                 # initial scale of the weights of the convolutions
    s2: float = 0.0                  # initial scale of the biases of the convolutions
    s3: float = 1e-2                 # initial scale of the Fourier kernel
    # domain subdivision, sub_projection only
    partition: str = "cos"           # cos, smoothstep (partitions of unity) or indicator (constant 1 on every subdomain, K ignored)
    H: float = 0.4                   # subdomain size
    M: int = 4                       # (M+1)^2 subdomains
    K: int = 3                       # smoothness of the partition of unity
    # data
    N_train: int = 1000
    N_val: int = 100
    N_test: int = 100
    # optimisation
    optim: str = "adam"              # adam or lion
    learning_rate: float = 1e-4
    gamma: float = 0.5               # learning rate is multiplied by gamma every N_drop epochs (exponential decay)
    N_drop: int = 100
    N_batch: int = 10
    N_epoch: int = 500
    key: int = 14                    # seed of the model initialisation and of the batches
    # evaluation, checkpoints, stopping
    stop_each: int = 50              # evaluate on the validation set and write a checkpoint every stop_each epochs
    patience: Optional[int] = None   # stop if the validation error did not improve for this many evaluations
    keep_checkpoints: bool = False   # keep all checkpoints on disk; if false only the best one is kept


def main(argv):
    cfg = load_config(Config, argv)
    exp_hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
    run_path = os.path.join(cfg.results_path, exp_hash)
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    data = np.load(cfg.dataset_path)
    train, val, test = preprocessing.split_slices(cfg.N_train, cfg.N_val, cfg.N_test)
    assert test.stop <= data["u"].shape[0], "N_train + N_val + N_test exceeds the number of samples"
    problem = setups.build_problem(cfg.setup, data, cfg.N_train, cfg.partition, cfg.H, cfg.M, cfg.K)

    D = problem.arrays["coords"].shape[0]
    if cfg.setup == "regression":
        Model, N_out = FNO, 1
    else:
        Model, N_out = FNO_normalised, cfg.N_basis
    if cfg.setup == "sub_projection":
        n = setups.min_subdomain_side(problem.arrays["subdomains"])
        assert cfg.N_modes <= n // 2 + 1, f"N_modes={cfg.N_modes} is too large for subdomains with {n} points, use at most {n // 2 + 1}"
    keys = random.split(random.PRNGKey(cfg.key), 2)
    model = Model(cfg.N_layers, [D + problem.N_features, cfg.N_processor, N_out], cfg.N_modes, D, keys[0],
                  s1=cfg.s1, s2=cfg.s2, s3=cfg.s3)
    model_size = engine.count_parameters(model)

    steps_per_epoch = cfg.N_train // cfg.N_batch
    optim = engine.make_optimizer(cfg.optim, cfg.learning_rate, cfg.N_drop * steps_per_epoch, cfg.gamma)
    indices = lambda s: jnp.arange(s.start, s.stop)
    result = engine.fit(
        model, optim, problem.loss_fn, problem.error_fn, problem.arrays, cfg.N_train, indices(val),
        N_batch=cfg.N_batch, N_epoch=cfg.N_epoch, stop_each=cfg.stop_each, key=keys[1], ckpt_dir=run_path,
        patience=cfg.patience, keep_checkpoints=cfg.keep_checkpoints,
    )
    print(f"stopped: {result.stop_reason}, best evaluation {result.best_n}")

    errors = {name: np.asarray(eqx.filter_jit(problem.error_fn)(result.model, problem.arrays, indices(s)))
              for name, s in [("train", slice(0, cfg.N_train)), ("val", val), ("test", test)]}
    print({name: float(e.mean()) for name, e in errors.items()})
    np.savez(os.path.join(run_path, "metrics.npz"), history=result.history, val_error_history=result.val_errors,
             **{f"{name}_errors": e for name, e in errors.items()},
             **{f"scale_{k}": np.asarray(v) for k, v in problem.scales._asdict().items()})

    row = {**asdict(cfg), "hash": exp_hash, "final_loss": result.history[-1], "model_size": model_size,
           "training_time": result.training_time, "stop_reason": result.stop_reason, "best_n": result.best_n,
           **{f"{name}_error": e.mean() for name, e in errors.items()}}
    csv_path = os.path.join(cfg.results_path, "results.csv")
    new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main(sys.argv[1:])
