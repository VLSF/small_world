from typing import NamedTuple

import jax.numpy as jnp

from ..partition import cos_partition, indicator_partition, smoothstep_partition
from . import basis, losses, metrics, preprocessing

# Glue between a dataset and the generic training engine (training/engine.py): builds the
# arrays and the loss/error functions for one of the three learning setups.
#
#   regression      the model predicts the solution (1 output channel)
#   projection      the model predicts N_basis vectors, global basis
#   sub_projection  the model predicts N_basis vectors on every subdomain of a partition
#
# loss_fn(model, arrays, indices) -> scalar    loss on a batch of samples
# error_fn(model, arrays, indices) -> (n,)     evaluation error per sample

SETUPS = ("regression", "projection", "sub_projection")
PARTITIONS = {"cos": cos_partition, "smoothstep": smoothstep_partition, "indicator": indicator_partition}


class Problem(NamedTuple):
    arrays: dict     # device arrays: pass them to loss_fn / error_fn (and to jit as arguments)
    loss_fn: callable
    error_fn: callable
    scales: preprocessing.Scales
    N_features: int  # number of input channels of the features (without coordinates)


def build_problem(setup, data, N_train, partition="cos", H=0.4, M=4, K=3):
    """Arrays (preprocessed with the train-set scales) and functions for `setup`;
    H, M, K, partition are the parameters of the domain subdivision (sub_projection only)."""
    if setup not in SETUPS:
        raise ValueError(f"unknown setup {setup!r}, expected one of {SETUPS}")
    features, targets, coords, scales = preprocessing.preprocess(data, N_train)
    w = preprocessing.get_quadrature_weights(data)
    arrays = dict(features=features, targets=targets, coords=coords, w=w)

    if setup == "regression":
        loss_fn = lambda model, a, ind: losses.regression_loss(model, a["features"][ind], a["targets"][ind], a["coords"], a["w"])
        error_fn = lambda model, a, ind: metrics.get_error_regression(model, a["features"][ind], a["targets"][ind], a["coords"], a["w"])
        return Problem(arrays, loss_fn, error_fn, scales, features.shape[1])

    A_data, A_indices, A_shape, rhs, sol = preprocessing.get_linear_systems(data)
    arrays.update(A_data=A_data, A_indices=A_indices, rhs=rhs, sol=sol)
    systems = lambda a, ind: (a["A_data"][ind], a["A_indices"], A_shape, a["rhs"][ind] if a["rhs"].ndim > 1 else a["rhs"], a["sol"][ind])

    if setup == "projection":
        loss_fn = lambda model, a, ind: losses.projection_loss(model, a["features"][ind], a["targets"][ind], a["coords"], a["w"])
        error_fn = lambda model, a, ind: metrics.get_error_projection(model, a["features"][ind], a["coords"], a["w"], *systems(a, ind))
    else:
        arrays["subdomains"] = basis.get_subdomains(data, scales, H, M, K, partition=PARTITIONS[partition])
        loss_fn = lambda model, a, ind: losses.sub_projection_loss(model, a["features"][ind], a["targets"][ind], a["subdomains"], a["w"])
        error_fn = lambda model, a, ind: metrics.get_error_sub_projection(model, a["features"][ind], a["subdomains"], a["w"], *systems(a, ind))
    return Problem(arrays, loss_fn, error_fn, scales, features.shape[1])


def min_subdomain_side(subdomains):
    """Number of points in the shortest side of the smallest subdomain (limits N_modes: N_modes <= n // 2 + 1)."""
    return min(min(c.shape[1:]) for c in subdomains.coords)
