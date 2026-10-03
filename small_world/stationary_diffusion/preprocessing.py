from typing import NamedTuple

import jax.numpy as jnp

# Preprocessing of datasets produced by dataset_generation.py (see the header there).
# The samples are split as: first N_train -> train, next N_val -> validation,
# next N_test -> test. Scales are computed on the train set only.


class Scales(NamedTuple):
    features: jnp.ndarray  # (1, C_in, 1, ..., 1)
    targets: jnp.ndarray   # (1, 1, 1, ..., 1)
    coords: jnp.ndarray    # (D, 1, ..., 1)


def get_features(data):
    """Features (K, C_in, N_x, N_y) with channels (k[, f]), targets (K, 1, N_x, N_y), coords (D, N_x, N_y)."""
    if 'f' in data:
        features = jnp.stack([jnp.asarray(data['k']), jnp.asarray(data['f'])], axis=1)
    else:
        features = jnp.expand_dims(jnp.asarray(data['k']), axis=1)
    targets = jnp.expand_dims(jnp.asarray(data['u']), axis=1)
    coords = jnp.asarray(data['coordinates'])
    return features, targets, coords


def split_slices(N_train, N_val, N_test):
    train = slice(0, N_train)
    val = slice(N_train, N_train + N_val)
    test = slice(N_train + N_val, N_train + N_val + N_test)
    return train, val, test


def _max_abs(field, axis):
    scale = jnp.max(jnp.abs(field), axis=axis, keepdims=True)
    return scale + (scale == 0)


def compute_scales(features, targets, coords, N_train):
    """Per-channel max-abs scales of features and targets over the first N_train samples
    (and all grid points); coords do not depend on the sample."""
    axis = (0,) + tuple(range(2, features.ndim))
    return Scales(
        features=_max_abs(features[:N_train], axis),
        targets=_max_abs(targets[:N_train], axis),
        coords=_max_abs(coords, tuple(range(1, coords.ndim))),
    )


def apply_scales(features, targets, coords, scales):
    return features / scales.features, targets / scales.targets, coords / scales.coords


def preprocess(data, N_train):
    """Features, targets, coords rescaled with the train-set scales, and the scales themselves."""
    features, targets, coords = get_features(data)
    scales = compute_scales(features, targets, coords, N_train)
    return (*apply_scales(features, targets, coords, scales), scales)


def get_quadrature_weights(data):
    """Weights w (N_x * N_y,) of the trapezoidal rule on the grid of unknowns, so that
    sum(w * a * b) is the L2((0,1)^2) scalar product of grid functions that vanish on the
    eliminated Dirichlet nodes. Same lexicographic order as the unknown vector.

    dirichlet: all unknowns are interior, w = h^2.
    neumann:   x_0 and y_0, y_N are Neumann boundary nodes (half weight); the node x_N = 1 is
               eliminated (u = 0 there), so x_{N-1} has the full weight h."""
    h = float(data['h'])
    bc = str(data['bc']).lower()
    N_x, N_y = data['u'].shape[1:]
    w_x, w_y = jnp.full(N_x, h), jnp.full(N_y, h)
    if bc == 'neumann':
        w_x = w_x.at[0].set(h / 2)
        w_y = w_y.at[jnp.array([0, -1])].set(h / 2)
    elif bc != 'dirichlet':
        raise ValueError(f"unknown bc {bc!r}")
    return jnp.outer(w_x, w_y).reshape(-1,)


def get_linear_systems(data):
    """Sparse systems A u = rhs: A_data (K, nnz), A_indices (nnz, 2), A_shape (tuple),
    rhs (K, n) = h^2 f, sol (K, n). Physical (unscaled) quantities."""
    h = float(data['h'])
    K = data['u'].shape[0]
    n = data['u'][0].size
    if 'f' in data:
        rhs = h**2 * jnp.asarray(data['f']).reshape(K, n)
    else:
        rhs = jnp.full((K, n), h**2)
    sol = jnp.asarray(data['u']).reshape(K, n)
    A_shape = tuple(int(s) for s in data['A_shape'])
    return jnp.asarray(data['A_data']), jnp.asarray(data['A_indices']), A_shape, rhs, sol
