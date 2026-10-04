from typing import NamedTuple

import jax.numpy as jnp
from jax import tree

from ..partition import cos_partition

# Learnable basis: the model maps (features, coords) to N_basis channels, which are
# the (not yet orthonormal) basis vectors, flattened to (N_basis, N_x * N_y).


def predict_basis(model, feature, coords):
    """Global basis: (N_basis, n)."""
    prediction = model(feature, coords)
    return prediction.reshape(prediction.shape[0], -1)


class Subdomains(NamedTuple):
    indices: list  # flat grid indices of each subdomain (rectangle in lexicographic order)
    bumps: list    # weights on each subdomain (partition of unity, or ones for indicator_partition)
    coords: list   # (D, n_1, n_2) subdomain coordinates, shifted to start at 0, global scale


def get_subdomains(data, scales, H, M, K, partition=cos_partition):
    """Partition of the grid of `data` into overlapping subdomains.

    The model receives the coordinates of a subdomain relative to its corner, divided by
    the same global scale as the coordinates of the whole domain."""
    bc = str(data['bc']).lower()
    N = int(data['N'])
    if not partition.validate_parameters(H, N, M, K):
        raise ValueError(f"invalid partition parameters H={H}, N={N}, M={M}, K={K}")
    indices, bumps, _, shifted_coords = partition.get_partition(N, H, M, K, bc=bc.capitalize())
    n = data['u'][0].size
    # jax silently clamps out-of-range gather indices, so check the grids match
    assert max(i.max() for i in indices) == n - 1, "partition grid does not match the dataset grid"
    to_jax = lambda t: tree.map(jnp.asarray, t)
    return Subdomains(to_jax(indices), to_jax(bumps), [jnp.asarray(c) / scales.coords for c in shifted_coords])


def predict_sub_basis(model, feature, subdomains):
    """Basis on the subdomains: the shared model is applied on each subdomain, the output is
    multiplied by the weights of the partition (partition of unity, or 1 for indicator_partition) and extended by zero to the whole grid.
    Returns the stacked basis of all subdomains, (N_subdomains * N_basis, n)."""
    C = feature.shape[0]
    n = feature[0].size
    flat = feature.reshape(C, -1)
    basis = []
    for ind, bump, coords in zip(*subdomains):
        local = model(flat[:, ind].reshape((C,) + coords.shape[1:]), coords)
        local = local.reshape(local.shape[0], -1) * bump[None]
        basis.append(jnp.zeros((local.shape[0], n), local.dtype).at[:, ind].set(local))
    return jnp.concatenate(basis)
