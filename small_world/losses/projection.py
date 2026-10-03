import jax.numpy as jnp
from jax.experimental import sparse

# Projection onto a learned subspace, flattened fields, discrete L2 scalar product
# (a, b) = sum_i w_i a_i b_i with quadrature weights w of shape (n,).
#
# Instead of working with L2-orthonormal vectors q = Q / sqrt(w) directly, we
# orthonormalise in the weighted coordinates sqrt(w) * a, where the L2 product is the
# Euclidean one. The projector is then P t = q q^T W t.


def orthonormalise(basis, w):
    """Weighted coordinates of an L2-orthonormal basis of span(basis).

    basis: (m, n). Returns Q_w of shape (n, m) with Q_w^T Q_w = I, such that
    q = Q_w / sqrt(w)[:, None] are L2-orthonormal."""
    return jnp.linalg.qr((jnp.sqrt(w) * basis).T)[0]


def projection_loss(basis, target, w):
    """||t - sum_i (t, q_i) q_i||_{L2} / ||t||_{L2}, basis: (m, n), target: (n,)."""
    Q_w = orthonormalise(basis, w)
    t_w = jnp.sqrt(w) * target
    residual = t_w - Q_w @ (Q_w.T @ t_w)
    return jnp.linalg.norm(residual) / jnp.linalg.norm(t_w)


def petrov_galerkin_solve(basis, w, A_data, A_indices, A_shape, rhs):
    """Solve A u = rhs in span(basis): u = q c, q^T A q c = q^T rhs (test = trial space).

    basis: (m, n), A is given as BCOO data (nnz,) with shared indices (nnz, 2), rhs: (n,)."""
    q = orthonormalise(basis, w) / jnp.sqrt(w)[:, None]
    A = sparse.BCOO((A_data, A_indices), shape=A_shape)
    return q @ jnp.linalg.solve(q.T @ (A @ q), q.T @ rhs)
