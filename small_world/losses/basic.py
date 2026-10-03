import jax.numpy as jnp

# All functions work on flattened fields: arrays of shape (..., n) and quadrature
# weights w of shape (n,). The weighted sum  sum_i w_i a_i b_i  is the discrete L2
# scalar product, and the weighted norm is the discrete L2 norm.


def l2_norm(a, w):
    """L2 norm over the last axis."""
    return jnp.sqrt(jnp.sum(w * a**2, axis=-1))


def relative_l2_error(prediction, target, w):
    """||prediction - target||_{L2} / ||target||_{L2} over the last axis."""
    return l2_norm(prediction - target, w) / l2_norm(target, w)
