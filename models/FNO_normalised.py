import jax.numpy as jnp

from .FNO import FNO as _FNO


def trapezoid_weights_1d(x, d):
    """1D trapezoid weights (incl. spacing) along axis d for coordinates x of shape (D, n_1, ..., n_D)."""
    h = jnp.take(x[d], 1, axis=d).ravel()[0] - jnp.take(x[d], 0, axis=d).ravel()[0]
    return jnp.ones(x.shape[1 + d]).at[jnp.array([0, -1])].set(0.5) * h


def l2_norm(f, x):
    """Per-channel L2 norm of f (C, n_1, ..., n_D) by the trapezoidal rule on a uniform grid with coordinates x."""
    for d in range(x.shape[0]):
        f = jnp.tensordot(f**2 if d == 0 else f, trapezoid_weights_1d(x, d), axes=([1], [0]))
    return jnp.sqrt(f)


class FNO(_FNO):
    """FNO whose output channels are divided by their discrete L2 norm (trapezoidal rule, uniform grid)."""

    eps: float = 1e-8

    def __call__(self, u, x):
        out = super().__call__(u, x)
        return out / (l2_norm(out, x) + self.eps).reshape((-1,) + (1,) * (out.ndim - 1))
