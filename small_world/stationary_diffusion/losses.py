from jax import vmap
import jax.numpy as jnp

from ..losses import basic, projection
from . import basis

# Losses on batches (features, targets) of preprocessed data; w are the quadrature weights
# from preprocessing.get_quadrature_weights. Not jitted: jit at the highest level.


def regression_loss(model, features, targets, coords, w):
    """Mean relative L2 error of the predicted solution."""
    def loss(feature, target):
        prediction = model(feature, coords).reshape(-1,)
        return basic.relative_l2_error(prediction, target.reshape(-1,), w)
    return jnp.mean(vmap(loss)(features, targets))


def projection_loss(model, features, targets, coords, w):
    """Mean relative L2 distance of the target to the span of the predicted basis."""
    def loss(feature, target):
        return projection.projection_loss(basis.predict_basis(model, feature, coords), target.reshape(-1,), w)
    return jnp.mean(vmap(loss)(features, targets))


def sub_projection_loss(model, features, targets, subdomains, w):
    """Same as projection_loss, the basis is predicted on subdomains and glued with the partition of unity."""
    def loss(feature, target):
        return projection.projection_loss(basis.predict_sub_basis(model, feature, subdomains), target.reshape(-1,), w)
    return jnp.mean(vmap(loss)(features, targets))
