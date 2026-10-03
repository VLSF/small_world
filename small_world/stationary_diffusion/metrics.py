from jax.lax import scan

from ..losses import basic, projection
from . import basis

# Evaluation protocols, one relative L2 error per sample. Samples are processed
# sequentially with scan to keep memory low. Not jitted: jit at the highest level.
# The linear systems (A_data, A_indices, A_shape, rhs, sol) are in physical units
# (see preprocessing.get_linear_systems; rhs is (n,) if shared by all samples, else (K, n)),
# the features are the rescaled ones.


def get_error_regression(model, features, targets, coords, w):
    def step(_, xs):
        feature, target = xs
        prediction = model(feature, coords).reshape(-1,)
        return None, basic.relative_l2_error(prediction, target.reshape(-1,), w)
    return scan(step, None, (features, targets))[1]


def _get_error_galerkin(predict_basis, features, w, A_data, A_indices, A_shape, rhs, sol):
    shared_rhs = rhs.ndim == 1

    def step(_, xs):
        feature, A_data_, sol_, *rhs_ = xs
        approx = projection.petrov_galerkin_solve(predict_basis(feature), w, A_data_, A_indices, A_shape, rhs if shared_rhs else rhs_[0])
        return None, basic.relative_l2_error(approx, sol_, w)
    xs = (features, A_data, sol) if shared_rhs else (features, A_data, sol, rhs)
    return scan(step, None, xs)[1]


def get_error_projection(model, features, coords, w, A_data, A_indices, A_shape, rhs, sol):
    predict = lambda feature: basis.predict_basis(model, feature, coords)
    return _get_error_galerkin(predict, features, w, A_data, A_indices, A_shape, rhs, sol)


def get_error_sub_projection(model, features, subdomains, w, A_data, A_indices, A_shape, rhs, sol):
    predict = lambda feature: basis.predict_sub_basis(model, feature, subdomains)
    return _get_error_galerkin(predict, features, w, A_data, A_indices, A_shape, rhs, sol)
