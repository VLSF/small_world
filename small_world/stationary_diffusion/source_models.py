import numpy as np
from .conductivity_models import kl_factor

def get_source_field(X, Y, p, sigma, l1, l2, rng):
    # Standardized Gaussian random field with the Karhunen-Loeve construction used in
    # conductivity_models.get_conductivity_field. The field has zero mean and unit
    # standard deviation over the grid, so the best constant approximation
    # (the mean, 0) has relative L2 error 1: f is never close to a constant.
    x, y = X[:, 0], Y[0, :]

    W1 = kl_factor(x, l1, p)
    W2 = kl_factor(y, l2, p)
    c = sigma * rng.standard_normal((W1.shape[1], W2.shape[1]))
    g = W1 @ c @ W2.T

    return (g - g.mean()) / g.std()

def smooth_source(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.4
    l2 = 0.4
    return get_source_field(X, Y, p, sigma, l1, l2, rng)

def rough_source(X, Y, rng):
    # exponential kernel (p=1) gives non-differentiable sample paths
    p = 1
    sigma = 1
    l1 = 0.05
    l2 = 0.05
    return get_source_field(X, Y, p, sigma, l1, l2, rng)
