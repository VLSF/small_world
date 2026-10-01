import numpy as np

def kl_factor(x, l, p, tol=1e-12):
    K = np.exp(-(np.abs(x[:, None] - x[None, :]) / l) ** p)
    w, V = np.linalg.eigh(K)
    keep = w > tol * w.max()
    return V[:, keep] * np.sqrt(w[keep])

def get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng):
    x, y = X[:, 0], Y[0, :]
    
    W1 = kl_factor(x, l1, p)
    W2 = kl_factor(y, l2, p)
    c = sigma * rng.standard_normal((W1.shape[1], W2.shape[1]))
    g = W1 @ c @ W2.T
    
    z = (g - g.mean()) / g.std()
    logC = np.log(10.0) * rng.uniform(a, b)
    logk = 0.5 * logC * np.tanh(beta * z)
    k = np.exp(logk - logk.mean())
    return k

def isotropic_1(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.2
    l2 = 0.2
    beta = 5
    a = 0.1
    b = 2
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def isotropic_2(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.7
    l2 = 0.7
    beta = 5
    a = 0.1
    b = 2
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def isotropic_3(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.2
    l2 = 0.2
    beta = 5
    a = 3
    b = 4
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def isotropic_4(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.7
    l2 = 0.7
    beta = 5
    a = 3
    b = 4
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def anisotropic_1(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.6
    l2 = 0.01
    beta = 1.0
    a = 3
    b = 4
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def anisotropic_2(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.6
    l2 = 0.1
    beta = 1.0
    a = 3
    b = 4
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def anisotropic_3(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.8
    l2 = 0.2
    beta = 1.0
    a = 0.1
    b = 2
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)

def anisotropic_4(X, Y, rng):
    p = 2
    sigma = 1
    l1 = 0.6
    l2 = 0.01
    beta = 1.0
    a = 0.1
    b = 2
    return get_conductivity_field(X, Y, p, sigma, l1, l2, beta, a, b, rng)