import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# Discretization of -(a1 u_x)_x - (a2 u_y)_y = f on (0, 1)^2 with
# homogeneous Neumann BCs on x=0, y=0, y=1 and homogeneous Dirichlet BC on x=1:
#   u'(0, y) = u'(x, 0) = u'(x, 1) = 0,  u(1, y) = 0.
#
# Grid: nodes x_i = i*h, i = 0, ..., N (h = 1/N). x=1 (i=N) carries the Dirichlet
# value 0 and is eliminated, leaving N unknowns i = 0, ..., N-1 (i=0 is the
# Neumann boundary x=0 and is kept as an unknown).
# Nodes y_j = j*h, j = 0, ..., N. Both y=0 (j=0) and y=1 (j=N) are Neumann and
# kept as unknowns, giving N+1 unknowns j = 0, ..., N.
#
# Neumann boundaries are handled with the standard mirror-ghost-node trick:
# the ghost value across a Neumann boundary equals the mirror image of the
# first interior value, and the flux coefficient at the (fictitious) mirrored
# face is taken equal to the coefficient at the real face next to the
# boundary. This turns the ghost connection into a doubled connection to the
# first interior neighbour.

def get_discretization_data(N):
    h = 1.0 / N
    N_x, N_y = N, N + 1

    x = np.arange(N_x) * h
    y = np.linspace(0, 1, N_y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    x_half = np.linspace(h/2, 1 - h/2, N_x)
    y_half = np.linspace(h/2, 1 - h/2, N_y - 1)
    X_half, _ = np.meshgrid(x_half, y, indexing="ij")
    X_y_half, Y_half = np.meshgrid(x, y_half, indexing="ij")

    i, j = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int)
    I, J = np.meshgrid(i, j, indexing="ij")
    lex = lambda i, j: j + i * N_y
    diag = lex(I, J)
    i_m = lex(I - 1, J)
    i_p = lex(I + 1, J)
    j_m = lex(I, J - 1)
    j_p = lex(I, J + 1)

    mask_i_m = (I - 1) >= 0
    mask_i_p = (I + 1) < N_x
    mask_j_m = (J - 1) >= 0
    mask_j_p = (J + 1) < N_y

    rows = np.concatenate([diag.reshape(-1,), diag[mask_i_m], diag[mask_i_p], diag[mask_j_m], diag[mask_j_p]])
    cols = np.concatenate([diag.reshape(-1,), i_m[mask_i_m], i_p[mask_i_p], j_m[mask_j_m], j_p[mask_j_p]])
    indices = np.stack([rows, cols], 1)
    return X_half, X_y_half, Y_half, X, Y, indices, N_x, N_y

def get_csc_matrix(X_half, X_y_half, Y_half, X, Y, indices, N_x, N_y, a1, a2, params):
    a1_face = a1(X_half, Y, params)              # faces k = 0, ..., N_x-1 (between node k and k+1)
    a2_face = a2(X_y_half, Y_half, params)       # faces k = 0, ..., N_y-2 (between node k and k+1)

    diag_x = a1_face.copy()
    diag_x[1:] += a1_face[:-1]
    diag_x[0] += a1_face[0]        # Neumann mirror at x=0

    diag_y = np.zeros((N_x, N_y))
    diag_y[:, :-1] += a2_face
    diag_y[:, 1:] += a2_face
    diag_y[:, 0] += a2_face[:, 0]     # Neumann mirror at y=0
    diag_y[:, -1] += a2_face[:, -1]   # Neumann mirror at y=1

    ip_weight = a1_face[:-1].copy()
    ip_weight[0] *= 2               # doubled connection 0 -> 1 (Neumann mirror at x=0)
    im_weight = a1_face[:-1].copy()

    jp_weight = a2_face.copy()
    jp_weight[:, 0] *= 2             # doubled connection 0 -> 1 (Neumann mirror at y=0)
    jm_weight = a2_face.copy()
    jm_weight[:, -1] *= 2            # doubled connection (N_y-1) -> (N_y-2) (Neumann mirror at y=1)

    data = np.concatenate([
        (diag_x + diag_y).reshape(-1,),
        -im_weight.reshape(-1,),
        -ip_weight.reshape(-1,),
        -jm_weight.reshape(-1,),
        -jp_weight.reshape(-1,),
    ])

    N_total = N_x * N_y
    A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N_total, N_total)).tocsc()
    return A

def discretization_test():
    x, y, p = sp.symbols('x, y, p')
    a1 = sp.exp(1 + 0.5*sp.cos(sp.pi*(x + 3*y)))
    a2 = 1 + 0.5*sp.sin(sp.pi*(2*x - y))

    X_part = sp.cos(sp.pi*x/2) + sp.Rational(2, 5)*sp.cos(3*sp.pi*x/2)
    Y_part = sp.cos(sp.pi*y) - sp.Rational(3, 10)*sp.cos(2*sp.pi*y) + sp.Rational(3, 2)
    u = X_part * Y_part

    f = - (a1 * u.diff('x')).diff('x') - (a2 * u.diff('y')).diff('y')

    a1 = sp.lambdify((x, y, p), a1, 'numpy')
    a2 = sp.lambdify((x, y, p), a2, 'numpy')
    u = sp.lambdify((x, y), u, 'numpy')
    f = sp.lambdify((x, y), f, 'numpy')

    params = None
    Ns = [16, 32, 64, 128, 256]
    errors = []
    for N in Ns:
        X_half, X_y_half, Y_half, X, Y, indices, N_x, N_y = get_discretization_data(N)
        A = get_csc_matrix(X_half, X_y_half, Y_half, X, Y, indices, N_x, N_y, a1, a2, params)
        A = splu(A)
        h = 1.0 / N
        rhs = f(X, Y) * h**2
        sol = A.solve(rhs.reshape(-1,))
        exact = u(X, Y).reshape(-1,)
        error = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
        errors.append(error)
    errors = np.array(errors)
    slope, _ = np.polyfit(np.log10(Ns), np.log10(errors), 1)
    print("expecting convergence order", -2)
    print("observed convergence order", slope)

if __name__ == "__main__":
    discretization_test()
