import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# Discretization of -(a1 u_x)_x - (a2 u_y)_y = f on (0, 1)^2 with homogeneous
# Dirichlet BCs on all four sides: u(x, 0) = u(0, y) = u(x, 1) = u(1, y) = 0.
#
# Grid: nodes x_i = i*h_x, i = 0, ..., N+1 (similarly for y), with h_x = 1/(N+1).
# The boundary nodes i=0 and i=N+1 carry the Dirichlet value 0 and are
# eliminated, leaving N unknowns i = 1, ..., N per direction (indexed 0, ...,
# N-1 in the code below). Fluxes are evaluated at the N+1 half-points between
# consecutive nodes (boundary nodes included), giving the standard central
# finite-difference stencil for a variable-coefficient Laplacian.

def get_discretization_data(N):
    N_x = N_y = N

    x, y = np.linspace(0, 1, N_x+2)[1:-1], np.linspace(0, 1, N_y+2)[1:-1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    h_x, h_y = x[1] - x[0], y[1] - y[0]
    x_half, y_half = np.linspace(h_x/2, 1-h_x/2, N_x+1), np.linspace(h_y/2, 1-h_y/2, N_y+1)
    X_half, _ = np.meshgrid(x_half, y, indexing="ij")
    _, Y_half = np.meshgrid(x, y_half, indexing="ij")

    i, j = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int)
    I, J = np.meshgrid(i, j, indexing="ij")
    lex = lambda i, j, N_x: (j + i*N_x)
    diag = lex(I, J, N_x)
    i_m = lex(I-1, J, N_x)
    i_p = lex(I+1, J, N_x)
    j_m = lex(I, J-1, N_x)
    j_p = lex(I, J+1, N_x)

    mask_i_m = (I-1) >= 0
    mask_i_p = (I+1) < N_x
    mask_j_m = (J-1) >= 0
    mask_j_p = (J+1) < N_y

    rows = np.concatenate([diag.reshape(-1,), diag[mask_i_m], diag[mask_i_p], diag[mask_j_m], diag[mask_j_p]])
    cols = np.concatenate([diag.reshape(-1,), i_m[mask_i_m], i_p[mask_i_p], j_m[mask_j_m], j_p[mask_j_p]])
    indices = np.stack([rows, cols], 1)
    return mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices

def get_csc_matrix(mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices, a1, a2, params):
    a1_p = a1(X_half[1:], Y, params)
    a1_m = a1(X_half[:-1], Y, params)
    a2_p = a2(X, Y_half[:, 1:], params)
    a2_m = a2(X, Y_half[:, :-1], params)

    data = np.concatenate([
        (a1_p + a1_m + a2_p + a2_m).reshape(-1,),
        -a1_m[mask_i_m],
        -a1_p[mask_i_p],
        -a2_m[mask_j_m],
        -a2_p[mask_j_p],
    ])

    N = X[0].size
    A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
    return A

def discretization_test():
    x, y, p = sp.symbols('x, y, p')
    a = sp.exp(1 + 0.5*sp.cos(sp.pi*(x + 3*y)))
    u = (x*y + 3*y + sp.exp(2*x + y/2))*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    f = - (a * u.diff('x')).diff('x') - (a * u.diff('y')).diff('y')

    a = sp.lambdify((x, y, p), a, 'numpy')
    u = sp.lambdify((x, y), u, 'numpy')
    f = sp.lambdify((x, y), f, 'numpy')

    params = None
    Ns = [16, 32, 64, 128, 256]
    errors = []
    for N in Ns:
        mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices = get_discretization_data(N)
        A = get_csc_matrix(mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices, a, a, params)
        A = splu(A)
        rhs = f(X, Y) * (X[1, 0] - X[0, 0])**2
        sol = A.solve(rhs.reshape(-1,))
        exact = u(X, Y).reshape(-1, )
        error = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
        errors.append(error)
    errors = np.array(errors)
    slope, _ = np.polyfit(np.log10(Ns), np.log10(errors), 1)
    print("expecting convergence order", -2)
    print("observed convergence order", slope)


def discretization_test_anisotropic():
    x, y, p = sp.symbols('x, y, p')
    a1 = sp.exp(1 + 0.5*sp.cos(sp.pi*(x + 3*y)))
    a2 = 1 + 0.5*sp.sin(sp.pi*(2*x - y))
    u = (x*y + 3*y + sp.exp(2*x + y/2))*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    f = - (a1 * u.diff('x')).diff('x') - (a2 * u.diff('y')).diff('y')

    a1 = sp.lambdify((x, y, p), a1, 'numpy')
    a2 = sp.lambdify((x, y, p), a2, 'numpy')
    u = sp.lambdify((x, y), u, 'numpy')
    f = sp.lambdify((x, y), f, 'numpy')

    params = None
    Ns = [16, 32, 64, 128, 256]
    errors = []
    for N in Ns:
        mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices = get_discretization_data(N)
        A = get_csc_matrix(mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices, a1, a2, params)
        A = splu(A)
        rhs = f(X, Y) * (X[1, 0] - X[0, 0])**2
        sol = A.solve(rhs.reshape(-1,))
        exact = u(X, Y).reshape(-1, )
        error = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
        errors.append(error)
    errors = np.array(errors)
    slope, _ = np.polyfit(np.log10(Ns), np.log10(errors), 1)
    print("expecting convergence order", -2)
    print("observed convergence order", slope)

if __name__ == "__main__":
    discretization_test()
    discretization_test_anisotropic()
