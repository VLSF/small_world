import numpy as np
from math import comb


def smootherstep(t, K):
    """Generalized smoothstep polynomial of order K (Perlin's "smootherstep" family).

    For t in [0, 1] this is the unique degree-(2K+1) polynomial S_K with
    S_K(0) = 0, S_K(1) = 1, and the first K derivatives of S_K vanishing at
    both t = 0 and t = 1. K=0 gives the linear ramp t, K=1 gives the classic
    smoothstep 3t^2 - 2t^3, K=2 gives the classic smootherstep 6t^5-15t^4+10t^3,
    etc. Values outside [0, 1] are clamped.
    """
    t = np.clip(t, 0.0, 1.0)
    coeffs = sum(comb(K + n, n) * comb(2 * K + 1, K - n) * (-t) ** n for n in range(K + 1))
    return t ** (K + 1) * coeffs


def compute_bump(x, H, K=2):
    """Compactly supported bump on (-H/2, H/2) built from smootherstep(K).

    Unlike cos(pi*x/H)^K -- which takes one fixed smooth shape f(x) and
    raises it to the K-th power to sharpen the contact at the edges -- this
    applies a *different polynomial for every K* (increasing degree,
    2K+1) to the smooth, edge-vanishing profile r(x)^2 = (2x/H)^2. The
    smoothness order at the boundary is controlled by which polynomial is
    used, not by exponentiation of a fixed shape.
    """
    mask = (x < H / 2) * (x > -H / 2)
    r2 = (2 * x / H) ** 2
    t = np.clip(1 - r2, 0.0, 1.0)
    return smootherstep(t, K) * mask


def validate_parameters(H, N, M, K):
    h = 1 / M
    cover_domain = H > h
    reasonable_K = K <= 8
    return cover_domain and reasonable_K


def get_partition(N, H, M, K, bc='Dirichlet'):
    h = 1.0 / M
    if bc == 'Dirichlet':
        N_x = N_y = N
        x, y = np.linspace(0, 1, N_x + 2)[1:-1], np.linspace(0, 1, N_y + 2)[1:-1]
    elif bc == 'Neumann':
        h_ = 1.0 / N
        N_x, N_y = N, N + 1
        x, y = np.arange(N_x) * h_, np.linspace(0, 1, N_y)
    else:
        raise ValueError("Supported bc are Dirichlet and Neumann")

    i, j = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int)
    I, J = np.meshgrid(i, j, indexing="ij")
    X, Y = np.meshgrid(x, y, indexing="ij")

    lex = lambda i, j: j + i * N_y
    k = lex(I, J)

    centers = np.arange(M + 1) * h
    Centers = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=2).reshape(-1, 2)
    indices = []
    bumps = []
    coords = []
    norm_factor = np.zeros((N_x * N_y,))
    for c in Centers:
        mask_x = np.logical_and(X - c[0] < H / 2, X - c[0] > -H / 2)
        mask_y = np.logical_and(Y - c[1] < H / 2, Y - c[1] > -H / 2)
        coord = np.stack(np.meshgrid(x[np.logical_and(x - c[0] < H / 2, x - c[0] > -H / 2)],
                                      y[np.logical_and(y - c[1] < H / 2, y - c[1] > -H / 2)],
                                      indexing="ij"), axis=0)
        domain_mask = np.logical_and(mask_x, mask_y)
        ind = k[domain_mask]
        bump = compute_bump(X.reshape(-1,)[ind] - c[0], H, K) * compute_bump(Y.reshape(-1,)[ind] - c[1], H, K)
        norm_factor[ind] += bump
        indices.append(ind)
        bumps.append(bump)
        coords.append(coord)

    for i in range(len(bumps)):
        bumps[i] = bumps[i] / norm_factor[indices[i]]

    normalised_coords = [c - np.min(c, axis=(1, 2), keepdims=True) for c in coords]
    return indices, bumps, coords, normalised_coords


if __name__ == "__main__":
    N = 128
    BCs = ['Dirichlet', 'Neumann', 'Dirichlet', 'Neumann']
    Ks = [2, 4, 3, 2]
    Hs = [0.4, 0.2, 0.1, 0.8]
    Ms = [3, 6, 11, 2]
    for i in range(4):
        bc = BCs[i]
        H = Hs[i]
        K = Ks[i]
        M = Ms[i]
        fine = validate_parameters(H, N, M, K)
        print("boundary conditions", bc, f"H = {H}", f"K = {K}", f"M = {M}")
        print("reasonable parameters?", fine)
        if fine:
            indices, bumps, coords, normalised_coords = get_partition(N, H, M, K, bc=bc)

        if bc == 'Dirichlet':
            N_x = N_y = N
        else:
            N_x, N_y = N, N + 1
        unity = np.zeros((N_x * N_y,))

        for i in range(len(bumps)):
            unity[indices[i]] += bumps[i]

        partition_of_unity = np.allclose(unity, np.ones_like(unity))
        cover_all = np.all(np.unique(np.concatenate(indices)) == np.arange(N_x * N_y))
        print("partition of unity is valid?", partition_of_unity)
        print("cover all points?", cover_all)
        print()
