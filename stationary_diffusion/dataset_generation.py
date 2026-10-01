import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import d2_dirichlet
import d2_neumann

# Dataset generation for operator learning (k, f) -> u, where
#   -div k grad u = f   on (0, 1)^2
# is discretized by d2_dirichlet.py or d2_neumann.py.
#
# The discretizations need k at cell faces. The model is sampled once on a grid of
# step h/2 that contains the nodes (even, even), the x-faces (odd, even) and the
# y-faces (even, odd) of the (extended) node grid, which includes the nodes eliminated
# by the Dirichlet BC. The matrix uses the face values exactly. The k stored in the
# dataset is the model value at the nodes of the grid of unknowns.
#
# Output dictionary (to be stored with np.savez):
#   "k"       (K, N_x, N_y)    conductivity on the grid of unknowns
#   "f"       (K, N_x, N_y)    source on the grid of unknowns (only if f_model is given)
#   "u"       (K, N_x, N_y)    solution of A u = h^2 f on the grid of unknowns
#   "A_data"  (K, nnz)         BCOO data of A, one row per sample
#   "A_indices" (nnz, 2)       BCOO indices of A, shared by all samples
#   "A_shape" (2,)             shape of A
#   "h", "N", "bc"             grid step, grid size parameter, boundary condition type
# A_k = jax.experimental.sparse.BCOO((A_data[k], A_indices), shape=tuple(A_shape)).
# Flattening is lexicographic: u.reshape(-1) is the unknown vector, rhs = h^2 f.reshape(-1).

def _faces(k_fine):
    # k_fine lives on the grid of step h/2 over the extended node grid: nodes at
    # (even, even), x-faces at (odd, even), y-faces at (even, odd).
    return k_fine[1::2, ::2], k_fine[::2, 1::2]

def generate_dataset(K, N, bc, k_model, f_model, rng):
    """K samples, grid parameter N, bc in {"dirichlet", "neumann"},
    k_model(X, Y, rng) -> k on the grid, f_model(X, Y, rng) -> f on the grid or None
    (then f = 1), rng: numpy random generator."""
    bc = bc.lower()
    if bc == "dirichlet":
        mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices = d2_dirichlet.get_discretization_data(N)
        h = 1.0 / (N + 1)
        ext_nodes = np.linspace(0, 1, N + 2)
        crop = np.s_[2:-2:2, 2:-2:2]
        # faces restricted to the unknowns in the tangential direction
        get_data = lambda kx, ky: d2_dirichlet.get_matrix_data_from_faces(
            mask_i_m, mask_i_p, mask_j_m, mask_j_p, kx[:, 1:-1], ky[1:-1, :])
    elif bc == "neumann":
        X_half, X_y_half, Y_half, X, Y, indices, N_x, N_y = d2_neumann.get_discretization_data(N)
        h = 1.0 / N
        ext_nodes = np.linspace(0, 1, N + 1)
        crop = np.s_[:-2:2, ::2]
        get_data = lambda kx, ky: d2_neumann.get_matrix_data_from_faces(N_x, N_y, kx, ky[:-1, :])
    else:
        raise ValueError(f"unknown bc {bc!r}, expected 'dirichlet' or 'neumann'")

    fine_nodes = np.linspace(0, 1, 2 * (ext_nodes.size - 1) + 1)
    X_fine, Y_fine = np.meshgrid(fine_nodes, fine_nodes, indexing="ij")
    n = X.size

    k_all, f_all, u_all, data_all = [], [], [], []
    for _ in range(K):
        k_fine = k_model(X_fine, Y_fine, rng)
        kx, ky = _faces(k_fine)
        data = get_data(kx, ky)
        A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(n, n)).tocsc()
        f = np.ones_like(X) if f_model is None else f_model(X, Y, rng)
        u = splu(A).solve((h**2 * f).reshape(-1,)).reshape(X.shape)

        k_all.append(k_fine[crop])
        f_all.append(f)
        u_all.append(u)
        data_all.append(data)

    dataset = {
        "k": np.stack(k_all),
        "u": np.stack(u_all),
        "A_data": np.stack(data_all),
        "A_indices": indices,
        "A_shape": np.array([n, n]),
        "h": np.array(h),
        "N": np.array(N),
        "bc": np.array(bc),
    }
    if f_model is not None:
        dataset["f"] = np.stack(f_all)
    return dataset

def residual_test():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.experimental import sparse

    from conductivity_models import isotropic_1
    from source_models import smooth_source

    for N in [32, 128]:
        for bc in ["dirichlet", "neumann"]:
            for f_name, f_model in [("f = 1", None), ("f = smooth GRF", smooth_source)]:
                print(f"N = {N}, {bc}, {f_name}")
                rng = np.random.default_rng(0)
                data = generate_dataset(1, N, bc, isotropic_1, f_model, rng)
                u = data["u"][0].reshape(-1,)
                f = data["f"][0] if "f" in data else np.ones_like(data["u"][0])
                rhs = data["h"] ** 2 * f.reshape(-1,)
                shape = tuple(data["A_shape"])

                # scipy
                A = coo_matrix((data["A_data"][0], (data["A_indices"][:, 0], data["A_indices"][:, 1])), shape=shape).tocsc()
                res = np.linalg.norm(A @ u - rhs) / np.linalg.norm(rhs)
                print("  scipy relative residual", res)

                # jax BCOO, everything transferred to jax
                A_jax = sparse.BCOO((jnp.asarray(data["A_data"][0]), jnp.asarray(data["A_indices"])), shape=shape)
                res_jax = jnp.linalg.norm(A_jax @ jnp.asarray(u) - jnp.asarray(rhs)) / jnp.linalg.norm(jnp.asarray(rhs))
                print("  jax relative residual  ", float(res_jax), "dtype", A_jax.dtype)
                # roundoff of the direct solve grows with the condition number of A ~ N^2
            tol = 1e-15 * N**2
            assert res < tol and res_jax < tol

if __name__ == "__main__":
    residual_test()
