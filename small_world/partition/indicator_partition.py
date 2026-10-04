import numpy as np

from .cos_partition import validate_parameters, get_partition as _get_cos_partition


def get_partition(N, H, M, K=None, bc='Dirichlet'):
    """Same subdomains as cos_partition.get_partition, but the weight on every subdomain is the
    constant 1 (indicator function of the subdomain). Overlapping subdomains are not normalised,
    so this is a cover of the domain, not a partition of unity: the weights sum to the number of
    subdomains that contain a point. K is accepted for compatibility and ignored."""
    indices, bumps, coords, normalised_coords = _get_cos_partition(N, H, M, 2, bc=bc)
    bumps = [np.ones_like(bump) for bump in bumps]
    return indices, bumps, coords, normalised_coords


if __name__ == "__main__":
    N = 128
    for bc, H, M in [('Dirichlet', 0.4, 3), ('Neumann', 0.2, 6), ('Dirichlet', 0.1, 11), ('Neumann', 0.8, 2)]:
        N_x, N_y = (N, N) if bc == 'Dirichlet' else (N, N + 1)
        print("boundary conditions", bc, f"H = {H}", f"M = {M}")
        if not validate_parameters(H, N, M, 1):
            print("invalid parameters\n")
            continue
        indices, bumps, coords, normalised_coords = get_partition(N, H, M, bc=bc)
        count = np.zeros((N_x * N_y,))
        for ind, bump in zip(indices, bumps):
            count[ind] += bump
        print("weights are constant 1?", all(np.all(bump == 1) for bump in bumps))
        print("cover all points?", np.all(count >= 1))
        print("max number of overlapping subdomains", int(count.max()))
        print()
