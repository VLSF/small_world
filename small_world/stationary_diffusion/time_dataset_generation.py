import time
import numpy as np

from .dataset_generation import generate_dataset
from .conductivity_models import isotropic_1
from .source_models import smooth_source

K = 10

if __name__ == "__main__":
    for N in [128, 256]:
        for bc in ["dirichlet", "neumann"]:
            rng = np.random.default_rng(0)
            t = time.perf_counter()
            generate_dataset(K, N, bc, isotropic_1, smooth_source, rng)
            elapsed = time.perf_counter() - t
            print(f"{bc}: {K} samples, N = {N}: {elapsed:.2f} s total, {elapsed / K:.2f} s per sample")
