# Dataset generation timing

Measured with `time_dataset_generation.py`: `generate_dataset` for `K = 10` samples, conductivity model `isotropic_1`, source model `smooth_source`. Three runs per case, seed 0, machine with 256 CPU cores (NumPy/SciPy default threading).

| N | BC | total for 10 samples, s | per sample, s |
|---|---|---|---|
| 128 | Dirichlet | 3.97, 4.43, 3.95 | 0.40 – 0.44 |
| 128 | Neumann | 3.79, 4.70, 4.19 | 0.38 – 0.47 |
| 256 | Dirichlet | 17.89, 17.20, 18.03 | 1.72 – 1.80 |
| 256 | Neumann | 16.51, 18.04, 17.81 | 1.65 – 1.80 |

Both boundary conditions cost about the same: ~0.4 s per sample at N = 128 (1000 samples ≈ 7 minutes) and ~1.75 s per sample at N = 256 (1000 samples ≈ 30 minutes).

Doubling N increased the time per sample by about 4.2×, i.e. roughly N² scaling (the number of unknowns grows 4×). This is not the N³ of the dense `eigh` used in the Karhunen–Loève factorization, so at these sizes the cost is not dominated by `eigh`; where the time actually goes was not profiled.
