# Losses and evaluation for learning stationary diffusion solutions

This note describes how to go from a dataset (see `dataset_generation.py`) to losses and evaluation metrics for three ways of using a neural operator.

## Where things live

| Module | Content |
|---|---|
| `stationary_diffusion/preprocessing.py` | features/targets/coords, train-set scales, data split, quadrature weights, sparse linear systems |
| `stationary_diffusion/basis.py` | predicted basis: global (`predict_basis`) and on subdomains (`get_subdomains`, `predict_sub_basis`) |
| `stationary_diffusion/losses.py` | `regression_loss`, `projection_loss`, `sub_projection_loss` |
| `stationary_diffusion/metrics.py` | `get_error_regression`, `get_error_projection`, `get_error_sub_projection` |
| `losses/basic.py`, `losses/projection.py` | equation-independent math on flattened fields: L2 norm, weighted QR, projection loss, Petrov–Galerkin solve |
| `partition/` | partitions of unity (and the indicator cover) used by the subdomain case |

Dependencies are one-way: `stationary_diffusion` uses `losses` and `partition`, never the reverse.

## Conventions

* A **sample** has `features` of shape `(C_in, N_x, N_y)` with channels `(k)` or `(k, f)` (`f` only if the dataset has a source), `targets` of shape `(1, N_x, N_y)`, shared `coords` of shape `(2, N_x, N_y)`. A model acts on one sample: `model(feature, coords)`; the losses and metrics vmap/scan over the batch.
* **Scales** are per-channel max-abs values computed on the first `N_train` samples only. Features, targets and coordinates are divided by them. The linear systems (`A`, `rhs`, `sol`) stay in **physical units**, because the Petrov–Galerkin system is solved with the original matrix.
* **Data split:** first `N_train` samples are train, next `N_val` validation, next `N_test` test (`preprocessing.split_slices`).
* **L2 scalar product.** All norms and scalar products are the discrete $L_2((0,1)^2)$ ones, $(a,b)=\sum_i w_i a_i b_i$, with trapezoidal quadrature weights `w = get_quadrature_weights(data)`. For Dirichlet data the weights are uniform ($h^2$), so L2 and plain $\ell_2$ coincide. For the mixed Neumann/Dirichlet data the nodes on the Neumann sides ($x_0$, $y_0$, $y_N$) get half weight; $x_{N-1}$ keeps the full weight because the node $x_N=1$ is eliminated ($u=0$ there).

## Preprocessing

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx

from small_world.models.FNO import FNO
from small_world.models.FNO_normalised import FNO as FNO_normalised
from small_world.stationary_diffusion import preprocessing, losses, metrics, basis
from small_world.stationary_diffusion import dataset_generation, conductivity_models, source_models

# small dataset generated right here (or np.load a file made by dataset_collection.py)
data = dataset_generation.generate_dataset(
    100, 64, "dirichlet", conductivity_models.isotropic_2, source_models.smooth_source, np.random.default_rng(0)
)
N_train, N_val, N_test = 60, 20, 20
train, val, test = preprocessing.split_slices(N_train, N_val, N_test)

features, targets, coords, scales = preprocessing.preprocess(data, N_train)   # scales: train set only
w = preprocessing.get_quadrature_weights(data)                                # (N_x * N_y,)
A_data, A_indices, A_shape, rhs, sol = preprocessing.get_linear_systems(data)  # physical units
```

`scales` is a `Scales(features, targets, coords)` named tuple; keep it to rescale new data (`preprocessing.apply_scales`) or to map predictions back (`prediction * scales.targets`).

`rhs` has shape `(K, n)` if the dataset has a source `f`; for `f = 1` it is the single vector `h^2 * ones` of shape `(n,)` shared by all samples.

## 1. Standard regression

The model predicts the solution: `model(feature, coords)` returns `(1, N_x, N_y)`. The loss is the mean relative L2 error, and the evaluation returns one relative L2 error per sample.

```python
D = coords.shape[0]
model = FNO(4, [D + features.shape[1], 32, 1], 16, D, random.PRNGKey(0))

loss = losses.regression_loss(model, features[train][:10], targets[train][:10], coords, w)
errors = metrics.get_error_regression(model, features[val], targets[val], coords, w)   # (N_val,)
```

## 2. Learnable basis

The model predicts `N_basis` vectors $q_i(x)$ (use `FNO_normalised`, whose output channels have unit L2 norm). The loss is the relative L2 distance from the target to the span of the basis,

$$
\frac{\left\|t - \sum_i (t, q_i) q_i\right\|}{\|t\|},
$$

where the $q_i$ are first L2-orthonormalised (weighted QR). The evaluation solves $A u = \mathrm{rhs}$ inside the span (Petrov–Galerkin: $q^T A q\,c = q^T \mathrm{rhs}$, $u = qc$) and returns the relative L2 error with respect to the reference solution.

```python
N_basis = 5
model = FNO_normalised(4, [D + features.shape[1], 32, N_basis], 16, D, random.PRNGKey(0))

loss = losses.projection_loss(model, features[train][:10], targets[train][:10], coords, w)

errors = metrics.get_error_projection(
    model, features[val], coords, w,
    A_data[val], A_indices, A_shape,
    rhs[val] if rhs.ndim > 1 else rhs,          # shared rhs for f = 1
    sol[val],
)
```

## 3. Learnable basis on domain subdomains

The domain is split into overlapping subdomains by `small_world.partition` (centres on an $(M+1)\times(M+1)$ grid, side $H > 1/M$, bump smoothness $K$). The **same** model is applied on every subdomain (it sees the subdomain's features and its coordinates relative to the subdomain corner, divided by the global coordinate scale) and predicts `N_basis` vectors. Each is multiplied by the partition-of-unity weight of its subdomain and extended by zero to the whole grid. All $(M+1)^2 N_{\rm basis}$ vectors are stacked into one global basis, which is then used exactly as in case 2: projection loss for training, Petrov–Galerkin for evaluation.

```python
from small_world.partition import cos_partition   # or smoothstep_partition, indicator_partition

H, M, K = 0.4, 4, 3
subdomains = basis.get_subdomains(data, scales, H, M, K, partition=cos_partition)

N_basis = 6
model = FNO_normalised(4, [D + features.shape[1], 32, N_basis], 6, D, random.PRNGKey(0))

loss = losses.sub_projection_loss(model, features[train][:10], targets[train][:10], subdomains, w)

errors = metrics.get_error_sub_projection(
    model, features[val], subdomains, w,
    A_data[val], A_indices, A_shape,
    rhs[val] if rhs.ndim > 1 else rhs,
    sol[val],
)
```

`get_subdomains` takes `bc` and the grid size from the dataset and checks that the partition matches the grid. Subdomains at the boundary are cut by the domain, so they are smaller than interior ones (the model is called once per subdomain shape, which `jit` handles by tracing each shape). The partitions `cos_partition` and `smoothstep_partition` give a partition of unity. `indicator_partition` uses the same subdomains but the constant weight $1$ on each (no normalisation, so it is a cover, not a partition of unity, and `K` is ignored); in the training script this is `partition=indicator`. `N_modes` is limited by the smallest subdomain (a corner one): with $n$ points in its shortest side, the last Fourier axis has only $n//2+1$ modes, so use `N_modes <= n//2 + 1`. The smallest side can be read off `min(c.shape[1:] for c in subdomains.coords)`.

## Using a loss in a training step (sketch)

The losses are scalar functions of the model, so they plug into Equinox/optax as usual. In a `scan` loop the step is the scan body, so no separate `jit` is needed:

```python
import optax
from jax.lax import scan

optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))
train_features, train_targets = features[train], targets[train]

def step(carry, batch_indices):
    model, opt_state = carry
    loss, grads = eqx.filter_value_and_grad(losses.sub_projection_loss)(
        model, train_features[batch_indices], train_targets[batch_indices], subdomains, w)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    return (eqx.apply_updates(model, updates), opt_state), loss

batches = random.choice(random.PRNGKey(1), N_train, shape=(5, 10))   # 5 steps, batch of 10
(model, opt_state), history = scan(step, (model, opt_state), batches)
```

## Sanity check: the oracle model

A model that returns the true solution must give zero error in all three protocols. This is a convenient test after changing anything (the error is ~1e-5 in float32 and ~1e-15 if float64 is enabled at the top of the script with `jax.config.update("jax_enable_x64", True)`):

```python
oracle = lambda feature, coords: feature[:1]       # feed the targets as features
err = metrics.get_error_projection(oracle, targets[:3], coords, w, A_data[:3], A_indices, A_shape,
                                   rhs[:3] if rhs.ndim > 1 else rhs, sol[:3])   # ~0
```

For the subdomain setup this check holds for the partitions of unity (`cos`, `smoothstep`): the basis contains $w_c u$ and $\sum_c w_c u = u$. It does **not** hold for `indicator_partition`: $u$ is in the span of $\{\mathbb 1_c u\}$ only if $\sum_c a_c \mathbb 1_c \equiv 1$ for some constants $a_c$, which is impossible for overlapping hard-edged subdomains, so the error is not zero there even for a perfect local model.
