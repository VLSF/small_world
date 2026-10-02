# Fourier neural operators: `FNO` and `FNO_normalised`

`FNO.py` implements a Fourier neural operator (FNO) for functions on the unit cube $[0,1]^D$, $D=1,2,3$, sampled on a uniform grid with periodic Fourier modes. `FNO_normalised.py` subclasses it and rescales the output. Both are [Equinox](https://docs.kidger.site/equinox/) modules written in JAX and are fully compatible with `jit`, `vmap` and `grad`.

## Notation

* $D$ — spatial dimension, grid $n_1\times\dots\times n_D$.
* $C_u$ — number of channels of the input function, $x\in\mathbb R^{D\times n_1\times\dots\times n_D}$ — grid coordinates.
* $n_{\rm in}=D+C_u$, $n_p$ — width of the processor, $n_{\rm out}$ — number of output channels (`N_features = (n_in, n_p, n_out)`).
* $L$ — number of layers (`N_layers`), $M$ — number of retained Fourier modes per direction (`N_modes`).
* $\sigma$ — GELU.

A "$1\times1$ convolution" below is a pointwise linear map of the channel dimension with bias, applied independently at every grid point (`eqx.nn.Conv` with kernel size 1).

## Architecture of `FNO`

The model acts on **one sample**: `model(u, x)` with `u` of shape $(C_u, n_1,\dots,n_D)$ and `x` of shape $(D, n_1,\dots,n_D)$ returns an array of shape $(n_{\rm out}, n_1,\dots,n_D)$. Use `jax.vmap` for batches.

1. **Input.** Coordinates and the input function are stacked along the channel axis, $v_0=[x~u]$, so the network sees $n_{\rm in}=D+C_u$ channels.
2. **Encoder.** $v_0\leftarrow P v_0$, where $P$ is a $1\times1$ convolution $n_{\rm in}\to n_p$.
3. **$L$ Fourier layers.** For $l=0,\dots,L-1$

$$
v_{l+1}=v_l+\sigma\Big(W^{(2)}_l~\sigma\big(W^{(1)}_l~\mathcal K_l v_l\big)\Big),
$$

   where $\mathcal K_l$ is a spectral convolution (below), and $W^{(1)}_l,W^{(2)}_l$ are $1\times1$ convolutions $n_p\to n_p$.
4. **Decoder.** $Q v_L$, with $Q$ a $1\times1$ convolution $n_p\to n_{\rm out}$ and no activation.

Differences from the original FNO, following F-FNO:

* There is **no linear bypass** $W v_l$ next to the spectral convolution. Instead the spectral convolution is followed by a two-layer pointwise MLP ($W^{(1)},W^{(2)}$ with GELU after each), and the **residual connection wraps this whole block** (including the final nonlinearity).
* Since the MLP is applied after $\mathcal K_l$, the nonlinearity acts on the spectral output rather than on the sum of spectral and bypass terms.

Unlike F-FNO, the spectral weights are **not factorised** over dimensions: each layer has a full tensor of size $n_p\times n_p\times M^D$.

### Spectral convolution $\mathcal K_l$

For a field $v\in\mathbb R^{n_p\times n_1\times\dots\times n_D}$

1. Take the real FFT over the spatial axes, $\hat v=\mathrm{rfft}_D(v)$ (normalised by $1/N$, $N=n_1\cdots n_D$, in the forward direction and unnormalised in the inverse direction, so the transform pair is consistent).
2. **Truncate to $M$ modes per direction.** On the last axis (real FFT) the modes $0,\dots,M-1$ are kept. On every other axis the $M$ modes closest to zero frequency are kept, $-\lfloor M/2\rfloor,\dots,M-\lfloor M/2\rfloor-1$~ this is done by rolling the axis by $\lfloor M/2\rfloor$ and slicing the first $M$ entries.
3. Multiply every retained mode by its own complex $n_p\times n_p$ matrix,

$$
\hat w(k)=A_l[:,:,k]~\hat v(k),\qquad k\in\lbrace \text{retained modes}\rbrace,\quad A_l\in\mathbb C^{n_p\times n_p\times M^D}.
$$

4. Zero-pad to the full spectrum, undo the roll (by $-\lfloor M/2\rfloor$) and apply the inverse real FFT to the original grid size, $w=\mathrm{irfft}_D(\hat w)$.

Consequences:

* The parameters do not depend on the grid size, so the model can be evaluated on a different resolution than it was trained on (as long as $M\le n_D/2+1$ along the last axis and $M\le n_i$ along the others).
* The cost of a layer is $O(N\log N)$ for the FFTs plus $O(n_p^2M^D)$ for the mode mixing.
* All spectral weights of the model are stored in one complex array `A` of shape $(L,n_p,n_p,M,\dots,M)$.
* Only $D=1,2,3$ are implemented~ `spectral_conv` dispatches on the number of spatial axes (any $D\ge3$ falls into the 3D branch).

### Initialisation

| Parameter | Initialisation |
|---|---|
| $1\times1$ convolutions (encoder, decoder, $W^{(1,2)}_l$) | default Equinox initialisation, weights multiplied by `s1`, biases by `s2` |
| spectral weights $A$ | standard complex Gaussian (unit variance) multiplied by `s3 / n_p` |

The factor $1/n_p$ keeps the output of the mode mixing at $O(1)$ scale independently of the width, because each output channel is a sum over $n_p$ input channels. All of `s1`, `s2`, `s3` default to $1$.

### Constructor

```python
FNO(N_layers, N_features, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0)
```

```python
import jax, jax.numpy as jnp
from small_world.models.FNO import FNO

model = FNO(N_layers=4, N_features=(3, 32, 1), N_modes=12, D=2, key=jax.random.PRNGKey(0))
# x: (2, n, n) coordinates, u: (1, n, n) input, e.g. conductivity k
out = jax.vmap(model, in_axes=(0, None))(u_batch, x)   # (batch, 1, n, n)
```

Here `N_features[0]` must equal $D+C_u$ (here $2+1=3$).

## Architecture of `FNO_normalised`

`FNO_normalised.FNO` is a subclass of `FNO` with the same parameters and the same network~ it only post-processes the output. Each output channel $c$ is divided by its discrete $L_2$ norm over the domain:

$$
\tilde y_c=\frac{y_c}{\Vert y_c\Vert_{L_2}+\varepsilon},\qquad
\Vert y_c\Vert_{L_2}^2=\int_{[0,1]^D}y_c^2~dx~\approx~\sum_{i_1,\dots,i_D}\Big(\prod_{d=1}^D w^{(d)}_{i_d}\Big)~y_c[i_1,\dots,i_D]^2 .
$$

The integral is approximated by the **trapezoidal rule on a uniform tensor-product grid**:

* the 1D weights along direction $d$ are $w^{(d)}_i=h_d$ for interior points and $h_d/2$ for the two end points~
* the spacing $h_d$ is read from the supplied coordinates, as the difference of `x[d]` between its first two points along axis $d$ (so the grid must be uniform and have at least two points per direction)~
* the $D$-dimensional sum is computed by contracting the squared output with the 1D weight vectors one axis at a time (`trapezoid_weights_1d`, `l2_norm`), so no $D$-dimensional weight array is formed~
* $\varepsilon$ is the class attribute `eps` (default $10^{-8}$) and prevents division by zero.

Properties and caveats:

* The norm approximates the continuous $L_2$ norm, so it does not grow with the number of grid points (unlike a plain sum).
* Every output channel has unit $L_2$ norm (up to $\varepsilon$), hence the model **cannot represent the amplitude of the solution**. The target data must be normalised in the same way, or the loss has to be scale-invariant, and predictions have to be rescaled by a separately known amplitude.
* The normalisation is part of the forward pass, so gradients flow through it.
* The decoder bias is included in the normalised output.
