# Partition of the domain and partition of unity

`cos_partition.py` and `smoothstep_partition.py` both split the domain $[0,1]^2$ into overlapping square subdomains and build a smooth *partition of unity* subordinate to that cover: a set of weight functions $w_c(x,y)$, one per subdomain, that are non-negative, are each supported only on their own subdomain, and sum to exactly $1$ everywhere on $[0,1]^2$. This is the standard construction used to smoothly stitch together quantities (e.g. local solutions or local predictions) defined independently on overlapping patches into a single global field. The two files share the same domain-decomposition and grid logic and differ only in the 1D bump function used to build the weights, `compute_bump`.

## Grid

Both files reuse the same grids as the two discretizations in `stationary_diffusion`: the interior grid $x_i = (i+1)h$, $i=0,\dots,N-1$, $h=1/(N+1)$, for `bc='Dirichlet'` (see [`d2_dirichlet.md`](../stationary_diffusion/d2_dirichlet.md)), and the mixed grid $N_x=N$, $N_y=N+1$ for `bc='Neumann'` (see [`d2_neumann.md`](../stationary_diffusion/d2_neumann.md)). Grid points are ordered with the same lexicographic index

$$
k(i,j) = j + i~N_y,
$$

so the local indices returned for a subdomain index directly into the same unknown vector $\mathbf u$ used by the corresponding discretization.

## Domain decomposition

Subdomains are centered on a coarse $(M+1)\times(M+1)$ grid covering $[0,1]^2$, spaced $h_M = 1/M$ apart:

$$
c_{m,n} = (m~h_M,~ n~h_M), \qquad m, n = 0, \dots, M.
$$

Each subdomain is the open square of side $H$ around its center,

$$
\Omega_{m,n} = \left\\{(x,y) \in [0,1]^2 : |x - m h_M| < H/2 ~\text{and}~ |y - n h_M| < H/2\right\\}.
$$

Requiring $H > h_M$ (checked by `validate_parameters`) makes neighboring subdomains overlap, so every grid point lies in at least one $\Omega_{m,n}$ and the union of all subdomains covers $[0,1]^2$; `get_partition` returns, for every $\Omega_{m,n}$, the flat grid indices it contains, its weight values there, its $(x,y)$ coordinates, and a normalized copy of the coordinates shifted so the subdomain's own bounding box starts at $(0,0)$.

## Partition of unity from a separable bump

For a subdomain centered at $c=(c_x, c_y)$ define the raw weight as a product of a 1D bump function evaluated along each axis,

$$
\varphi_c(x,y) = \mathrm{bump}(x - c_x; H, K)~\mathrm{bump}(y - c_y; H, K),
$$

where $\mathrm{bump}(\cdot; H, K)$ is compactly supported on $(-H/2, H/2)$ and $K$ controls how smoothly it decays to $0$ at the edges. Normalizing by the sum of all subdomains' raw weights at each point,

$$
w_{m,n}(x,y) = \frac{\varphi_{c_{m,n}}(x,y)}{\displaystyle\sum_{m',n'} \varphi_{c_{m',n'}}(x,y)},
$$

makes $\\{w_{m,n}\\}$ a genuine partition of unity: $\sum_{m,n} w_{m,n} \equiv 1$ wherever the denominator is non-zero, which is guaranteed everywhere on $[0,1]^2$ by the overlap condition $H > h_M$ above. This normalization is exactly what both `get_partition` implementations do, and is what the `__main__` block in each file checks numerically (`partition_of_unity`, via `np.allclose`) together with full coverage of the grid (`cover_all`).

The two files differ only in the choice of $\mathrm{bump}$.

## Construction 1: raised cosine (`cos_partition.py`)

$$
\mathrm{bump}_{\cos}(x; H, K) = \cos\left(\frac{\pi x}{H}\right)^{K} \cdot \mathbb{1}_{|x| < H/2}.
$$

This takes a single fixed smooth shape, $\cos(\pi x / H)$, which already vanishes at the subdomain edges $x = \pm H/2$, and sharpens its contact with $0$ there by raising it to the power $K$: near the edge, writing $s = H/2 - |x| \to 0^+$, $\cos(\pi x/H) = \sin(\pi s / H) = O(s)$, so $\mathrm{bump}_{\cos} = O(s^K)$ — the function and its first $K-1$ derivatives vanish at the boundary, giving a globally $C^{K-1}$ weight once combined with the indicator.

## Construction 2: generalized smoothstep (`smoothstep_partition.py`)

Instead of exponentiating a fixed shape, this construction uses a **different polynomial for every $K$**. Let

$$
S_K(t) = t^{K+1}\sum_{n=0}^{K} \binom{K+n}{n}\binom{2K+1}{K-n}(-t)^n, \qquad t \in [0,1],
$$

the unique degree-$(2K+1)$ polynomial with $S_K(0)=0$, $S_K(1)=1$, and $K$ vanishing derivatives at both $t=0$ and $t=1$ (Perlin's "smootherstep" family; $S_0(t)=t$, $S_1(t)=3t^2-2t^3$, $S_2(t)=6t^5-15t^4+10t^3$). Apply it to the smooth, edge-vanishing quadratic profile $t(x) = 1-(2x/H)^2$:

$$
\mathrm{bump}_{S}(x; H, K) = S_K\left(\max\left(0,~ 1-\left(\frac{2x}{H}\right)^2\right)\right) \cdot \mathbb{1}_{|x| < H/2}.
$$

Because $t(x)$ vanishes linearly at the edge ($t = O(s)$ with $s = H/2-|x|\to0^+$) and $S_K(t) = O(t^{K+1})$ near $t=0$, $\mathrm{bump}_{S} = O(s^{K+1})$ — one order of vanishing higher than $\mathrm{bump}_{\cos}$ at the same $K$, giving a globally $C^{K}$ weight. This was confirmed numerically by fitting the decay exponent of both bumps as $x \to (H/2)^-$: the raised cosine decays as $s^{K}$ and the smoothstep bump as $s^{K+1}$, for $K = 1, 2, 3$.

## Testing

Both files' `__main__` blocks run the same four `(bc, H, K, M)` parameter combinations through `get_partition` and check two properties for each:

* **`partition_of_unity`** — accumulating every subdomain's returned weights back into a full-size array and comparing it to the all-ones array with `np.allclose`.
* **`cover_all`** — checking that the union of all subdomains' returned indices is exactly `range(N_x*N_y)`, i.e. every grid point belongs to at least one subdomain.

`validate_parameters(H, N, M, K)` is run first and gates whether `get_partition` is called at all, rejecting parameters where the subdomains would be too small to overlap ($H \le 1/M$) or where $K$ is unreasonably large ($K > 8$).

## Indicator cover (`indicator_partition.py`)

For experiments without a partition of unity, `indicator_partition.get_partition` returns the same subdomains, indices and coordinates as `cos_partition.get_partition` but with the constant weight $w_{m,n}\equiv 1$ on every subdomain (`K` is accepted and ignored). The weights are not normalised, so they sum to the number of subdomains containing a point (up to 4 for $H<2/M$) instead of $1$: this is a cover of the domain, not a partition of unity. Its `__main__` block checks that the weights are all $1$ and that every grid point is covered.
