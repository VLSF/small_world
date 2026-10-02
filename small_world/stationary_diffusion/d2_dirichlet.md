# Discretization of the stationary diffusion equation (Dirichlet BCs)

`d2_dirichlet.py` discretizes the stationary (anisotropic) diffusion equation

$$-\frac{\partial}{\partial x}\left(a_1(x, y)\frac{\partial u(x, y)}{\partial x}\right) -\frac{\partial}{\partial y}\left(a_2(x, y)\frac{\partial u(x, y)}{\partial y}\right) = f(x, y), \qquad (x, y) \in (0, 1)^2,$$

with homogeneous Dirichlet boundary conditions on all four sides of the square

$$
u(x, 0) = u(0, y) = u(x, 1) = u(1, y) = 0.
$$

The diffusion coefficients $a_1, a_2 > 0$ may differ from one another and are allowed to depend on both $x$ and $y$.

## Grid

Let $N$ be the number of interior grid points per direction and

$$
h = \frac{1}{N+1}.
$$

Nodes are placed on a uniform grid covering $[0, 1]$, including the boundary:

$$
x_i = (i+1)~h, \qquad y_j = (j+1)~h, \qquad i, j = -1, 0, 1, \dots, N,
$$

so that $x_{-1} = y_{-1} = 0$ and $x_N = y_N = 1$ are the boundary nodes, and $i, j = 0, \dots, N-1$ enumerate the $N \times N$ interior unknowns $u_{i,j} \approx u(x_i, y_j)$ (this is the 0-based indexing used in the code). The boundary values are known from the Dirichlet condition:

$$
u_{-1,j} = u_{N,j} = u_{i,-1} = u_{i,N} = 0.
$$

Fluxes are evaluated at the $N+1$ half-points lying between consecutive nodes (boundary nodes included):

$$
x_{i+1/2} = x_i + \frac{h}{2}, \qquad i = -1, 0, \dots, N-1,
$$

and likewise for $y_{j+1/2}$. Correspondingly, define the coefficients sampled at these faces:

$$
a_1^{i+1/2,~j} = a_1(x_{i+1/2}, y_j), \qquad a_2^{i,~j+1/2} = a_2(x_i, y_{j+1/2}).
$$

## Finite-difference approximation

The scheme is the standard conservative (flux-form) second-order finite difference approximation of a divergence of a flux. For the $x$-derivative term, the flux $-a_1 u_x$ is approximated at the half-points by a central difference,

$$
\left.a_1 \frac{\partial u}{\partial x}\right|_{x_{i+1/2}, y_j} \approx a_1^{i+1/2,~j}~\frac{u_{i+1,j} - u_{i,j}}{h},
$$

and the second derivative is obtained by differencing these fluxes again:

$$
\left.-\frac{\partial}{\partial x}\left(a_1 \frac{\partial u}{\partial x}\right)\right|_{x_i, y_j}
\approx -\frac{a_1^{i+1/2,~j}\left(u_{i+1,j}-u_{i,j}\right) - a_1^{i-1/2,~j}\left(u_{i,j}-u_{i-1,j}\right)}{h^2}.
$$

The $y$-derivative term is discretized analogously with $a_2$ and half-points in $y$. Summing both contributions and multiplying through by $h^2$ gives, for every interior node $(i, j)$, $i, j = 0, \dots, N-1$, the five-point stencil equation

$$\left(a_1^{i+1/2,j} + a_1^{i-1/2,j} + a_2^{i,j+1/2} + a_2^{i,j-1/2}\right)~ u_{i,j} - a_1^{i+1/2,j}~ u_{i+1,j} - a_1^{i-1/2,j}~ u_{i-1,j} - a_2^{i,j+1/2}~ u_{i,j+1} - a_2^{i,j-1/2}~ u_{i,j-1} = f_{i,j}~ h^2,$$

where $f_{i,j} = f(x_i, y_j)$ and boundary terms ($u_{-1,j}$, $u_{N,j}$, $u_{i,-1}$, $u_{i,N}$) are replaced by $0$ and simply dropped from the equation.

## Linear system

Collecting the unknowns $u_{i,j}$ into a single vector with the lexicographic (row-major) ordering

$$
k(i, j) = j + i N, \qquad i, j = 0, \dots, N-1,
$$

the stencil above defines a sparse linear system

$$
A~ \mathbf{u} = \mathbf{f}~ h^2,
$$

where $A$ is an $N^2 \times N^2$ matrix with, for each row $k(i,j)$:

* a diagonal entry $a_1^{i+1/2,j} + a_1^{i-1/2,j} + a_2^{i,j+1/2} + a_2^{i,j-1/2}$,
* an off-diagonal entry $-a_1^{i+1/2,j}$ in column $k(i+1,j)$ (when $i+1 \le N-1$),
* an off-diagonal entry $-a_1^{i-1/2,j}$ in column $k(i-1,j)$ (when $i-1 \ge 0$),
* an off-diagonal entry $-a_2^{i,j+1/2}$ in column $k(i,j+1)$ (when $j+1 \le N-1$),
* an off-diagonal entry $-a_2^{i,j-1/2}$ in column $k(i,j-1)$ (when $j-1 \ge 0$).

The matrix $A$ is symmetric positive definite since $a_1, a_2 > 0$ (each off-diagonal pair $A_{k(i,j),~k(i+1,j)}$ and $A_{k(i+1,j),~k(i,j)}$ shares the same coefficient $-a_1^{i+1/2,j}$).

## Order of accuracy

Both the flux approximation and the outer difference are central and second-order accurate, so the local truncation error is $O(h^2)$, and for a smooth solution $u$ the discretization converges as

$$
\|\mathbf{u}_h - u\| = O(h^2)
$$

in the discrete norms used in `discretization_test()` and `discretization_test_anisotropic()`, which is confirmed numerically by fitting the observed error decay in $\log N$ vs $\log(\mathrm{error})$.
