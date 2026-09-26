# Discretization of the stationary diffusion equation (mixed Neumann/Dirichlet BCs)

`d2_neumann.py` discretizes the same stationary anisotropic diffusion equation as in
[`d2_dirichlet.md`](d2_dirichlet.md),

$$
-\frac{\partial}{\partial x}\left(a_1(x, y)\frac{\partial u(x, y)}{\partial x}\right)
-\frac{\partial}{\partial y}\left(a_2(x, y)\frac{\partial u(x, y)}{\partial y}\right) = f(x, y),
\qquad (x, y) \in (0, 1)^2,
$$

but with homogeneous Neumann conditions on three sides of the square and a homogeneous Dirichlet condition on the fourth:

$$
\frac{\partial u}{\partial x}(0, y) = 0, \qquad
\frac{\partial u}{\partial y}(x, 0) = 0, \qquad
\frac{\partial u}{\partial y}(x, 1) = 0, \qquad
u(1, y) = 0.
$$

## Grid

Let $N$ be the number of grid intervals per direction and

$$
h = \frac{1}{N}.
$$

Nodes are placed on a uniform grid covering $[0, 1]$ in each direction:

$$
x_i = i\,h, \quad i = 0, \dots, N, \qquad\qquad y_j = j\,h, \quad j = 0, \dots, N.
$$

* In $x$, the node $x_N = 1$ carries the known Dirichlet value $u(1, y) = 0$ and is eliminated from the system, leaving $N$ unknowns $i = 0, \dots, N-1$. The node $x_0 = 0$ is the Neumann boundary and **is** kept as an unknown.
* In $y$, both endpoints $y_0 = 0$ and $y_N = 1$ are Neumann boundaries, so all $N+1$ nodes $j = 0, \dots, N$ are kept as unknowns.

The set of unknowns is therefore $u_{i,j} \approx u(x_i, y_j)$ for $i = 0, \dots, N_x - 1$ and $j = 0, \dots, N_y - 1$, with $N_x = N$ and $N_y = N + 1$ (the notation used in the code).

As in the Dirichlet discretization, fluxes are evaluated at half-points between consecutive nodes,

$$
x_{i+1/2} = x_i + \frac{h}{2}, \qquad y_{j+1/2} = y_j + \frac{h}{2},
$$

with coefficients sampled there:

$$
a_1^{i+1/2,\,j} = a_1(x_{i+1/2}, y_j), \qquad a_2^{i,\,j+1/2} = a_2(x_i, y_{j+1/2}).
$$

There are $N_x$ faces in $x$ ($i = 0, \dots, N_x-1$, the last one $x_{N_x - 1/2}$ lying next to the eliminated Dirichlet node) and $N_y - 1$ faces in $y$ ($j = 0, \dots, N_y - 2$).

## Interior stencil and the Dirichlet side

Away from the Neumann boundaries the discretization is the same conservative, central, second-order finite-difference stencil used for the Dirichlet problem: for $1 \le i \le N_x - 1$ and $1 \le j \le N_y - 2$,

$$
\Big(a_1^{i+1/2,j} + a_1^{i-1/2,j} + a_2^{i,j+1/2} + a_2^{i,j-1/2}\Big)\, u_{i,j}
- a_1^{i+1/2,j}\, u_{i+1,j} - a_1^{i-1/2,j}\, u_{i-1,j}
- a_2^{i,j+1/2}\, u_{i,j+1} - a_2^{i,j-1/2}\, u_{i,j-1}
= f_{i,j}\, h^2 .
$$

At $i = N_x - 1$ (the node next to the Dirichlet boundary $x_N = 1$) the term $a_1^{i+1/2,j}\,u_{i+1,j}$ is dropped because $u_{N_x,j} = 0$, while the coefficient $a_1^{i+1/2,j}$ still contributes to the diagonal — exactly as at the Dirichlet boundaries in `d2_dirichlet.py`.

## Neumann boundaries: mirror ghost-node trick

At a Neumann boundary the derivative, not the value, is prescribed, so there is no boundary value to substitute. Instead, a standard second-order technique is used: introduce a fictitious *ghost* node just outside the domain, and choose its value to be the mirror image of the first interior value across the boundary; the flux coefficient on the (fictitious) face connecting the ghost node is likewise mirrored, i.e. taken equal to the coefficient at the real face right next to the boundary.

**Boundary $x = 0$ ($i = 0$).** The ghost node $i = -1$ is set to $u_{-1,j} := u_{1,j}$, and $a_1^{-1/2,j} := a_1^{1/2,j}$. The $x$-part of the flux balance at $i=0$ reads, before substitution,

$$
-\Big(a_1^{1/2,j}(u_{1,j}-u_{0,j}) - a_1^{-1/2,j}(u_{0,j}-u_{-1,j})\Big),
$$

and substituting $u_{-1,j}=u_{1,j}$, $a_1^{-1/2,j}=a_1^{1/2,j}$ collapses the two flux terms into a single, doubled connection to the interior neighbor $i=1$:

$$
2\,a_1^{1/2,j}\, u_{0,j} \;-\; 2\,a_1^{1/2,j}\, u_{1,j} .
$$

Adding the (unmodified) $y$-part gives the full equation at $(0,j)$, $=f_{0,j}h^2$.

**Boundary $y = 0$ ($j = 0$)** and **boundary $y = 1$ ($j = N_y-1$)** are treated the same way, mirroring across each respective side, and again contribute only to the $y$-part of the equation:

$$
j=0: \qquad 2\,a_2^{i,1/2}\, u_{i,0} - 2\,a_2^{i,1/2}\, u_{i,1},
$$

$$
j=N_y-1: \qquad 2\,a_2^{i,N_y-3/2}\, u_{i,N_y-1} - 2\,a_2^{i,N_y-3/2}\, u_{i,N_y-2} .
$$

Since the $x$- and $y$-contributions are formed independently (the code builds `diag_x`/`diag_y` and the four off-diagonal weights separately and adds them), a node at a corner (e.g. $i=0, j=0$) simply combines the doubled rule from each direction, giving diagonal $2a_1^{1/2,0} + 2a_2^{0,1/2}$ and two doubled off-diagonal connections, to $(1,0)$ and to $(0,1)$.

## Full assembled system

Collecting the unknowns with the lexicographic ordering used in the code,

$$
k(i,j) = j + i\, N_y, \qquad i = 0,\dots,N_x-1,\quad j = 0,\dots,N_y-1,
$$

gives a sparse linear system

$$
A\, \mathbf{u} = \mathbf{f}\, h^2 .
$$

The diagonal entry at row $k(i,j)$ is always the sum of an $x$-contribution and a $y$-contribution, each computed independently by the following rule (this is exactly what `diag_x`/`diag_y` do in the code):

| direction | case | contribution to the diagonal |
|---|---|---|
| $x$ | interior, $0 < i < N_x - 1$ | $a_1^{i+1/2,j} + a_1^{i-1/2,j}$ |
| $x$ | Neumann side, $i = 0$ | $2\,a_1^{1/2,j}$ |
| $x$ | Dirichlet side, $i = N_x - 1$ | $a_1^{i+1/2,j} + a_1^{i-1/2,j}$ (unchanged; the boundary value is $0$) |
| $y$ | interior, $0 < j < N_y - 1$ | $a_2^{i,j+1/2} + a_2^{i,j-1/2}$ |
| $y$ | Neumann side, $j = 0$ | $2\,a_2^{i,1/2}$ |
| $y$ | Neumann side, $j = N_y - 1$ | $2\,a_2^{i,N_y - 3/2}$ |

and the off-diagonal entries of row $k(i,j)$ are:

| connection | column | weight |
|---|---|---|
| to $i+1$ (exists for $i < N_x - 1$) | $k(i+1,j)$ | $-a_1^{i+1/2,j}$, doubled to $-2a_1^{1/2,j}$ when $i=0$ |
| to $i-1$ (exists for $i > 0$) | $k(i-1,j)$ | $-a_1^{i-1/2,j}$ (never doubled; the $i=0$ row has no such column) |
| to $j+1$ (exists for $j < N_y - 1$) | $k(i,j+1)$ | $-a_2^{i,j+1/2}$, doubled to $-2a_2^{i,1/2}$ when $j=0$ |
| to $j-1$ (exists for $j > 0$) | $k(i,j-1)$ | $-a_2^{i,j-1/2}$, doubled to $-2a_2^{i,N_y-3/2}$ when $j=N_y-1$ |

A corner node, e.g. $(i,j) = (0,0)$, simply combines the $x=0$ and $y=0$ rules: diagonal $2a_1^{1/2,0} + 2a_2^{0,1/2}$, with doubled off-diagonal connections to $(1,0)$ and to $(0,1)$.

**Note on symmetry.** Unlike the pure-Dirichlet discretization, $A$ is generally **not symmetric**: e.g. the connection from node $(0,j)$ to $(1,j)$ carries weight $-2a_1^{1/2,j}$, while the connection from $(1,j)$ back to $(0,j)$ carries the standard, undoubled weight $-a_1^{1/2,j}$. This is the well-known behavior of the ghost/mirror technique for Neumann boundaries (equivalently, the boundary node's associated control volume is half the width of an interior one, and the equation there is not divided by this smaller volume before assembling $A\mathbf u = \mathbf f h^2$ with a uniform right-hand-side scaling). The scheme remains a consistent, second-order accurate approximation; only its matrix representation loses symmetry.

## Order of accuracy

The mirror technique reproduces the Neumann condition to first order pointwise at the boundary row, but — as is standard for this construction — this does not degrade the global accuracy of the scheme: with a manufactured solution satisfying the prescribed boundary conditions exactly, `discretization_test()` in `d2_neumann.py` observes

$$
\|\mathbf{u}_h - u\| = O(h^2),
$$

confirming second-order convergence, consistent with the Dirichlet discretization in `d2_dirichlet.py`.
