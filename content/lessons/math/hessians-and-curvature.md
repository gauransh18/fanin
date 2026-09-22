---
summary: The second derivative tells you the shape of the loss surface. Why nobody computes it exactly, and what Adam is approximating instead.
prereqs: [derivatives-gradients-jacobians, eigenvalues-eigenvectors]
---

The gradient tells you which way is downhill. It says nothing about how far you can
safely step. That information lives in the second derivative, and every adaptive
optimizer is a cheap guess at it.

## The Hessian

For $f : \mathbb{R}^n \to \mathbb{R}$, the Hessian is the matrix of second partials:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

It is symmetric whenever $f$ is twice continuously differentiable, because mixed
partials commute. Symmetry matters: it means all the machinery of lesson 1.05 applies —
real eigenvalues, orthogonal eigenvectors, a clean diagonalisation.

The second-order Taylor expansion is

$$
f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f^\top \boldsymbol{\delta} + \tfrac12 \boldsymbol{\delta}^\top H \boldsymbol{\delta}
$$

The quadratic term is the curvature: how fast the gradient itself changes as you move.

## Eigenvalues classify critical points

At a point where $\nabla f = \mathbf{0}$, the Hessian's eigenvalues say what kind of
point it is:

| Eigenvalues | Point | Shape |
|---|---|---|
| All $> 0$ | Local minimum | Bowl |
| All $< 0$ | Local maximum | Dome |
| Mixed signs | Saddle | Pringle |
| Some $= 0$ | Degenerate | Flat direction |

In two dimensions, saddles are rare curiosities. In $10^9$ dimensions they are
overwhelmingly the common case: for a critical point to be a true minimum, *all*
$10^9$ eigenvalues must happen to be positive. Random-matrix arguments say that almost
never happens.

::: insight
The old worry that neural networks get "stuck in bad local minima" was largely
misplaced. High-dimensional loss surfaces are dominated by saddles, not minima, and
saddles are escapable — the gradient is zero but any perturbation along a negative
eigenvalue direction falls away. Stochastic gradient noise supplies that perturbation
for free, which is one reason SGD works better than its deterministic counterpart.
:::

::: check
A loss surface in $10^9$ dimensions has a point where the gradient is exactly zero. What is it most likely to be?

- [x] A saddle — for a true minimum, all $10^9$ eigenvalues would have to be positive at once
  > Random-matrix arguments say that essentially never happens. This is why the old worry about networks getting stuck in bad local minima was largely misplaced: saddles dominate, and a perturbation along a negative-curvature direction escapes them.
- [ ] A local minimum, since the loss has been decreasing
  > A decreasing loss says the path was downhill, not that the destination is a bowl in every one of a billion directions.
- [ ] A local maximum, if the learning rate overshot
  > A maximum needs *all* eigenvalues negative, which is exactly as improbable as all positive.
- [ ] A global minimum, since overparameterised networks reach them easily
  > Overparameterisation makes many low-loss regions reachable, but it says nothing about the curvature at a given critical point.
:::

## Curvature sets your step size

Take a quadratic $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top H\mathbf{x}$ and run gradient
descent with step $\eta$. In the eigenbasis the problem decouples, and along direction
$i$ the error is multiplied by $(1 - \eta\lambda_i)$ each step.

Convergence requires $|1 - \eta\lambda_i| < 1$ for every $i$, so

$$
\eta < \frac{2}{\lambda_{\max}}
$$

The **largest** curvature caps your learning rate globally, while the **smallest**
curvature determines how slowly you converge. Their ratio $\kappa = \lambda_{\max}/\lambda_{\min}$
is the condition number, and the number of steps needed scales like $\kappa$.

This single inequality explains an enormous amount of practice:

- **Learning-rate warmup** exists because $\lambda_{\max}$ is large and erratic early in
  training; a small initial step avoids diverging before the surface settles.
- **Gradient clipping** bounds the damage when a batch lands in a high-curvature region.
- **Normalization layers** (lesson 3.06) reduce $\kappa$ directly by rescaling
  activations, which is the main reason they let you train at higher learning rates.

::: check
Gradient descent on a quadratic converges only when $|1 - \eta\lambda_i| < 1$ for every direction $i$. Which eigenvalue therefore caps the learning rate?

- [x] The largest, giving $\eta < 2/\lambda_{\max}$
  > One over-curved direction is enough to diverge, so the steepest direction sets the global cap — while the flattest sets how slowly you converge. Their ratio is the condition number, and warmup, clipping and normalization layers are all attacks on it.
- [ ] The smallest, since flat directions need the biggest steps
  > Flat directions would indeed like a bigger step; they do not get one, and that tension is precisely the problem.
- [ ] The mean eigenvalue, since the step acts on all directions at once
  > An average would permit divergence in the directions above it. Stability has to hold everywhere.
- [ ] None; on a quadratic any positive step size converges
  > Step past $2/\lambda_{\max}$ and the error in that direction grows every step.
:::

## Newton's method, and why you cannot use it

Minimising the quadratic approximation exactly gives the Newton step:

$$
\boldsymbol{\delta} = -H^{-1}\nabla f
$$

This rescales each direction by its own curvature, so the condition number becomes 1 and
convergence is quadratic. On a genuine quadratic it lands on the minimum in a single
step.

It is also completely impractical for neural networks:

- $H$ has $N^2$ entries. For $N = 7\times10^9$, that is $4.9\times10^{19}$ numbers.
- Inverting it costs $O(N^3)$.
- At a saddle, $H^{-1}$ points *toward* the saddle along negative-curvature directions —
  Newton's method converges to critical points of any type, including the ones you want
  to escape.

::: key
Every practical optimizer is a cheap approximation to $H^{-1}$.

- **SGD**: $H^{-1} \approx \eta I$. Ignores curvature entirely.
- **Momentum**: accumulates gradient history, which damps oscillation in
  high-curvature directions — an implicit smoothing of $\kappa$.
- **Adam**: $H^{-1} \approx \text{diag}(1/\sqrt{\mathbb{E}[g^2]})$. A diagonal
  approximation using gradient second moments as a curvature proxy. This is why Adam
  is robust to badly scaled parameters and why it costs 2× parameters in state.
- **Shampoo / K-FAC**: block or Kronecker-factored approximations, closer to the true
  $H$ at meaningfully higher cost.
:::

## Hessian-vector products are cheap

You cannot form $H$, but you can compute $H\mathbf{v}$ for any $\mathbf{v}$ at roughly
the cost of one extra backward pass, because

$$
H\mathbf{v} = \nabla_{\mathbf{x}} \left( \nabla f(\mathbf{x})^\top \mathbf{v} \right)
$$

Differentiate the scalar $\nabla f \cdot \mathbf{v}$ once more and you get the
Hessian-vector product without ever touching the matrix. This is the trick behind
power-iteration estimates of $\lambda_{\max}$, behind influence functions, and behind
every second-order method that is actually deployed.

```python
import torch

def hvp(f, x, v):
    """Hessian-vector product without materialising the Hessian."""
    grad = torch.autograd.grad(f(x), x, create_graph=True)[0]
    return torch.autograd.grad(grad @ v, x)[0]

x = torch.randn(5, requires_grad=True)
A = torch.randn(5, 5); A = A + A.T          # symmetric, so H = A exactly
f = lambda z: 0.5 * z @ A @ z

v = torch.randn(5)
torch.allclose(hvp(f, x, v), A @ v, atol=1e-5)     # True

# Power iteration for the top eigenvalue: only HVPs, never H.
u = torch.randn(5)
for _ in range(50):
    u = hvp(f, x, u)
    u = u / u.norm()
print((u @ hvp(f, x, u)).item(), torch.linalg.eigvalsh(A).abs().max().item())
```

::: exercise
Sharpness-aware minimisation claims that flat minima generalise better than sharp ones.
State that claim in terms of the Hessian, and give one reason it is plausible.
:::

::: solution
Flatness means the Hessian eigenvalues at the minimum are small: the loss changes
slowly in every direction, so $\mathbf{x}^\top H\mathbf{x}$ is small for all unit
$\mathbf{x}$. Sharpness is the opposite — at least one large $\lambda_i$.

The plausibility argument: the test loss surface is a perturbed version of the train
surface. If the minimum is flat, a shift of the surface by $\boldsymbol{\epsilon}$
raises the loss by only about $\tfrac12\boldsymbol{\epsilon}^\top H\boldsymbol{\epsilon}$,
which is small. At a sharp minimum the same shift can raise the loss dramatically.

The caveat worth knowing: flatness is not reparameterisation-invariant. Rescaling a
layer's weights up and the next layer's down leaves the function unchanged while
changing the Hessian's eigenvalues, so "flat" needs a careful definition before the
claim can be made precise.
:::

::: check
Adam approximates $H^{-1}$ with $\text{diag}(1/\sqrt{\mathbb{E}[g^2]})$. Compared with the exact Newton step, what does that buy and what does it give up?

- [x] $O(N)$ memory instead of $O(N^2)$, ignoring every off-diagonal coupling
  > Newton's $H$ has $N^2$ entries — $4.9\times10^{19}$ at 7B parameters — and inverting it is $O(N^3)$. A diagonal proxy is the most curvature you can afford per parameter.
- [ ] It converges quadratically, like Newton, but with less memory
  > Quadratic convergence comes from the *exact* inverse Hessian. A diagonal approximation does not deliver it.
- [ ] It avoids being attracted to saddle points
  > Newton's genuine defect is that $H^{-1}$ points towards critical points of any type. Adam is not immune by design; gradient noise is what escapes saddles in practice.
- [ ] It removes the need for a learning rate
  > Adam still has one. The per-parameter scaling changes the relative step sizes, not the need for a global one.
:::

## What to carry forward

- The Hessian is symmetric; its eigenvalues classify critical points.
- High-dimensional surfaces are full of saddles, not bad minima.
- $\eta < 2/\lambda_{\max}$ caps learning rate; $\kappa$ sets convergence speed.
- Newton's method is exact and unusable; Adam is a diagonal approximation to it.
- $H\mathbf{v}$ costs one extra backward pass and is how curvature is measured in practice.
