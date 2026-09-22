---
summary: The directions a matrix does not rotate, why repeated multiplication is governed by the largest one, and where that shows up in training stability.
prereqs: [matrices-as-linear-maps, rank-span-subspaces]
---

Most vectors get rotated when you apply a matrix. A few do not — they only get
stretched. Those special directions expose the structure of the map, and they control
what happens when you apply it over and over, which is exactly what a deep network and
an optimizer both do.

## The definition

A non-zero vector $\mathbf{v}$ is an **eigenvector** of a square matrix $A$ with
**eigenvalue** $\lambda$ if

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

The map does nothing to $\mathbf{v}$ except scale it. The eigenvalue records the
scaling: $|\lambda| > 1$ stretches, $|\lambda| < 1$ shrinks, $\lambda < 0$ flips,
$\lambda = 0$ means $\mathbf{v}$ is in the null space.

Rearranging gives $(A - \lambda I)\mathbf{v} = \mathbf{0}$, which has a non-zero
solution only when $A - \lambda I$ is singular. So the eigenvalues are the roots of the
**characteristic polynomial** $\det(A - \lambda I) = 0$. That is how you find them by
hand for a $2\times 2$; for anything larger, use an iterative algorithm and never a
determinant.

```python
import torch

A = torch.tensor([[4.0, 1.0],
                  [2.0, 3.0]])

values, vectors = torch.linalg.eig(A)
values.real                       # tensor([5., 2.])
# Check the definition on the first eigenpair.
v = vectors[:, 0].real
torch.allclose(A @ v, 5.0 * v, atol=1e-5)   # True
```

## Diagonalisation: the change-of-basis view

When $A \in \mathbb{R}^{n\times n}$ has $n$ independent eigenvectors, stack them as the
columns of $V$ and put the eigenvalues on the diagonal of $\Lambda$:

$$
A = V \Lambda V^{-1}
$$

Read right to left, this says: **change into the eigenbasis, scale each axis
independently, change back.** Every diagonalisable matrix is a diagonal matrix wearing
a disguise.

The payoff is powers. Because the inner $V^{-1}V$ pairs cancel,

$$
A^k = V \Lambda^k V^{-1}
$$

Raising a matrix to the hundredth power costs one eigendecomposition and $n$ scalar
powers. More importantly, it tells you what happens asymptotically.

::: key
Repeated multiplication is governed entirely by the largest eigenvalue in magnitude,
the **spectral radius** $\rho(A) = \max_i |\lambda_i|$.

- $\rho(A) > 1$ — components blow up exponentially.
- $\rho(A) < 1$ — everything decays to zero exponentially.
- $\rho(A) = 1$ — the marginal case, and the only one that is stable.
:::

::: check
A matrix $A$ is applied one hundred times in a row to a vector. Which quantity decides whether the result blows up or dies?

- [x] The spectral radius $\rho(A) = \max_i|\lambda_i|$
  > $A^k = V\Lambda^k V^{-1}$, so each eigendirection scales by $\lambda_i^k$. The largest magnitude dominates everything else within a few steps.
- [ ] The determinant of $A$
  > The determinant is the *product* of the eigenvalues. A matrix with eigenvalues $10$ and $0.1$ has determinant 1 and still explodes along the first direction.
- [ ] The trace of $A$
  > The trace is the sum of the eigenvalues, which can be small or zero while an individual eigenvalue is large.
- [ ] The Frobenius norm of $A$
  > It bounds the spectral radius from above but does not determine it, and a matrix can have a large norm with every eigenvalue inside the unit circle.
:::

## This is the vanishing gradient problem

A recurrent network applies roughly the same weight matrix at every timestep. Over $T$
steps, gradients flowing backward are multiplied by something like $(W^\top)^T$. Apply
the rule above:

- If $\rho(W) < 1$, gradients decay like $\rho^T$. At $\rho = 0.9$ and $T = 100$, that
  is $2.6\times10^{-5}$ — the model cannot learn long-range dependencies because no
  signal reaches back that far.
- If $\rho(W) > 1$, gradients explode. At $\rho = 1.1$ and $T = 100$, the factor is
  $13{,}780$, and training diverges in one step.

The window between "vanishes" and "explodes" is vanishingly narrow, which is why plain
RNNs are so hard to train and why LSTMs (lesson 3.12) introduce a gated path whose
effective multiplier sits near 1 by construction. Residual connections (lesson 3.08)
solve the same problem in feed-forward networks by making the per-block Jacobian
$I + \partial F$, whose eigenvalues cluster around 1 instead of around 0.

```python
import torch

W = torch.randn(64, 64) * 0.15            # deliberately small spectral radius
rho = torch.linalg.eigvals(W).abs().max()
print(f'spectral radius {rho:.3f}')

x = torch.randn(64)
for t in [1, 10, 50, 100]:
    y = torch.linalg.matrix_power(W, t) @ x
    print(f'step {t:3d}  norm {y.norm():.3e}')
# norms fall off like rho**t -- this is gradient vanishing, in miniature
```

::: check
An RNN's recurrent weight has $\rho(W) = 0.9$. Over 100 timesteps, roughly what factor multiplies the gradient flowing back?

- [x] About $2.6\times10^{-5}$ — the signal effectively does not reach
  > $0.9^{100} \approx 2.6\times10^{-5}$. The model cannot learn a dependency across that span because nothing arrives to learn from. This is the vanishing gradient problem, exactly.
- [ ] About $0.9$ — the decay is per-application, not cumulative
  > It compounds. Each step multiplies again, so the exponent is the number of steps.
- [ ] About $90$ — repeated application accumulates magnitude
  > A factor below 1 shrinks. Above 1 is the exploding case: $1.1^{100} \approx 13{,}780$.
- [ ] It depends on the input, not on $W$
  > The input sets the starting vector, but the growth rate over many steps is set by $W$'s spectrum.
:::

## Symmetric matrices are the nice case

If $A = A^\top$, three things become true at once, and they are worth knowing because
covariance matrices, Hessians and Gram matrices are all symmetric:

1. All eigenvalues are **real** — no complex arithmetic.
2. Eigenvectors for distinct eigenvalues are **orthogonal**, so $V$ can be chosen
   orthogonal and $V^{-1} = V^\top$.
3. The decomposition $A = V\Lambda V^\top$ always exists.

A symmetric matrix with all $\lambda_i > 0$ is **positive definite**, equivalent to
$\mathbf{x}^\top A \mathbf{x} > 0$ for all non-zero $\mathbf{x}$. That condition is
exactly what makes a quadratic form a bowl rather than a saddle, and lesson 1.09 uses
it to classify critical points of a loss surface.

::: warning
Eigenvalues only exist for square matrices, and diagonalisation can fail even then — a
matrix with repeated eigenvalues may not have enough independent eigenvectors. The SVD
in lesson 1.06 has neither restriction: it exists for every matrix of every shape. When
you are unsure which tool you want, you want the SVD.
:::

::: exercise
The condition number of a symmetric positive definite matrix is $\kappa =
\lambda_{\max}/\lambda_{\min}$. Gradient descent on a quadratic with Hessian $H$
converges at a rate governed by $\kappa$. Explain why a large $\kappa$ is slow, in
terms of eigenvectors.
:::

::: solution
In the eigenbasis of $H$ the problem decouples into $n$ independent one-dimensional
problems, where direction $i$ has curvature $\lambda_i$. Gradient descent with step
size $\eta$ multiplies the error in direction $i$ by $(1 - \eta\lambda_i)$ each step.

Stability requires $\eta < 2/\lambda_{\max}$, so the step size is capped by the
*steepest* direction. But convergence along the *flattest* direction goes like
$(1 - \eta\lambda_{\min})$, and with $\eta \approx 1/\lambda_{\max}$ that factor is
$1 - 1/\kappa$ — arbitrarily close to 1 when $\kappa$ is large.

Geometrically, the loss surface is a long narrow valley: you must take small steps to
avoid oscillating across the narrow axis, so you creep along the long one. Momentum,
Adam's per-parameter scaling, and every normalization layer in lesson 3.06 are attacks
on exactly this problem.
:::

::: check
You have a non-square matrix and want its principal directions. Which tool applies?

- [x] The SVD, which exists for every matrix of every shape
  > Eigenvalues need a square matrix, and even then diagonalisation can fail with repeated eigenvalues. The SVD has neither restriction — lesson 1.06 is the next page for a reason.
- [ ] Eigendecomposition, after padding the matrix to square with zeros
  > Padding changes the map. The eigenvectors you would get describe the padded matrix, not the original.
- [ ] Eigendecomposition of $A^\top A$, which is equivalent
  > This is closer than it looks — the eigenvectors of $A^\top A$ *are* the right singular vectors — but forming $A^\top A$ squares the condition number and loses precision. Compute the SVD directly.
- [ ] Neither; a non-square matrix has no principal directions
  > It has two sets of them, one in the input space and one in the output space. That is precisely what the SVD gives you.
:::

## What to carry forward

- Eigenvectors are the directions a matrix only scales; eigenvalues are the scale factors.
- $A^k = V\Lambda^k V^{-1}$, so repeated application is governed by the spectral radius.
- $\rho > 1$ explodes and $\rho < 1$ vanishes — the RNN training problem in one line.
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors; positive definite means a bowl.
- Condition number $\lambda_{\max}/\lambda_{\min}$ predicts how slowly gradient descent will crawl.
