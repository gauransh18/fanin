---
summary: What convexity guarantees, why neural networks have none of it, and which of the intuitions survive anyway.
prereqs: [hessians-and-curvature, derivatives-gradients-jacobians]
---

Optimisation theory is built on convexity, and neural network losses are not convex.
That does not make the theory useless — it tells you precisely which guarantees you have
given up, and which behaviours you should still expect.

## The definition

A set is convex if the straight line between any two of its points stays inside it. A
function is **convex** if the chord between any two points on its graph lies above it:

$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})
$$

for all $\lambda \in [0,1]$. Twice-differentiable functions have a cleaner test: $f$ is
convex exactly when its Hessian is positive semi-definite everywhere — no negative
curvature in any direction, at any point.

## What convexity buys

**Every local minimum is global.** There is nowhere to get stuck. This is the guarantee
that everything else rests on.

**Gradient descent converges, with a known rate.** For an $L$-smooth convex function with
step size $\eta = 1/L$:

$$
f(\mathbf{x}_k) - f^* \le \frac{L\lVert \mathbf{x}_0 - \mathbf{x}^*\rVert^2}{2k}
$$

Error falls like $O(1/k)$. Add strong convexity — curvature bounded below by $\mu > 0$ —
and the rate becomes geometric:

$$
f(\mathbf{x}_k) - f^* \le \left(1 - \frac{\mu}{L}\right)^k \left(f(\mathbf{x}_0) - f^*\right)
$$

with $L/\mu$ being the condition number $\kappa$ of lesson 1.09. Every intuition about
$\kappa$ from that lesson is formally justified here.

**A zero gradient means you are done.** In the non-convex world it means nothing of the
sort.

## Convex problems you already use

Least squares, logistic regression, softmax regression, SVMs, LASSO, and any
exponential-family MLE (lesson 1.12) are all convex. So is every one of these with L1 or
L2 regularisation added, since both penalties are convex and sums of convex functions
are convex.

That closure property is worth remembering: convexity survives non-negative weighted
sums, composition with affine maps, and pointwise maxima. It is how you check a new
objective quickly.

## Then you add a hidden layer

A single linear layer with cross-entropy is convex in its weights. Compose two layers and
it is not, for a reason that has nothing to do with the loss: **permutation symmetry**.

Swap two hidden units — along with their incoming and outgoing weights — and the network
computes exactly the same function. So every minimum has at least $h!$ identical copies
for $h$ hidden units. A function with multiple separated global minima cannot be convex,
because the midpoint between two of them would have to lie below the chord.

::: warning
This means the loss landscape has an astronomically large number of equivalent global
minima — $h!$ is beyond $10^{100}$ for a layer of width 70. Non-convexity here is not a
sign of a hard problem; it is a sign of redundant parameterisation. The interesting
question is not "is it convex" but "are the minima you can reach good ones", and
empirically they mostly are.
:::

::: check
A two-layer network's loss is non-convex. What does permutation symmetry contribute to that?

- [x] Swapping two hidden units with their weights gives an identical function, so every minimum has at least $h!$ copies — and a function with separated global minima cannot be convex
  > At width 70 that is past $10^{100}$ equivalent optima. Non-convexity from this source is a sign of redundant parameterisation rather than a hard problem, which is why the useful question is whether reachable minima are good, not whether the loss is convex.
- [ ] It makes the Hessian indefinite at every point
  > The Hessian is indefinite in many places, but permutation symmetry is an argument about *multiple global minima*, which rules out convexity on its own.
- [ ] It creates spurious local minima that trap gradient descent
  > The copies are all global minima of equal value. Landing in any of them is the same outcome.
- [ ] It only applies to networks with more than one hidden layer
  > One hidden layer of width $h$ already has $h!$ orderings.
:::

## Which intuitions survive

**Condition number still governs speed.** Even without global guarantees, the local
quadratic approximation around your current point has a condition number, and it predicts
progress. This is why normalization and adaptive methods help.

**Smoothness still caps the step size.** $\eta < 2/L$ remains the local stability bound.
Exceeding it produces the loss spike everyone recognises.

**Saddles, not minima, are the obstacle.** As lesson 1.09 argued, high-dimensional
critical points are overwhelmingly saddles. SGD's gradient noise is enough to escape
them, which is a real advantage over exact methods.

**Convergence is to a region, not a point.** With constant step size and stochastic
gradients, SGD converges to a noise ball of radius proportional to $\eta\sigma$. This is
exactly why learning-rate decay is necessary — shrinking $\eta$ shrinks the ball. Lesson
2.09 covers the schedules.

```python
import torch

# Strongly convex quadratic: the theoretical rate is observable.
torch.manual_seed(0)
n = 50
A = torch.randn(n, n); H = A.T @ A / n + 0.1 * torch.eye(n)
lam = torch.linalg.eigvalsh(H)
L, mu = lam.max(), lam.min()
print(f'L={L:.3f}  mu={mu:.3f}  kappa={L/mu:.1f}')

x = torch.randn(n)
eta = 1.0 / L
for k in range(1, 201):
    x = x - eta * (H @ x)
    if k in (10, 50, 100, 200):
        observed = 0.5 * (x @ H @ x)
        bound = (1 - mu / L) ** k * 0.5 * 1.0    # up to the initial constant
        print(f'step {k:3d}  f(x)={observed:.3e}  rate factor {(1-mu/L)**k:.3e}')
```

::: check
With a constant step size and stochastic gradients, SGD converges to a noise ball of radius proportional to $\eta\sigma$ rather than to a point. What follows?

- [x] The learning rate has to decay, since shrinking $\eta$ is what shrinks the ball
  > This is the reason every schedule in lesson 2.09 exists. A constant step size leaves you orbiting the optimum at a radius you chose without meaning to.
- [ ] Training should stop as soon as the loss plateaus
  > A plateau at constant $\eta$ often *is* the noise floor. Decaying the rate typically drops the loss again, which is exactly what a plateau-then-decay curve shows.
- [ ] The batch size must grow without bound
  > Growing the batch shrinks $\sigma$ and is a real alternative, but at $1/\sqrt{n}$ it is a far more expensive lever than decaying $\eta$.
- [ ] The model has not converged and the run is broken
  > It has converged, in the only sense available to constant-step-size SGD. The ball is the expected outcome, not a fault.
:::

## Non-convexity that is actually a problem

Not all of it is benign. Two cases genuinely hurt:

- **Poorly conditioned or badly initialised networks** can land in flat regions where
  gradients are near zero and no progress is made. This is what proper initialisation
  (lesson 3.05) exists to prevent.
- **The RL objective** in track 6 is non-convex *and* non-stationary — the data
  distribution changes as the policy changes. That combination is far harder than
  supervised non-convexity, and it is why RL training is so much less reliable.

::: exercise
Is cross-entropy loss convex in the model's *logits*? Is it convex in the model's
*parameters*? Explain the difference and why it matters.
:::

::: solution
**In the logits: yes.** For a fixed true class $c$, the loss is
$\mathcal{L}(\mathbf{z}) = \log\sum_j e^{z_j} - z_c$. The log-sum-exp function is convex
(its Hessian is $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, which is positive
semi-definite), and subtracting a linear term preserves convexity.

**In the parameters: only for a linear model.** If $\mathbf{z} = W\mathbf{x}$, then the
loss is a convex function composed with an affine map, which stays convex — this is why
softmax regression is a convex problem. If $\mathbf{z} = f_\theta(\mathbf{x})$ for a deep
$f$, convexity is destroyed, because a convex function composed with a *non-affine* map
is generally non-convex.

**Why it matters:** the convexity in logit space is what makes the loss well-behaved
locally — gradients point sensibly and the per-example landscape has no spurious
structure. All the difficulty is contributed by the parameter-to-logit map. That
separation is exactly why techniques acting on the logit side (label smoothing, logit
clipping, temperature) are reliable, while techniques acting on the parameter side
(initialisation, normalization) are the ones that need care.
:::

## What to carry forward

- Convex means every local minimum is global, and rates are $O(1/k)$ or geometric.
- Neural networks are non-convex mostly because of permutation symmetry, not difficulty.
- Condition number, smoothness limits and saddle-dominance all survive into the non-convex case.
- Constant-step SGD converges to a noise ball, which is why you must decay the learning rate.
