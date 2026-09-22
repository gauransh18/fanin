---
summary: The smallest useful model, what it can and cannot represent, and why its failure defined the field for a decade.
prereqs: [matrices-as-linear-maps, maximum-likelihood]
---

Everything in this track is an elaboration of one equation. Understanding exactly what
that equation can and cannot do is what makes the elaborations feel necessary rather than
arbitrary.

## The model

$$
\hat{y} = f(\mathbf{w}^\top\mathbf{x} + b)
$$

A weighted sum of inputs, a bias, and a function applied to the result. Three choices of
$f$ give three classical models:

| $f$ | Model | Output | Loss |
|---|---|---|---|
| identity | Linear regression | any real | MSE |
| sign | Perceptron | $\pm 1$ | perceptron rule |
| sigmoid | Logistic regression | $(0,1)$ | binary cross-entropy |

From lesson 1.02, $\mathbf{w}^\top\mathbf{x}$ is a dot product: it measures alignment
between the input and a learned direction. The bias shifts the threshold. So the model
asks one question — *how much does this input point along $\mathbf{w}$?* — and turns the
answer into a prediction.

## The decision boundary is a hyperplane

The set where $\mathbf{w}^\top\mathbf{x} + b = 0$ is a hyperplane: a line in 2-D, a plane
in 3-D, a $(n{-}1)$-dimensional flat in $\mathbb{R}^n$. $\mathbf{w}$ is its normal
vector, and the signed distance from a point to it is

$$
d = \frac{\mathbf{w}^\top\mathbf{x} + b}{\lVert\mathbf{w}\rVert_2}
$$

This is the entire geometry. A linear model can only ever split space with a flat cut.

## The perceptron learning rule

Rosenblatt's 1958 algorithm, which is still the simplest online learning rule there is:

```python
import numpy as np

def perceptron(X, y, epochs=100, lr=1.0):
    """y in {-1, +1}. Returns (w, b)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:          # misclassified
                w += lr * yi * xi               # rotate toward the example
                b += lr * yi
                errors += 1
        if errors == 0:
            return w, b                          # separated; stop
    return w, b
```

The update has a clean reading: on a mistake, add the input (scaled by its label) to the
weight vector, which rotates $\mathbf{w}$ toward correctly classifying it.

::: key
**The perceptron convergence theorem**: if the data is linearly separable with margin
$\gamma$ and inputs bounded by $R$, the algorithm makes at most $(R/\gamma)^2$ mistakes,
regardless of dataset size or dimension.

Note what it does *not* promise. If the data is not separable, the algorithm never
terminates and the weights oscillate forever. There is no "best-effort" behaviour — it
either solves the problem exactly or fails to converge at all. That brittleness is why
gradient-based losses replaced it.
:::

## What a linear model cannot do

XOR. Four points, two classes:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No line separates $\{(0,1),(1,0)\}$ from $\{(0,0),(1,1)\}$. Suppose one existed, with
$w_1x_1 + w_2x_2 + b$ positive for the first pair and negative for the second. Then

$$
b < 0, \quad w_1 + b > 0, \quad w_2 + b > 0, \quad w_1 + w_2 + b < 0
$$

The middle two give $w_1 + w_2 + 2b > 0$, so $w_1 + w_2 + b > -b > 0$ — contradicting the
fourth. No such line exists.

Minsky and Papert published this in 1969 and the field's funding collapsed for a decade.
The irony is that the fix was already understood: stack another layer. What was missing
was a way to *train* the stack, which arrived with backpropagation in 1986.

```python
import numpy as np

XOR = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y   = np.array([-1, 1, 1, -1])

w, b = perceptron(XOR, y, epochs=1000)
preds = np.sign(XOR @ w + b)
print('accuracy', (preds == y).mean())     # 0.5 -- chance, after 1000 epochs

# One extra feature makes it separable: the product x1*x2.
lifted = np.c_[XOR, XOR[:, 0] * XOR[:, 1]]
w2, b2 = perceptron(lifted, y, epochs=1000)
print('lifted accuracy', (np.sign(lifted @ w2 + b2) == y).mean())    # 1.0
```

That last trick is the whole idea behind kernels — and behind hidden layers. A hidden
layer *learns* which lifted features to build instead of requiring you to name them.

::: check
A linear model's decision boundary is the set where $\mathbf{w}^\top\mathbf{x} + b = 0$. What is the geometric limit that follows?

- [x] It can only ever split space with a single flat cut — a line in 2-D, a hyperplane in $\mathbb{R}^n$
  > $\mathbf{w}$ is the normal to that flat. Everything the model can express is "which side", which is exactly why XOR defeats it and why the rest of the track exists.
- [ ] It can produce any boundary, but only after enough training
  > Training moves the hyperplane. It cannot bend it.
- [ ] It can produce curved boundaries when the activation is nonlinear
  > A sigmoid changes the *output* into a probability; the level set where it crosses one half is still $\mathbf{w}^\top\mathbf{x} + b = 0$.
- [ ] It is limited to two classes
  > Softmax regression handles $K$ classes with $K$ hyperplanes. The flatness is the limit, not the class count.
:::

## Logistic regression, and why it is the better ancestor

Replacing the sign with a sigmoid gives probabilities rather than hard labels:

$$
p(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}+b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

By lesson 1.14, maximising the likelihood of Bernoulli outputs gives binary
cross-entropy, and its gradient is strikingly simple:

$$
\nabla_{\mathbf{w}}\mathcal{L} = (\sigma(z) - y)\,\mathbf{x}
$$

Error times input — the same outer-product form as every linear layer's weight gradient
in lesson 1.07. Three properties make this the better foundation:

- It is **convex** (lesson 1.16), so there is one optimum and gradient descent finds it.
- It is **smooth**, so it degrades gracefully on non-separable data instead of refusing
  to converge.
- It produces **calibrated probabilities**, not just decisions.

::: warning
On perfectly separable data, logistic regression's weights diverge to infinity: the
likelihood keeps improving as the boundary sharpens, with no finite optimum. L2
regularisation (lesson 1.13's Gaussian prior) bounds the weights and restores a unique
solution. This is a real failure on small, clean, high-dimensional datasets, and it is
the same phenomenon as the confidence blow-up that label smoothing addresses in lesson
1.15.
:::

::: check
The perceptron convergence theorem bounds mistakes at $(R/\gamma)^2$ for data separable with margin $\gamma$ and inputs bounded by $R$. What happens when the data is *not* separable?

- [x] The bound says nothing and the algorithm never settles — it keeps updating forever
  > Which is the practical reason logistic regression is the better ancestor: it has a well-defined optimum whether or not the data separates, and it produces calibrated probabilities rather than a hard sign.
- [ ] It converges to the minimum-error separator instead
  > Nothing in the update rule optimises error count. It reacts to individual mistakes.
- [ ] It converges more slowly, proportional to the overlap
  > It does not converge at all.
- [ ] It raises an error once the mistake bound is exceeded
  > The bound is an analysis, not a check the algorithm performs.
:::

## The softmax generalisation

For $K$ classes, use $K$ weight vectors and normalise:

$$
p(y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top\mathbf{x}+b_k}}{\sum_j e^{\mathbf{w}_j^\top\mathbf{x}+b_j}}
$$

This is `nn.Linear(d, K)` followed by `F.cross_entropy` — and it is exactly the output
head of every classifier and every language model in this curriculum. The final layer of
a 70B-parameter transformer is a softmax regression over the vocabulary. Everything
before it exists to compute a representation $\mathbf{x}$ in which the classes are
linearly separable.

::: exercise
A softmax classifier has weights $\mathbf{w}_1,\ldots,\mathbf{w}_K$. Show that adding the
same vector $\mathbf{c}$ to every $\mathbf{w}_k$ leaves all predictions unchanged. What
does that imply about the parameterisation?
:::

::: solution
Adding $\mathbf{c}$ to each weight adds $\mathbf{c}^\top\mathbf{x}$ to every logit. In
the softmax,

$$
\frac{e^{z_k + \mathbf{c}^\top\mathbf{x}}}{\sum_j e^{z_j + \mathbf{c}^\top\mathbf{x}}}
= \frac{e^{\mathbf{c}^\top\mathbf{x}} e^{z_k}}{e^{\mathbf{c}^\top\mathbf{x}} \sum_j e^{z_j}}
= \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

The common factor cancels. Predictions are identical.

**The implication: softmax regression is over-parameterised.** There is a $d$-dimensional
direction (plus one for the biases) along which the loss is exactly flat, so the Hessian
has a zero eigenvalue and the optimum is not unique — a whole subspace of weights gives
the same function.

Three consequences you will meet again:

- Weight decay breaks the degeneracy by preferring the minimum-norm member of that
  subspace, which is one reason it is applied to the output layer.
- The flat direction means the loss surface is *degenerate*, not merely non-convex — the
  "some eigenvalues are zero" row of lesson 1.09's table.
- Binary classification with $K=2$ is genuinely redundant: one weight vector suffices,
  which is why `BCEWithLogitsLoss` takes a single logit while `CrossEntropyLoss` takes two.
:::

## What to carry forward

- A linear model cuts space with one hyperplane; $\mathbf{w}$ is its normal.
- XOR is not linearly separable, and the fix is a learned feature, not a better line.
- The perceptron converges only on separable data; logistic regression degrades gracefully.
- Its gradient is (prediction − target) × input, the same form as every linear layer.
- Every deep classifier ends in softmax regression; the depth exists to make the classes linearly separable.
