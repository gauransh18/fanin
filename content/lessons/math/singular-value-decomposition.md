---
summary: The one factorisation that works for every matrix, and the theorem that makes it the basis of compression, PCA and low-rank adaptation.
prereqs: [eigenvalues-eigenvectors, rank-span-subspaces]
---

The SVD is the most useful theorem in applied linear algebra. It says every matrix —
any shape, any rank, no conditions at all — is a rotation, then a scaling, then another
rotation. From that single statement you get PCA, optimal low-rank compression, the
pseudoinverse, and a principled way to measure how "big" a matrix is.

## The statement

For any $A \in \mathbb{R}^{m \times n}$ there exist orthogonal $U \in \mathbb{R}^{m\times m}$,
orthogonal $V \in \mathbb{R}^{n\times n}$, and diagonal $\Sigma \in \mathbb{R}^{m\times n}$
with non-negative entries, such that

$$
A = U \Sigma V^\top
$$

The diagonal entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ are the **singular
values**. Columns of $V$ are right singular vectors (input directions); columns of $U$
are left singular vectors (output directions).

Geometrically, reading right to left: $V^\top$ rotates the input into a convenient
frame, $\Sigma$ stretches each axis by $\sigma_i$, and $U$ rotates into the output
frame. **Every linear map is rotate–stretch–rotate.** There is nothing else a matrix
can do.

```python
import torch

A = torch.randn(5, 3)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

U.shape, S.shape, Vh.shape          # (5,3), (3,), (3,3)
S                                   # sorted descending, all >= 0
torch.allclose(A, U @ torch.diag(S) @ Vh, atol=1e-5)   # True
```

## Relation to eigenvalues

The singular values of $A$ are the square roots of the eigenvalues of $A^\top A$, and
$V$ holds that matrix's eigenvectors:

$$
A^\top A = V\Sigma^\top \Sigma V^\top
$$

This is why the SVD always exists: $A^\top A$ is symmetric positive semi-definite for
any $A$, so it always diagonalises with real non-negative eigenvalues, no matter how
badly behaved $A$ itself is. All the caveats from lesson 1.05 disappear.

Two corollaries fall out immediately:

- **Rank** is the number of non-zero singular values. Numerical rank is the number
  above a tolerance — and now "look at the spectrum" has a precise meaning.
- **The spectral norm** $\lVert A \rVert_2 = \sigma_1$ is the largest factor by which
  $A$ can stretch any vector. This is the quantity spectral normalization controls, and
  the one that bounds how much a layer can amplify its input.

## The theorem that matters: optimal truncation

Keep only the largest $k$ singular values and zero the rest:

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

The **Eckart–Young theorem** says $A_k$ is the *best possible* rank-$k$ approximation to
$A$, in both the spectral and Frobenius norms. Not a good heuristic — provably optimal.
No other rank-$k$ matrix gets closer.

The error is exactly the tail you discarded:

$$
\lVert A - A_k \rVert_F^2 = \sum_{i=k+1}^{\min(m,n)} \sigma_i^2
$$

So the spectrum is a direct readout of compressibility. If $\sigma$ decays fast, a few
components capture almost everything; if it is flat, the matrix is genuinely
high-dimensional and truncation will hurt.

::: key
Rank-$k$ truncation stores $k(m + n + 1)$ numbers instead of $mn$. For a
$4096\times4096$ weight at $k=64$, that is 524K instead of 16.8M — a 32× reduction,
with an error you can compute in advance from the singular values you dropped.
:::

```python
import torch

A = torch.randn(512, 64) @ torch.randn(64, 512)   # true rank 64
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

for k in (8, 32, 64, 128):
    Ak = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    rel = (A - Ak).norm() / A.norm()
    tail = (S[k:] ** 2).sum().sqrt() / A.norm()    # predicted by Eckart-Young
    print(f'k={k:4d}  error {rel:.2e}  predicted {tail:.2e}')
# At k=64 both hit ~1e-6: the matrix had no rank beyond 64 to lose.
```

## Where you will meet it

**PCA is the SVD of centred data.** Subtract the mean from your data matrix and take
the SVD; the right singular vectors are the principal components, and $\sigma_i^2/(n-1)$
is the variance explained along each. Everything people say about PCA is a statement
about the SVD in different vocabulary.

**LoRA initialisation.** Lesson 5.06's adapters are exactly a rank-$r$ factorisation.
Eckart–Young is the guarantee that a well-chosen rank-$r$ update is the best you could
possibly do with that parameter budget.

**Model compression.** Truncating weight matrices is one of the oldest compression
methods. It works less well than quantization (lesson 5.12) on modern transformers,
precisely because their weight spectra turn out to be flatter than people expected —
which is itself a finding you can only state because of Eckart–Young.

**Conditioning.** $\kappa(A) = \sigma_1/\sigma_{\min}$ generalises the condition number
to non-square matrices, and it tells you how much a numerical error in the input can be
amplified in the output.

::: warning
The full SVD of an $m\times n$ matrix costs $O(mn \min(m,n))$ — for a $4096^2$ weight
that is on the order of $10^{11}$ operations. Use `torch.svd_lowrank` when you only need
the top few components; it is a randomised algorithm that costs a small multiple of one
matrix product.
:::

::: exercise
You have a $1000 \times 800$ matrix whose singular values are approximately
$\sigma_i = 100 / i$. How many components do you need to capture 95% of the Frobenius
norm squared, and what compression does that give?
:::

::: solution
$\lVert A \rVert_F^2 = \sum_i \sigma_i^2 = 10^4 \sum_{i=1}^{800} 1/i^2$. The sum
converges quickly to $\pi^2/6 \approx 1.6449$, so the total is about $1.645\times10^4$.

The tail beyond $k$ is $10^4 \sum_{i>k} 1/i^2 \approx 10^4/k$. Requiring the tail to be
under 5% of the total gives $10^4/k \le 0.05 \times 1.645\times10^4$, so $k \ge 12.2$ —
about **13 components**.

Storage: $13 \times (1000 + 800 + 1) = 23{,}413$ numbers against $800{,}000$, a **34×
reduction** for 5% of the squared norm. The $1/i$ decay is what makes this work; a flat
spectrum would have needed 760 components for the same fidelity.
:::

## What to carry forward

- $A = U\Sigma V^\top$ exists for every matrix: rotate, stretch, rotate.
- Singular values are $\sqrt{\text{eigenvalues of } A^\top A}$, always real and non-negative.
- Eckart–Young: truncating the spectrum is the provably optimal low-rank approximation.
- $\sigma_1$ is the spectral norm; $\sigma_1/\sigma_{\min}$ is the condition number.
- Spectral decay rate *is* compressibility — look at it before trusting any rank claim.
