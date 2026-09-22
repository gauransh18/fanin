---
summary: Rank measures how much information a matrix can carry. Low-rank structure is the reason adapters, compression and bottlenecks work.
prereqs: [matrices-as-linear-maps]
---

Rank is the most practically useful number attached to a matrix. It tells you how much
of the input space survives the map, how much redundancy the weights contain, and
whether you can replace a big matrix with two small ones.

## Span and linear independence

The **span** of a set of vectors is everything you can reach by scaling and adding
them:

$$
\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{ \textstyle\sum_i c_i \mathbf{v}_i \;:\; c_i \in \mathbb{R} \right\}
$$

Two vectors in $\mathbb{R}^3$ that point in different directions span a plane. If the
second is a multiple of the first, they span only a line — the second added nothing.
That is **linear dependence**: a set is dependent if one member is expressible from the
others, independent if none is.

The **dimension** of a subspace is the size of the smallest set that spans it.

## Rank

The **rank** of $A$ is the dimension of the span of its columns. Equivalently — and
this is a genuinely surprising theorem — it is also the dimension of the span of its
rows. Row rank equals column rank, always.

For $A \in \mathbb{R}^{m \times n}$, rank is at most $\min(m, n)$. A matrix hitting
that bound is **full rank**; anything less is **rank-deficient**.

```python
import torch

A = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0],      # exactly 2x row 1
                  [1.0, 1.0, 1.0]])

torch.linalg.matrix_rank(A)              # tensor(2), not 3
```

::: insight
Rank is the number of genuinely independent directions the map can produce. A
$4096 \times 4096$ weight matrix of rank 8 has 16.7M parameters but only
$8 \times (4096 + 4096) = 65{,}536$ degrees of freedom. Everything else is redundancy —
and redundancy is compressible.
:::

## The four fundamental subspaces

Every $A \in \mathbb{R}^{m\times n}$ carries four subspaces, and they partition both
the input and output spaces cleanly.

| Subspace | Lives in | Dimension | What it is |
|---|---|---|---|
| Column space $\mathcal{C}(A)$ | $\mathbb{R}^m$ | $r$ | Everything $A$ can output |
| Null space $\mathcal{N}(A)$ | $\mathbb{R}^n$ | $n - r$ | Inputs crushed to zero |
| Row space $\mathcal{C}(A^\top)$ | $\mathbb{R}^n$ | $r$ | Inputs that survive |
| Left null space $\mathcal{N}(A^\top)$ | $\mathbb{R}^m$ | $m - r$ | Outputs unreachable by $A$ |

The relation $\text{rank} + \text{nullity} = n$ is the **rank–nullity theorem**, and it
says something intuitive: every input dimension either survives or gets destroyed, and
the books balance.

The two subspaces inside $\mathbb{R}^n$ are orthogonal complements: the row space and
the null space meet only at zero, and together they span everything. So any input
splits uniquely into a part the matrix acts on and a part it annihilates.

::: check
An autoencoder maps $\mathbb{R}^{768} \to \mathbb{R}^{32} \to \mathbb{R}^{768}$ with two linear layers. How many input directions does the composite map send to zero?

= 736
> Rank is capped by the narrowest link, so the composite has rank at most 32. Rank–nullity then gives a null space of $768 - 32 = 736$ dimensions. No amount of training widens that while the bottleneck holds.
:::

## Why low rank is everywhere

**Bottlenecks force it.** An autoencoder that maps $\mathbb{R}^{768} \to \mathbb{R}^{32}
\to \mathbb{R}^{768}$ has an end-to-end map of rank at most 32, because rank cannot
exceed the smallest dimension in the chain. The bottleneck *is* a rank constraint, and
lesson 3.14 builds on that.

**Trained weights tend toward it.** Empirically, the update a model undergoes during
fine-tuning has far lower rank than the weights themselves. That observation is the
entire premise of LoRA: freeze $W$ and learn $\Delta W = BA$ with $r \ll d$, cutting
trainable parameters by a factor of hundreds. Lesson 5.06 does the arithmetic.

**Attention creates it.** With $h$ heads of dimension $d_h$ each, the per-head map
$W_Q W_K^\top$ has rank at most $d_h$, typically 64 or 128, even though it operates on
a $d$-dimensional residual stream of 4096. Multi-head attention is a sum of low-rank
maps by construction.

```python
import torch

d, r = 4096, 8
W_full = torch.randn(d, d)               # 16,777,216 parameters
B = torch.randn(d, r)
A = torch.randn(r, d)                    # 65,536 parameters total

torch.linalg.matrix_rank(B @ A)          # tensor(8)
(B.numel() + A.numel()) / W_full.numel() # 0.0039 -- 0.4% of the parameters
```

::: check
Why does LoRA freeze $W$ and learn $\Delta W = BA$ with $r \ll d$, rather than learning $\Delta W$ directly?

- [x] Fine-tuning updates are observed to be close to low rank, so the extra degrees of freedom in a full $\Delta W$ mostly go unused
  > At $d = 4096$ and $r = 8$, the factored form has 65,536 parameters against 16.7M — 0.4% — and empirically loses little, because the update was never using the rest.
- [ ] A low-rank matrix is faster to multiply, which is the main saving
  > Applying the factors in sequence is indeed cheaper, but the headline win is the count of *trainable* parameters and the optimizer state that comes with them.
- [ ] A full $\Delta W$ would not fit in memory
  > It is the same size as $W$, which is already resident. Memory for optimizer states is the pressure, not the matrix itself.
- [ ] Low rank regularises the model, preventing overfitting
  > There is a regularising side effect, but the premise is empirical redundancy, not a deliberate capacity constraint.
:::

## Rank in floating point is a threshold, not a fact

Exact rank is a discrete property, and floating-point noise destroys it. A matrix that
is mathematically rank 2 will compute as rank 3 the moment rounding perturbs the third
singular value from $0$ to $10^{-16}$.

Real implementations therefore report **numerical rank**: the number of singular values
above a tolerance, usually $\max(m,n) \cdot \epsilon \cdot \sigma_{\max}$. Lesson 1.06
shows why singular values are the right thing to threshold.

::: warning
Never test `matrix_rank(A) == k` on a trained weight matrix and conclude anything. Look
at the *spectrum* — the sorted singular values — and ask where it decays. A matrix with
values $[100, 90, 80, 0.001, 0.0009]$ is rank 5 by any tolerance test and rank 3 for
every practical purpose.
:::

::: exercise
A network applies $W_2 W_1$ with $W_1 \in \mathbb{R}^{64 \times 512}$ and $W_2 \in
\mathbb{R}^{512 \times 64}$. What is the maximum rank of the composite map, how many
input directions are destroyed, and what does that mean for information flow?
:::

::: solution
The composite is $512 \times 512$, but its rank is at most $\min(512, 64, 512) = 64$,
since rank cannot exceed that of any factor.

By rank–nullity, the null space has dimension $512 - 64 = 448$: there are 448
independent input directions that the pair maps to zero, and no amount of training can
change that while the bottleneck stays at 64.

Information in those directions is irrecoverably lost. That is the point of a
bottleneck — it forces the network to spend its 64 surviving directions on whatever
matters most — but it also means this block cannot pass through a residual signal
unchanged, which is one reason transformer MLP blocks expand to $4d$ rather than
contracting, and why the residual connection of lesson 3.08 bypasses the block entirely.
:::

::: check
`torch.linalg.matrix_rank` returns 5 for a trained weight matrix whose singular values are $[100, 90, 80, 0.001, 0.0009]$. What should you conclude?

- [x] Practically it behaves as rank 3; the rank count alone is not informative
  > Exact rank is discrete and floating-point noise destroys it. Read the spectrum and find where it decays — two directions carrying $10^{-5}$ of the leading scale are doing nothing.
- [ ] The matrix is full rank, so it cannot be compressed
  > It compresses beautifully: dropping the last two singular values costs almost nothing and is exactly what low-rank approximation does.
- [ ] The rank function is buggy and should be reported
  > It is behaving as documented, thresholding at a tolerance the matrix happens to clear. The number is correct and still not the number you want.
- [ ] The last two values indicate numerical instability in training
  > Small singular values are ordinary in trained weights. They signal redundancy, not instability.
:::

## What to carry forward

- Rank is the count of independent directions the map produces; row rank equals column rank.
- $\text{rank} + \text{nullity} = n$: inputs either survive or are destroyed.
- Rank of a product is bounded by every factor, so bottlenecks impose rank ceilings.
- Low-rank structure is what makes LoRA, compression and multi-head attention efficient.
- In floating point, look at the spectrum rather than trusting a rank count.
