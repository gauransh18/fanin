---
summary: Three ways to read a matrix product, and the FLOP count that decides what your training run costs.
prereqs: [matrices-as-linear-maps]
---

Matrix multiplication is the single operation that consumes almost all the compute in
deep learning. Understanding it three different ways — and knowing its exact cost —
is what lets you look at an architecture and predict its runtime before writing any
code.

## The definition, and three readings

For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, the product
$C = AB$ is $m \times n$ with

$$
C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}
$$

The inner dimension $k$ must match, and it vanishes from the result. Three readings of
that same formula are each useful in different situations.

**As dot products.** $C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of
$B$. This is the view to hold when reading attention: $QK^\top$ produces a matrix whose
$(i,j)$ entry is "how much does query $i$ align with key $j$".

**As a sum of outer products.** Writing $\mathbf{a}_l$ for column $l$ of $A$ and
$\mathbf{b}_l^\top$ for row $l$ of $B$:

$$
AB = \sum_{l=1}^{k} \mathbf{a}_l \mathbf{b}_l^\top
$$

Each term is a rank-1 matrix. The product is a sum of $k$ rank-1 pieces — which is why
the rank of a product can never exceed $\min(\text{rank}(A), \text{rank}(B))$, and why
low-rank adapters work at all.

**As a batch of maps.** Column $j$ of $C$ is $A$ applied to column $j$ of $B$. So
$AB$ is "run the map $A$ on each of $n$ input vectors at once". This is the view that
makes batching obvious: a batch of 64 tokens through a linear layer is one matrix
product, not 64 matrix–vector products.

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = A @ B
C.shape                                   # torch.Size([3, 5])

# Reading 2: the same product as a sum of outer products.
C2 = sum(torch.outer(A[:, l], B[l, :]) for l in range(4))
torch.allclose(C, C2, atol=1e-6)          # True
```

## The FLOP count you should have memorised

Each of the $mn$ output entries needs $k$ multiplications and $k-1$ additions. Counting
a multiply-accumulate as two operations, the total is

$$
\text{FLOPs}(AB) = 2mnk
$$

That factor of two is a convention, but it is the convention every hardware vendor and
every scaling-law paper uses, so adopt it. Now apply it.

A linear layer with input dimension $d_{\text{in}}$, output dimension $d_{\text{out}}$,
processing $B$ tokens, costs $2 B\, d_{\text{in}} d_{\text{out}}$ FLOPs. A transformer's
feed-forward block is two such layers with hidden size $4d$:

$$
2 \cdot B \cdot d \cdot 4d \;+\; 2 \cdot B \cdot 4d \cdot d \;=\; 16 B d^2
$$

Run the same arithmetic over attention projections and you arrive at the rule of thumb
that underpins all of lesson 5.03: **a forward pass costs about $2N$ FLOPs per token
for a model with $N$ parameters, and a full training step costs about $6N$** — one unit
forward, two backward.

::: key
Training compute $\approx 6 \times N_{\text{params}} \times N_{\text{tokens}}$.

For a 7B model on 2T tokens: $6 \times 7\times10^9 \times 2\times10^{12} \approx
8.4\times10^{22}$ FLOPs. An H100 sustains roughly $4\times10^{14}$ FLOP/s on bf16
in practice, so that is about $2\times10^8$ GPU-seconds — near 6,500 GPU-days, or
a month on 220 GPUs. Lesson 7.14 turns this into a budget.
:::

::: check
You train a 3-billion-parameter model on 1 trillion tokens. Using the $6N$ rule, how many FLOPs is that, in units of $10^{21}$?

= 18
> $6 \times 3\times10^{9} \times 1\times10^{12} = 1.8\times10^{22}$, which is $18 \times 10^{21}$. Lesson 5.03 turns this number into a training budget.
:::

## Why the cost model is a lie (and when)

$2mnk$ counts arithmetic. It does not count *memory movement*, and on modern hardware
memory movement is usually what you are actually paying for.

A matrix product reads $mk + kn$ numbers and writes $mn$, while doing $2mnk$ FLOPs. The
ratio of arithmetic to bytes moved — the **arithmetic intensity** — is roughly

$$
I = \frac{2mnk}{(mk + kn + mn) \cdot \text{bytes per element}}
$$

When all three dimensions are large, $I$ is large and you are compute-bound: the FLOP
count predicts runtime well. When one dimension is small — a batch size of 1 during
inference, say — $I$ collapses and you become memory-bound. The GPU sits idle waiting
for weights to arrive, and doubling your FLOPs costs nothing extra.

::: warning
This is why decoding one token at a time is so inefficient, and why every trick in
lesson 7.09 and 7.10 — continuous batching, speculative decoding — is really a trick
for raising arithmetic intensity. The FLOPs were never the problem.
:::

```python
import torch, time

def timed(m, k, n, iters=50):
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / iters
    return 2 * m * n * k / seconds / 1e12      # TFLOP/s

# Square and large: compute-bound, near peak.
timed(4096, 4096, 4096)
# One skinny dimension: memory-bound, a fraction of peak despite the same kernel.
timed(1, 4096, 4096)
```

::: check
The same matmul kernel hits near peak TFLOP/s at $4096 \times 4096 \times 4096$ but a small fraction of peak at $1 \times 4096 \times 4096$. Why?

- [x] The second has low arithmetic intensity
  > The GPU waits on memory rather than computing. With one row, the weights still have to be read in full but there is almost no arithmetic to do with them. The FLOP count no longer predicts runtime because FLOPs were never the bottleneck.
- [ ] The second does far fewer FLOPs, so it finishes faster
  > It does do fewer FLOPs — but the measure here is FLOPs *per second*, and that rate collapses. Doing less work more slowly is the whole problem.
- [ ] Non-square matrices cannot use tensor cores
  > Tensor cores handle non-square shapes fine. The limit is bandwidth, not instruction support.
- [ ] Numerical error accumulates differently in skinny matrices
  > Accuracy is not what is being measured, and bf16 error does not depend on shape this way.
:::

## Associativity is free performance

Matrix multiplication is associative: $(AB)C = A(BC)$. The results are identical; the
costs are not.

Take $A \in \mathbb{R}^{1000 \times 5}$, $B \in \mathbb{R}^{5 \times 1000}$, and
$C \in \mathbb{R}^{1000 \times 20}$.

- $(AB)C$: forming $AB$ costs $2 \cdot 1000 \cdot 1000 \cdot 5 = 10^7$ FLOPs and
  materialises a $1000 \times 1000$ matrix; then multiplying by $C$ costs
  $2 \cdot 1000 \cdot 20 \cdot 1000 = 4\times10^7$. Total: $5\times10^7$.
- $A(BC)$: forming $BC$ costs $2 \cdot 5 \cdot 20 \cdot 1000 = 2\times10^5$; then
  $A(BC)$ costs $2 \cdot 1000 \cdot 20 \cdot 5 = 2\times10^5$. Total: $4\times10^5$.

A **125× difference** from re-bracketing, with no change in the answer. This is not a
toy: it is exactly the optimisation behind linear attention, and exactly why LoRA
computes $B(A\mathbf{x})$ rather than $(BA)\mathbf{x}$.

::: exercise
A LoRA adapter adds $\Delta W = BA$ to a frozen weight $W \in \mathbb{R}^{4096 \times
4096}$, where $B \in \mathbb{R}^{4096 \times 8}$ and $A \in \mathbb{R}^{8 \times 4096}$.
For a batch of 512 tokens, compare the FLOP cost of materialising $\Delta W$ first
against applying the factors in sequence.
:::

::: solution
**Materialising first.** Computing $BA$ costs $2 \cdot 4096 \cdot 4096 \cdot 8 \approx
2.7\times10^8$ FLOPs, and then applying the $4096 \times 4096$ result to 512 tokens
costs $2 \cdot 512 \cdot 4096 \cdot 4096 \approx 1.7\times10^{10}$.

**In sequence.** $A\mathbf{x}$ for 512 tokens costs $2 \cdot 512 \cdot 4096 \cdot 8
\approx 3.4\times10^7$, and $B$ applied to that result costs the same again, for
roughly $6.7\times10^7$ total.

That is about **250× cheaper**, and it also avoids allocating a 16M-element temporary.
The materialised form is still useful at deployment time, when you merge the adapter
into $W$ once and pay nothing per token thereafter.
:::

::: check
A LoRA adapter computes $\Delta W\mathbf{x}$ where $\Delta W = BA$, with $B \in \mathbb{R}^{4096\times8}$ and $A \in \mathbb{R}^{8\times4096}$. Why does the implementation compute $B(A\mathbf{x})$ rather than $(BA)\mathbf{x}$?

- [x] Bracketing right keeps the rank-8 bottleneck, avoiding both the $4096^2$ product and the temporary that holds it
  > Two skinny multiplies through an 8-dimensional waist cost about 250× less than forming the full $4096 \times 4096$ matrix first, and allocate nothing large. Same answer, different bill.
- [ ] $(BA)\mathbf{x}$ would give a different result
  > Matrix multiplication is associative, so the two agree exactly. Only the cost differs.
- [ ] $BA$ cannot be formed because the inner dimension is only 8
  > It forms perfectly well — $B$ is $4096\times8$ and $A$ is $8\times4096$, giving a $4096\times4096$ result. That it *can* be formed is the trap.
- [ ] Materialising $\Delta W$ loses precision in bf16
  > There is a small extra rounding step, but it is not the reason. At deployment you do merge $\Delta W$ into $W$ once, precisely because the cost is then paid only once.
:::

## What to carry forward

- $2mnk$ FLOPs, and $6N$ per parameter per token for a training step.
- A product is a sum of $k$ rank-1 outer products, which bounds its rank.
- FLOPs predict runtime only when arithmetic intensity is high; small batches are memory-bound.
- Re-bracketing a chain of products can change the cost by orders of magnitude.
