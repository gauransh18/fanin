---
summary: Turning discrete symbols into vectors, what the resulting geometry does and does not mean, and why the embedding layer is usually the largest single matrix in a model.
prereqs: [indexing-gather-scatter, vectors-norms-geometry]
---

Neural networks need vectors. Language, categories and IDs are discrete symbols. An
embedding is a lookup table that bridges them — and the geometry of what it learns is the
foundation of everything from retrieval to interpretability.

## A lookup table that learns

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=50_000, embedding_dim=768)
tokens = torch.tensor([[42, 1337, 7]])
emb(tokens).shape                       # torch.Size([1, 3, 768])

torch.allclose(emb(tokens), emb.weight[tokens])   # True -- it is just indexing
```

From lesson 2.04: the forward pass is a row gather, and the backward is a `scatter_add_`
— each occurrence of token $i$ adds its gradient to row $i$. Frequent tokens' embeddings
therefore receive many more updates per step than rare ones, which is why rare-token
embeddings remain close to their initialisation for most of training.

## Why not one-hot

A one-hot vector times a weight matrix gives exactly the same result:

```python
import torch
import torch.nn.functional as F

W = torch.randn(1000, 64)
i = torch.tensor([42])
torch.allclose(F.one_hot(i, 1000).float() @ W, W[i])   # True
```

The lookup is preferred for three reasons, in ascending order of importance: it avoids a
$V \times d$ matrix product per token; it avoids materialising a 50,000-element mostly-zero
vector; and it gives a **dense, low-dimensional** representation in which similarity is
meaningful. One-hot vectors are all equidistant — every pair has the same dot product of
zero — so they encode no relationships at all.

## The geometry, and its limits

Trained embeddings place related items near each other. The classic demonstration is
analogy by vector arithmetic:

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}
$$

Similarity is measured with cosine (lesson 1.01), because direction carries the meaning
and magnitude largely tracks frequency.

::: warning
The analogy result is weaker than it is usually presented. The standard evaluation
*excludes the three input words* from the candidate set — and without that exclusion, the
nearest vector to $\text{king} - \text{man} + \text{woman}$ is usually **king** itself.
The arithmetic mostly recovers the region of the space, and the exclusion does the rest.

Treat embedding geometry as a useful approximation, not as evidence that the model has
learned compositional semantics.
:::

The concentration result from lesson 1.01 explains why embeddings can be so informative:
in 768 dimensions there is room for an enormous number of *nearly* orthogonal directions,
so a model can encode far more than 768 distinguishable features by tolerating small
interference between them. That is the superposition hypothesis, and it is why
interpretability work on embeddings uses sparse dictionaries rather than reading
individual coordinates.

## Sizing the table

$$
\text{params} = V \times d
$$

For a 128,000-token vocabulary at $d = 4096$: **524 million parameters** in the embedding
alone. In a 7B model that is 7.5% of the total; in a 1B model it would be half.

```python
import torch.nn as nn

for V, d in [(32_000, 4096), (128_000, 4096), (256_000, 8192)]:
    print(f'V={V:7,}  d={d:5,}  ->  {V * d / 1e6:7.1f}M params, '
          f'{V * d * 2 / 1e9:.2f} GB in bf16')
```

**Weight tying** shares the embedding matrix with the output projection:

```python
import torch.nn as nn

class LM(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight       # one matrix, two uses
```

This halves the vocabulary-related parameters and regularises: a token's input and output
representations become the same vector. It was near-universal in smaller models. Very
large models increasingly untie them, because at scale the embedding is a small fraction
of parameters and the extra flexibility is worth more than the saving.

Note that tying relies on the gradient accumulation of lesson 1.08 — the shared matrix
receives gradients from both the lookup and the output projection, summed.

## Beyond tokens

The same mechanism covers any discrete input:

```python
import torch.nn as nn

user_emb = nn.Embedding(n_users, 64)       # recommender systems
item_emb = nn.Embedding(n_items, 64)
# A prediction is a dot product: score = (user_emb(u) * item_emb(i)).sum(-1)

pos_emb  = nn.Embedding(max_len, d)        # learned positional encodings
type_emb = nn.Embedding(2, d)              # segment or speaker IDs
```

For a very large vocabulary with limited capacity, **hashing** maps IDs into a fixed table
via $h(\text{id}) \bmod N$. Collisions are tolerated: two items sharing a row is usually a
smaller cost than having no representation at all. This is standard in production
recommender systems with billions of item IDs.

::: exercise
You train a recommender with user and item embeddings of dimension 64. New users appear
constantly. What is the problem, and what are two architectural fixes?
:::

::: solution
**The problem is cold start.** A new user has no row in the table — or a freshly
initialised, untrained one. The lookup returns noise, so predictions are arbitrary. Worse,
the table's shape is fixed at construction, so a genuinely new ID cannot be represented at
all without resizing and retraining.

**Fix 1: embed features, not identities.** Replace the user ID lookup with an encoder over
user *attributes* — signup country, device, the first few items they interacted with:

```python
user_vec = feature_encoder(torch.cat([country_emb(c), device_emb(dv), recent_items.mean(0)]))
```

A new user with known attributes gets a sensible vector immediately, because the encoder
generalises across users rather than memorising each one. This is the two-tower
architecture used throughout industrial recommendation.

**Fix 2: a learned fallback plus rapid adaptation.** Reserve row 0 as an explicit
`<unknown user>` embedding, trained by randomly masking a small fraction of user IDs during
training so the model learns to make reasonable predictions without identity. Then update
the new user's row online as interactions arrive. This preserves the ID-embedding
architecture and merely gives it a sensible starting point.

**In practice both are used together:** the feature encoder provides the cold-start
representation, and a per-user residual embedding is learned on top as data accumulates —
which is exactly the structure of lesson 5.06's LoRA, applied to users instead of weights.
:::

## What to carry forward

- An embedding is a gather; its backward is a scatter-add, so frequent tokens update faster.
- One-hot vectors are equidistant and encode no relationships; dense vectors do.
- Analogy arithmetic is real but weaker than usually stated — the input words are excluded.
- $V \times d$ can dominate a small model; weight tying halves it.
- Cold start is an architecture problem: embed features, not identities.
