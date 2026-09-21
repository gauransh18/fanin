---
summary: Three ways to tell a permutation-equivariant model about order, and why rotary encodings won.
prereqs: [multi-head-attention, seq2seq-to-attention]
---

Self-attention is permutation-equivariant (lesson 4.01), so order must be supplied
explicitly. The three approaches differ in a way that matters enormously for long context.

## Sinusoidal

The original transformer added fixed sinusoids of geometrically spaced frequencies:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

```python
import math, torch

def sinusoidal(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

x = x + sinusoidal(x.size(1), x.size(-1)).to(x)     # added to the embedding
```

The design has one elegant property: $PE_{pos+k}$ is a **linear function** of $PE_{pos}$
for fixed $k$, because shifting a sinusoid's phase is a rotation. In principle this lets
the model learn relative offsets. In practice it works moderately well and extrapolates
poorly beyond the trained length.

## Learned absolute

Just make it an embedding table:

```python
import torch.nn as nn

pos_emb = nn.Embedding(max_len, d)
x = x + pos_emb(torch.arange(x.size(1), device=x.device))
```

Simple, and used by GPT-2, BERT and most 2018–2021 models. Two problems.

::: warning
**It cannot extrapolate at all.** Position 2,049 has no row in a table built for 2,048.
Not "degrades gracefully" — there is literally no vector to look up. Extending context
requires adding rows and retraining them.

**Absolute positions are the wrong invariant.** What matters linguistically is that a verb
is three tokens after its subject, not that the subject is at index 847. An absolute scheme
must learn the same relationship separately at every offset.
:::

## Relative position encodings

Bias the attention scores by the *distance* between positions:

$$
S_{ij} = \frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}} + b_{i-j}
$$

T5 learns a bucketed bias per relative distance. **ALiBi** goes simpler still — a fixed
linear penalty with a per-head slope:

$$
S_{ij} = \frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}} - m_h\,|i-j|
$$

```python
import torch

def alibi_slopes(n_heads):
    """Geometric sequence: head 1 decays fastest, head n slowest."""
    start = 2 ** (-8 / n_heads)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])

def alibi_bias(T, n_heads, device):
    pos = torch.arange(T, device=device)
    dist = (pos[None, :] - pos[:, None]).abs()            # (T, T)
    return -alibi_slopes(n_heads).to(device)[:, None, None] * dist
```

ALiBi adds no parameters and extrapolates well past the training length, because a linear
penalty is defined for any distance. Its cost is a hard recency bias: distant tokens are
always penalised, so genuine long-range retrieval is harder.

## RoPE, and why it won

Rotary position embedding rotates queries and keys by an angle proportional to position,
in 2-D subspaces of the head dimension:

$$
\tilde{\mathbf{q}}_m = R_{\Theta, m}\,\mathbf{q}_m, \qquad
\tilde{\mathbf{k}}_n = R_{\Theta, n}\,\mathbf{k}_n
$$

where $R_{\Theta,m}$ is a block-diagonal rotation with block $i$ turning by
$m\theta_i$, $\theta_i = 10000^{-2i/d}$.

The property that makes it work falls out of how rotations compose:

$$
\tilde{\mathbf{q}}_m \cdot \tilde{\mathbf{k}}_n = \mathbf{q}_m^\top R_{\Theta,n-m}\,\mathbf{k}_n
$$

::: key
The dot product depends **only on the relative offset $n-m$**, even though each vector was
rotated by its absolute position. You get relative-position behaviour from an operation
applied to queries and keys individually — so there is no $T\times T$ bias matrix to
materialise, and it composes with FlashAttention (lesson 4.12), which never forms the
score matrix at all.

That last point is why RoPE beat ALiBi and T5-style biases in practice: a relative scheme
requiring an explicit bias matrix is incompatible with the kernels everyone now uses.
:::

```python
import torch

def rope_tables(T, d_h, device, base=10000.0):
    theta = base ** (-torch.arange(0, d_h, 2, device=device).float() / d_h)
    angles = torch.outer(torch.arange(T, device=device).float(), theta)  # (T, d_h/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """x: (B, h, T, d_h). Rotates each adjacent coordinate pair."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)

# Applied to q and k only -- never to v, which carries content, not position.
cos, sin = rope_tables(T, d_h, q.device)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

Note that RoPE touches queries and keys but **not values**. Position should influence
*which* tokens are retrieved, not *what* they contain.

## Extending RoPE's context

RoPE does not extrapolate for free either — beyond the trained length, the rotation angles
are out of distribution. Three standard extensions:

- **Position interpolation.** Scale positions down by $L_{\text{new}}/L_{\text{train}}$ so
  they land back in the trained range. Needs a short fine-tune and works surprisingly well.
- **NTK-aware scaling.** Increase the base from 10,000, which stretches low-frequency
  components more than high-frequency ones — preserving local resolution while extending
  range.
- **YaRN.** Combines interpolation with a temperature correction and per-frequency
  treatment. Currently the strongest of the three.

## The comparison

| Scheme | Parameters | Extrapolates | Works with FlashAttention | Used by |
|---|---|---|---|---|
| Sinusoidal | 0 | Poorly | Yes | Original transformer |
| Learned absolute | $L \times d$ | No | Yes | GPT-2, BERT |
| T5 relative bias | small | Somewhat | Awkwardly | T5 |
| ALiBi | 0 | Well | Awkwardly | BLOOM, MPT |
| RoPE | 0 | With scaling | Yes | Llama, Mistral, Qwen, most current models |

::: exercise
RoPE rotates queries and keys but not values. Why would rotating values be wrong?
:::

::: solution
**The roles differ.** From lesson 4.02: the query–key dot product decides *which* positions
to attend to; the values decide *what content* is retrieved. Position is relevant to the
first question and not the second — a token's meaning does not change because it sits at
index 500.

**Mechanically, it would break the output.** The attention output is
$\mathbf{z}_i = \sum_j A_{ij}\mathbf{v}_j$. If each $\mathbf{v}_j$ were rotated by its own
absolute position $j$, the sum would mix vectors rotated by *different* angles. The
relative-offset property does not apply here — there is no second rotation to cancel
against — so the output would depend on absolute positions in an uncontrolled way, and the
same content at two positions would produce different contributions.

**The cancellation is specific to the dot product.** $R_m^\top R_n = R_{n-m}$ works because
two rotated vectors are being compared. A weighted sum of rotated vectors has no such
structure.

**A useful consistency check:** consider a sequence where the same token appears at
positions 5 and 500, and attention puts equal weight on both. With values unrotated, they
contribute identically, which is correct — the model already encoded "where" in the
attention weights. With values rotated, they contribute differently, and the model would
have to learn to undo a rotation it cannot observe.
:::

## What to carry forward

- Learned absolute encodings cannot extrapolate — there is no row to look up.
- Relative schemes encode the linguistically meaningful invariant, but most need a bias matrix.
- RoPE gets relative behaviour by rotating q and k individually, so it works with fused kernels.
- Never rotate values: position decides retrieval, not content.
- Extending RoPE's context needs interpolation or NTK-aware scaling plus a short fine-tune.
