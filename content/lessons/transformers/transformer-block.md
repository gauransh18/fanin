---
summary: Assembling attention and an MLP into the unit that repeats, where the parameters and FLOPs actually go, and the two design decisions that matter.
prereqs: [multi-head-attention, residual-connections, normalization]
---

Everything so far assembles into one block, repeated $L$ times. The block has exactly two
sublayers, and the separation between them is the transformer's central design decision.

## The block

```python
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, n_heads, expansion=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn  = MultiHeadAttention(d, n_heads, dropout)
        self.norm2 = nn.RMSNorm(d)
        self.mlp   = nn.Sequential(
            nn.Linear(d, expansion * d, bias=False),
            nn.GELU(),
            nn.Linear(expansion * d, d, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, is_causal=True):
        x = x + self.attn(self.norm1(x), is_causal)    # mix across positions
        x = x + self.mlp(self.norm2(x))                # transform each position
        return x
```

Two sublayers, each wrapped in a pre-norm residual (lesson 3.06):

$$
\mathbf{x} \leftarrow \mathbf{x} + \text{Attn}(\text{Norm}(\mathbf{x})), \qquad
\mathbf{x} \leftarrow \mathbf{x} + \text{MLP}(\text{Norm}(\mathbf{x}))
$$

## The separation of concerns

::: key
**Attention mixes across positions. The MLP transforms within a position.**

Attention is the *only* operation in the entire architecture where information moves
between tokens. The MLP runs independently on each position — the same weights, applied
position-wise, with no knowledge that other positions exist.

This factorisation is why transformers parallelise so well, why you can reason about the
two sublayers separately, and why mechanistic interpretability work treats "attention
heads move information" and "MLPs process it" as distinct claims.
:::

The same split appears elsewhere: depthwise-separable convolutions (lesson 3.09) separate
spatial mixing from channel mixing for exactly the same reason.

::: check
Which operation in a transformer moves information between token positions?

- [x] Attention, and only attention
  > The MLP runs position-wise with the same weights and no knowledge that other positions exist. That factorisation is why transformers parallelise, why the two sublayers can be reasoned about separately, and why interpretability work treats "heads move information" and "MLPs process it" as distinct claims.
- [ ] Both attention and the MLP, since the MLP sees the whole sequence tensor
  > It sees a tensor with a position axis, but it is applied independently along it. Nothing crosses.
- [ ] The normalization layer, which computes statistics over the sequence
  > LayerNorm and RMSNorm normalise across the *feature* dimension, per position. Normalising across the sequence is what BatchNorm-style layers do, and transformers avoid it.
- [ ] The residual connection, which carries earlier positions forward
  > The residual carries a position's own value forward through depth, not across positions.
:::

## The full model

```python
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks  = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm_f  = nn.RMSNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying, lesson 3.13

        self.apply(self._init)
        # Scale residual writes by 1/sqrt(2L) -- lesson 3.05.
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('mlp.2.weight'):
                nn.init.normal_(p, std=0.02 / (2 * n_layers) ** 0.5)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok_emb(idx)                            # (B, T, d)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))                 # (B, T, vocab)
```

Note the **final norm** before the head. Because pre-norm leaves the residual stream
un-normalised (lesson 3.08), its magnitude grows through depth — without `norm_f` the
logits would be scaled by an uncontrolled factor.

## Where the parameters go

Per block, with $d$ the model dimension:

| Component | Parameters |
|---|---|
| $W_Q, W_K, W_V$ | $3d^2$ |
| $W_O$ | $d^2$ |
| MLP up + down ($4\times$) | $8d^2$ |
| Norms | $2d$ |
| **Total** | $\approx 12d^2$ |

So **two-thirds of a transformer's parameters live in its MLPs** and one third in
attention. That ratio is why mixture-of-experts (lesson 4.14) replaces the MLP
specifically, and why quantization work (lesson 5.12) focuses there first.

```python
def gpt_params(vocab, d, n_layers):
    embedding = vocab * d
    per_block = 12 * d * d + 2 * d
    return embedding + n_layers * per_block + d

# Llama-2 7B: 32 layers, d=4096, vocab=32000 (with SwiGLU the MLP ratio differs)
print(f'{gpt_params(32_000, 4096, 32) / 1e9:.2f}B')
```

## Where the FLOPs go

Per token, forward:

$$
\underbrace{24 d^2}_{\text{matmuls}} + \underbrace{4 T d}_{\text{attention scores}}
$$

using the $2mnk$ rule from lesson 1.03. The crossover is at $T \approx 6d$: below it, the
projections dominate; above it, the quadratic attention term does.

For Llama-2 7B with $d = 4096$, that is $T \approx 24{,}576$. So at typical context lengths
of 4k–8k, **attention is not the bottleneck** — the dense matrix products are. This
surprises people who assume $O(T^2)$ means attention dominates everywhere.

The memory story is different: $T^2$ attention scores blow up long before the FLOPs do,
which is lesson 4.12's subject.

## The two decisions that matter

**Pre-norm over post-norm.** Covered in lesson 3.06 — a clean residual path is what makes
deep transformers trainable. Every model since GPT-2 uses pre-norm.

**The $4\times$ expansion.** Wide enough that the nonlinearity has room to separate
features, narrow enough to afford. With SwiGLU (lesson 3.02) the convention shifts to
about $\tfrac{8}{3}d$ per matrix, since there are three matrices instead of two — keeping
the parameter count matched.

::: check
Why does the block apply the MLP with an expansion of $4d$ rather than keeping width $d$ throughout?

- [x] A narrow-to-wide-to-narrow sandwich gives the position-wise transform somewhere to compute before projecting back into the residual stream
  > Lesson 1.02's shape table read forwards: a $4d\times d$ lift, a nonlinearity, then a $d\times4d$ squash. Contracting instead would impose a rank ceiling on the block.
- [ ] It matches the number of attention heads
  > Head count and MLP expansion are independent; models routinely have 32 heads and an expansion of 4.
- [ ] Wider layers are cheaper per FLOP on tensor cores
  > Larger matmuls do have better arithmetic intensity, which is a reason the shape is *tolerable*, not the reason it exists.
- [ ] It makes the block's parameter count match attention's
  > It does land near a 2:1 ratio in practice, which is a consequence of the choice rather than its motivation.
:::

## Scaling the shape

| Model | $L$ | $d$ | heads | Params |
|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 | 124M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| Llama-2 7B | 32 | 4096 | 32 | 7B |
| Llama-2 70B | 80 | 8192 | 64 | 70B |

Depth and width grow together, roughly as $d \propto \sqrt{N}$ and $L \propto \sqrt[3]{N}$.
Extremely deep-and-narrow or shallow-and-wide models underperform at matched parameter
count — there is a broad optimum, and everything above sits inside it.

::: warning
`bias=False` throughout is deliberate and now standard. Biases add $d$ parameters per
projection for a measured quality difference near zero, and they interfere with the
tensor-parallel sharding of lesson 7.06 — a bias must be added on exactly one rank, which
is an annoying special case. Removing them is free.
:::

::: exercise
Estimate parameters, training FLOPs and activation memory for a model with $L=24$,
$d=2048$, $h=16$, vocabulary 50k, trained on 100B tokens at batch 1M tokens.
:::

::: solution
**Parameters.**
- Embedding: $50{,}000 \times 2048 = 102$M
- Per block: $12 \times 2048^2 = 50.3$M, times 24 blocks = $1{,}208$M
- Total: about **1.31B**, of which the embedding is 8%.

**Training FLOPs.** Using $6N$ per parameter per token (lesson 1.03):

$$
6 \times 1.31\times10^9 \times 10^{11} = 7.9\times10^{20}\ \text{FLOPs}
$$

At an H100 sustaining $4\times10^{14}$ FLOP/s, that is $2\times10^6$ GPU-seconds — about
**23 GPU-days**, or under a day on 32 GPUs. (Lesson 7.14 adds the MFU correction, which
typically makes the real figure 2–3× this.)

**Activation memory** for one microbatch of 8 sequences at $T=2048$, in bf16. The dominant
term is the stored activations per layer, roughly $\sim 10\,B\,T\,d$ elements with the
usual set of saved tensors:

$$
10 \times 8 \times 2048 \times 2048 \times 2\ \text{bytes} \times 24\ \text{layers} \approx 16\ \text{GB}
$$

**The check that matters:** compare against parameter-related memory. From lesson 2.01,
mixed-precision AdamW costs about 18 bytes per parameter: $1.31\times10^9 \times 18
\approx 24$ GB. So activations (16 GB) and optimizer state (24 GB) are the same order —
40 GB total, which fits on an 80 GB card without sharding but leaves little headroom.

To raise the batch size you would reach for gradient checkpointing (lesson 7.08) first,
since activations are the term that scales with batch.
:::

## What to carry forward

- Two sublayers: attention mixes positions, the MLP transforms each one independently.
- $12d^2$ parameters per block, two-thirds of them in the MLP.
- FLOPs per token are $24d^2 + 4Td$; attention dominates only past $T \approx 6d$.
- Pre-norm plus a final norm before the head; no biases.
- Depth and width scale together — the optimum is broad but the extremes are bad.
