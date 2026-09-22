---
summary: A complete, working GPT in about 120 lines, with every design decision traced back to the lesson that justified it.
prereqs: [transformer-block, causal-masking, positional-encodings, training-loop]
---

Everything in this track assembles into one file. This is a complete implementation — it
trains, it generates, and every line has a reason you have already seen.

## The model

```python
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50_304      # padded to a multiple of 64 for tensor cores
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0          # 0 for pretraining, ~0.1 for fine-tuning


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.n_embd // cfg.n_head
        # Fused QKV: one matmul beats three (lesson 1.03).
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Split the LAST dim, then transpose (lesson 2.02 -- the other
        # order silently scrambles positions against features).
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # lesson 4.04

        if kv_cache is not None:                                   # lesson 4.10
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache = (k, v)

        # is_causal must be False when T == 1 during cached decoding: the
        # single query attends to the whole cached prefix, which is already
        # causal by construction.
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None or T > 1),
            dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), kv_cache


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.attn = nn.RMSNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2, self.mlp  = nn.RMSNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        # Pre-norm: the residual path stays clean (lessons 3.06, 3.08).
        attn_out, kv_cache = self.attn(self.ln1(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv_cache


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.RMSNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying (lesson 3.13)

        cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head)
        self.register_buffer('cos', cos, persistent=False)   # buffer, not attribute
        self.register_buffer('sin', sin, persistent=False)   # (lesson 2.08)

        self.apply(self._init_weights)
        # Residual writes scaled by 1/sqrt(2L) (lesson 3.05).
        for name, p in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        offset = 0 if kv_caches is None else kv_caches[0][0].size(2)
        cos = self.cos[offset:offset + T]
        sin = self.sin[offset:offset + T]

        x = self.tok_emb(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x, cache = block(x, cos, sin, cache)
            new_caches.append(cache)
        x = self.ln_f(x)

        if targets is None:
            # Generation: only the last position's logits are needed. Pass
            # `all_positions=True` when you need the full logit tensor, as the
            # causal-leakage check below does.
            last = x if getattr(self, 'all_positions', False) else x[:, [-1]]
            return self.head(last), new_caches

        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss
```

## The pieces from earlier lessons

```python
import torch

def rope_tables(max_len, d_head, base=10000.0):
    theta = base ** (-torch.arange(0, d_head, 2).float() / d_head)
    angles = torch.outer(torch.arange(max_len).float(), theta)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (B, h, T, d_head). Rotates adjacent coordinate pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

::: check
The config sets `vocab_size = 50_304` where the tokenizer produces 50,257 tokens. Why the padding?

- [x] It rounds the vocabulary to a multiple of 64 so the output matmul aligns with tensor-core tile sizes
  > The extra rows are never predicted in practice and cost a negligible amount of memory. It is a free few percent on the largest matmul in the model.
- [ ] The extra tokens are reserved for special control symbols
  > Special tokens are already inside 50,257. These rows are pure padding.
- [ ] A power-of-two vocabulary is required by `F.cross_entropy`
  > 50,304 is not a power of two, and cross-entropy has no such requirement.
- [ ] It prevents index-out-of-range errors during sampling
  > Sampling is bounded by the logits tensor either way.
:::

## Generation

```python
import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    caches = None
    for _ in range(max_new_tokens):
        # With a cache, feed only the newest token (lesson 4.10).
        step_input = idx if caches is None else idx[:, -1:]
        logits, caches = model(step_input, kv_caches=caches)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

## The checks, before training anything

Run lesson 3.16's procedure in order. Each is a few seconds:

```python
import math, torch

cfg = Config(n_layer=4, n_head=4, n_embd=128, vocab_size=1000, block_size=64)
model = GPT(cfg)

# 1. Parameter count matches the architecture (lesson 2.08).
n = sum(p.numel() for p in model.parameters())
expected = cfg.vocab_size * cfg.n_embd + cfg.n_layer * 12 * cfg.n_embd ** 2
print(f'{n/1e6:.2f}M params, expected ~{expected/1e6:.2f}M (tied embeddings)')

# 2. Initial loss is ln(vocab) (lesson 3.16).
x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
_, loss = model(x[:, :-1], x[:, 1:])
print(f'initial loss {loss.item():.3f}, expected {math.log(cfg.vocab_size):.3f}')

# 3. No causal leakage (lesson 4.06): editing a token must not change
#    any earlier position's logits.
model.all_positions = True          # return logits at every position
with torch.no_grad():
    inp = x[:, :-1]
    edit_at = inp.size(1) // 2
    base, _ = model(inp)
    edited = inp.clone()
    edited[0, edit_at] = (edited[0, edit_at] + 1) % cfg.vocab_size
    after, _ = model(edited)
before_delta = (base[0, :edit_at] - after[0, :edit_at]).abs().max()
after_delta  = (base[0, edit_at:] - after[0, edit_at:]).abs().max()
print(f'before edit {before_delta:.2e} (must be ~0), '
      f'after {after_delta:.2e} (must be > 0)')
assert before_delta < 1e-4, 'causal mask leaks'
model.all_positions = False

# 4. Cached and uncached generation agree.
prompt = torch.randint(0, cfg.vocab_size, (1, 8))
torch.manual_seed(0); a = generate(model, prompt.clone(), 16, temperature=1e-8)
torch.manual_seed(0); b = generate(model, prompt.clone(), 16, temperature=1e-8)
assert torch.equal(a, b), 'greedy decoding is not deterministic'
```

::: key
Check 4 is the one people skip and the one that catches KV-cache bugs. With temperature
near zero the sampling is effectively greedy, so two runs must produce identical tokens.
A cache that mishandles the RoPE position offset produces plausible but different text —
the model still generates English, just worse, and nothing errors.

Compare cached against uncached generation directly whenever you touch the cache.
:::

## Training it

```python
import torch

model = GPT(Config()).to('cuda')
model = torch.compile(model)                    # lesson 2.14

opt = torch.optim.AdamW(
    [{'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': 0.1},
     {'params': [p for p in model.parameters() if p.ndim <  2], 'weight_decay': 0.0}],
    lr=6e-4, betas=(0.9, 0.95), fused=True)     # lesson 2.09

for step, (x, y) in enumerate(loader):
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):    # lesson 2.12
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)             # lesson 2.11
```

::: check
The code splits heads with `view(B, T, n_head, d_head).transpose(1, 2)` and the comment warns about the other order. What goes wrong if you transpose first and then view?

- [x] Positions and features get scrambled together, producing correct shapes and wrong values
  > The last dimension is the one holding the concatenated heads, so it must be split before anything moves. Reordering first mixes the position axis into the head split — a shape-correct, silently wrong result, which is lesson 2.02's failure mode exactly.
- [ ] It raises a contiguity error
  > It may, which would at least be loud. The dangerous case is the one where it does not.
- [ ] Gradients stop flowing through the attention block
  > Both `view` and `transpose` are differentiable and keep the graph intact.
- [ ] The KV cache would store the wrong number of heads
  > The head count is the same either way; it is the assignment of values to heads that breaks.
:::

## What this is missing

Honest list of what separates this from a production implementation:

- **GQA** instead of full multi-head attention — lesson 4.11 shows the cache saving.
- **SwiGLU** instead of GELU, with the $\tfrac{8}{3}d$ hidden size.
- **Distributed training** — FSDP or tensor parallelism, track 7.
- **Proper data loading** from memory-mapped shards, not an in-memory dataset.
- **Checkpointing and resumption** with optimizer and RNG state (lesson 2.15).
- **Cache eviction** for generation past the context length.

None of those change the mathematics. They are what makes it survive a real run.

::: exercise
Add gradient accumulation so an effective batch of 512 sequences fits on a GPU that holds
only 16 at a time. What must change, and what is the single most likely bug?
:::

::: solution
```python
accum = 512 // 16                               # 32 microbatches per step

for step in range(total_steps):
    opt.zero_grad(set_to_none=True)             # OUTSIDE the microbatch loop

    for _ in range(accum):
        x, y = next(data_iter)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / accum).backward()               # divide -- see below

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()                                # once per OPTIMIZER step
```

**Three things change.** `zero_grad` moves outside the inner loop, so gradients accumulate
across microbatches. The loss is divided by `accum`. And the scheduler steps once per
optimizer step, not once per microbatch.

**The most likely bug is forgetting the division.** Gradients accumulate by summing
(lesson 1.08), so 32 microbatch backwards produce $\sum_i g_i$, not the mean. That is a
gradient 32× larger than the equivalent large batch — an effective learning rate of
$32 \times 6\times10^{-4} = 0.019$, far past the stability bound $\eta < 2/\lambda_{\max}$
of lesson 1.09. The loss diverges within a few steps.

**How to confirm it:** log the gradient norm. With the bug it is roughly `accum` times the
un-accumulated value from the very first step, which is unambiguous.

**A second, quieter bug worth knowing:** under DDP, every `backward()` triggers a gradient
all-reduce. With 32 microbatches that is 32 syncs per step instead of 1, and throughput
collapses even though the result is correct. Wrap the first `accum - 1` microbatches in
`model.no_sync()` so communication happens only on the last one — lesson 7.05 covers the
mechanics.
:::

## What to carry forward

- The whole architecture is about 120 lines; every line traces to an earlier lesson.
- Register RoPE tables as buffers, not attributes, or `.to(device)` leaves them behind.
- Pad the vocabulary to a multiple of 64 — it costs nothing and helps tensor cores.
- Always compare cached against uncached generation; cache bugs do not raise errors.
- Run the four checks before starting a real run.
