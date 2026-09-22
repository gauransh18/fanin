---
summary: Why bf16 replaced fp16, what autocast actually does, and the ordering rules that make AMP correct.
prereqs: [tensors-dtypes-devices, training-loop]
---

Mixed precision runs the expensive operations in 16 bits and the delicate ones in 32,
roughly halving memory and roughly doubling throughput on tensor-core hardware. The
"mixed" is the important word — getting the split wrong is how precision bugs happen.

## The two 16-bit formats, again

From lesson 2.01: fp16 has 10 mantissa bits and 5 exponent bits, maxing out at 65,504.
bf16 has 7 mantissa bits and 8 exponent bits — the **same exponent range as fp32**.

That difference decides everything about how you use them:

| | fp16 | bf16 |
|---|---|---|
| Overflow risk | Real, at 65,504 | Same as fp32 |
| Loss scaling needed | Yes | No |
| Precision | Better (10 bits) | Worse (7 bits) |
| Hardware | Volta and later | Ampere and later |

::: key
If your hardware supports bf16, use bf16. The extra mantissa bits of fp16 do not
compensate for an entire class of failure mode that requires dynamic loss scaling to
work around. Every large transformer training run of the last several years uses bf16.
:::

## autocast

`torch.autocast` maintains a per-operation policy. It does not cast your model; it
intercepts operations and chooses their compute dtype.

```python
import torch

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(x)           # matmuls and convs run in bf16
    loss = criterion(out, y) # softmax/cross-entropy stay in fp32
```

The policy has three tiers, and they follow directly from numerical stability:

- **16-bit**: `matmul`, `conv`, `linear`, `bmm`, `einsum` — dominated by accumulation
  over many terms, which tensor cores do in fp32 internally anyway.
- **fp32**: `softmax`, `log_softmax`, `layer_norm`, `sum`, `exp`, `log`, `pow`,
  anything computing a loss. These either exponentiate (risking overflow) or reduce over
  many values (losing precision to rounding).
- **Widest input type**: `add`, `cat`, and other operations where the answer should not
  depend on which operand happened to be narrower.

::: warning
Autocast applies to the **forward pass only**. Put the backward call outside the context.
Gradients automatically use the dtype chosen for each operation's forward, so wrapping
`backward()` does nothing useful and can confuse the caching allocator.

```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x, y)
loss.backward()              # outside -- this is correct
```
:::

::: check
Under autocast, which operations are kept in fp32 rather than run in 16 bits?

- [x] `softmax` and `log_softmax`
  > They exponentiate, which risks overflow, and they feed losses where precision matters most.
- [x] `layer_norm` and `sum`
  > Reductions over many values lose precision to rounding, and a normalization's statistics are exactly such a reduction.
- [ ] `matmul` and `linear`
  > These are the ones you *want* in 16 bits: they are dominated by accumulation over many terms, which tensor cores do in fp32 internally anyway. This is where the speed comes from.
- [ ] `einsum` and `bmm`
  > Same story — they dispatch to the same tensor-core paths and belong in the 16-bit tier.
:::

## Loss scaling, if you must use fp16

Gradients are typically much smaller than activations. In fp16, values below about
$6\times10^{-8}$ flush to zero, so small gradients vanish entirely before the optimizer
sees them.

`GradScaler` multiplies the loss by a large constant before backward — scaling every
gradient up into representable range by the chain rule — then divides it out before the
optimizer step.

```python
import torch

scaler = torch.amp.GradScaler('cuda')

for x, y in loader:
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(x, y)

    scaler.scale(loss).backward()        # gradients are now scale * true
    scaler.unscale_(opt)                 # MUST precede clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                     # skips the step if any grad is inf/nan
    scaler.update()                      # adjusts the scale factor
    opt.zero_grad(set_to_none=True)
```

The scale is chosen dynamically: it doubles every 2,000 successful steps and halves
whenever an overflow is detected, converging on the largest safe value.

::: warning
`scaler.unscale_(opt)` before clipping is not optional. Clipping scaled gradients to a
norm of 1.0 clips them to $1.0/\text{scale}$ in true units — with a typical scale of
65,536 that is a clip threshold of $1.5\times10^{-5}$, which annihilates the entire
gradient. This is a genuinely common bug and it presents as "the model trains extremely
slowly for no reason".

With bf16 you need none of this. That is the argument in one paragraph.
:::

## Master weights

Parameters are kept in fp32 and cast to bf16 for each forward. The reason is
accumulation: bf16 has ~3 decimal digits, so an update smaller than about $10^{-3}$
relative to the weight simply does not change it.

```python
import torch

w = torch.tensor([1.0], dtype=torch.bfloat16)
w + 1e-4      # tensor([1.0000]) -- the update vanished entirely
```

With a learning rate of $10^{-4}$ and weights near 1, *every single update* would be lost
if weights were stored in bf16. The fp32 master copy is what makes the arithmetic work,
and it is why lesson 2.01's memory accounting has both a 2-byte and a 4-byte copy of
every parameter.

::: check
Where does `loss.backward()` belong relative to the `autocast` context?

- [x] Outside it — autocast applies to the forward pass only
  > Each operation's backward automatically uses the dtype chosen for its forward, so wrapping the backward call does nothing useful and can confuse the caching allocator.
- [ ] Inside it, so gradients are computed in bf16
  > Gradient dtypes are already decided by the forward policy. Wrapping the backward does not change them.
- [ ] Inside it, but only when using `GradScaler`
  > `GradScaler` wraps the *step*, not the context, and is an fp16 concern that bf16 does not need at all.
- [ ] Either; the context has no effect on backward
  > It has no *useful* effect, which is not the same as none — the advice is to keep it outside.
:::

## Verifying it worked

```python
import torch

with torch.autocast('cuda', dtype=torch.bfloat16):
    h = model.layer1(x)
    print(h.dtype)           # torch.bfloat16 -- the matmul was cast
    n = model.norm(h)
    print(n.dtype)           # torch.float32  -- layer_norm stayed wide
```

Two things to check beyond dtypes. First, that memory actually fell — if it did not, your
model is dominated by something autocast does not touch. Second, that the loss curve
matches an fp32 baseline for the first few hundred steps. A curve that tracks and then
diverges points at an accumulation done in the wrong precision, usually a hand-written
reduction.

::: exercise
You enable autocast and your custom attention implementation produces `nan` in fp16,
though the same code is fine in fp32. Where would you look first?
:::

::: solution
**The softmax input.** Attention scores are $QK^\top/\sqrt{d}$, summed over $d$ terms. For
$d = 128$ and unit-scale entries, individual scores are order 10 — fine. But if your
implementation forgets the $1/\sqrt{d}$ scaling, or if the inputs are not unit-scale, the
scores can easily reach the hundreds. Then $e^{\text{score}}$ overflows fp16's 65,504
limit and produces `inf`, and `inf/inf` in the normalisation gives `nan`.

**The diagnosis** is to print the max absolute score before the softmax. Anything above
about 11 will overflow on exponentiation in fp16 ($e^{11} \approx 60{,}000$).

**Three fixes, in order of preference:**

1. **Use bf16.** Its exponent range matches fp32, so the overflow cannot occur.
2. **Subtract the max before exponentiating** — the standard stable-softmax trick from
   lesson 2.07. PyTorch's own `softmax` does this; hand-rolled ones often do not.
3. **Force fp32 for the softmax** explicitly:
   ```python
   attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
   ```
   or wrap that section in `with torch.autocast('cuda', enabled=False):`.

The deeper point: autocast's fp32 list already contains `softmax` for exactly this
reason. If your implementation is hitting the problem, you have probably computed the
softmax with primitive `exp` and `sum` calls rather than calling `softmax`, which took
you off the policy list.
:::

## What to carry forward

- Prefer bf16; it removes loss scaling and an entire failure mode.
- autocast wraps the forward only; backward inherits the dtypes.
- Under fp16, unscale before clipping or you clip to nothing.
- fp32 master weights exist because bf16 cannot accumulate small updates.
- Hand-written reductions fall outside the autocast policy — check them.
