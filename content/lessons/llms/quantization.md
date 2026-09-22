---
summary: Serving a model in 4 bits, the outlier problem that makes naive quantization fail, and how to choose a method.
prereqs: [tensors-dtypes-devices, peft-lora]
seealso: [memory-hierarchy-roofline]
---

A 70B model is 140 GB in bf16 — two 80 GB GPUs before the KV cache. In 4 bits it is 35 GB
and fits on one. Since decode is memory-bandwidth-bound (lesson 4.10), quantization also
makes it **faster**, not just smaller.

## The mapping

Affine quantization maps a float range onto integers:

$$
q = \text{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s(q - z)
$$

with scale $s$ and zero-point $z$. **Symmetric** quantization sets $z = 0$ and is cheaper;
**asymmetric** handles skewed ranges better.

```python
import torch

def quantize_symmetric(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

w = torch.randn(4096, 4096)
q, s = quantize_symmetric(w)
print(f'error {(dequantize(q, s) - w).abs().mean():.5f}, '
      f'range {w.abs().max():.2f}, step {s:.5f}')
```

## Granularity

One scale for a whole tensor is wasteful when values vary across it:

| Granularity | Scales | Quality | Overhead |
|---|---|---|---|
| Per-tensor | 1 | Poor | Negligible |
| Per-channel | $d_{\text{out}}$ | Good | Negligible |
| Per-group (128) | $d^2/128$ | Very good | ~0.4 bits/weight |

Per-group at 128 elements is the standard for 4-bit. The scales themselves cost bits, which
is why "4-bit" methods are typically 4.5 effective bits — and why double quantization
(quantizing the scales) exists.

## The outlier problem

::: key
Transformer activations contain **outliers**: a small number of channels with magnitudes
10–100× the rest, appearing consistently in the same channels across inputs. They emerge
during training and are functionally important — zeroing them collapses quality.

Naive quantization is destroyed by them. Setting the scale from the maximum means the
typical value uses a fraction of the available range:

With a max of 100 and typical values near 1, int8's step size is $100/127 \approx 0.79$ —
so a value of 1.0 quantizes to 1 and a value of 1.4 also quantizes to 1. The information in
ordinary weights is lost to accommodate a handful of extremes.

This is why 8-bit weight quantization works easily and 8-bit **activation** quantization
does not.
:::

Three responses:

**LLM.int8()** keeps outlier channels in fp16 and quantizes the rest to int8. Exact for the
outliers, and the mixed-precision matmul is slower than a uniform one.

**SmoothQuant** migrates the difficulty from activations to weights. Scale activations down
by a per-channel factor $s$ and weights up by the same factor — the product is unchanged,
and both are now quantizable:

$$
Y = (X \oslash s)\,(s \odot W)
$$

**AWQ** observes that not all weights matter equally. Identify the ~1% of weight channels
that are salient (by activation magnitude, not weight magnitude) and scale them to protect
them from quantization error.

::: check
Quantizing a 70B model from bf16 to 4 bits cuts memory from 140 GB to about 35 GB. Why does it also make decoding *faster*?

- [x] Decode is memory-bandwidth-bound
  > Reading a quarter as many bytes per token is a direct speedup. One new token against a large cache has almost no arithmetic per byte read. Shrinking the bytes is the whole lever, which is also why KV-cache quantization pays off.
- [ ] Integer arithmetic is faster than bf16 on tensor cores
  > Weights are typically dequantised to a floating format for the matmul, so the arithmetic is not what changes.
- [ ] Smaller models need fewer layers
  > Quantization changes the representation, not the architecture. Every layer is still there.
- [ ] It enables larger batches, which is where the speedup comes from
  > Larger batches are a real second-order benefit, and the per-token saving happens at any batch size.
:::

## The methods

**GPTQ** quantizes weights column by column, using second-order information to adjust the
remaining columns to compensate for each rounding error. Needs a small calibration set;
4-bit GPTQ typically loses well under a point of perplexity.

**AWQ** protects salient channels by scaling. Faster to run than GPTQ and comparable in
quality.

**NF4** (from QLoRA, lesson 5.06) uses a 4-bit format whose levels sit at the quantiles of a
normal distribution — information-theoretically optimal when the weights are Gaussian, which
they approximately are.

**GGUF** is a family of formats used by llama.cpp, mixing bit-widths across layers with
names like `Q4_K_M`. Widely used for local deployment.

```python
# pip install auto-gptq
from transformers import AutoModelForCausalLM, GPTQConfig

cfg = GPTQConfig(
    bits=4,
    group_size=128,
    dataset='c4',                 # calibration data
    desc_act=True,                # order columns by activation magnitude
)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf',
                                             quantization_config=cfg)
```

::: check
Transformer activations contain a few features with magnitudes far above the rest. Why does that break naive quantization?

- [x] A single outlier stretches the scale for the whole tensor
  > Every ordinary value is crushed into a handful of quantization levels. The responses are all about granularity: per-channel scales, keeping outlier channels in higher precision, or rotating the representation so the outliers are spread out.
- [ ] Outliers overflow the integer range and wrap around
  > Clamping handles the range; the damage is to the resolution of everything else.
- [ ] Outliers are noise and should simply be clipped
  > They carry real signal — clipping them measurably degrades quality, which is why so much work goes into preserving them.
- [ ] They make the zero-point non-integer
  > A non-integer zero-point is a representational detail with a standard fix.
:::

## What it costs

| Precision | 70B size | Typical perplexity change |
|---|---|---|
| bf16 | 140 GB | baseline |
| int8 | 70 GB | +0.01 |
| int4 (GPTQ/AWQ) | 35 GB | +0.1 to +0.3 |
| int3 | 26 GB | +1.0 or worse |
| int2 | 18 GB | usually unusable |

4-bit is the practical floor for general use. Below it, quality falls faster than size.

::: warning
**A larger model at 4 bits beats a smaller model at 16 bits, at equal memory.** A 70B model
at 4 bits (35 GB) outperforms a 13B model at bf16 (26 GB) on essentially every benchmark.

When memory-constrained, quantize the biggest model that fits rather than running a smaller
one at full precision. This is one of the most reliable results in deployment practice and
it is frequently got backwards.
:::

## Quantizing the KV cache

Lesson 4.10 showed the cache can exceed the weights. It quantizes too:

```python
# Keys are more sensitive than values -- they feed a dot product whose result
# is exponentiated, so error is amplified by the softmax.
kv_config = {'key_bits': 8, 'value_bits': 4, 'group_size': 64}
```

The asymmetry is real and worth exploiting: int8 keys with int4 values is a common and
effective choice.

## Quantization-aware training

Post-training quantization is the usual approach. When it loses too much, QAT simulates
quantization during training so the model learns weights robust to it — using the
straight-through estimator of lesson 2.07 to get gradients through the rounding.

More expensive, and worth it mainly below 4 bits or for models being deployed at very
large scale.

::: exercise
You quantize a 7B model to int8 and perplexity is unchanged. You quantize activations to
int8 as well and it collapses. Why?
:::

::: solution
**Weights and activations have very different distributions.**

**Weights** are approximately Gaussian, centred near zero, with a bounded range — roughly
$[-0.5, 0.5]$ after training. The max is close to typical magnitudes, so int8's 256 levels
spread evenly over the useful range. Quantization error is small relative to the values,
and it is also *unbiased* across many weights, so errors partly cancel in the sum.

**Activations contain systematic outliers.** Certain channels consistently carry values
10–100× the rest. With a max of 100 and typical values near 1:

$$
\text{step} = \frac{100}{127} \approx 0.79
$$

A typical activation of 1.0 is represented by a single quantization level, and anything
between 0.4 and 1.2 maps to the same value. The relative error on ordinary activations is
near 100%.

**Why it collapses rather than degrades.** The outlier channels are functionally important
— they are not noise. And because they are the same channels across inputs, the error is
*systematic*, not random: it does not cancel in the matrix product, it accumulates
coherently across layers.

**The fixes, in order of ease:**

1. **Keep activations in bf16 and quantize only weights.** This is what almost everyone
   does, and it captures most of the memory and bandwidth benefit — in the decode regime
   the weights are what you are reading from HBM, not the activations.

2. **SmoothQuant.** Migrate the range difficulty into the weights with a per-channel factor:
   $Y = (X \oslash s)(s \odot W)$. Weights have headroom to absorb it; activations do not.
   Both become quantizable and the product is mathematically unchanged.

3. **Mixed precision on outlier channels** (LLM.int8()). Keep the ~0.1% of channels
   identified as outliers in fp16. Exact, and the mixed matmul costs throughput.

**How to confirm the diagnosis in two minutes:**

```python
acts = capture_activations(model, calibration_batch)   # lesson 2.08 hooks
for name, a in acts.items():
    ratio = a.abs().max() / a.abs().median()
    print(f'{name:40s} max/median = {ratio:8.1f}')
```

A ratio above ~50 marks a layer where activation quantization will fail. Weights will show
ratios around 5–10.
:::

## What to carry forward

- Per-group scales at 128 elements are the standard for 4-bit; the scales cost bits too.
- Activation outliers are systematic and functionally important — they break naive quantization.
- Weight-only int8 is nearly free; activation int8 needs SmoothQuant or mixed precision.
- 4-bit is the practical floor; a bigger model at 4 bits beats a smaller one at 16.
- Quantize the KV cache too, with more bits for keys than values.
