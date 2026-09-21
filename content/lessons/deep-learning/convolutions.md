---
summary: Convolution as a constrained linear layer, the three priors it encodes, and why transformers can beat it given enough data.
prereqs: [matrices-as-linear-maps, mlps-universal-approximation]
---

A convolution is a matrix multiplication with two constraints imposed: weights are shared
across positions, and each output depends on only a local neighbourhood. Those constraints
are a *prior* about the data, and understanding them as a prior explains both why
convolutions dominated vision for a decade and why they were displaced.

## The operation

For a 1-D input $x$ and kernel $w$ of size $k$:

$$
y_i = \sum_{j=0}^{k-1} w_j \, x_{i+j} + b
$$

In 2-D with multiple channels:

$$
y_{c_{\text{out}}, i, j} = b_{c_{\text{out}}} + \sum_{c_{\text{in}}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} w_{c_{\text{out}}, c_{\text{in}}, u, v}\; x_{c_{\text{in}},\, i+u,\, j+v}
$$

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                 stride=1, padding=1, bias=False)
# Parameters: 64 * 3 * 3 * 3 = 1,728 -- independent of image size.
```

That last comment is the point. A fully connected layer mapping a $224\times224\times3$
image to 64 channels of the same size would need $224^2\cdot3 \times 224^2\cdot64 \approx
4.8\times10^{11}$ parameters. The convolution needs 1,728.

## The three priors

::: key
**1. Locality.** An output depends only on a $k\times k$ neighbourhood. This asserts that
nearby pixels are more related than distant ones — true for images, false for arbitrary
tabular data.

**2. Translation equivariance.** Shift the input, and the output shifts identically:
$f(T_\delta x) = T_\delta f(x)$. A cat detector works anywhere in the frame because the
same weights are applied everywhere.

**3. Parameter sharing.** One kernel serves every position, so the parameter count is
independent of input size and the layer generalises across positions automatically.
:::

These are strong, correct assumptions about images. They are wrong for many other things —
which is exactly why convolutions never took over language, where the relevant dependency
can be a thousand tokens away and is not translation-equivariant in any useful sense.

## Output size

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(k-1) - 1}{s} \right\rfloor + 1
$$

with padding $p$, dilation $d$, stride $s$. The case worth memorising: with $s=1$, $d=1$,
and $p = (k-1)/2$ for odd $k$, the output size equals the input size. That is why 3×3
convolutions with padding 1 are ubiquitous — they compose without changing resolution.

## Convolution as a matrix

A convolution *is* a linear map (lesson 1.02), so it has a matrix. For 1-D input of length
5 and kernel $[a, b, c]$ with no padding:

$$
\begin{bmatrix} a & b & c & 0 & 0 \\ 0 & a & b & c & 0 \\ 0 & 0 & a & b & c \end{bmatrix}
$$

A **Toeplitz** matrix: constant along diagonals, mostly zeros. So a convolution is a fully
connected layer whose weight matrix is constrained to be sparse and to have tied entries.
Every property follows from that: fewer parameters (tying), locality (sparsity), and
equivariance (the diagonal structure).

This is also how convolutions are implemented on GPUs — `im2col` unfolds the input into a
matrix so the convolution becomes a single dense matrix product, which tensor cores handle
far better than a direct sliding-window loop.

## Efficient variants

**Depthwise separable.** Split into a per-channel spatial convolution and a $1\times1$
cross-channel mixing:

```python
import torch.nn as nn

# Standard: C_in * C_out * k * k parameters
standard = nn.Conv2d(256, 256, 3, padding=1)             # 589,824

# Separable: C_in * k * k + C_in * C_out
separable = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1, groups=256),       # 2,304 depthwise
    nn.Conv2d(256, 256, 1),                              # 65,536 pointwise
)                                                        # 67,840 total -- 8.7x fewer
```

MobileNet and EfficientNet are built on this. Note the structure: spatial mixing and
channel mixing are separated — the same factorisation a transformer makes between
attention (position mixing) and the MLP (feature mixing).

**Dilated.** Insert gaps in the kernel to grow the receptive field without more
parameters. Stacking dilations 1, 2, 4, 8 gives exponential receptive-field growth, which
is how WaveNet modelled raw audio.

**1×1.** No spatial extent at all — purely a linear map across channels, applied
position-wise. Identical in function to a transformer's position-wise MLP.

## Where transformers won

Vision transformers (lesson 4.15) throw away all three priors and treat an image as a
sequence of patches with global attention.

With limited data, ViTs lose — they must *learn* locality and equivariance from examples
that a CNN gets for free. With enough data (roughly 100M+ images, or heavy augmentation
and distillation), they win, because the learned attention patterns are better adapted to
the data than the hand-imposed prior.

::: insight
This is the bias–variance tradeoff of lesson 1.11 at the level of architecture. A strong
prior is high bias and low variance: it helps when data is scarce and **caps** performance
when data is abundant. The history of deep learning is largely a sequence of replacing
hand-designed priors with learned ones as datasets grew.

Modern hybrids get both: convolutional stems for early layers where locality is clearly
correct, attention for later layers where long-range relationships matter.
:::

::: exercise
A 3×3 convolution with stride 1 has a receptive field of 3 pixels. What is the receptive
field after stacking $n$ such layers, and how many layers do you need to see a whole
$224\times224$ image?
:::

::: solution
Each 3×3 layer extends the receptive field by 2 — one pixel on each side. So after $n$
layers:

$$
r_n = 1 + 2n
$$

For $r_n \ge 224$: $n \ge 111.5$, so **112 layers**. That is impractically deep for the
purpose, and it is why plain stacks of 3×3 convolutions are not how global context is
obtained.

**The three standard fixes:**

**Stride or pooling.** Downsampling by 2 doubles the effective receptive field of every
subsequent layer. With five stride-2 stages, the receptive field grows roughly
$2^5 = 32\times$ faster, and about 20 layers suffice. This is the ResNet design.

**Dilation.** Dilations 1, 2, 4, 8, 16 give receptive field $1 + 2(1+2+4+8+16) = 63$ from
five layers, with no downsampling and no extra parameters.

**Attention.** One attention layer has a receptive field of the entire input, by
construction. This is the structural advantage transformers have: global context in one
layer rather than a hundred.

**The caveat worth knowing:** the *effective* receptive field is much smaller than the
theoretical one. Contributions decay roughly as a Gaussian from the centre, so a layer
with a theoretical field of 224 pixels has most of its sensitivity concentrated in a
region of order $\sqrt{n}$. Theoretical receptive field is an upper bound, not a
description of behaviour.
:::

## What to carry forward

- A convolution is a Toeplitz matrix — a linear layer with tied, sparse weights.
- Its three priors are locality, translation equivariance and parameter sharing.
- Output size $= \lfloor (H + 2p - k)/s \rfloor + 1$; 3×3 with padding 1 preserves resolution.
- Separable convolutions split spatial from channel mixing, as transformers do.
- Strong priors win on small data and cap performance on large data.
