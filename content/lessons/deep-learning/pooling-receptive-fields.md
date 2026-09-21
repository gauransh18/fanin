---
summary: How downsampling buys receptive field, why the effective field is far smaller than the theoretical one, and what replaced pooling.
prereqs: [convolutions]
---

Lesson 3.09's exercise showed that stacking 3×3 convolutions grows the receptive field far
too slowly to see a whole image. Downsampling is the fix, and understanding *how much* it
buys is what lets you design a network that actually sees what it needs to.

## Pooling

**Max pooling** takes the maximum over each window; **average pooling** takes the mean.

```python
import torch.nn as nn

nn.MaxPool2d(kernel_size=2, stride=2)     # halves both spatial dimensions
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d(1)                   # global: (B, C, H, W) -> (B, C, 1, 1)
```

Max pooling gives small **translation invariance**: shifting a feature by one pixel within
a window leaves the output unchanged. Note the distinction from lesson 3.09 — convolution
is *equivariant* (the output shifts with the input), pooling is *invariant* (the output
does not change). Classification wants invariance at the end and equivariance in the
middle.

The gradient of max pooling routes entirely to the argmax position; the other elements in
the window receive zero. Average pooling distributes gradient evenly.

## Strided convolution replaced it

Modern architectures mostly use `stride=2` convolutions instead of pooling:

```python
import torch.nn as nn

nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # downsample + learn
```

The downsampling is then **learned** rather than fixed, and it happens in the same
operation as the feature extraction. Pooling survives in two places: as global average
pooling before a classifier head, and inside architectures where a parameter-free
operation is wanted deliberately.

## Receptive field arithmetic

Track two quantities layer by layer: the receptive field $r$ and the cumulative stride
$j$ (the input-pixel spacing between adjacent output positions).

$$
j_{\text{out}} = j_{\text{in}} \cdot s, \qquad
r_{\text{out}} = r_{\text{in}} + (k-1)\cdot d \cdot j_{\text{in}}
$$

```python
def receptive_field(layers):
    """layers: list of (kernel, stride, dilation)."""
    r, j = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * j
        j *= s
        print(f'k={k} s={s} d={d}  ->  receptive field {r:4d}, jump {j:3d}')
    return r

# A ResNet-ish stem plus four stages.
receptive_field([(7,2,1), (3,2,1)] + [(3,1,1)] * 4 + [(3,2,1)] + [(3,1,1)] * 4)
```

The key effect: **each stride-2 layer doubles $j$**, so every subsequent layer contributes
twice as much receptive field as the one before. That is what turns lesson 3.09's 112
layers into about 20.

## The effective receptive field

Theoretical receptive field is an upper bound on which pixels *can* influence an output.
It is not a description of which ones *do*.

::: key
Contributions from the edge of the receptive field pass through far fewer paths than
contributions from the centre. Summing over all paths gives an approximately Gaussian
weighting, and the **effective** receptive field grows like $O(\sqrt{n})$ with depth while
the theoretical one grows like $O(n)$.

Consequence: a network with a theoretical receptive field of 224 pixels may be genuinely
sensitive to a region of 60. If your task needs true global context, do not assume depth
will supply it — measure it, or use an operation that is global by construction.
:::

Measuring it takes one backward pass:

```python
import torch

def effective_receptive_field(model, size=224):
    x = torch.zeros(1, 3, size, size, requires_grad=True)
    out = model(x)
    # Gradient of the centre output unit with respect to the input.
    centre = out[0, :, out.shape[2] // 2, out.shape[3] // 2].sum()
    centre.backward()
    grad = x.grad[0].abs().sum(0)
    support = (grad > grad.max() * 0.01).float()
    print(f'effective radius ~{support.sum().sqrt().item()/2:.1f} px')
    return grad
```

## The resolution–channels trade

Every downsampling stage typically doubles the channel count. That is not arbitrary — it
keeps the per-layer FLOP count roughly constant:

$$
\text{FLOPs} \propto H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}}
$$

Halving $H$ and $W$ divides by 4; doubling both channel counts multiplies by 4. The
budget per stage stays level while the representation shifts from "many positions, few
features" to "few positions, many features".

| Stage | Resolution | Channels | Relative FLOPs |
|---|---|---|---|
| 1 | 56×56 | 64 | 1.00 |
| 2 | 28×28 | 128 | 1.00 |
| 3 | 14×14 | 256 | 1.00 |
| 4 | 7×7 | 512 | 1.00 |

## Where it went in transformers

Vision transformers do the downsampling **once**, in the patch embedding: a
$224\times224$ image becomes $14\times14 = 196$ patches of 16×16 pixels, via a stride-16
convolution. After that, resolution is constant and every layer is global.

Hierarchical variants such as Swin reintroduce stages — merging patches to halve
resolution and double dimension — recovering the pyramid for dense prediction tasks where
multi-scale features matter.

::: exercise
You need per-pixel segmentation at full resolution, but you also need global context.
These pull in opposite directions. How do standard architectures resolve it?
:::

::: solution
The tension is real: global context requires downsampling (or attention), and per-pixel
output requires full resolution. Three standard resolutions, each with a different
tradeoff.

**1. Encoder–decoder with skip connections (U-Net).** Downsample to a low-resolution
bottleneck with a large receptive field, then upsample back, concatenating the
corresponding encoder feature map at each level. The bottleneck supplies semantics, the
skips supply spatial precision. This is the dominant design for medical imaging and
diffusion models.

**2. Dilated convolutions (DeepLab).** Keep resolution at 1/8 and grow the receptive field
with dilation instead of stride. No information is lost to downsampling, so no upsampling
is needed — at the cost of running later layers at 64× the spatial positions, which is
expensive. Atrous spatial pyramid pooling adds parallel branches at several dilations to
capture multiple scales at once.

**3. Feature pyramid networks.** Predict from several stages at once, with a top-down path
that adds upsampled semantic features into the higher-resolution ones. Standard for object
detection, where objects appear at many scales in the same image.

**What transformers do:** attention is global at every layer, so the receptive-field half
of the problem disappears. The remaining problem is that attention costs $O(T^2)$ and full
resolution means a very large $T$. Hierarchical ViTs (Swin) solve it by restricting
attention to local windows and shifting the windows between layers so information still
propagates globally — which is, in the end, the same pyramid idea reached from the other
direction.
:::

## What to carry forward

- Pooling gives invariance; convolution gives equivariance. You want both, in that order.
- $r \mathrel{+}= (k-1)\,d\,j$ and $j \mathrel{*}= s$: stride is what makes receptive field grow fast.
- The effective receptive field grows like $\sqrt{n}$, not $n$ — measure it before assuming.
- Doubling channels at each halving of resolution keeps per-stage FLOPs constant.
- Full-resolution output plus global context needs an encoder–decoder, dilation, or attention.
