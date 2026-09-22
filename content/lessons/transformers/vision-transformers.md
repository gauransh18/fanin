---
summary: Images as sequences of patches, why ViTs need far more data than CNNs, and what closed the gap.
prereqs: [transformer-block, convolutions, embeddings]
---

A ViT discards every prior a convolution encodes (lesson 3.09) and treats an image as a
sequence. That is a strictly worse inductive bias — and past a data threshold, it wins
anyway.

## Patches as tokens

Split the image into non-overlapping patches, flatten each, project to $d$:

$$
\mathbf{x}_p \in \mathbb{R}^{P^2 C} \;\longrightarrow\; \mathbf{z}_p = W\mathbf{x}_p \in \mathbb{R}^{d}
$$

For $224\times224$ at $P = 16$, that is $(224/16)^2 = 196$ tokens.

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """A stride-P, kernel-P convolution IS patch extraction plus projection."""
    def __init__(self, img=224, patch=16, in_ch=3, d=768):
        super().__init__()
        self.n_patches = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)

    def forward(self, x):                        # (B, C, H, W)
        x = self.proj(x)                         # (B, d, H/P, W/P)
        return x.flatten(2).transpose(1, 2)      # (B, n_patches, d)
```

Implementing patch embedding as a convolution is not a trick — a stride-$P$, kernel-$P$
convolution is exactly "cut into patches and apply the same linear map to each", which is
the definition.

## The rest is a standard transformer

```python
import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, img=224, patch=16, d=768, depth=12, heads=12, n_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img, patch, 3, d)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, is_causal=False)          # bidirectional: it is an encoder
        return self.head(self.norm(x)[:, 0])     # classify from the CLS token
```

Three differences from a language model:

- **Bidirectional attention.** There is no causal ordering among patches.
- **A CLS token** — a learned vector prepended to the sequence, whose final representation
  is used for classification. Global average pooling over patch tokens works about as well
  and is used by many later models.
- **Learned absolute positions.** Images have fixed size at training, so the extrapolation
  problem of lesson 4.04 is less pressing — though changing input resolution requires
  interpolating the position embeddings.

::: check
`PatchEmbed` implements patch extraction as a stride-16, kernel-16 `Conv2d`. Is that a trick?

- [x] No — a stride-$P$, kernel-$P$ convolution *is* "cut into non-overlapping patches and apply the same linear map to each", which is the definition
  > It is the same operation written in the vocabulary the framework already has, and it is why the implementation is three lines.
- [ ] Yes — it reintroduces the convolutional prior the ViT was meant to discard
  > With stride equal to kernel size there is no overlap, so neither locality across patches nor translation equivariance survives. The prior is genuinely gone.
- [ ] Yes — it makes the model equivariant to patch-sized translations
  > Equivariance would need overlapping windows. Shifting the image by 8 pixels changes every patch.
- [ ] No, but it is slower than an explicit reshape and matmul
  > It dispatches to the same kind of matmul and is typically as fast or faster.
:::

## Why it needs so much data

::: key
A CNN gets locality, translation equivariance and parameter sharing **for free** from its
architecture (lesson 3.09). A ViT must learn all three from data.

Observed behaviour: ViT loses to a ResNet on ImageNet-1k (1.3M images), matches it on
ImageNet-21k (14M), and beats it decisively on JFT-300M (300M).

This is the bias–variance tradeoff at the architectural level (lesson 1.11): a strong prior
is high bias and low variance, which helps when data is scarce and **caps** you when it is
not. Attention maps in trained ViTs turn out to be local in early layers — the model
*rediscovers* convolution-like behaviour, then goes beyond it in later layers.
:::

## What closed the gap

The data requirement was largely eliminated by training recipe rather than architecture:

- **DeiT** — heavy augmentation (RandAugment, Mixup, CutMix), stochastic depth, and
  distillation from a CNN teacher. Matched CNNs on ImageNet-1k alone.
- **Hybrid stems** — a few convolutional layers before patchification, supplying locality
  where it is clearly correct and leaving attention for the rest.
- **Hierarchical ViTs (Swin)** — reintroduce the resolution pyramid of lesson 3.10, with
  attention restricted to local windows that shift between layers so information still
  propagates globally. Essential for dense prediction tasks.
- **Self-supervised pretraining** — MAE masks 75% of patches and reconstructs them, a
  denoising autoencoder (lesson 3.14) over image patches. DINO uses self-distillation and
  produces features whose attention maps segment objects without any segmentation labels.

::: check
A ViT discards locality, translation equivariance and parameter sharing across positions. Why does it win anyway, past a data threshold?

- [x] Those priors are a constraint as well as a help
  > With enough data the model learns the structure that actually holds rather than the structure a convolution assumes. Below that threshold the convolutional prior is worth more than the data, which is why ViTs underperform CNNs on small datasets and overtake them on very large ones.
- [ ] Attention has a larger receptive field, so it needs fewer layers
  > A global receptive field from layer one is real and is not the reason it needs so much data to pay off.
- [ ] Patch embedding is cheaper than convolution
  > A stride-$P$ kernel-$P$ convolution *is* the patch embedding, at the same cost.
- [ ] Transformers are easier to optimise than CNNs
  > ViTs are notoriously more sensitive to augmentation, regularisation and schedule.
:::

## Multimodal models

The practical importance of ViTs now is as the vision half of a multimodal model. The
pattern is consistent:

1. A ViT encodes the image into patch tokens.
2. A projection maps them into the language model's embedding space.
3. Those tokens are prepended to the text sequence.
4. The language model attends over both.

CLIP trains an image encoder and a text encoder jointly with the InfoNCE loss of lesson
3.04, so that matching pairs have high cosine similarity. The resulting image encoder is
what most vision-language models start from, because its representations are already
aligned with language.

::: warning
Patch count grows quadratically with resolution, and attention cost quadratically with
patch count — so cost grows as the **fourth power** of resolution. Going from 224 to 448
pixels is a 16× increase in tokens and up to 256× in attention cost.

This is why high-resolution vision models use hierarchical designs, windowed attention, or
token merging. Naively raising resolution is not affordable.
:::

::: exercise
You fine-tune a ViT pretrained at 224×224 on 384×384 images. Accuracy collapses. Why, and
what is the fix?
:::

::: solution
**The position embeddings are wrong.** At 224 with patch 16 there are 196 patches; at 384
there are 576. `self.pos_embed` was learned with 197 rows (196 patches plus CLS) and there
is now no row for patches 197–576.

Depending on the implementation you get a shape error, or — worse — silent truncation or
broadcasting that leaves most patches with no positional information at all. The patch
embedding itself is fine, since a convolution applies to any input size; only the position
table is tied to the resolution.

**The fix: interpolate the position embeddings.** Treat them as a $14\times14$ grid of
$d$-dimensional vectors, resize to $24\times24$ bicubically, and flatten back:

```python
import torch
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, old_grid, new_grid):
    cls, patches = pos_embed[:, :1], pos_embed[:, 1:]
    d = patches.shape[-1]
    patches = patches.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=(new_grid, new_grid),
                            mode='bicubic', align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, new_grid ** 2, d)
    return torch.cat([cls, patches], dim=1)

model.pos_embed = nn.Parameter(interpolate_pos_embed(model.pos_embed, 14, 24))
```

Keep the CLS token's embedding separate — it has no spatial position, and interpolating it
along with the grid corrupts it.

**This works because the learned embeddings are spatially smooth**: neighbouring patches
have similar embeddings, so interpolation produces sensible values for the new positions. A
short fine-tune afterwards recovers the remaining gap.

**Two alternatives worth knowing.** Use a relative or 2-D rotary scheme (lesson 4.04)
instead of a learned table, which handles any resolution natively — this is what newer
models do. Or keep 224 and tile the larger image into several 224 crops, processing each
and combining, which is what many high-resolution vision-language models do to bound cost.
:::

## What to carry forward

- A ViT is a standard encoder over patch tokens; patch embedding is a stride-$P$ convolution.
- It must learn locality and equivariance from data, so it needs far more of it.
- Augmentation, distillation, hybrid stems and hierarchy closed the data gap.
- Cost grows as the fourth power of resolution.
- Changing resolution requires interpolating the position embeddings.
