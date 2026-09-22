---
summary: Compression as a learning objective, why the vanilla version learns a disappointing subspace, and the variational and sparse variants that fixed it.
prereqs: [singular-value-decomposition, rank-span-subspaces]
---

An autoencoder learns to reconstruct its input through a bottleneck. The bottleneck is the
whole mechanism: it forces the model to keep only what is needed, and what it chooses to
keep is the representation you were after.

## The architecture

$$
\mathbf{z} = f_\theta(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\phi(\mathbf{z}), \qquad
\mathcal{L} = \lVert\mathbf{x}-\hat{\mathbf{x}}\rVert_2^2
$$

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, d_in=784, d_latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(),
            nn.Linear(256, d_latent))
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 256), nn.GELU(),
            nn.Linear(256, d_in))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

By lesson 1.04, a bottleneck of width $r$ caps the end-to-end map at rank $r$ — at least
$d - r$ input directions are destroyed and cannot be recovered. The training objective
decides *which* directions get spent.

## The linear case is just PCA

::: key
A linear autoencoder with MSE loss and latent dimension $k$ learns exactly the subspace
spanned by the top $k$ principal components — the same answer as the SVD in lesson 1.06,
by Eckart–Young.

Two caveats. It finds the same *subspace*, but not the same *basis*: any invertible
$k\times k$ transform of the latent gives identical loss, so the axes are arbitrary and
unordered. And it is a far more expensive way to compute something `torch.linalg.svd`
gives in closed form.

The lesson: **a vanilla autoencoder is only interesting when it is nonlinear.** If yours
is behaving like PCA, the nonlinearities are not doing anything.
:::

::: check
Your autoencoder's reconstructions look exactly like PCA's. What does that tell you?

- [x] The nonlinearities are not doing anything — a linear autoencoder with MSE learns the top-$k$ principal subspace and nothing more
  > By Eckart–Young it recovers the same subspace the SVD gives in closed form, at far greater cost. A vanilla autoencoder is only interesting when it is genuinely nonlinear.
- [ ] It has converged to the global optimum, which is the best possible result
  > It may well be at the optimum *of the linear problem*. That is the complaint, not the reassurance.
- [ ] The latent dimension is too large
  > A larger latent would reconstruct better, not more linearly.
- [ ] PCA and autoencoders always agree, so this is expected
  > They agree only in the linear case, and even there only on the subspace: any invertible $k\times k$ transform of the latent gives identical loss, so the axes are arbitrary and unordered.
:::

## What goes wrong

**The latent space has no structure.** Nothing in the objective says nearby latents should
decode to similar outputs. Sampling a random $\mathbf{z}$ and decoding it typically
produces noise, because the encoder maps the data onto a thin, irregular manifold and
everything between is unvisited. A vanilla autoencoder is a compressor, not a generator.

**It can learn the identity.** If the latent is wide enough and the network flexible
enough, it will simply copy — reconstruction is perfect and the representation is useless.

The fixes all amount to adding a second pressure that competes with reconstruction.

## Denoising

Corrupt the input, reconstruct the clean version:

```python
import torch

noisy = x + 0.3 * torch.randn_like(x)         # or random masking
recon, _ = model(noisy)
loss = F.mse_loss(recon, x)                   # target is the CLEAN x
```

Now the identity is not a solution, because the identity would reproduce the noise. The
model must learn the structure that distinguishes signal from corruption.

This is the direct ancestor of masked language modelling — BERT is a denoising
autoencoder over tokens, where the corruption is masking 15% of them — and of diffusion
models, where the corruption is Gaussian noise at a schedule of levels.

## Variational autoencoders

A VAE makes the latent space a probability distribution, so it can be sampled from.

The encoder outputs a mean and log-variance rather than a point; the decoder reads a
sample; and a KL term pulls the posterior toward a standard normal:

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\!\left[-\log p(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction}} + \underbrace{\beta\, D_{\text{KL}}\!\left(q(\mathbf{z}\mid\mathbf{x}) \,\|\, \mathcal{N}(0,I)\right)}_{\text{regularise the latent}}
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.GELU())
        self.mu, self.logvar = nn.Linear(256, d_latent), nn.Linear(256, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, 256), nn.GELU(),
                                 nn.Linear(256, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        # Reparameterisation: sample as mu + sigma * eps so the gradient
        # flows through mu and sigma rather than through a sampling op.
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    rec = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
    return rec + beta * kl
```

The **reparameterisation trick** is the essential piece. Sampling is not differentiable,
so writing $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$
with $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ moves the randomness to a term with no
parameters, leaving a differentiable path to $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

::: warning
**Posterior collapse.** If the decoder is powerful enough to model the data unconditionally
— an autoregressive decoder, for instance — the cheapest way to minimise the KL term is to
make $q(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(0,I)$ exactly, ignoring $\mathbf{x}$. The
latent carries zero information and the VAE degenerates into an unconditional model.

Symptoms: KL term near zero, reconstructions that look plausible but bear no relation to
the specific input. Fixes: anneal $\beta$ from 0 upward, use free bits (a floor on the KL
per dimension), or weaken the decoder.
:::

## Sparse autoencoders, and why they returned

Constrain the latent to be sparse rather than narrow:

$$
\mathcal{L} = \lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2 + \lambda\lVert\mathbf{z}\rVert_1
$$

By lesson 1.01, the $\ell_1$ penalty's corners push values exactly to zero. The latent may
be *wider* than the input — an over-complete dictionary — as long as few entries are
active at once.

This is now one of the main tools in mechanistic interpretability. Train a sparse
autoencoder on a transformer's residual stream (lesson 3.08) and the learned dictionary
directions turn out to be far more interpretable than the raw neurons — which is what you
would expect if the model is using superposition (lesson 3.13) to pack many features into
fewer dimensions.

## VQ-VAE

Quantise the latent to entries of a learned codebook, making the representation discrete:

```python
# Nearest codebook entry, with a straight-through estimator for the gradient
# (lesson 2.07) since argmin is not differentiable.
distances = (z.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
indices = distances.argmin(dim=1)
z_q = codebook[indices]
z_q = z + (z_q - z).detach()          # forward: quantised; backward: through z
```

Discrete latents let you model images and audio with an autoregressive transformer over
code indices, which is how the first generation of image and audio generation models were
built.

::: exercise
Your VAE produces blurry reconstructions. Explain why MSE reconstruction loss causes this,
and give two fixes.
:::

::: solution
**Why MSE blurs.** From lesson 1.14, MSE is the negative log-likelihood of a Gaussian, and
the value minimising expected squared error is the **conditional mean**. When several
outputs are plausible for one latent — the digit could be a 3 or an 8, the edge could be
here or two pixels over — the loss-minimising prediction is their *average*, which is
blurry. It is not a failure of optimisation; the blur **is** the optimum of that objective.

The VAE's sampling makes it worse: the decoder must produce something reasonable for every
$\mathbf{z}$ in a neighbourhood, so it hedges across all of them.

**Fix 1: an adversarial or perceptual loss.** Add a discriminator that must tell real from
reconstructed (VAE-GAN), or compare features from a pretrained network rather than pixels
(LPIPS). Both penalise *implausibility* rather than pixel distance, so averaging two
plausible outputs is no longer a good answer. This is what modern latent diffusion
autoencoders use.

**Fix 2: a discrete latent with an autoregressive decoder.** VQ-VAE quantises the latent
and models the code indices with a transformer. The decoder then produces one specific
sample rather than a conditional mean, because the autoregressive factorisation
(lesson 1.10) never averages over modes — it commits at each step.

**Why not just "use a better decoder":** a more powerful decoder under MSE still optimises
toward the conditional mean. The objective has to change, not just the capacity.
:::

## What to carry forward

- A bottleneck caps rank; a linear autoencoder recovers PCA's subspace and nothing more.
- Vanilla autoencoders compress but do not generate — the latent space has no structure.
- Denoising removes the identity solution and is the ancestor of BERT and diffusion.
- VAEs need the reparameterisation trick, and watch for posterior collapse.
- MSE reconstruction blurs because the conditional mean is its optimum.
