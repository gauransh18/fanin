---
summary: The recurrence, why backpropagation through time cannot be parallelised, and the spectral-radius argument that limits what an RNN can remember.
prereqs: [eigenvalues-eigenvectors, chain-rule-backprop]
---

Recurrent networks process sequences by carrying a hidden state forward. They are worth
understanding in a transformer curriculum for two reasons: the gradient argument that
killed them is the same one that motivates residual connections, and state space models
(lesson 4.16) are recurrence returning with the problems fixed.

## The recurrence

$$
\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})
$$

The same weights at every timestep — parameter sharing across time, exactly analogous to
a convolution's sharing across space.

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.W_hh = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_xh = nn.Linear(d_in, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, x):                       # x: (B, T, d_in)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d_hidden)
        outputs = []
        for t in range(T):                      # inherently sequential
            h = torch.tanh(self.W_hh(h) + self.W_xh(x[:, t]))
            outputs.append(h)
        return torch.stack(outputs, dim=1)      # (B, T, d_hidden)
```

## Backpropagation through time

Unroll the recurrence and the network becomes a $T$-layer feedforward net with tied
weights. The gradient of the loss at step $T$ with respect to an early state is a product
of Jacobians:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k} = \prod_{t=k+1}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
= \prod_{t=k+1}^{T} \text{diag}\!\left(\tanh'(\mathbf{z}_t)\right) W_{hh}^\top
$$

Lesson 1.05's spectral-radius argument applies directly. Writing $\rho$ for the spectral
radius of $W_{hh}$ and noting $\tanh' \le 1$:

- $\rho < 1$ — the product decays like $\rho^{T-k}$. At $\rho = 0.9$ over 100 steps,
  $2.6\times10^{-5}$: **no gradient reaches back**, so long-range dependencies cannot be
  learned.
- $\rho > 1$ — the product grows like $\rho^{T-k}$. At $\rho = 1.1$ over 100 steps,
  $1.4\times10^4$: training diverges in one update.

::: key
The stable window is $\rho \approx 1$ exactly, and there is no mechanism keeping it there.
Worse, $\tanh'$ is strictly below 1 away from the origin, so the effective multiplier is
*less* than $\rho$ — a saturated RNN vanishes even with a well-conditioned $W_{hh}$.

Exploding gradients are easy to fix: clip by global norm (lesson 2.09). Vanishing
gradients cannot be clipped — there is nothing to rescale. That asymmetry is why the
solution had to be architectural, and the answer was gating (lesson 3.12) and later the
residual path of lesson 3.08.
:::

```python
import torch

# The argument, empirically.
for rho in (0.9, 1.0, 1.1):
    W = torch.randn(64, 64)
    W = W * (rho / torch.linalg.eigvals(W).abs().max())   # set spectral radius
    g = torch.randn(64)
    for t in (10, 50, 100):
        norm = (torch.linalg.matrix_power(W.T, t) @ g).norm()
        print(f'rho={rho}  t={t:3d}  gradient norm {norm:.3e}')
```

## The real problem: no parallelism

Even with the gradient problem solved, recurrence has a structural limitation that matters
more on modern hardware.

$\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$, so the $T$ steps must run **in sequence**.
Training a transformer on a sequence of 2,048 tokens is one large matrix product per
layer; training an RNN on the same sequence is 2,048 small ones, each too small to fill a
GPU.

This is the arithmetic-intensity argument from lesson 1.03. An RNN step is a
matrix–vector product — memory-bound, a fraction of peak FLOP/s. A transformer layer is a
matrix–matrix product — compute-bound, near peak.

::: insight
The transformer's decisive advantage was never that attention is a better inductive bias
for language. It is that attention is **parallel over positions during training**, so it
can consume the full compute of a modern accelerator, and RNNs cannot.

The tradeoff reverses at inference: generating token by token, a transformer must attend
over the whole growing context ($O(T)$ work per token, $O(T)$ memory for the KV cache),
while an RNN carries a fixed-size state and does $O(1)$ work per token. Lesson 4.16's
state space models are an attempt to have both — parallel training, recurrent inference.
:::

## Truncated BPTT

Full backpropagation through a long sequence needs every hidden state stored, which is
$O(T)$ memory. Truncation caps it:

```python
h = torch.zeros(B, d_hidden, device=device)
for chunk in chunks(sequence, size=128):
    h = h.detach()                              # cut the graph, keep the value
    out, h = model(chunk, h)
    loss = criterion(out, targets)
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
```

The `detach()` is the whole technique: the forward pass carries state across chunk
boundaries, but gradients do not flow past them. You get $O(k)$ memory at the cost of
never learning dependencies longer than $k$.

## Bidirectional and stacked

```python
import torch.nn as nn

rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=3,
              bidirectional=True, batch_first=True, dropout=0.1)
```

Bidirectional RNNs run a second pass in reverse and concatenate, so each position sees
both directions. This requires the whole sequence up front, so it is unusable for
generation — it is the RNN analogue of a bidirectional encoder like BERT, and it fails to
be a language model for the same reason.

::: exercise
An RNN is trained on sequences of length 50 and evaluated on length 500. It performs far
worse than expected. Give two mechanisms.
:::

::: solution
**1. The hidden state drifts outside its trained range.** During training the state only
ever evolves for 50 steps from its initialisation. Over 500 steps it can accumulate into a
region of state space the network never saw — saturating $\tanh$, or growing without
bound if $\rho > 1$. The network's behaviour there is undefined in the strict sense: it
was never trained on those inputs. This is extrapolation failure, and it is the same
phenomenon as a transformer failing beyond its trained context length (lesson 4.13).

**2. It never learned long-range dependencies to begin with.** With truncated BPTT at
length 50, gradients never propagated further than 50 steps, so the model cannot have
learned any dependency longer than that — regardless of how long the sequences at
evaluation are. It is not that the ability degrades; it was never acquired.

**How to distinguish them.** Plot the hidden state norm against position at evaluation. A
steady drift or saturation points to mechanism 1. A stable norm with performance that
degrades only on tasks requiring long context points to mechanism 2.

**Fixes.** For mechanism 1: train on variable-length sequences including long ones, and
add normalization to the recurrence (LayerNorm inside the cell) to keep the state in
range. For mechanism 2: nothing at evaluation time helps — you must train with longer
truncation windows, which costs memory linear in the window.
:::

## What to carry forward

- BPTT multiplies $T$ Jacobians, so the spectral radius decides vanish versus explode.
- Exploding gradients clip; vanishing ones cannot — the fix has to be architectural.
- Recurrence cannot be parallelised over positions, which is why transformers won training.
- Recurrence wins at inference: $O(1)$ per token against a growing KV cache.
- Truncated BPTT caps memory and caps the longest dependency you can learn.
