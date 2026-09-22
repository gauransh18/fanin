---
summary: Recurrence with the parallelism problem fixed — the selective mechanism that made SSMs competitive, and where they still lose to attention.
prereqs: [recurrent-networks, flash-attention, eigenvalues-eigenvectors]
---

Lesson 3.11 identified two problems with recurrence: vanishing gradients, and no
parallelism during training. Gating solved the first. State space models solve the second,
which makes recurrence competitive again — with one important limitation intact.

## The continuous system

A linear state space model maps a signal through a hidden state:

$$
\mathbf{h}'(t) = A\mathbf{h}(t) + B x(t), \qquad y(t) = C\mathbf{h}(t)
$$

Discretising with step size $\Delta$ gives a recurrence:

$$
\mathbf{h}_t = \bar{A}\mathbf{h}_{t-1} + \bar{B}x_t, \qquad y_t = C\mathbf{h}_t
$$

with $\bar{A} = \exp(\Delta A)$ and $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$.

This is a **linear** RNN: no nonlinearity inside the recurrence. That restriction is what
buys everything.

## Linearity means parallelism

Unrolling a linear recurrence gives a closed form:

$$
y_t = \sum_{k=0}^{t} C\bar{A}^{k}\bar{B}\, x_{t-k}
$$

which is a **convolution** with kernel $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$.
So the whole sequence can be computed at once with an FFT in $O(T\log T)$ — fully parallel.

::: key
This is the trick. A nonlinear recurrence must be evaluated step by step because
$\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$ through a nonlinearity you cannot unroll. A
**linear** one has a closed form, so training parallelises over the sequence — exactly the
property that made transformers win (lesson 3.11).

And at inference you switch back to the recurrent form: $O(1)$ work per token and a
fixed-size state, rather than attention's growing KV cache. Parallel training, recurrent
inference.
:::

## Making $\bar{A}$ well-behaved

From lesson 1.05, repeated multiplication by $\bar{A}$ is governed by its spectral radius:
above 1 it explodes, below 1 it decays. A random $A$ gives a model that forgets almost
immediately.

**HiPPO** provides a specific $A$ derived from the problem of optimally compressing a
signal's history onto orthogonal polynomials. It gives $\bar{A}$ eigenvalues that produce a
principled memory decay across many timescales rather than one. S4's performance came
largely from this initialisation — the architecture is simple, and the matrix is where the
work is.

::: check
A state space model keeps its recurrence **linear**. What does that restriction buy?

- [x] A closed form that is a convolution
  > Training parallelises over the sequence in $O(T\log T)$ with an FFT — while inference can switch back to the recurrent form with $O(1)$ work per token. A nonlinear recurrence has to be evaluated step by step. Parallel training and recurrent inference is exactly the combination transformers do not have.
- [ ] Guaranteed stability, since linear systems cannot diverge
  > They diverge readily when the spectral radius exceeds 1, which is why $\bar A$ needs careful parameterisation.
- [ ] Lower parameter count than a gated RNN
  > Parameter counts are comparable. Parallelism is the prize.
- [ ] The ability to attend to arbitrary positions
  > It is the opposite: a fixed-size state is what an SSM carries, and selecting arbitrary past positions is what attention does and SSMs approximate.
:::

## The limitation: no selectivity

$\bar{A}$, $\bar{B}$ and $C$ are **fixed** — the same for every token. So the model
compresses history the same way regardless of content, and it cannot decide that a
particular token matters and should be retained.

This shows up concretely on **selective copying**: given a sequence with a few marked
tokens scattered among noise, reproduce the marked ones. Attention does this trivially — it
retrieves by content. A fixed-dynamics SSM cannot, because the marking is content and the
dynamics ignore content.

## Mamba: make the parameters input-dependent

Mamba makes $\Delta$, $B$ and $C$ functions of the input:

$$
\Delta_t = \text{softplus}(W_\Delta \mathbf{x}_t), \qquad
B_t = W_B\mathbf{x}_t, \qquad
C_t = W_C\mathbf{x}_t
$$

Now the recurrence is $\mathbf{h}_t = \bar{A}_t\mathbf{h}_{t-1} + \bar{B}_t x_t$, and the
model can choose per token how much to remember and what to write.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveScan(nn.Module):
    """Readable reference. The real implementation is a fused kernel that
    keeps the state in SRAM -- the same hardware argument as lesson 4.12."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        # A is stored in log space and negated to keep eigenvalues in (0,1),
        # so the recurrence can only decay -- lesson 1.05's stability condition.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float())
                                  .repeat(d_model, 1))
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):                                   # (B, T, d)
        B_, T, d = x.shape
        A = -torch.exp(self.A_log)                          # (d, d_state), negative
        dbc = self.x_proj(x)
        delta, Bm, Cm = dbc[..., :1], dbc[..., 1:1 + self.d_state], dbc[..., 1 + self.d_state:]
        delta = F.softplus(delta)                           # (B, T, 1), positive

        A_bar = torch.exp(delta.unsqueeze(-1) * A)          # (B, T, d, d_state)
        B_bar = delta.unsqueeze(-1) * Bm.unsqueeze(2)       # (B, T, 1, d_state)

        h = x.new_zeros(B_, d, self.d_state)
        ys = []
        for t in range(T):                                  # a parallel scan in reality
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]
            ys.append((h * Cm[:, t, None]).sum(-1))
        return torch.stack(ys, 1) + x * self.D
```

**Input-dependent parameters break the convolution form** — the kernel now differs at every
position, so the FFT trick is gone. Mamba recovers parallelism with a **parallel scan**
(the associative-scan algorithm), which computes the recurrence in $O(\log T)$ depth, and a
fused kernel that keeps the state in SRAM rather than writing $\mathbf{h}_t$ to HBM for
every $t$. That second part is precisely FlashAttention's argument (lesson 4.12) applied to
a different operation.

::: check
What does a fixed-size recurrent state give up relative to attention's KV cache?

- [x] Exact recall of arbitrary earlier tokens
  > The state is a lossy summary, where a KV cache keeps every key and value. This is why hybrids exist: a few attention layers among many SSM layers recover the exact-recall behaviour while keeping most of the cost advantage.
- [ ] Nothing; a sufficiently large state is equivalent to a cache
  > A state of fixed size cannot hold a sequence of unbounded length without loss, whatever the constant.
- [ ] The ability to process variable-length sequences
  > Recurrence handles variable length particularly naturally.
- [ ] Parallel training, which only attention has
  > Linear SSMs parallelise training via the convolutional form. That is the whole point of keeping the recurrence linear.
:::

## The comparison

| | Transformer | Mamba |
|---|---|---|
| Training cost | $O(T^2 d)$ | $O(Td)$ |
| Inference per token | $O(T)$, growing cache | $O(1)$, fixed state |
| Memory at inference | $O(T)$ KV cache | $O(d \cdot d_{\text{state}})$ |
| Exact retrieval from context | Yes | No — state is lossy |
| In-context learning | Strong | Weaker |

::: warning
The fixed-size state is a **hard** information bottleneck, exactly as in lesson 3.12. A
transformer can retrieve any past token exactly, because the KV cache stores them all.
Mamba compresses history into $d \cdot d_{\text{state}}$ numbers, so a long context is
necessarily lossy.

This shows in practice: SSMs match transformers on language modelling perplexity while
lagging on tasks requiring precise recall from long context — which is a meaningful
fraction of what people actually use long context for.
:::

## Hybrids

The practical answer has been to mix them. Jamba, Zamba and Samba interleave Mamba layers
with a few attention layers — typically one attention layer per six or eight SSM layers.

The SSM layers do the cheap sequence modelling; the attention layers provide exact
retrieval where it is needed. Most of the memory saving, most of the retrieval ability.
This is the same compromise as the sliding-window-plus-full-attention designs in lesson
4.13, reached from the other direction.

::: exercise
Mamba is $O(T)$ in training and $O(1)$ per token at inference, against a transformer's
$O(T^2)$ and $O(T)$. Why has it not replaced transformers?
:::

::: solution
**1. The asymptotics do not bind where people operate.** From lesson 4.05, attention
overtakes the dense projections only past $T \approx 6d$ — about 24k tokens at $d=4096$.
Below that, a transformer's cost is dominated by matrix products that are $O(T)$ anyway, so
Mamba's advantage is small. Most training and most inference happens below the crossover.

**2. Hardware utilisation.** Matrix multiplication is what accelerators are built for.
Attention is almost entirely matmul and runs near peak FLOP/s. A selective scan is
elementwise and sequential in structure, so even with a good fused kernel it achieves a
lower fraction of peak. Mamba's FLOP advantage is partly given back in utilisation —
lesson 1.03's point, that FLOPs predict runtime only when intensity is high.

**3. The lossy state is a real capability gap.** Exact retrieval from a long context is not
an edge case; it is what long context is largely used for — RAG, code, document analysis.
An architecture that cannot do it reliably is not a drop-in replacement whatever its
perplexity.

**4. Ecosystem inertia, which is not a trivial reason.** Every serving stack, quantization
method, LoRA implementation, inference kernel and interpretability tool assumes attention.
A new architecture must be substantially better to overcome that, not merely competitive.

**Where it does win, today:** very long sequences where exact retrieval is not required —
audio, genomics, time series — and edge deployment where the fixed-size state matters more
than recall.

**The likely outcome is the hybrid.** Jamba-style interleaving gets most of the efficiency
with most of the retrieval, and does not require the ecosystem to abandon attention. Pure
architectural replacement is a much higher bar than "better on some axis".
:::

## What to carry forward

- A linear recurrence has a closed form, so it trains in parallel and runs recurrently.
- $\bar{A}$'s eigenvalues set the memory horizon; HiPPO gives a principled choice.
- Fixed dynamics cannot select by content — Mamba makes the parameters input-dependent.
- That breaks the convolution form, recovered by a parallel scan plus a SRAM-resident kernel.
- The fixed state is a hard bottleneck on exact retrieval; hybrids are the practical answer.
