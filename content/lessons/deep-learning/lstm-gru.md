---
summary: Gating as an additive path through time, why the cell state is the RNN's residual connection, and what the gates actually learn.
prereqs: [recurrent-networks, residual-connections]
---

The LSTM's solution to vanishing gradients is the same idea as the residual connection,
arriving eighteen years earlier: give the signal an **additive** path that multiplication
cannot destroy.

## The cell state is the point

An LSTM carries two things forward: a hidden state $\mathbf{h}_t$ and a cell state
$\mathbf{c}_t$. The cell state's update is what matters:

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

Compare to a plain RNN, where the state is passed through a weight matrix and a
nonlinearity at every step. Here the previous cell state is only **multiplied elementwise
by a gate** and added to. There is no $W_{hh}$ in the path.

The gradient consequence is immediate:

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t)
$$

so over $T$ steps the factor is $\prod_t \mathbf{f}_t$ elementwise. If a forget gate stays
near 1, that product stays near 1 **for arbitrarily many steps**. This is exactly the
$I + \partial F$ structure of lesson 3.08, with a learned gate in place of the identity.

## The full cell

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) && \text{forget: what to keep} \\
\mathbf{i}_t &= \sigma(W_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) && \text{input: what to write} \\
\tilde{\mathbf{c}}_t &= \tanh(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) && \text{candidate: what value} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t && \text{update the cell} \\
\mathbf{o}_t &= \sigma(W_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) && \text{output: what to expose} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) && \text{the visible state}
\end{aligned}
$$

Sigmoid for gates — bounded in $(0,1)$, so they behave as soft switches — and $\tanh$ for
values. This is the one place sigmoid remains the right choice (lesson 3.02).

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        # One matrix for all four gates: a single larger matmul beats four small ones.
        self.linear = nn.Linear(d_in + d_hidden, 4 * d_hidden)
        self.d = d_hidden
        # Forget-gate bias at 1: the gate starts open, so gradients flow from step one.
        nn.init.zeros_(self.linear.bias)
        nn.init.ones_(self.linear.bias[:d_hidden])

    def forward(self, x, state):
        h, c = state
        gates = self.linear(torch.cat([h, x], dim=-1))
        f, i, g, o = gates.chunk(4, dim=-1)
        f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, (h, c)
```

::: key
**Initialise the forget-gate bias to 1.** With a zero bias, the gate starts at
$\sigma(0) = 0.5$, so the cell state is halved every step — memory decays with a half-life
of one timestep and the network cannot learn long dependencies before it learns to open
the gate.

Starting at $\sigma(1) \approx 0.73$, or $\sigma(2) \approx 0.88$, means memory persists
from the beginning. This single line is worth several points on long-sequence tasks, and
it is the recurrent analogue of initialising a residual block near zero.
:::

## GRU: the same idea, fewer parts

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(W_z[\mathbf{h}_{t-1},\mathbf{x}_t]) && \text{update gate} \\
\mathbf{r}_t &= \sigma(W_r[\mathbf{h}_{t-1},\mathbf{x}_t]) && \text{reset gate} \\
\tilde{\mathbf{h}}_t &= \tanh(W[\mathbf{r}_t\odot\mathbf{h}_{t-1},\mathbf{x}_t]) \\
\mathbf{h}_t &= (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t
\end{aligned}
$$

Three differences: one state instead of two, three gates instead of four, and the forget
and input gates are **tied** — what you forget is exactly what you write, via
$(1-\mathbf{z})$ and $\mathbf{z}$.

About 25% fewer parameters, comparable accuracy on most tasks. LSTMs retain a small edge
where very long memory is needed, because decoupled forget and input gates let the cell
hold a value while still accepting new information.

::: check
The LSTM cell state updates as $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$. Why does this fix vanishing gradients?

- [x] $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1} = \text{diag}(\mathbf{f}_t)$ — no weight matrix in the path, so a forget gate near 1 carries gradient for arbitrarily many steps
  > It is the $I + \partial F$ structure of residual connections, eighteen years earlier, with a learned gate in place of the identity.
- [ ] The tanh bounds the state, preventing explosion
  > The tanh applies to the candidate and to the output view, not to the path between successive cell states. Bounding would not stop vanishing anyway.
- [ ] The gates are trained to keep the spectral radius at exactly 1
  > Nothing constrains them to 1. What matters is that the path is elementwise and additive, so the gate *can* stay open.
- [ ] Four gates give four independent gradient paths, so at least one survives
  > The gates are not parallel paths. One of them, the forget gate, defines the single path that matters.
:::

## What the gates learn

Trained LSTMs develop interpretable gate behaviour, and this was some of the earliest
mechanistic interpretability work:

- Cells that track **quote and bracket depth**, with the forget gate held open across the
  entire span and closed at the matching delimiter.
- Cells that fire on **line position**, resetting at newlines.
- Cells that act as **counters**, incrementing through a loop.

The reading that matters: the forget gate implements *when to remember*, learned
per-dimension and per-timestep. That is a strictly more expressive memory mechanism than a
fixed decay, and strictly less expressive than attention's ability to retrieve any past
position directly.

::: check
The example initialises the forget-gate bias to 1 rather than 0. Why?

- [x] The gate starts open, so gradient flows from the first step instead of being throttled while the bias learns its way up
  > $\sigma(0) = 0.5$, so a zero bias halves the cell-state path at every step before training has taught it otherwise — $0.5^{100}$ is not a promising start.
- [ ] It makes the forget gate ignore the input initially
  > The gate still reads its input; the bias shifts where it sits, not whether it looks.
- [ ] It prevents the cell state from saturating the tanh
  > Saturation of the output tanh is a separate concern and is not what the bias addresses.
- [ ] It matches the input gate, keeping the two balanced
  > The input gate is conventionally initialised at zero bias. They are not meant to match.
:::

## Why they were replaced anyway

The gradient problem is solved. The parallelism problem is not — the recurrence is still
sequential, so lesson 3.11's argument stands unchanged.

Two further limits:

**Fixed-size state.** Everything the model knows about the past is compressed into
$\mathbf{h}_t$ and $\mathbf{c}_t$. Attention has no such bottleneck: it can retrieve any
past position at full fidelity. For tasks requiring exact recall — copying a name from
2,000 tokens back — a fixed state is a hard ceiling.

**Path length.** Information from position 1 reaching position 1,000 traverses 1,000
recurrent steps, each a chance to be overwritten. In attention it is one hop.

::: warning
LSTMs are not obsolete for every purpose. They remain a reasonable choice for streaming
inference with hard latency budgets, for very long sequences where $O(T^2)$ attention is
infeasible, and for small models on edge hardware where the $O(1)$-per-token inference
cost dominates. Know why you are choosing one, rather than reaching for a transformer
reflexively.
:::

::: exercise
Show that the LSTM cell state's gradient path is $\prod_t \mathbf{f}_t$, and explain why
this solves vanishing gradients without creating an exploding-gradient problem.
:::

::: solution
**The derivation.** Treating the gates as approximately constant with respect to
$\mathbf{c}_{t-1}$ (they depend on $\mathbf{h}_{t-1}$, which is a secondary path):

$$
\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\tilde{\mathbf{c}}_t
\quad\Longrightarrow\quad
\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t)
$$

Chaining over $T$ steps gives $\prod_{t} \text{diag}(\mathbf{f}_t)$, which is elementwise
$\prod_t f_t^{(j)}$ for each dimension $j$.

**Why it solves vanishing.** Each $f_t \in (0,1)$ and can be *learned* to sit near 1. With
$f_t = 0.99$ over 1,000 steps the product is $e^{-10} \approx 4.5\times10^{-5}$ — already
far better than a plain RNN — and at $f_t = 0.999$ it is $0.37$. Critically, the network
controls this: it can hold a gate open for exactly the dimensions and durations the task
requires, rather than being subject to a fixed spectral radius.

**Why it does not explode.** The sigmoid bounds every gate below 1, so the product is
bounded by 1 and monotonically non-increasing. Unlike a plain RNN's $W_{hh}^T$, which can
have $\rho > 1$ and grow without bound, this path can only decay — never amplify.

**The honest caveat.** The cell state path is clean, but the *full* gradient includes the
secondary path through $\mathbf{h}_{t-1}$ into the gates, which does involve weight
matrices and can misbehave. In practice LSTM training still uses gradient clipping. The
gating removes the dominant failure mode, not every one of them.
:::

## What to carry forward

- The cell state's additive update is a residual connection, eighteen years early.
- $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1} = \text{diag}(\mathbf{f}_t)$ — a learned, bounded decay.
- Initialise the forget-gate bias to 1 so memory persists from step one.
- GRU ties forget and input into one gate: fewer parameters, similar quality.
- Gating fixed gradients, not parallelism or the fixed-size state bottleneck.
