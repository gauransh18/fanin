---
summary: Decoupling parameter count from compute per token, the load-balancing problem that makes it hard, and what it costs in memory.
prereqs: [transformer-block, indexing-gather-scatter]
---

Lesson 4.05 established that two-thirds of a transformer's parameters live in its MLPs and
that every token passes through all of them. Mixture of experts changes the second part: it
replaces one MLP with $N$, and routes each token to only $k$ of them.

## The construction

$$
\mathbf{y} = \sum_{i \in \text{top-}k} g_i(\mathbf{x})\, E_i(\mathbf{x})
$$

where the router $g$ scores experts and only the top $k$ contribute.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d, d_hidden, n_experts=8, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_hidden, bias=False), nn.GELU(),
                          nn.Linear(d_hidden, d, bias=False))
            for _ in range(n_experts)])

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)                                # (N, d)

        logits = self.router(flat)                             # (N, n_experts)
        weights, idx = torch.topk(logits, self.k, dim=-1)      # (N, k)
        weights = F.softmax(weights, dim=-1)                   # renormalise over k

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit_tok, hit_slot = (idx == e).nonzero(as_tuple=True)
            if hit_tok.numel() == 0:
                continue
            # Each expert sees only its own tokens -- one matmul, not N.
            out.index_add_(0, hit_tok,
                           expert(flat[hit_tok]) * weights[hit_tok, hit_slot, None])
        return out.reshape(B, T, d)
```

Note `index_add_` rather than indexed assignment: a token routed to two experts must
**accumulate** both contributions, and lesson 2.04 explains why `out[idx] +=` would silently
keep only one.

::: key
**Softmax over the top $k$, not over all experts.** Taking the softmax first and then
selecting the top $k$ leaves weights that do not sum to 1, so the layer's output magnitude
varies with how confident the router happened to be. Select first, then normalise over the
selected.
:::

## What it buys

| | Dense 7B | MoE 8×7B, top-2 |
|---|---|---|
| Total parameters | 7B | 47B |
| Active per token | 7B | 13B |
| Training FLOPs per token | $6 \times 7$B | $6 \times 13$B |
| Memory to serve | 14 GB | 94 GB |
| Typical quality | 7B | ≈ 40B dense |

Parameter count and compute per token are **decoupled**. Knowledge scales with total
parameters; cost scales with active parameters. For a fixed FLOP budget, an MoE gets
substantially better quality.

The catch is in the memory row. Every expert must be resident even though most are unused
for any given token, so serving an MoE costs the memory of the full parameter count. MoE
trades memory for compute, which is a good trade during training and often a bad one for
single-stream inference.

::: check
An MoE layer with 8 experts and top-$k$ of 2 replaces one MLP. What does it change about parameters and compute per token?

- [x] Parameters grow roughly eightfold while per-token compute stays near the top-2 cost
  > Capacity is decoupled from compute. Every token passes through all of a dense MLP. Routing means each token pays for two experts while the model as a whole holds eight, which is the entire reason MoE exists.
- [ ] Both grow eightfold, which is why MoE models are so expensive to serve
  > Serving is expensive because the *weights* must all be resident, not because each token does eightfold arithmetic.
- [ ] Parameters stay the same and compute falls by a factor of four
  > The experts are additional parameters; nothing is removed.
- [ ] Parameters grow eightfold and compute falls, since only two experts run
  > Compute per token does not fall below the original dense MLP — top-2 of eight experts of the same width costs about twice one MLP, not a quarter.
:::

## Load balancing

Nothing in the objective prevents the router from sending every token to one expert. This
**collapse** is a real failure mode and it is self-reinforcing: a popular expert gets more
gradient, improves faster, and becomes more popular.

The standard fix is an auxiliary loss penalising imbalance:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean router
probability for it. The product is minimised when both are uniform.

```python
def load_balance_loss(router_logits, idx, n_experts, alpha=0.01):
    probs = F.softmax(router_logits, dim=-1)                   # (N, n_experts)
    P = probs.mean(0)                                          # mean probability
    f = torch.zeros(n_experts, device=probs.device)
    f.scatter_add_(0, idx.reshape(-1),
                   torch.ones_like(idx.reshape(-1), dtype=probs.dtype))
    f = f / idx.numel()                                        # fraction routed
    return alpha * n_experts * (f * P).sum()
```

$f$ is not differentiable (it comes from `topk`), but $P$ is — so the gradient flows
through the probabilities and pushes the router toward uniformity.

::: warning
**Expert capacity** is the other half of the problem. Batched execution needs a fixed
buffer per expert, so implementations set a capacity $C = \text{capacity\_factor} \times
\frac{\text{tokens}}{N}$ and **drop** tokens beyond it. A dropped token skips the MoE layer
entirely, passing through on the residual path alone.

At capacity factor 1.0 with imperfect balance, drop rates of 10–20% are common. Typical
production settings are 1.25 during training and higher at inference. Always log the drop
rate — a run with 30% of tokens bypassing every MoE layer trains, and trains badly.
:::

::: check
Why does an MoE layer need an explicit load-balancing loss?

- [x] Routing is self-reinforcing
  > An expert that gets more tokens trains faster and becomes more attractive, so without pressure the router collapses onto a few experts. The unused experts are then dead parameters, and the model has the capacity of a much smaller one. An auxiliary loss penalising imbalance is standard.
- [ ] Without it the top-$k$ selection is not differentiable
  > Top-$k$ is not differentiable in its selection either way; the gradient flows through the softmax weights on the chosen experts.
- [ ] Because experts must see equal token counts for the matmuls to be the same shape
  > Capacity factors and padding handle the shape question. Balance is wanted for learning reasons first.
- [ ] To prevent the router from overfitting to the training distribution
  > Router overfitting is a real concern and a different one; collapse happens even on the training distribution.
:::

## Routing strategies

- **Top-$k$ token choice** (above) — each token picks $k$ experts. Simple; needs capacity
  limits and balancing losses.
- **Expert choice** — invert it: each expert picks its top $C$ tokens. Perfect balance by
  construction, and no dropped tokens. The cost is that it needs the whole batch at once,
  so it does not work for autoregressive decoding.
- **Shared experts** (DeepSeek) — a few experts every token uses, plus routed ones. The
  shared experts capture general computation so the routed ones can specialise, and it
  reduces the redundancy that otherwise appears across experts.
- **Fine-grained experts** — many small experts rather than few large ones. More
  combinations for the same parameter count.

## Distributed training

Experts are typically sharded across devices — **expert parallelism** (lesson 7.06). Each
device holds a few experts, and tokens are sent to wherever their expert lives via an
all-to-all collective.

That all-to-all is the dominant cost of MoE training, and it is why MoEs are more sensitive
to interconnect bandwidth than dense models of the same active size. A cluster with slow
node-to-node links can spend more time in communication than in compute.

## What experts learn

Less than the name suggests. Studies of trained MoEs find specialisation by **token-level
surface features** — punctuation, numbers, code syntax, specific languages — rather than by
topic or domain. "An expert for biology" is not what emerges; "an expert for tokens
following a newline" is closer.

Expert assignment is also unstable across layers: the same token routes to unrelated
experts at different depths. Treat the experts as a learned sparse factorisation of one big
MLP rather than as a committee of specialists.

::: exercise
You train an 8-expert top-2 MoE. The load-balancing loss stays near zero but validation
perplexity is worse than a dense model with the same active parameters. What is happening?
:::

::: solution
A near-zero balancing loss means the router is perfectly uniform — which sounds good and is
the symptom of the problem. **A perfectly uniform router is one that has stopped
discriminating.**

The balancing loss pushes toward uniform routing. If $\alpha$ is too large, it dominates
the task loss and the optimal router becomes one that ignores the input entirely, assigning
tokens round-robin. Every expert then sees an identical distribution of tokens, so every
expert learns the same function, and you have paid for 8 copies of one MLP while using two
of them per token — strictly worse than a dense model of the same active size, since the
effective capacity is that of a single expert.

**How to confirm it, in order:**

1. **Check the router entropy per token.** Uniform routing means entropy near $\ln 8 =
   2.08$ nats for every token. A healthy router has high entropy *averaged over the
   dataset* but low entropy *per token* — confident about individual tokens, balanced
   overall. Measure both; they are different quantities and conflating them is the
   underlying mistake.

2. **Check expert similarity.** Compute pairwise cosine similarity between experts' weight
   matrices. If they have converged to near-identical functions, this is confirmed.

3. **Ablate.** Run inference with a random router instead of the learned one. If quality is
   unchanged, the router is not doing anything.

**The fix:** lower $\alpha$. Typical values are 0.01 or lower — it is a tiebreaker, not a
primary objective. Then re-check: you want per-token entropy well below $\ln N$ with
dataset-level balance near uniform.

**Two secondary causes worth ruling out** if $\alpha$ turns out to be fine. Router
z-loss — a penalty on router logit magnitude — may be over-weighted and flattening the
logits directly. And if the drop rate is high, tokens are bypassing the layer regardless of
routing quality, which produces the same symptom for a different reason. Log the drop rate
alongside the balancing loss.
:::

## What to carry forward

- MoE decouples total parameters from compute per token; memory follows total, not active.
- Softmax over the selected top-$k$, and accumulate with `index_add_` for multi-routing.
- Balancing loss prevents collapse; too much of it causes uniform routing, which is collapse's mirror image.
- Log the token drop rate — silent dropping is a common and invisible failure.
- Experts specialise by surface features, not by topic.
