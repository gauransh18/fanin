---
summary: The power laws that let you predict a model's loss before training it, the Chinchilla correction, and what the laws do not tell you.
prereqs: [matrix-multiplication-cost, pretraining-data, expectation-variance-concentration]
---

Scaling laws are the closest thing this field has to an engineering formula. They let you
answer "if I spend $10^{23}$ FLOPs, what loss will I get, and how should I split that
budget between model size and data?" before committing the money.

## The empirical finding

Loss falls as a power law in each of three quantities, with the other two held
non-limiting:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

for parameters $N$, data $D$ and compute $C$. Straight lines on a log-log plot, holding
over more than seven orders of magnitude of compute.

Two facts make this remarkable. The exponents are **small** — around 0.05 to 0.1 — so
progress is slow and expensive: a 10× compute increase buys perhaps a 15% loss reduction.
And the relationship is **smooth and predictable**, which is what makes it possible to
plan a frontier run rather than gamble on one.

## Chinchilla

The 2020 Kaplan analysis concluded that model size should grow much faster than data. The
2022 Chinchilla work found this was an artefact of holding the learning-rate schedule fixed
across model sizes — a methodological bug — and redid the analysis properly:

$$
L(N, D) = E + \frac{A}{N^{0.34}} + \frac{B}{D^{0.28}}
$$

Minimising this subject to $C = 6ND$ (the FLOP rule from lesson 1.03) gives

$$
N_{\text{opt}} \propto C^{0.5}, \qquad D_{\text{opt}} \propto C^{0.5}
$$

::: key
**Scale parameters and data equally.** The practical rule that came out of it: roughly
**20 tokens per parameter**.

Chinchilla at 70B parameters and 1.4T tokens outperformed Gopher at 280B parameters and
300B tokens, using the *same* compute. A model 4× smaller beat one 4× larger because the
larger one was badly undertrained.

Every model before 2022 was substantially undertrained, and correcting that was worth more
than several years of architectural work.
:::

::: check
Chinchilla found that the 2020 Kaplan analysis had over-weighted model size. What was the methodological error?

- [x] The learning-rate schedule was held fixed across model sizes, which handicapped the larger-data runs
  > Redoing it properly gave $N_{\text{opt}} \propto C^{0.5}$ and $D_{\text{opt}} \propto C^{0.5}$ — scale both equally, roughly 20 tokens per parameter.
- [ ] The compute budget was measured in GPU-hours rather than FLOPs
  > Both analyses worked in FLOPs via the $6ND$ rule.
- [ ] The models were evaluated on a contaminated test set
  > Contamination is a real hazard elsewhere; it is not what the Chinchilla correction was about.
- [ ] Loss was measured in bits rather than nats
  > A unit change rescales the axis and cannot change which allocation is optimal.
:::

## Inference-optimal is a different question

Chinchilla minimises training loss for a training budget. It ignores inference entirely.

If you will serve a model billions of times, a smaller model trained on *more* data than
Chinchilla-optimal is better overall: slightly worse loss, far cheaper to serve.

Llama-2 7B trained on 2T tokens is about 285 tokens per parameter — 14× past
Chinchilla-optimal. That was a deliberate and correct choice for a model meant to be
deployed widely.

```python
def optimal_allocation(flops, tokens_per_param=20):
    """C = 6ND with D = ratio * N."""
    n = (flops / (6 * tokens_per_param)) ** 0.5
    return n, n * tokens_per_param

for c in (1e21, 1e23, 1e25):
    n, d = optimal_allocation(c)
    print(f'{c:.0e} FLOPs -> {n/1e9:6.1f}B params, {d/1e12:6.2f}T tokens')

# Same budget, deployment-oriented:
for c in (1e23,):
    for ratio in (20, 100, 300):
        n, d = optimal_allocation(c, ratio)
        print(f'ratio {ratio:3d}: {n/1e9:5.1f}B params, {d/1e12:5.2f}T tokens')
```

## What the laws assume

::: warning
Every scaling law is fitted under conditions that are easy to violate:

- **Fresh data.** Repeated data gives less than the law predicts (lesson 5.02). If your
  corpus is smaller than $D_{\text{opt}}$, the curve bends.
- **Optimal hyperparameters at every scale.** Learning rate and batch size must be tuned
  per size, which is what Kaplan got wrong. μP (maximal update parameterisation) provides a
  principled way to transfer hyperparameters across widths and is now common.
- **A fixed architecture and data distribution.** Changing either refits the constants. A
  law fitted on one corpus does not transfer to another.
- **Loss as the target.** The laws predict loss, not capability.
:::

That last point is the most important limitation.

## Emergence

Some capabilities appear abruptly: near-chance performance up to a scale, then sharp
improvement. Multi-digit arithmetic and multi-step reasoning are the usual examples.

This looks like a discontinuity that scaling laws cannot predict. The best current
explanation is that it is largely a **metric artefact**. Exact-match accuracy on a 5-step
task is a thresholded function of per-step accuracy: if per-step accuracy improves smoothly
from 0.3 to 0.9, then $p^5$ goes from 0.002 to 0.59 — a sharp-looking jump produced by a
smooth underlying improvement.

Measuring with a continuous metric — per-token log-probability of the correct answer, or
partial credit — usually reveals the smooth curve underneath.

Not all emergence dissolves this way. Induction-head formation (lesson 4.03) is a genuine
phase transition in the model's internal structure, visible as a bump in the loss curve
itself. The honest position: most reported emergence is metric choice; some is real; you
cannot tell which without measuring continuously.

::: check
Scaling-law exponents are around 0.05 to 0.1. What does that mean in practice?

- [x] A 10× compute increase buys perhaps a 15% loss reduction
  > Progress is slow, expensive, and above all *predictable*. Predictability is what makes a frontier run plannable rather than a gamble. The smallness of the exponents is why the bills are what they are.
- [ ] A 10× compute increase roughly halves the loss
  > That would be an exponent near 0.3. The measured values are far smaller.
- [ ] The exponents are too small to be useful for planning
  > They are small and remarkably stable over seven orders of magnitude, which is precisely what makes them useful.
- [ ] Loss falls linearly with compute
  > It falls as a power law — a straight line on a log-log plot, not a linear one.
:::

## Using them

```python
def predicted_loss(n_params, n_tokens, E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28):
    """Chinchilla's fitted constants. Refit for your own setup."""
    return E + A / n_params ** alpha + B / n_tokens ** beta

# Sanity: Chinchilla 70B on 1.4T
print(f'{predicted_loss(70e9, 1.4e12):.3f}')

# Does doubling parameters or doubling data help more here?
base = predicted_loss(7e9, 2e12)
print(f'2x params: {base - predicted_loss(14e9, 2e12):+.4f}')
print(f'2x tokens: {base - predicted_loss(7e9, 4e12):+.4f}')
```

The practical workflow: train a ladder of small models (10M to 1B), fit the constants on
**your** data and architecture, extrapolate, and commit. Trust the fitted curve about one
order of magnitude beyond your largest measured point, and no further.

::: exercise
You have $10^{23}$ FLOPs. Compare a Chinchilla-optimal model against one at 200 tokens per
parameter, assuming you will serve 10 billion tokens of inference.
:::

::: solution
**Chinchilla-optimal (20 tokens/param).**
$N = \sqrt{10^{23}/120} \approx 28.9$B parameters, $D \approx 578$B tokens.
Predicted loss ≈ 1.92 by the formula above.

**Inference-oriented (200 tokens/param).**
$N = \sqrt{10^{23}/1200} \approx 9.1$B parameters, $D \approx 1.83$T tokens.
Predicted loss ≈ 1.97 — about 2.6% worse.

**Inference cost.** At $2N$ FLOPs per generated token (lesson 1.03):

- 28.9B: $2 \times 28.9\times10^9 \times 10^{10} = 5.8\times10^{20}$ FLOPs
- 9.1B: $2 \times 9.1\times10^9 \times 10^{10} = 1.8\times10^{20}$ FLOPs

The smaller model saves $4\times10^{20}$ FLOPs — **0.4% of the training budget**. On FLOPs
alone this looks marginal.

**But FLOPs are the wrong unit for serving.** Three effects dominate:

1. **Memory.** 28.9B in bf16 is 58 GB, so it needs two 80 GB GPUs once you account for the
   KV cache (lesson 4.10). 9.1B is 18 GB and fits comfortably on one. That is a 2×
   difference in hardware cost per replica, and it changes what you can deploy at all.
2. **Batch size.** With more free memory, the smaller model serves a much larger batch.
   Since decode is memory-bandwidth-bound, throughput scales close to linearly with batch —
   often 3–4× more tokens per second per GPU.
3. **Latency.** Time per token is roughly proportional to model size in the bandwidth-bound
   regime, so the smaller model is about 3× faster per token.

**The verdict.** At only 10B inference tokens, the compute argument is a wash and you
should take the Chinchilla model for its better loss. The inference-oriented choice becomes
correct around $10^{12}$ inference tokens and above, or immediately if the deployment
constraint is "must run on one GPU" — which for most production systems it is.

**The general rule:** Chinchilla-optimal if the model is a research artefact or will be
distilled; heavily overtrained if it will be served at scale. Llama-2 7B's 285 tokens per
parameter was the latter judgement, and it was right.
:::

## What to carry forward

- Loss falls as a power law with small exponents: 10× compute buys roughly 15% loss.
- Chinchilla: scale $N$ and $D$ together, about 20 tokens per parameter.
- Serving at scale justifies going far past that — Llama-2 7B is at 285.
- The laws assume fresh data and per-scale tuning; violate either and the curve bends.
- Most reported emergence is a thresholded metric; measure continuously to see through it.
