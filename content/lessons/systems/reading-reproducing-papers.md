---
summary: A procedure for extracting what matters from a paper quickly, deciding whether to trust it, and reproducing it when you must.
prereqs: [evaluating-llms, experiment-tracking, testing-ml-code]
---

The field publishes faster than anyone can read. The skill is not reading more papers; it is
deciding quickly which ones deserve a careful read, and knowing what to distrust in the rest.

## Three passes

**First pass — five minutes.** Title, abstract, figures, conclusion. Answer three questions:

- What problem?
- What is the claimed contribution?
- Is the improvement large enough to matter to me?

Most papers stop here. That is the correct outcome, not a failure.

**Second pass — thirty minutes.** Method section and experiments. Answer:

- What is the actual mechanism, in one sentence?
- What is the baseline, and is it a fair one?
- How large is the effect relative to the error bars?

**Third pass — hours.** Re-derive the equations, check the shapes, and reconstruct the
implementation mentally. Reserve this for papers you intend to build on.

## Reading the method

::: key
Find the **one sentence** that is the contribution. Almost every paper has one, and it is
usually buried in the middle of a section.

- FlashAttention: "tile the computation so the score matrix never reaches HBM."
- LoRA: "the fine-tuning update is low-rank, so parameterise it that way."
- PPO: "clip the probability ratio so a single batch cannot move the policy too far."
- GQA: "share key and value heads across query heads to shrink the KV cache."

If you cannot state it in one sentence after the second pass, either the paper is unclear or
the contribution is marginal. Both are useful signals.
:::

Then check the shapes. Papers contain shape errors and ambiguous notation constantly, and
working through the dimensions of the central equation catches both — and forces you to
understand it.

## What to distrust

Every item here appears in published work regularly. None requires assuming bad faith; all of
them inflate results.

**A weak baseline.** The most common problem by far. Check whether the baseline was tuned as
carefully as the proposed method. A paper that sweeps ten hyperparameters for its method and
uses the baseline's published defaults has not made a fair comparison.

**Missing error bars.** From lesson 5.16: a 1.7-point difference on 400 examples is noise. If
the paper reports one seed and no interval, the result is an anecdote. Look for the seed count
— it is often only in the appendix, and often one.

**Unequal compute.** A method that trains 2× longer and scores higher has not demonstrated
anything about the method. Match FLOPs, not epochs.

**Contamination.** Benchmarks leak into training corpora (lesson 5.02). A paper reporting
strong MMLU numbers without describing decontamination has probably not done it.

**Cherry-picked qualitative examples.** Generated samples in a paper are selected. Assume the
median output is substantially worse.

**Ablations that do not ablate.** An ablation removing a component *and* retuning the
remainder is measuring two things at once.

::: warning
**The strongest signal of reliability is not in the results section.** It is whether the
authors released code that runs, reported negative results, and described what did not work.

A paper with a "limitations" section naming real limitations is more trustworthy than one
claiming uniform improvement, almost regardless of the numbers.
:::

## Deciding to reproduce

Reproduction is expensive. Reasons that justify it:

- You intend to build on the method and need to understand it precisely.
- The result contradicts something you believe, and the contradiction matters.
- You need the method in production and must know its failure modes.

Reasons that do not: the paper is famous, or the number is impressive.

## The reproduction procedure

**1. Start from the released code if there is any.** Run it unchanged on the authors' setup
first. If it does not reproduce their own numbers, stop and contact the authors — you have
learned something already.

**2. Reproduce at small scale first.** A 10M-parameter version of the experiment costs
minutes and catches most implementation errors. If the effect does not appear at small scale,
either it is scale-dependent (which the paper should have said) or your implementation is
wrong.

**3. Implement the baseline before the method.** You cannot detect an improvement without a
trustworthy reference point, and building the baseline forces you to understand the setup.

**4. Match everything you can, and record what you cannot.**

```python
reproduction_log = {
    'matched': ['architecture', 'optimizer', 'lr_schedule', 'batch_size', 'data'],
    'differed': {
        'hardware': 'A100 vs paper H100',
        'seed': 'paper does not report',
        'tokenizer': 'paper does not specify version',
    },
    'guessed': {
        'warmup_steps': 'not stated; used 2000 (1% of training)',
        'weight_decay': 'not stated; used 0.1 (standard for AdamW)',
    },
}
```

Papers routinely omit warmup length, weight decay, gradient clipping and initialisation
details. Guess reasonably, **record the guesses**, and check their sensitivity if the
reproduction fails.

**5. Run multiple seeds.** From lesson 2.15, a single run tells you nothing about whether a
difference is real.

## When it does not reproduce

::: key
Work through the causes in order of prior probability, not in order of how interesting they
are:

1. **Your implementation has a bug.** By a wide margin the most likely explanation. Run
   lesson 7.12's test suite — especially the overfit-one-batch check.
2. **A hyperparameter differs.** Sweep the ones you guessed. Learning rate and warmup are the
   usual culprits.
3. **The data differs.** Same-named datasets have different versions, different preprocessing
   and different splits.
4. **Seed variance.** Run more seeds before concluding anything.
5. **The paper is wrong or selectively reported.** Last, not first — and it does happen.

Assuming (5) before exhausting (1) through (4) is the most common way reproduction attempts
go wrong.
:::

Contact the authors. Most respond, most are helpful, and the missing detail is usually
something they did not think to write down.

## Keeping up without drowning

- **Follow a few people whose filtering you trust** rather than reading feeds. Curation is
  the scarce resource.
- **Read the ablations before the headline result.** They tell you which component matters,
  which is what you need to know.
- **Prefer papers with released, runnable code.** A strong correlation with reproducibility,
  and it makes the third pass much cheaper.
- **Keep notes with the one-sentence contribution.** In six months that sentence is all you
  will retain anyway, and a searchable list of them is genuinely useful.
- **Revisit older papers.** The field rediscovers ideas constantly. Reading the 1990s
  literature on gating, normalization and momentum explains a surprising amount of what is
  presented as new.

## A worked example

Take FlashAttention (lesson 4.12) through the passes:

**First pass.** Attention is slow and memory-hungry; they claim exact attention with $O(T)$
memory and 2–4× speedup. The memory claim is the striking one — exactness plus asymptotic
improvement is unusual.

**Second pass.** The mechanism: tile the computation so score tiles stay in SRAM, and use an
online softmax so normalisation can be done incrementally. The baseline is standard PyTorch
attention, which is fair. The speedup grows with sequence length, which is consistent with a
memory-traffic explanation rather than an arithmetic one.

**Third pass.** Derive the online softmax rescaling and confirm it is algebraically exact.
Check that the backward pass recomputes rather than stores, and count the memory: $O(T)$ for
the row statistics instead of $O(T^2)$ for the matrix. Confirm the FLOP count *rises* and the
runtime falls — which only makes sense if the operation was memory-bound, and lesson 7.02
says it was.

The paper survives all three passes: the mechanism is clearly stated, the baseline is fair,
the claim is exact rather than approximate, and the explanation is consistent with the
hardware model. That combination is what a trustworthy paper looks like.

::: exercise
A paper claims a new optimizer beats AdamW by 15% on training loss. What would convince you,
and what would you check first?
:::

::: solution
**15% on training loss is a very large claim.** For calibration: lesson 5.03's scaling laws
say a 10× compute increase buys roughly 15% loss reduction. An optimizer delivering the same
for free would be among the most significant results in the field. Extraordinary claims
warrant checking the ordinary explanations first.

**Check first, in this order:**

1. **Was AdamW tuned?** The single most likely explanation. If the paper swept its method's
   hyperparameters and used AdamW at default $\beta_2 = 0.999$ and an untuned learning rate,
   the comparison is not meaningful. Look for the baseline's sweep — if it is not described,
   assume it did not happen.

2. **Training loss or validation loss?** Training loss is not the objective. An optimizer
   that reaches lower training loss while generalising worse is a worse optimizer. If only
   training loss is reported, that omission is itself informative.

3. **Matched compute?** Many proposed optimizers cost more per step — extra memory,
   preconditioner updates, second-order information. A fair comparison is loss against
   wall-clock or against FLOPs, not against step count. Check which axis the figures use.

4. **What scale?** Optimizer results on small models frequently do not transfer. The relevant
   evidence is a scaling study showing the gap persists or widens with model size. A result
   on a 100M model says little about 7B.

5. **Seeds and error bars.** How many runs? What was the spread? At one seed per
   configuration this is not a measurement (lesson 2.15).

**What would convince me:**

- The gap **holds or grows** across at least three model scales, ideally spanning an order of
  magnitude.
- Both methods tuned with **equal search budget**, and the search described.
- **Validation** loss and downstream benchmarks, not just training loss.
- Loss plotted against **wall-clock time**, so the per-step cost is visible.
- **Multiple seeds** with reported spread.
- Working code that reproduces one headline figure.
- An **independent** reproduction by someone with no stake in the result.

**The base rate is worth stating plainly.** Many optimizers have claimed to beat Adam. Almost
none have displaced it, and the usual reason is that the gap vanishes with a properly tuned
baseline or at larger scale. That prior should be explicit in how you read the paper — not as
cynicism, but as the correct starting point given the evidence to date.

**The cheapest decisive test:** reproduce at small scale with both methods tuned by the same
automated search. A day of compute, and it settles the question for your setting.
:::

## What to carry forward

- Three passes: five minutes, thirty minutes, hours — and most papers stop at the first.
- Find the one-sentence contribution; if you cannot, that is a signal.
- Distrust weak baselines, missing error bars, unequal compute and absent decontamination.
- When reproduction fails, suspect your implementation before the paper.
- A real limitations section is stronger evidence of reliability than any number.
