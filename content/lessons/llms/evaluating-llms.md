---
summary: Why benchmark numbers are mostly noise, the contamination and selection problems behind them, and what to measure instead.
prereqs: [expectation-variance-concentration, pretraining-data, reward-modeling]
---

Most reported LLM evaluations are not measuring what they claim. This lesson is about the
specific ways that happens and what to do instead.

## Error bars, first

From lesson 1.11, accuracy on $n$ examples has standard error $\sqrt{p(1-p)/n}$, and a 95%
interval is roughly twice that.

| $n$ | 95% interval at $p=0.7$ |
|---|---|
| 100 | ±9.0 points |
| 500 | ±4.0 points |
| 2,000 | ±2.0 points |
| 10,000 | ±0.9 points |

::: key
A model scoring 71.2% against 69.8% on a 500-example benchmark has **not** been shown to be
better. The difference is well under one standard error.

Most benchmark suites have 200–1,000 examples per task. Most reported differences are noise.
Report intervals, and use **paired** comparisons on identical examples — which, as lesson
1.11's exercise showed, can turn an underpowered comparison into a decisive one at no extra
cost.
:::

## Contamination

Benchmarks are on the public web; training corpora are scraped from the public web. A model
that memorised the test set scores well and has learned nothing.

Detecting it after the fact is hard, but three signals help:

```python
def contamination_signals(model, benchmark):
    return {
        # 1. Does the model complete a test question verbatim from its first half?
        'verbatim_completion': measure_prefix_continuation(model, benchmark),
        # 2. Does performance drop on reordered multiple-choice options?
        'order_sensitivity': accuracy(benchmark) - accuracy(shuffle_options(benchmark)),
        # 3. Does it drop on semantically identical rewrites?
        'paraphrase_gap': accuracy(benchmark) - accuracy(paraphrase(benchmark)),
    }
```

A large `paraphrase_gap` is the clearest signal: a model that understands the task should be
nearly indifferent to phrasing, and a model that memorised the string is not.

The only real defence is **held-out evaluation you construct yourself**, after the model's
training cutoff, and never publish.

::: check
Model A scores 71.2% and model B 69.8% on a 500-example benchmark. What has been shown?

- [x] Nothing — the 95% interval at $n = 500$ is about ±4 points, so the difference is well under one standard error
  > Most benchmark suites have 200–1,000 examples per task, and most reported differences are noise. Report intervals, and use paired comparisons on identical examples, which can make an underpowered comparison decisive at no extra cost.
- [ ] That A is better, since it scored higher
  > A higher number on a noisy measurement is not evidence of a difference.
- [ ] That A is better by 1.4 points, with the uncertainty applying to the absolute scores only
  > Uncertainty on each score propagates into the difference; it does not cancel because the scores are independent draws.
- [ ] That the benchmark is saturated
  > Saturation would show as both models near the ceiling. At 70% there is plenty of headroom.
:::

## Selection effects

Even with clean data, reported numbers are optimistically biased by how they were produced:

- **Best checkpoint.** Selecting the best of 50 evaluations is a maximisation over noise,
  worth roughly one standard deviation.
- **Best prompt.** Trying ten prompts and reporting the best is the same effect.
- **Best of several runs.** Seed variance (lesson 2.15) is often larger than the effect
  being claimed.
- **Publication bias.** Configurations that did not work are not reported.

Each is individually small and they compound. A paper reporting a 2-point improvement,
selected across checkpoints, prompts and seeds, has demonstrated very little.

## What the standard benchmarks measure

| Benchmark | Measures | Main weakness |
|---|---|---|
| MMLU | Multiple-choice knowledge | Heavily contaminated; format-sensitive |
| GSM8K | Grade-school maths | Contaminated; saturated |
| HumanEval | Python from docstrings | 164 problems — ±7 points |
| MT-Bench | Chat quality, LLM-judged | Judge biases, position bias |
| Chatbot Arena | Human preference, paired | Preference is not correctness |
| SWE-bench | Real GitHub issue resolution | Expensive; harness-sensitive |

Multiple choice deserves particular scepticism. It measures discrimination among given
options, not generation. Models are also sensitive to option *order* and to the letter
labels themselves — a real effect, and a sign the format is measuring something other than
knowledge.

## LLM as judge

Using a strong model to score outputs is cheap and scales. It has consistent, measurable
biases:

- **Position bias** — the first-presented response wins more often. Always evaluate both
  orderings and average.
- **Length bias** — longer responses score higher, the same effect as lesson 5.07's reward
  models.
- **Self-preference** — models rate their own outputs and their own style higher.
- **Style over substance** — confident, well-formatted, wrong answers beat hesitant correct
  ones.

```python
def judge_pairwise(judge, question, a, b):
    """Both orderings, then combine. A single ordering is not a measurement."""
    first  = judge.compare(question, a, b)
    second = judge.compare(question, b, a)
    if first == 'A' and second == 'B':
        return 'a'                      # consistent: a wins both orderings
    if first == 'B' and second == 'A':
        return 'b'
    return 'tie'                        # inconsistent -> no signal
```

The inconsistency rate is itself informative: if the judge disagrees with itself on 30% of
pairs, its resolution is about 30% and differences below that are not measurable.

Always calibrate the judge against human labels on a sample before trusting it at scale.

::: check
Why is benchmark contamination structurally hard to avoid?

- [x] Benchmarks are published on the public web and training corpora are scraped from the public web
  > A model that memorised the test set scores well and has learned nothing. Decontamination at the data stage and held-out private sets are the responses; detecting it after the fact is hard.
- [ ] Because benchmark licences forbid filtering them out
  > Licences are not the obstacle; finding every paraphrase and mirror is.
- [ ] Because a model memorises everything it is shown at least once in training
  > They memorise some things and not others, which is what makes detection statistical rather than certain.
- [ ] Because test sets are too small to remove from a corpus
  > Size makes them easy to *store* and hard to *find*, since they appear in many reformatted copies.
:::

## What to measure instead

**Build a task-specific evaluation set.** 200–500 examples from your actual use case,
labelled by people who understand the domain. This beats every public benchmark for
deciding whether a model works *for you*, and it cannot be contaminated because it does not
exist publicly.

**Measure what you care about**, not what is easy:

```python
metrics = {
    'accuracy':      exact_or_semantic_match(pred, gold),
    'faithfulness':  supported_by_sources(pred, context),   # for RAG
    'abstention':    says_unknown_when_unknown(pred, gold),
    'latency_p95':   ...,
    'cost_per_task': ...,
    'failure_mode':  classify_error(pred, gold),            # read them
}
```

**Read the failures.** Fifty failed cases, read by hand and classified, tells you more than
any aggregate score. Aggregates say *how much*; failure classes say *what to fix*.

**Track regressions.** When you change a prompt or upgrade a model, the question is not "is
the average better" but "what got worse". Keep a fixed regression set and diff per-example.

::: warning
**Never tune on the set you report.** If you selected a prompt, a checkpoint or a
hyperparameter using a set, that set is now a training set and its number is optimistic.

Three splits: development (tune freely), validation (check occasionally), test (look once,
at the end). This is elementary and routinely violated in practice because the test set is
the only labelled data available — in which case the honest report is "tuned and evaluated
on the same data", not a clean-sounding number.
:::

::: exercise
Your new model scores 73.1% on your benchmark against the old model's 71.4%. The benchmark
has 400 examples. Should you ship it?
:::

::: solution
**Not on this evidence.** Run the arithmetic first.

Standard error per model: $\sqrt{0.72 \times 0.28 / 400} \approx 2.2$ points. If the two
were evaluated independently, the difference has standard error $\sqrt{2.2^2 + 2.2^2}
\approx 3.2$ points. The observed 1.7-point difference is **half a standard error** —
entirely consistent with no difference at all.

**What to do instead, in order:**

1. **Use a paired comparison.** Both models on the same 400 examples, compared per example.
   Only the disagreements carry information (McNemar's test). If they disagree on 60
   examples split 40–20 in the new model's favour, that *is* significant ($p \approx 0.01$)
   even though the marginal difference looked like noise. This costs nothing and frequently
   resolves the question.

2. **Check seed variance.** If both are fine-tunes, run each with 3 seeds. If within-model
   spread is ±2 points, a 1.7-point between-model difference is not a finding (lesson 2.15).

3. **Expand the evaluation set** to 2,000 examples if the decision matters. That narrows the
   interval to ±2.0 points per model, at which point 1.7 might be detectable.

4. **Look at the failure classes.** Even without statistical significance, if the new model
   fixes a category of error you care about and introduces none, that is a better reason to
   ship than the aggregate. Conversely, a model with the same average that fails on your
   most important segment should not ship.

5. **Check the non-quality axes.** Latency, cost, context requirements and behaviour on
   adversarial input often decide the question. A 1.7-point gain that doubles cost per query
   is not obviously a gain.

**What would justify shipping:** a paired test showing significance, plus no regression on
the failure classes you care about, plus acceptable cost and latency. "The average went up"
on 400 examples is not evidence.

**The honest framing:** treat 1.7 points on 400 examples as "no measurable difference". If
the new model is better for another reason — simpler, cheaper, better licensed — ship it for
that reason and say so.
:::

## What to carry forward

- Compute the error bar before comparing; most reported differences are noise.
- Use paired comparisons on identical examples — they are far more powerful.
- Assume public benchmarks are contaminated; test with paraphrases to detect it.
- LLM judges have position, length and self-preference biases; evaluate both orderings.
- Build a private task-specific set, read the failures, and never tune on what you report.
