---
summary: What breaks after a model ships — the failure modes that do not raise errors, and the monitoring that catches them.
prereqs: [inference-serving, evaluating-llms, testing-ml-code]
---

A deployed model fails differently from a deployed service. The service is up, latency is
fine, and the outputs are getting worse. Monitoring has to be designed for that.

## The deployment checklist

Before serving traffic:

```python
def preflight(model, tokenizer, config):
    checks = {}
    # 1. The tokenizer matches the checkpoint. A mismatch produces fluent nonsense.
    checks['tokenizer'] = tokenizer.vocab_size == model.config.vocab_size
    # 2. Generation stops. The most common launch bug (lesson 5.05).
    out = model.generate(tokenizer.encode('Hello'), max_new_tokens=200)
    checks['terminates'] = tokenizer.eos_token_id in out
    # 3. The chat template round-trips exactly as trained.
    checks['template'] = render_template(EXAMPLE) == GOLDEN_TEMPLATE_STRING
    # 4. Known-good outputs on a fixed set of prompts.
    checks['smoke'] = all(passes(model, p, expected) for p, expected in SMOKE_SET)
    # 5. Memory headroom under the worst-case batch and context.
    checks['memory'] = peak_memory_under_max_load() < 0.85 * total_memory()
    return checks
```

Items 1 through 3 are the ones that ship broken. All three produce plausible output and none
raises an error.

## Shadow and canary

::: key
**Shadow deployment** sends production traffic to the new model without returning its
responses. You get real-distribution behaviour with zero user risk. Run it long enough to
cover the traffic's daily cycle.

**Canary deployment** routes a small fraction of real traffic and compares outcomes. Start at
1%, then 5%, 25%, 100%, with an automatic rollback on a metric regression.

Shadow catches crashes, latency regressions and output-distribution shifts. It cannot measure
user outcomes, because nobody sees the responses. Canary measures outcomes and carries real
risk. Use both, in that order.
:::

## What to monitor

Four layers, and most teams only instrument the first:

**Infrastructure** — latency percentiles, error rate, GPU utilisation, memory, queue depth.
Standard service monitoring.

**Model behaviour** — this is the layer that catches model-specific failures:

```python
def output_metrics(request, response, logprobs):
    return {
        'output_tokens': len(response),
        'finish_reason': 'stop' if response[-1] == eos else 'length',
        'mean_logprob': logprobs.mean(),          # confidence proxy
        'min_logprob': logprobs.min(),            # a very unlikely token was emitted
        'repetition_rate': max_ngram_repeat(response, n=4),
        'refusal': matches_refusal_pattern(response),
        'empty': len(response) < 3,
    }
```

::: warning
**`finish_reason == 'length'` is the metric people forget.** A rising rate means the model is
failing to stop, and the cause is usually upstream — a template change, a context overflow, or
a fine-tune that damaged the stop token (lesson 5.05).

**Mean log-probability** is the best cheap proxy for distribution shift. It falls when inputs
drift away from the training distribution, and it falls before human-visible quality does.
:::

**Input distribution** — prompt length percentiles, language mix, request-type mix, rate of
inputs outside the expected schema. Drift here explains most model-behaviour drift.

**Business outcomes** — user retention, thumbs up/down rate, escalation rate, task completion.
The only layer that measures what you actually care about, and the slowest to move.

## Drift

Two kinds, with different responses:

**Covariate shift** — $p(x)$ changes while $p(y\mid x)$ does not. Users start asking about a
new topic. The model may still be correct; it is just operating on inputs it saw less of.

**Concept drift** — $p(y\mid x)$ changes. The correct answer to "what is the latest version"
changes. The model is now wrong on inputs it handles confidently.

```python
import numpy as np

def population_stability_index(baseline, current, bins=10):
    """PSI > 0.2 conventionally indicates significant distribution shift."""
    edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(baseline, edges)[0] / len(baseline) + 1e-6
    c = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((c - b) * np.log(c / b)))
```

Run PSI on embedding coordinates, prompt lengths and output log-probabilities. It gives an
early warning that is cheap and does not need labels — which is the point, since labels in
production are scarce.

## Feedback

```python
def log_interaction(request_id, prompt, response, metadata):
    """Log enough to reconstruct and to build an eval set later."""
    store({
        'request_id': request_id,
        'model_version': MODEL_VERSION,           # essential for attributing regressions
        'prompt_hash': sha256(prompt),            # hash, not the prompt, by default
        'prompt': prompt if metadata['consented'] else None,
        'response': response if metadata['consented'] else None,
        'sampling': metadata['sampling_params'],
        'latency_ms': metadata['latency'],
        'metrics': output_metrics(prompt, response, metadata['logprobs']),
        'timestamp': time.time(),
    })
```

Implicit signals are more plentiful than explicit ones and usually more honest: did the user
rephrase immediately, copy the output, abandon the session, escalate to a human. Thumbs-up
rates are heavily biased toward users who had a strong reaction either way.

The value of this logging compounds: today's production traffic is tomorrow's evaluation set
(lesson 5.16), and it is the only eval set that cannot be contaminated.

## Rollback

::: key
**Be able to roll back in minutes, and practise it.** Model deployments regress in ways tests
do not catch, and the correct response to an unexplained quality drop is to roll back first
and diagnose afterwards.

Requirements: keep the previous version warm, make the version a config value rather than a
rebuild, version the prompt template alongside the weights, and alert on the model-behaviour
metrics rather than only on errors.

A rollback that takes an hour is not a rollback.
:::

## Cost

```python
def cost_per_request(prompt_tokens, output_tokens, model_size_b,
                     gpu_hourly=2.5, prefill_tok_s=8000, decode_tok_s=60):
    prefill_s = prompt_tokens / prefill_tok_s
    decode_s = output_tokens / decode_tok_s
    return (prefill_s + decode_s) / 3600 * gpu_hourly
```

Track cost per request and per successful task separately. A cheaper model that requires two
attempts is more expensive, and only the second metric shows it.

::: exercise
Users report the model has got worse. Your metrics — latency, error rate, GPU utilisation —
are all normal. How do you investigate?
:::

::: solution
**Normal infrastructure metrics is the expected finding.** Model quality regressions do not
show up as errors, which is exactly why the model-behaviour layer exists.

**In order:**

1. **Establish when.** Plot the model-behaviour metrics — mean log-probability, output
   length, `finish_reason` distribution, refusal rate — over the last few weeks. A step
   change dates the regression, and the date tells you what to look at.

2. **Check what changed on that date.** Model version, prompt template, tokenizer, serving
   library, sampling defaults, or a retrieval index rebuild if there is RAG in the path. A
   step change in metrics with no model deployment usually means a template or config change
   — and template changes are the most common cause of "it got worse" with nothing else
   moving.

3. **If nothing changed, check the inputs.** Run PSI on prompt length and embedding
   distribution against a baseline from before. Covariate shift means the model is fine and
   the traffic moved — the response is to extend the eval set to cover the new distribution,
   not to roll back.

4. **Read the outputs.** Pull 50 recent interactions with low mean log-probability and 50
   from before the regression, and read them side by side. This is unglamorous and it
   identifies the failure mode faster than any metric. Aggregate numbers tell you *how much*;
   reading tells you *what*.

5. **Run the regression set.** If you have a fixed set of prompts with known-good outputs
   (lesson 5.16), run it against the current deployment and against the previous version.
   This distinguishes "the model changed" from "the traffic changed" definitively.

6. **Check the whole path, not just the model.** Retrieval quality (lesson 5.14), a rate
   limiter silently truncating context, a caching layer serving stale responses. The model is
   one component and users are reporting on the system.

**If it is real and you cannot find the cause within an hour: roll back.** Diagnose against
the previous version while users are unaffected. The instinct to find the cause first is the
wrong order of operations.

**Also worth ruling out:** that nothing changed and expectations did. A user base that has
grown or shifted to harder tasks reports degradation without any regression. Check whether
the *distribution of requests* moved, which step 3 covers, before concluding there is a bug.
:::

## What to carry forward

- Tokenizer mismatch, failure to stop, and template drift all ship without erroring.
- Shadow first for safety, then canary for outcomes.
- Monitor model behaviour — log-probability, finish reason, repetition — not just infrastructure.
- PSI on embeddings and output distributions gives label-free drift warning.
- Roll back in minutes and diagnose afterwards; practise it before you need it.
