---
summary: Making a result something you can reproduce and explain three months later — what to log, what to version, and the discipline that makes ablations mean anything.
prereqs: [reproducibility, training-loop]
---

A number you cannot reproduce is not a result. Most of the work of making one reproducible is
mechanical, cheap, and skipped anyway.

## The four things to version

::: key
**1. Code.** A git commit hash, *and* whether the tree was dirty. A hash from a dirty tree
identifies nothing (lesson 2.15).

**2. Data.** A content hash of the processed dataset, not the download URL. Datasets get
re-scraped, re-filtered and silently updated.

**3. Configuration.** The complete resolved config — every default that was in effect, not
just the values you overrode.

**4. Environment.** Library versions, CUDA version, GPU model, world size. These change
kernel selection and therefore numerical results.

Miss any one and a future reproduction attempt has an unexplained variable.
:::

```python
import hashlib, json, os, subprocess, sys
import torch

def provenance(config, data_paths):
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()

    def file_hash(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()[:16]

    return {
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain')),
        'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'data_hashes': {p: file_hash(p) for p in data_paths},
        'config': config,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'world_size': int(os.environ.get('WORLD_SIZE', 1)),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
    }
```

## What to log during training

```python
def training_metrics(loss, grad_norm, opt, model, step, tokens_seen, step_time):
    lr = opt.param_groups[0]['lr']
    param_norm = torch.sqrt(sum(p.detach().pow(2).sum() for p in model.parameters()))
    return {
        'loss': loss.item(),
        'grad_norm': grad_norm.item(),
        'lr': lr,
        'param_norm': param_norm.item(),
        # How far parameters move per step, relative to their size.
        'update_ratio': (lr * grad_norm / param_norm).item(),
        'tokens_seen': tokens_seen,
        'tokens_per_sec': tokens_seen / step_time,
        'gpu_mem_gb': torch.cuda.max_memory_allocated() / 1e9,
    }
```

`update_ratio` deserves the attention it rarely gets. Healthy training sits near $10^{-3}$;
much larger means the learning rate is too high, much smaller means training has stalled. It
catches problems earlier than loss does (lesson 5.04).

Log **gradient norm every step**. It is the earliest warning of a data problem or an
impending divergence, and it costs nothing.

## Configuration

Use structured, composable configs rather than argparse for anything non-trivial:

```python
from dataclasses import dataclass, asdict, field

@dataclass
class ModelConfig:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lr: float = 3e-4
    batch_size: int = 32
    steps: int = 100_000
    seed: int = 0

    def __post_init__(self):
        # Validate here, not 40 minutes into the run.
        assert self.model.d_model % self.model.n_heads == 0
        assert self.lr > 0 and self.steps > 0
```

::: warning
**Log the resolved config, not the command line.** `python train.py --lr 3e-4` does not
record the other forty defaults that were in effect, and those defaults change when someone
edits the file.

Serialise the whole dataclass with `asdict()` and store it with the run. This is the single
most common reason a result cannot be reproduced six months later.
:::

::: check
An ablation changes two things at once and the result improves. What have you learned?

- [x] That the pair helps, and nothing about either one
  > The two could even be working against each other with one large enough to win. An ablation is only informative when exactly one thing changes and the comparison is against a baseline you ran yourself, on the same data, with the same budget.
- [ ] That both changes help, since the combined effect is positive
  > A positive sum is consistent with one helping a lot and the other hurting.
- [ ] That the larger of the two changes is responsible
  > Nothing in the result tells you which was larger in effect.
- [ ] Nothing at all; multi-factor experiments are uninformative
  > They tell you about the combination, which is sometimes exactly what you wanted to know.
:::

## Naming and organising runs

A name like `exp_v3_final_final2` tells you nothing in three months. Encode the variable:

```python
def run_name(cfg, provenance):
    return (f'{cfg.model.n_layers}L-{cfg.model.d_model}d'
            f'_lr{cfg.lr:.0e}_bs{cfg.batch_size}'
            f'_seed{cfg.seed}_{provenance["git_commit"][:7]}')
# 12L-768d_lr3e-04_bs32_seed0_a3f9c21
```

Tag runs by the question they answer — `ablation:norm`, `sweep:lr`, `baseline` — so a search
returns a comparable set rather than everything you have ever run.

::: check
Which of these must be versioned for a result to be reproducible?

- [x] The git commit hash *and* whether the working tree was dirty
  > A hash from a dirty tree identifies nothing.
- [x] A content hash of the processed dataset, not the download URL
  > Datasets get re-scraped, re-filtered and silently updated. The URL is not the data.
- [x] The complete resolved configuration, defaults included
  > Recording only the overrides leaves every default free to change underneath you.
- [x] Library, CUDA and GPU versions, and the world size
  > These change kernel selection and therefore numerical results — lesson 2.15's point, in operational form.
- [ ] The wall-clock time at which the run was started
  > Useful for bookkeeping, and it explains no variance in the result.
:::

## Ablations that mean something

::: key
An ablation with one seed per configuration measures seed variance, not the intervention
(lesson 2.15).

Minimum standard:
- **3+ seeds per configuration.** Report mean and standard deviation.
- **One variable at a time.** Changing two and observing an improvement attributes nothing.
- **Matched compute.** A configuration that trains longer is not a fair comparison; match
  FLOPs, not epochs.
- **Report the failures.** A config that diverged is a result, and omitting it makes the
  surviving ones look more reliable than they are.
:::

```python
import statistics

def ablation_report(results):
    """results: {config_name: [metric per seed]}"""
    for name, values in sorted(results.items(), key=lambda kv: -statistics.mean(kv[1])):
        m, sd = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
        print(f'{name:36s} {m:7.4f} ± {sd:.4f}  (n={len(values)})')
```

## Tooling

| Tool | Best for |
|---|---|
| Weights & Biases | Hosted, rich UI, good sweep support |
| MLflow | Self-hosted, model registry |
| TensorBoard | Local, minimal, no account |
| Plain JSONL | Zero dependencies, greppable, always works |

JSONL is underrated. One line per step, one file per run:

```python
import json, time

class JSONLLogger:
    def __init__(self, path, provenance):
        self.f = open(path, 'a', buffering=1)          # line-buffered: survives a crash
        self.log({'event': 'start', 'time': time.time(), **provenance})

    def log(self, record):
        self.f.write(json.dumps(record) + '\n')
```

It has no service to be down, no API to change, and `jq` handles the analysis. For a run that
must be reproducible in five years, this is the more robust choice.

## Checkpointing for resumption

From lesson 2.11 and 2.15, a resumable checkpoint needs the model, optimizer, scheduler, step
counter, RNG state and data-loader position. Write atomically:

```python
import os, torch

def save_checkpoint(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)        # atomic: a killed job never leaves a partial file
```

Keep the last few and every $N$-th, not all — checkpoints for a large model are the dominant
storage cost of a run.

::: exercise
A colleague cannot reproduce your result from three months ago. Your run logged the git hash,
the config and the final metric. What is missing?
:::

::: solution
**Four things, in rough order of how often they are the cause:**

**1. Whether the tree was dirty.** A commit hash with uncommitted changes identifies code
that no longer exists anywhere. If `git_dirty` was true and unrecorded, the exact code cannot
be recovered. *Fix going forward:* record the flag, and refuse to start a run on a dirty tree
for anything you intend to publish.

**2. The data.** A path is not a version. The file at `data/train.jsonl` may have been
regenerated with updated filtering. Without a content hash there is no way to tell whether
your colleague has the same corpus. *Fix:* hash the processed data and store it with the run.

**3. The environment.** PyTorch and CUDA versions change kernel selection, which changes
summation order, which changes results in the last bits — and over a long run those compound
(lesson 2.15). A different GPU model or world size changes it further. *Fix:* record all of
it, and pin versions in a lockfile.

**4. The seed, and how many you ran.** If only the best of five seeds was reported, your
colleague running one seed will usually get a worse number and there is nothing wrong. *Fix:*
report mean and spread over seeds, and say how many.

**What to do right now to resolve the specific dispute:**

- **Establish seed variance.** Have both of you run 3 seeds with the current code. If the
  spread covers the discrepancy, the mystery is resolved — the original number was the top of
  a range.
- **Diff the environments.** Compare `torch.__version__`, CUDA, and GPU model between the two
  runs.
- **Check the data.** Hash both copies. If they differ, you have found it.
- **Check the evaluation.** Best-checkpoint versus last-checkpoint selection can differ by
  about a standard deviation (lesson 5.16). Confirm you are both reporting the same thing.

**The honest conclusion in most real cases** is (4): the original number was selected across
seeds or checkpoints, and the "correct" value is a mean with a spread. That is worth saying
plainly rather than hunting for a bug that is not there.
:::

## What to carry forward

- Version code, data, config and environment — and record whether the tree was dirty.
- Log the resolved config, not the command line.
- Log gradient norm every step and watch `update_ratio` near $10^{-3}$.
- 3+ seeds per ablation, one variable at a time, matched compute, and report the failures.
- JSONL has no service to be down; write checkpoints atomically.
