# Fanin

A free AI research curriculum: **108 lessons across 7 tracks**, from vector norms to
sharded training runs. Every lesson is open — no tiers, no trial, no account.

**Live site:** https://gauransh18.github.io/fanin/ — once Pages is enabled, see below.

## Why

The usual model for a curriculum like this opens the first few lessons and puts
attention, RLHF and the systems material behind a subscription. The free part teaches
you what a tensor is; the paid part is the reason you came. This one is inverted: the
hardest material is the part most worth giving away.

There is no backend, no analytics, no account system, and nothing that could later grow
a paywall. Lesson progress is kept in your own browser and never leaves it.

## Learning by answering

Every lesson carries questions — 228 of them across the curriculum, placed at the end of
the section they test. Answering one gives immediate feedback and an explanation for
**every** option, right or wrong, so a miss teaches rather than just scoring zero. Clearing
a lesson's questions is what marks it complete; there is no box to tick.

Answers earn XP (20 first try, 8 after a miss, plus a combo bonus every fifth in a row),
XP earns levels, and each track ends in a **trial** — ten questions drawn from across it,
no explanations until the end, 80% to pass. A track badge takes both: every lesson cleared
and the trial passed.

The correct answer ships inside the page, base64-encoded, because there is no server to
check against. That is enough to stop a stray ctrl-F spoiling a question and is not
pretending to be more.

## The curriculum

| # | Track | Lessons |
|---|-------|---------|
| 1 | Mathematical Foundations | 16 |
| 2 | Python and PyTorch | 15 |
| 3 | Deep Learning Core | 16 |
| 4 | Transformers and Attention | 16 |
| 5 | Large Language Models | 16 |
| 6 | Reinforcement Learning | 14 |
| 7 | Systems and MLOps | 15 |

## Running it locally

```bash
node src/build.mjs     # writes dist/
node src/check.mjs     # verifies links, frontmatter and lesson ordering
node src/serve.mjs     # serves dist/ on http://localhost:4173
```

There are **no dependencies**. `npm install` is not required and there is no lockfile —
the generator is plain Node with a hand-written Markdown parser and syntax highlighter.

Set `BASE_PATH` when the site is served from a subdirectory:

```bash
BASE_PATH=/fanin node src/build.mjs
```

## Layout

```
content/
  curriculum.mjs        track and lesson manifest — the single source of structure
  lessons/<track>/*.md  lesson content, Markdown with frontmatter
src/
  build.mjs             static site generator
  check.mjs             integrity checks — run in CI
  markdown.mjs          Markdown subset: callouts, tables, TeX, solutions, checks
  test-progress.mjs     51 tests over the XP, streak, level, badge and trial engine
  highlight.mjs         build-time syntax highlighting
  templates.mjs         page shell
  assets/               styles.css, app.js, self-hosted fonts and KaTeX
```

Nothing is loaded from a CDN. IBM Plex and KaTeX are vendored under
`src/assets/`, so the site makes **no third-party requests** — which is what the
no-tracking promise on the About page actually requires.

### Writing a lesson

Create `content/lessons/<track>/<slug>.md` matching a slug in `content/curriculum.mjs`:

```markdown
---
summary: One sentence shown under the title and in search results.
prereqs: [vectors-norms-geometry, derivatives-gradients-jacobians]
seealso: [flash-attention]
---

## A section

Prose, with $inline$ maths and display blocks:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) A(s,a) \right]
$$

::: insight
Callout kinds: note, tip, warning, insight, key, history, exercise, solution.
:::

::: exercise
Show that softmax is invariant to adding a constant to every logit.
:::

::: solution
Subtracting the max is exactly this, used for numerical stability.
:::

::: check
Which of these is the reason for the $1/\sqrt{d_k}$ scaling?

- [x] Raw scores have standard deviation $\sqrt{d_k}$, and softmax would saturate
  > At $p \approx 1$ the softmax Jacobian vanishes and nothing upstream gets a gradient.
- [ ] It keeps the dot products inside float32's range
  > Range is not the problem; the shape of the distribution is.
:::
```

A `::: check` block holds a question, its options, which are correct, and — for every
option — why. Mark more than one option `[x]` and it becomes a select-all-that-apply. For a
number answer, replace the options with `= 64` (optionally `= 3.14 ± 0.01`) and a single
`>` explanation. Options are shuffled at build time from a seed of the lesson id and the
check's index, so the correct answer is not wherever you wrote it but does not move between
builds. A malformed check — no correct option, an option with no explanation, fewer than
two options — fails the build with the lesson and check named.

`prereqs` are lessons this one leans on and **must come earlier** in the
curriculum; `seealso` points **forwards**. `node src/check.mjs` enforces both
directions, so the "dependency order" claim on the About page stays true.

Lessons listed in the manifest without a Markdown file render as an honest
"being edited" page and are flagged in the index — never as locked content.
All 108 are currently written.

## Deploying

`.github/workflows/deploy.yml` runs on every push:

- **`verify`** builds the site and runs `src/check.mjs`. This is the job that
  tells you whether the site is correct, and it never touches Pages.
- **`deploy`** publishes `dist/` to GitHub Pages.

**One-time setup.** A workflow token cannot create a Pages site, so until Pages
is switched on for the repository the deploy job checks the Pages API, logs a
notice and **skips** — the run stays green rather than emailing a failure for a
setting nobody has flipped yet. Enable it once:

> **Settings → Pages → Build and deployment → Source: `GitHub Actions`**

The next push publishes. Nothing else is needed: `BASE_PATH` is resolved from
`actions/configure-pages`, so the same build works at `/fanin` on a project
site or at `/` behind a custom domain.

## Licence

Lesson content is **CC BY-SA 4.0**. Site code is **MIT**.
