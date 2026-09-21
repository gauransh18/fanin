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
  markdown.mjs          Markdown subset: callouts, tables, TeX, collapsible solutions
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
```

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
