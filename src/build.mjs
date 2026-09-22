import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { site, tracks, allLessons, totals } from '../content/curriculum.mjs';
import { render, renderCheck, parseFrontmatter, toPlainText } from './markdown.mjs';
import { layout, esc, makeUrl, ICON } from './templates.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const OUT = path.join(ROOT, 'dist');
const LESSON_DIR = path.join(ROOT, 'content', 'lessons');

// Two link modes.
//   absolute  links are BASE + "/curriculum/"  — what a web server wants.
//   relative  links are "../../curriculum/index.html" — works from any path,
//             including a file:// open or a preview host that serves published
//             paths literally with no directory-index resolution.
const RELATIVE = process.env.LINK_MODE === 'relative';
const BASE = RELATIVE ? '' : (process.env.BASE_PATH || '').replace(/\/$/, '');

// Depth of the page currently being written, so relative links can climb out
// of it. Pages are generated one at a time, so a module-level cursor is safe.
let pageDepth = 0;
const setPage = (canonical) => {
  pageDepth = canonical.split('/').filter(Boolean).length;
};

const absoluteUrl = makeUrl(BASE);

function u(target) {
  if (/^(https?:|mailto:|#)/.test(target)) return target;
  if (!RELATIVE) return absoluteUrl(target);
  const clean = target.replace(/^\//, '');
  const file = clean === '' || clean.endsWith('/') ? clean + 'index.html' : clean;
  const up = pageDepth ? '../'.repeat(pageDepth) : './';
  return up + file;
}

const byTrack = new Map(tracks.map((t) => [t.id, t]));
const lessonAt = (i) => (i >= 0 && i < allLessons.length ? allLessons[i] : null);
const lessonId = (l) => `${l.trackId}/${l.slug}`;
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function hours(minutes) {
  const h = minutes / 60;
  return h >= 10 ? `${Math.round(h)}h` : `${h.toFixed(1)}h`;
}

async function write(relPath, contents) {
  const target = path.join(OUT, relPath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, contents);
}

// -------------------------------------------------------------- fragments --

function trackDot(track) {
  return `<span class="track-dot" style="--track:${track.fill}" aria-hidden="true"></span>`;
}

function spine() {
  // Each bar is the track's own slice of one continuous 100%-wide line, placed
  // at its real offset through the curriculum. The seven bars step across the
  // rows, so the shape shows where a track sits in the whole path — not just
  // how many lessons it has.
  let elapsed = 0;
  const rows = tracks
    .map((track, i) => {
      const mins = track.lessons.reduce((s, [, , m]) => s + m, 0);
      const offset = (elapsed / totals.minutes) * 100;
      const width = (mins / totals.minutes) * 100;
      elapsed += mins;
      return `<a class="spine-row" href="${u(`/learn/${track.id}/`)}" style="--track:${track.fill}">
        <span class="spine-num">${String(i + 1).padStart(2, '0')}</span>
        <span class="spine-name">${esc(track.short)}</span>
        <span class="spine-track" aria-hidden="true"><i style="margin-left:${offset.toFixed(
          1
        )}%;width:${width.toFixed(1)}%"></i></span>
        <span class="spine-meta">${track.lessons.length}</span>
      </a>`;
    })
    .join('\n');
  return `<div class="spine">
    <p class="spine-title">The seven tracks, in order</p>
    ${rows}
    <p class="spine-foot">Bar position shows where each track falls across the ${hours(
      totals.minutes
    )} path.</p>
  </div>`;
}

function trackNav(track, currentSlug) {
  const items = track.lessons
    .map(([slug, title], i) => {
      const id = `${track.id}/${slug}`;
      const current = slug === currentSlug;
      const number = `${tracks.indexOf(track) + 1}.${String(i + 1).padStart(2, '0')}`;
      return `<li><a href="${u(`/learn/${track.id}/${slug}/`)}" data-lesson-id="${id}"${
        current ? ' aria-current="page"' : ''
      }><span class="n">${number}</span><span>${esc(title)}</span></a></li>`;
    })
    .join('\n');

  const others = tracks
    .filter((t) => t.id !== track.id)
    .map(
      (t) =>
        `<li><a href="${u(`/learn/${t.id}/`)}">${trackDot(t)}<span>${esc(t.short)}</span></a></li>`
    )
    .join('\n');

  return `<nav class="track-nav" aria-label="${esc(track.title)} lessons">
    <div class="track-nav-head">${trackDot(track)}<strong>${esc(track.title)}</strong></div>
    <ol>${items}</ol>
    <div class="track-nav-other">
      <p class="spine-title">Other tracks</p>
      <ol>${others}</ol>
    </div>
  </nav>`;
}

function lessonRows(track, present) {
  return track.lessons
    .map(([slug, title, minutes], i) => {
      const id = `${track.id}/${slug}`;
      const draft = present.has(id) ? '' : ' data-draft="1"';
      return `<a class="lesson-row" href="${u(`/learn/${track.id}/${slug}/`)}" data-lesson-id="${id}"${draft}>
        <span class="n">${tracks.indexOf(track) + 1}.${String(i + 1).padStart(2, '0')}</span>
        <span class="t">${esc(title)}</span>
        <span class="m">${minutes} min</span>
      </a>`;
    })
    .join('\n');
}

// ------------------------------------------------------------------ pages --

function landingPage(present) {
  setPage('/');
  const done = allLessons.filter((l) => present.has(lessonId(l))).length;

  const cards = tracks
    .map((track, i) => {
      const ids = track.lessons.map(([slug]) => `${track.id}/${slug}`).join(' ');
      const mins = track.lessons.reduce((s, [, , m]) => s + m, 0);
      return `<a class="track-card" href="${u(`/learn/${track.id}/`)}" style="--track:${track.fill}"
         data-track-progress data-track-lessons="${ids}">
        <span class="track-card-top">${trackDot(track)}<span class="track-card-num">TRACK ${String(
        i + 1
      ).padStart(2, '0')}</span></span>
        <h3>${esc(track.title)}</h3>
        <p>${esc(track.blurb)}</p>
        <span class="track-card-foot">
          <span>${track.lessons.length} lessons</span><span class="sep"></span>
          <span>${hours(mins)}</span><span class="sep"></span>
          <span data-progress-label>0 / ${track.lessons.length}</span>
        </span>
        <span class="progress"><i></i></span>
      </a>`;
    })
    .join('\n');

  const paths = [
    {
      who: 'You can code, but the maths is a wall',
      what: 'Start at the beginning. The maths track is written to be read, not drilled — every result is introduced because something later needs it.',
      href: '/learn/math/vectors-norms-geometry/',
      cta: 'Vectors, Norms, and Geometry',
    },
    {
      who: 'You know the maths, you want the models',
      what: 'Skip to PyTorch, then go straight at attention. Track 4 builds a working GPT from an empty file in nine lessons.',
      href: '/learn/transformers/scaled-dot-product-attention/',
      cta: 'Scaled Dot-Product Attention',
    },
    {
      who: 'You ship models and want the systems layer',
      what: 'Track 7 is the one most curricula stop before: memory hierarchies, sharding strategies, serving throughput, and what a training run actually costs.',
      href: '/learn/systems/gpu-architecture/',
      cta: 'GPU Architecture for ML Engineers',
    },
  ];

  const pathCards = paths
    .map(
      (p, i) => `<div class="path">
      <p class="path-num">Path ${i + 1}</p>
      <h3>${esc(p.who)}</h3>
      <p class="path-what">${esc(p.what)}</p>
      <a class="path-cta" href="${u(p.href)}">${esc(p.cta)} ${ICON.arrow}</a>
    </div>`
    )
    .join('\n');

  const body = `
<section class="hero">
  <div class="wrap hero-grid">
    <div>
      <p class="eyebrow">${totals.lessons} lessons · ${totals.tracks} tracks · no account</p>
      <h1>The AI research curriculum, <em>fully unlocked</em>.</h1>
      <p class="hero-lede">
        Linear algebra to distributed training, in the order the ideas actually depend
        on each other. Every lesson is open — there is no pro tier, no trial, and nothing
        behind an email form.
      </p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="${u('/learn/math/vectors-norms-geometry/')}">Start lesson 1.01 ${ICON.arrow}</a>
        <a class="btn btn-ghost" href="${u('/curriculum/')}">See all ${totals.lessons} lessons</a>
      </div>

      <!-- Populated from local storage; stays hidden for a first-time reader. -->
      <a class="resume" id="resume-card" href="#" hidden>
        <span class="resume-label">Pick up where you left off</span>
        <span class="resume-title"><b data-resume-number></b> <span data-resume-title></span></span>
        <span class="resume-meta"><span data-resume-done>0</span> of ${
          totals.lessons
        } complete ${ICON.arrow}</span>
      </a>
    </div>
    <div>${spine()}</div>
  </div>
</section>

<div class="wrap">
  <div class="stats">
    <div class="stat"><div class="stat-value">${totals.lessons}</div><div class="stat-label">lessons, all readable now</div></div>
    <div class="stat"><div class="stat-value">${hours(totals.minutes)}</div><div class="stat-label">of written material</div></div>
    <div class="stat"><div class="stat-value">${totals.tracks}</div><div class="stat-label">tracks, in dependency order</div></div>
    <div class="stat"><div class="stat-value">$0</div><div class="stat-label">now and permanently</div></div>
  </div>
</div>

<section class="section">
  <div class="wrap">
    <div class="section-head">
      <h2>Seven tracks, built to be read in order</h2>
      <p>
        Each track assumes the ones before it and nothing else. You can enter anywhere —
        every lesson names the handful of ideas it leans on and links straight to them.
      </p>
    </div>
    <div class="track-list">${cards}</div>
  </div>
</section>

<section class="section">
  <div class="wrap">
    <div class="section-head">
      <h2>Three ways in</h2>
      <p>The order below is the one the curriculum was designed for, but most people arrive mid-way through it.</p>
    </div>
    <div class="paths">${pathCards}</div>
  </div>
</section>

<section class="section">
  <div class="wrap">
    <div class="manifesto">
      <div class="section-head">
        <h2>Why there is no paid tier</h2>
      </div>
      <div class="manifesto-body">
        <p>
          The standard model for a curriculum like this is to open the first few lessons,
          then put attention, RLHF and the systems material behind a subscription. The
          free part teaches you what a tensor is. The paid part is the reason you came.
        </p>
        <p>
          This one is inverted on purpose. The hardest material — FlashAttention, ZeRO
          sharding, PPO, speculative decoding — is the part most worth giving away, because
          it is the part that is genuinely hard to assemble from scattered papers and blog
          posts. Everything here is written once and open to everyone.
        </p>
        <p>
          The whole site is static files in a public repository. There is no backend, no
          analytics, no account system, and nothing that could later grow a paywall. Your
          progress is stored in your own browser and never leaves it. If this project is
          ever abandoned, the content is ${site.license} — fork it and keep going.
        </p>
        <p class="manifesto-cta">
          <a class="btn btn-ghost" href="${site.repo}" target="_blank" rel="noopener noreferrer">Read the source</a>
          <a class="btn btn-ghost" href="${u('/about/')}">More about the project</a>
        </p>
      </div>
    </div>
  </div>
</section>`;

  const order = JSON.stringify(
    allLessons.map((l) => [lessonId(l), l.number, l.title])
  );

  return layout({
    title: site.name,
    description: site.description,
    extraHead: `<script type="application/json" id="lesson-order">${order.replace(
      /</g,
      '\\u003c'
    )}</script>`,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: '/',
    nav: 'home',
  });
}

// The path: every lesson in the curriculum as one node, in order, grouped by
// track and ending in that track's trial. It is the same information the
// lesson rows below carry, in a shape you can see all of at once.
function pathMap(present, trialTracks) {
  return `
<section class="path" aria-label="The whole curriculum">
  <div class="path-head">
    <h2>The path</h2>
    <p class="path-sub">
      <span data-overall-progress data-all-lessons="${allLessons.map(lessonId).join(' ')}">0</span>
      of ${totals.lessons} cleared. Jump in anywhere — nothing is locked.
    </p>
  </div>
  <ol class="path-tracks">
    ${tracks
      .map(
        (track, i) => `<li class="path-track"
        style="--track:${track.fill};--track-ink:${track.inkLight};--track-ink-dark:${track.inkDark}">
      <a class="path-label" href="${u(`/learn/${track.id}/`)}">
        <span class="path-num">${String(i + 1).padStart(2, '0')}</span>
        <span class="path-name">${esc(track.short)}</span>
        <span class="path-count" data-track-progress
              data-track-lessons="${track.lessons.map(([s]) => `${track.id}/${s}`).join(' ')}">
          <span data-progress-label>0 / ${track.lessons.length}</span>
        </span>
      </a>
      <ol class="path-nodes">
        ${track.lessons
          .map(([slug, title], n) => {
            const id = `${track.id}/${slug}`;
            const number = `${i + 1}.${String(n + 1).padStart(2, '0')}`;
            return `<li><a class="node" href="${u(`/learn/${track.id}/${slug}/`)}"
              data-lesson-id="${id}" data-path-node
              title="${esc(`${number} ${title}`)}"
              aria-label="${esc(`${number} ${title}`)}"><span>${String(n + 1).padStart(2, '0')}</span></a></li>`;
          })
          .join('')}
        ${
          trialTracks.has(track.id)
            ? `<li><a class="node node-trial" href="${u(`/learn/${track.id}/trial/`)}"
                 data-trial-node="${track.id}"
                 title="${esc(`${track.short} trial — pass it to earn ${track.badge}`)}"
                 aria-label="${esc(`${track.short} trial`)}">
                 <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l2.6 5.6 6.1.8-4.5 4.2 1.2 6.1L12 16.8 6.6 19.7l1.2-6.1L3.3 9.4l6.1-.8z"/></svg>
               </a></li>`
            : ''
        }
      </ol>
    </li>`
      )
      .join('')}
  </ol>
  <p class="path-key">
    <span class="key-item"><i class="node" data-done="1"></i> cleared</span>
    <span class="key-item"><i class="node" data-current="1"></i> up next</span>
    <span class="key-item"><i class="node node-trial"></i> track trial</span>
  </p>
</section>`;
}

function curriculumPage(present, trialTracks) {
  setPage('/curriculum/');
  const allIds = allLessons.map(lessonId).join(' ');

  const sections = tracks
    .map((track, i) => {
      const ids = track.lessons.map(([slug]) => `${track.id}/${slug}`).join(' ');
      const mins = track.lessons.reduce((s, [, , m]) => s + m, 0);
      return `<section class="curriculum-track" id="${track.id}" style="--track:${track.fill}">
      <div class="ct-head">
        <div>
          <h2 class="ct-title">${trackDot(track)}<a href="${u(`/learn/${track.id}/`)}">${esc(
        track.title
      )}</a></h2>
          <p class="ct-blurb">${esc(track.blurb)}</p>
        </div>
        <div class="ct-meta" data-track-progress data-track-lessons="${ids}">
          <div class="ct-meta-row"><span>TRACK ${String(i + 1).padStart(2, '0')}</span><span>${
        track.lessons.length
      } lessons · ${hours(mins)}</span></div>
          <div class="progress"><i></i></div>
          <div class="ct-meta-row"><span>completed</span><span data-progress-label>0 / ${
            track.lessons.length
          }</span></div>
        </div>
      </div>
      <div class="lesson-rows">${lessonRows(track, present)}</div>
    </section>`;
    })
    .join('\n');

  const body = `
<div class="wrap">
  <section class="page-head">
    <p class="eyebrow">Curriculum</p>
    <h1>Every lesson, in order</h1>
    <p class="page-lede">
      ${totals.lessons} lessons across ${totals.tracks} tracks, roughly ${hours(
    totals.minutes
  )} of reading.
      Nothing here is locked. Every lesson carries questions you answer on the page, and clearing
      one is what marks it off — that state lives in this browser only.
    </p>
    <div class="overall" data-overall-progress-wrap>
      <div class="overall-row">
        <span class="overall-count"><b data-overall-progress data-all-lessons="${allIds}">0</b> of ${
    totals.lessons
  } complete</span>
        <span class="overall-pct" data-overall-pct>0%</span>
      </div>
      <div class="progress"><i data-overall-bar></i></div>
    </div>
  </section>

  ${pathMap(present, trialTracks)}

  ${sections}
</div>`;

  return layout({
    title: 'Curriculum',
    description: `All ${totals.lessons} Fanin lessons across ${totals.tracks} tracks, from linear algebra to distributed training. Free, no account required.`,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: '/curriculum/',
    nav: 'curriculum',
  });
}

function trackPage(track, present, hasTrial) {
  setPage(`/learn/${track.id}/`);
  const i = tracks.indexOf(track);
  const ids = track.lessons.map(([slug]) => `${track.id}/${slug}`).join(' ');
  const mins = track.lessons.reduce((s, [, , m]) => s + m, 0);
  const first = track.lessons[0];
  const prev = tracks[i - 1];
  const next = tracks[i + 1];

  const body = `
<div class="wrap">
  <section class="page-head" style="--track:${track.fill}">
    <div class="crumbs">
      <a href="${u('/curriculum/')}">Curriculum</a><span class="sep">/</span>
      <span>Track ${String(i + 1).padStart(2, '0')}</span>
    </div>
    <h1 class="ct-title">${trackDot(track)}${esc(track.title)}</h1>
    <p class="page-lede">${esc(track.blurb)}</p>
    <p class="track-outcome"><span class="label">By the end</span> ${esc(track.outcome)}</p>
    <div class="overall" data-track-progress data-track-lessons="${ids}">
      <div class="overall-row">
        <span class="overall-count">${track.lessons.length} lessons · ${hours(mins)}</span>
        <span class="overall-pct" data-progress-label>0 / ${track.lessons.length}</span>
      </div>
      <div class="progress"><i></i></div>
    </div>
    <div class="hero-actions">
      <a class="btn btn-primary" href="${u(`/learn/${track.id}/${first[0]}/`)}">Start ${String(
    i + 1
  )}.01 ${ICON.arrow}</a>
      ${hasTrial ? `<a class="btn" href="${u(`/learn/${track.id}/trial/`)}">Take the trial</a>` : ''}
    </div>
  </section>

  <div class="lesson-rows">${lessonRows(track, present)}</div>

  ${
    hasTrial
      ? `<a class="trial-card" href="${u(`/learn/${track.id}/trial/`)}"
             data-trial-card="${track.id}" style="--track:${track.fill}">
           <span class="tc-tag">Trial</span>
           <span class="tc-title">${esc(track.title)} trial</span>
           <span class="tc-body">Ten questions drawn from the whole track. Pass it to
             earn <b>${esc(track.badge)}</b>.</span>
           <span class="tc-state" data-trial-state>Not attempted</span>
         </a>`
      : ''
  }

  <div class="pager track-pager">
    ${
      prev
        ? `<a href="${u(`/learn/${prev.id}/`)}"><span class="dir">← Previous track</span><span class="ttl">${esc(
            prev.title
          )}</span></a>`
        : ''
    }
    ${
      next
        ? `<a class="next" href="${u(`/learn/${next.id}/`)}"><span class="dir">Next track →</span><span class="ttl">${esc(
            next.title
          )}</span></a>`
        : ''
    }
  </div>
</div>`;

  return layout({
    title: track.title,
    description: track.blurb,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: `/learn/${track.id}/`,
    nav: 'curriculum',
  });
}

function draftBody(lesson, track) {
  return `<p>
    This lesson is written but not yet published — it is being edited into the same shape
    as the rest of the track. Nothing about it will be paid when it lands.
  </p>
  <p>
    In the meantime, the lessons around it in
    <a href="${u(`/learn/${track.id}/`)}">${esc(track.title)}</a> are complete and readable.
  </p>`;
}

async function lessonPage(lesson, raw) {
  setPage(lesson.href);
  const track = byTrack.get(lesson.trackId);
  const i = allLessons.indexOf(lesson);
  const prev = lessonAt(i - 1);
  const next = lessonAt(i + 1);
  const headings = [];

  let summary = '';
  let prereqs = [];
  let seealso = [];
  let contentHtml;
  let plain = '';
  let isDraft = false;
  const checks = [];

  if (raw) {
    const { data, body } = parseFrontmatter(raw);
    const list = (v) => (Array.isArray(v) ? v : v ? [v] : []);
    summary = data.summary || '';
    prereqs = list(data.prereqs);
    seealso = list(data.seealso);
    contentHtml = render(body, {
      url: u,
      headings,
      checks,
      seed: lessonId(lesson),
      where: `${lesson.number} ${lesson.title}`,
    });
    plain = toPlainText(body);
  } else {
    isDraft = true;
    summary = 'This lesson is being edited. The rest of the track is complete.';
    contentHtml = draftBody(lesson, track);
  }

  const tocItems = headings
    .filter((h) => h.level <= 3)
    .map((h) => `<li class="lv${h.level}"><a href="#${h.id}">${h.html}</a></li>`)
    .join('\n');

  // "Leans on" points backwards to results this lesson uses; "continues in"
  // points forward, so a cross-track dependency is never mislabelled as a
  // prerequisite the reader was supposed to have read already.
  const linkList = (slugs) =>
    slugs
      .map((slug) => {
        const target = allLessons.find((l) => l.slug === slug);
        if (!target) return null;
        return `<a href="${u(target.href)}">${target.number} ${esc(target.title)}</a>`;
      })
      .filter(Boolean)
      .join('<span class="sep">·</span>');

  const prereqLinks = linkList(prereqs);
  const seealsoLinks = linkList(seealso);

  const nav = trackNav(track, lesson.slug);

  const body = `
<div class="wrap lesson-shell" style="--track:${track.fill};--track-ink:${track.inkLight}">
  ${nav}
  <template id="drawer-content">${nav}</template>

  <article class="lesson-main">
    <button class="drawer-trigger" type="button" id="drawer-trigger">
      ${trackDot(track)}<span>${esc(track.title)}</span>
      <span class="count">${lesson.indexInTrack + 1} / ${track.lessons.length}</span>
    </button>

    <div class="crumbs">
      <a href="${u('/curriculum/')}">Curriculum</a><span class="sep">/</span>
      <a href="${u(`/learn/${track.id}/`)}">${esc(track.short)}</a><span class="sep">/</span>
      <span>${lesson.number}</span>
    </div>

    <header class="lesson-head">
      <div class="lesson-kicker">
        <span class="lesson-num">${lesson.number}</span>
        <a class="lesson-track" href="${u(`/learn/${track.id}/`)}">${trackDot(track)}${esc(
    track.title
  )}</a>
        <span class="lesson-time">${lesson.minutes} min read</span>
      </div>
      <h1>${esc(lesson.title)}</h1>
      ${summary ? `<p class="lesson-summary">${esc(summary)}</p>` : ''}
      ${
        prereqLinks
          ? `<p class="lesson-prereq"><span class="label">Leans on</span>${prereqLinks}</p>`
          : ''
      }
      ${
        seealsoLinks
          ? `<p class="lesson-prereq lesson-seealso"><span class="label">Continues in</span>${seealsoLinks}</p>`
          : ''
      }
    </header>

    ${
      tocItems
        ? `<details class="toc-mobile"><summary>On this page</summary><ol>${tocItems}</ol></details>`
        : ''
    }

    <div class="prose">${contentHtml}</div>

    <footer class="lesson-foot">
      <div class="complete-row">
        <button class="complete-btn" type="button" id="complete-btn" data-lesson="${lessonId(
          lesson
        )}" data-minutes="${lesson.minutes}" data-checks="${checks.length}"
                aria-pressed="false"${checks.length ? ' aria-describedby="check-tally"' : ''}>
          <span class="box" aria-hidden="true"></span>
          <span class="label">${checks.length ? 'Clear this lesson' : 'Mark complete'}</span>
          <span class="xp-hint"></span>
        </button>
        ${
          checks.length
            ? `<span class="check-tally" id="check-tally" data-tally>
                 <span class="tally-n">0</span> of ${checks.length} answered
               </span>`
            : ''
        }
        <span class="complete-note">Saved in this browser only. Nothing is sent anywhere.</span>
      </div>

      <div class="suggest-row">
        <button class="suggest-btn" type="button" data-suggest
                data-lesson-number="${lesson.number}"
                data-lesson-title="${esc(lesson.title)}">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12.5 3.5l4 4L8 16H4v-4z"/><path d="M11 5l4 4"/>
          </svg>
          Found a problem with this lesson?
        </button>
        <span class="suggest-note">
          Select any passage to quote it, or open a request below.
        </span>
      </div>
      <nav class="pager" aria-label="Lesson navigation">
        ${
          prev
            ? `<a href="${u(prev.href)}"><span class="dir">← ${esc(prev.number)}</span><span class="ttl">${esc(
                prev.title
              )}</span></a>`
            : ''
        }
        ${
          next
            ? `<a class="next" href="${u(next.href)}"><span class="dir">${esc(
                next.number
              )} →</span><span class="ttl">${esc(next.title)}</span></a>`
            : ''
        }
      </nav>
    </footer>
  </article>

  ${
    tocItems
      ? `<aside class="toc-rail"><p class="spine-title">On this page</p><ol>${tocItems}</ol></aside>`
      : '<aside class="toc-rail"></aside>'
  }
</div>`;

  const html = layout({
    title: `${lesson.title} · ${track.short}`,
    description: summary || `${lesson.title} — part of the free ${site.name} AI curriculum.`,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: lesson.href,
    nav: 'curriculum',
    math: /class="tex"/.test(contentHtml) || /class="tex"/.test(tocItems),
  });

  return { html, plain, summary, isDraft, checks };
}

// The end-of-track trial: questions drawn from every lesson in the track, no
// explanations until the end, and a score you have to beat to earn the badge.
// The pool ships whole and the page picks from it, so a retry is a different
// set rather than the same one again.
const TRIAL_SIZE = 10;

function trialPage(track, pool) {
  setPage(`/learn/${track.id}/trial/`);
  const i = tracks.indexOf(track);
  const size = Math.min(TRIAL_SIZE, pool.length);
  const headings = [];

  const questions = pool
    .map((q, n) =>
      renderCheck(
        q.check,
        { url: u, headings, seed: `${track.id}#trial#${n}` },
        `${track.id}#trial#${n}`,
        { trial: true, tag: `Question`, from: q.number }
      )
    )
    .join('\n');

  const body = `
<div class="wrap">
  <section class="page-head" style="--track:${track.fill}">
    <div class="crumbs">
      <a href="${u('/curriculum/')}">Curriculum</a><span class="sep">/</span>
      <a href="${u(`/learn/${track.id}/`)}">${esc(track.short)}</a><span class="sep">/</span>
      <span>Trial</span>
    </div>
    <h1 class="ct-title">${trackDot(track)}${esc(track.title)} trial</h1>
    <p class="page-lede">
      ${size} questions drawn from the ${track.lessons.length} lessons in this track.
      No explanations until the end. Score ${Math.ceil(size * 0.8)} or better to earn
      <strong>${esc(track.badge)}</strong>.
    </p>
  </section>

  <section class="trial" data-trial="${track.id}" data-size="${size}"
           data-track-lessons="${track.lessons.map(([s]) => `${track.id}/${s}`).join(' ')}"
           data-badge="${esc(track.badge)}">
    <div class="trial-intro" data-intro>
      <div class="trial-stat-row">
        <div class="trial-stat"><b>${size}</b><span>questions</span></div>
        <div class="trial-stat"><b>${Math.ceil(size * 0.8)}</b><span>to pass</span></div>
        <div class="trial-stat"><b data-trial-best>—</b><span>your best</span></div>
      </div>
      <p class="trial-warn" data-warn hidden>
        You have not cleared every lesson in this track yet. You can still take the
        trial — it just draws on lessons you may not have read.
      </p>
      <button class="btn btn-primary btn-lg" type="button" data-start>Begin the trial</button>
      <p class="trial-note">Nothing is timed. Nothing is sent anywhere.</p>
    </div>

    <div class="trial-run" data-run hidden>
      <div class="trial-progress">
        <p class="trial-count">Question <b data-at>1</b> of ${size}</p>
        <div class="trial-bar"><i data-bar style="width:0%"></i></div>
      </div>
      <div class="trial-pool" data-pool>${questions}</div>
      <div class="trial-controls">
        <button class="btn btn-primary" type="button" data-next disabled>Next question</button>
        <span class="trial-skip-note" data-picked>Pick an answer to continue.</span>
      </div>
    </div>

    <div class="trial-done" data-done hidden>
      <div class="trial-score">
        <p class="trial-verdict" data-verdict></p>
        <p class="trial-tally"><b data-score>0</b> <span>of ${size} right</span></p>
      </div>
      <div class="trial-review" data-review></div>
      <div class="trial-again">
        <button class="btn btn-primary" type="button" data-retry>Take it again</button>
        <a class="btn" href="${u(`/learn/${track.id}/`)}">Back to ${esc(track.short)}</a>
      </div>
    </div>

    <noscript>
      <p class="trial-note">The trial needs JavaScript. Every question in it also
      appears in the lesson it came from, which works without.</p>
    </noscript>
  </section>

  <nav class="pager" aria-label="Track navigation">
    <a href="${u(`/learn/${track.id}/`)}"><span class="dir">← Track ${String(i + 1).padStart(2, '0')}</span><span class="ttl">${esc(track.title)}</span></a>
  </nav>
</div>`;

  return layout({
    title: `${track.short} trial`,
    description: `A ${size}-question trial over the ${track.lessons.length} lessons of ${track.title}. Free, no account.`,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: `/learn/${track.id}/trial/`,
    nav: 'curriculum',
    math: /class="tex"/.test(questions),
  });
}

function aboutPage() {
  setPage('/about/');
  const body = `
<div class="wrap narrow">
  <section class="page-head">
    <p class="eyebrow">About</p>
    <h1>What this is, and what it is not</h1>
  </section>
  <div class="prose">
    <p>
      ${site.name} is a written curriculum in machine learning research, covering the path from
      vector norms to sharded training runs in ${totals.lessons} lessons. It exists because the
      good versions of this material are usually split across three places: a paid course that
      stops where it gets interesting, a textbook that assumes a maths degree, and a pile of
      papers with no through-line.
    </p>

    <h2>The rules this was written under</h2>
    <ul>
      <li><strong>Everything is free.</strong> Not a free tier. There is no tier.</li>
      <li><strong>No account, ever.</strong> Nothing asks who you are. There is no login to build.</li>
      <li><strong>No tracking.</strong> No analytics script, no pixel, no cookie, and no request
      to any third party — the fonts and the maths renderer are served from this site, not a CDN.</li>
      <li><strong>Your progress is yours alone.</strong> Lessons completed, XP, streak and badges are
      all derived from one record in your browser's local storage. It never leaves the device, so
      there is no leaderboard, no profile and nothing to compare. The scoreboard is private by
      construction.</li>
      <li><strong>Dependency order, not difficulty order.</strong> A lesson appears after the
      lessons whose results it uses, and it says at the top which those are.</li>
      <li><strong>Maths is shown, not waved at.</strong> Where a derivation matters, it is done.</li>
      <li><strong>Code runs.</strong> Snippets are written to be pasted into a file, not read past.</li>
    </ul>

    <h2>How to use it</h2>
    <p>
      If you are starting from scratch, read in order — the tracks are numbered for a reason, and
      track 1 takes about eight hours. If you are filling gaps, use search (<code>⌘K</code> or
      <code>/</code>) and follow the <em>Leans on</em> links at the top of each lesson backwards
      until you hit something you already know.
    </p>
    <p>
      Every lesson has questions in it — a few per page, placed at the end of the section they
      test. Answer one and you are told immediately whether you are right and, for every option,
      <em>why</em>. A wrong answer explains itself and leaves the question open, so nobody is ever
      stuck behind one; it simply pays less. Clearing all of a lesson's questions is what marks
      the lesson off.
    </p>
    <p>
      Answers earn XP, XP earns levels, and finishing a track means clearing its lessons
      <em>and</em> passing its trial — ten questions drawn from across the track with no
      explanations until the end. The <a href="${u('/curriculum/')}">curriculum index</a> shows
      the whole path at a glance and <a href="${u('/progress/')}">your progress page</a> has the
      rest. All of it is derived from one record in this browser. Clearing your browser data
      clears it; there is no server copy, by design.
    </p>
    <p>
      The answers ship inside the page, base64-encoded, because there is no server to check them
      against. That is enough to stop a stray <code>⌘F</code> spoiling a question and it is not
      pretending to be more — if you want to read them, you can, and the explanations are the
      part worth reading anyway.
    </p>

    <h2>Corrections and contributions</h2>
    <p>
      Every lesson carries a <strong>Found a problem with this lesson?</strong> control at the
      bottom, and selecting any passage offers to quote it. Both open a prefilled issue on GitHub
      for a <strong>correction</strong> (something is wrong), an <strong>addition</strong>
      (something is missing) or a <strong>deletion</strong> (something should go). The page posts
      nothing itself — it just saves you describing where the problem is.
    </p>
    <p>
      The whole site — content, generator, styles — is one public repository. Lessons are plain
      Markdown files under <code>content/lessons/</code>, and the build is a single Node script
      with no dependencies. Pull requests are welcome; <code>node src/check.mjs</code> will tell
      you if a link or a lesson dependency is wrong before CI does.
    </p>
    <p>
      Lesson text is licensed <strong>${site.license}</strong>: use it, translate it, teach from
      it, print it. The site code is MIT. If this project ever stops being maintained, the content
      is already yours.
    </p>

    <h2>What this is not</h2>
    <p>
      It is not a video course, a certificate programme, or a replacement for actually training
      models. Reading track 4 will teach you what every line of a transformer does; it will not
      substitute for the afternoon you spend finding out that your attention mask was transposed.
    </p>
  </div>
</div>`;

  return layout({
    title: 'About',
    description: `Why ${site.name} is free, how it is built, and how to use it.`,
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: '/about/',
    nav: 'about',
  });
}

function progressPage() {
  setPage('/progress/');
  const allIds = allLessons.map(lessonId).join(' ');

  // One ring per track. The stroke uses the contrast-safe ink variant, not the
  // raw viridis fill -- at the light end of the ramp the fill drops to 1.6:1
  // against white, which is unreadable for a thin mark.
  const rings = tracks
    .map((track, i) => {
      const ids = track.lessons.map(([slug]) => `${track.id}/${slug}`).join(' ');
      return `<a class="ring-card" href="${u(`/learn/${track.id}/`)}"
         data-track-progress data-track-lessons="${ids}"
         style="--track:${track.fill};--track-ink:${track.inkLight};--track-ink-dark:${track.inkDark}">
        <svg class="ring" viewBox="0 0 64 64" aria-hidden="true">
          <circle class="ring-bg" cx="32" cy="32" r="26" />
          <circle class="ring-fill" cx="32" cy="32" r="26" data-ring />
        </svg>
        <span class="ring-meta">
          <span class="ring-track">${trackDot(track)}${esc(track.short)}</span>
          <span class="ring-count" data-progress-label>0 / ${track.lessons.length}</span>
        </span>
        <span class="ring-pct" data-ring-pct>0%</span>
      </a>`;
    })
    .join('\n');

  const body = `
<div class="wrap">
  <section class="page-head">
    <p class="eyebrow">Progress</p>
    <h1>Where you are</h1>
    <p class="page-lede">
      Everything here is computed from your own browser's local storage and stays there.
      No account, no sync, nothing sent anywhere — so a new device starts from zero.
    </p>
  </section>

  <section class="level-card">
    <div class="level-head">
      <div>
        <p class="level-eyebrow">Level <span data-level-num>1</span></p>
        <h2 class="level-name" data-level-name>Randomly Initialized</h2>
      </div>
      <p class="level-xp"><b data-xp>0</b> XP</p>
    </div>
    <div class="progress level-bar"><i data-level-bar></i></div>
    <p class="level-next" data-next-level>600 XP to First Backward Pass</p>
  </section>

  <div class="stat-row">
    <div class="stat-tile">
      <span class="stat-num" data-overall-progress data-all-lessons="${allIds}">0</span>
      <span class="stat-cap">of ${totals.lessons} lessons</span>
    </div>
    <div class="stat-tile">
      <span class="stat-num" data-streak>0</span>
      <span class="stat-cap">day streak</span>
    </div>
    <div class="stat-tile">
      <span class="stat-num" data-longest-streak>0</span>
      <span class="stat-cap">longest streak</span>
    </div>
    <div class="stat-tile">
      <span class="stat-num" data-accuracy>&mdash;</span>
      <span class="stat-cap">right first try</span>
    </div>
    <div class="stat-tile">
      <span class="stat-num" data-badge-count>0</span>
      <span class="stat-cap">badges</span>
    </div>
  </div>

  <section class="panel quests-panel">
    <div class="panel-head">
      <h2>Today</h2>
      <p class="panel-sub"><span data-quests-done>0</span> of 3 done</p>
    </div>
    <ul class="quests" data-quests>
      <li class="quest quest-empty">Three goals, picked fresh each day.</li>
    </ul>
    <p class="panel-note">
      Quests reset at midnight in your own timezone. Nothing is timed and
      nothing is lost by skipping a day &mdash; only the streak.
    </p>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Activity</h2>
      <p class="cal-legend">
        Less
        <span class="cal-cell" data-step="1"></span>
        <span class="cal-cell" data-step="2"></span>
        <span class="cal-cell" data-step="3"></span>
        <span class="cal-cell" data-step="4"></span>
        More
      </p>
    </div>
    <div class="cal" id="activity-grid" data-weeks="18"></div>
    <p class="panel-note">
      Lessons completed per day over the last eighteen weeks. Days before you started
      show as empty.
    </p>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Tracks</h2></div>
    <div class="ring-grid">${rings}</div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Badges</h2>
      <p class="panel-sub"><span data-badge-count>0</span> earned</p>
    </div>
    <ul class="badge-grid" id="badge-grid"></ul>
  </section>

  <p class="progress-actions">
    <a class="btn btn-primary" href="${u('/curriculum/')}">Back to the curriculum</a>
    <button class="btn btn-ghost" type="button" id="reset-progress">Clear progress</button>
  </p>
</div>`;

  return layout({
    title: 'Progress',
    description: 'Your XP, streak, badges and per-track progress, stored locally in your browser.',
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: '/progress/',
    nav: 'progress',
  });
}

function notFoundPage() {
  setPage('/');
  const body = `
<div class="wrap narrow">
  <section class="page-head">
    <p class="eyebrow">404</p>
    <h1>That page is not here</h1>
    <p class="page-lede">
      The link may be old, or the lesson may have been renamed. Everything that exists is
      listed on the curriculum index — and all of it is free, so nothing is missing because
      you have not paid for it.
    </p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="${u('/curriculum/')}">All ${totals.lessons} lessons</a>
      <button class="btn btn-ghost" type="button" data-search-open>Search the curriculum</button>
    </div>
  </section>
</div>`;

  return layout({
    title: 'Not found',
    description: 'That page does not exist.',
    body,
    base: RELATIVE ? u('/').replace(/index\.html$/, '') : BASE,
    canonical: '/404',
  });
}

const FAVICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    ${tracks
      .map((t, i) => `<stop offset="${((i / (tracks.length - 1)) * 100).toFixed(0)}%" stop-color="${t.fill}"/>`)
      .join('')}
  </linearGradient></defs>
  <rect width="32" height="32" rx="6" fill="url(#g)"/>
</svg>`;

// ------------------------------------------------------------------- main --

async function readLesson(lesson) {
  const file = path.join(LESSON_DIR, lesson.trackId, `${lesson.slug}.md`);
  try {
    return await fs.readFile(file, 'utf8');
  } catch {
    return null;
  }
}

async function copyDir(from, to) {
  await fs.mkdir(to, { recursive: true });
  for (const entry of await fs.readdir(from, { withFileTypes: true })) {
    const src = path.join(from, entry.name);
    const dst = path.join(to, entry.name);
    if (entry.isDirectory()) await copyDir(src, dst);
    else await fs.copyFile(src, dst);
  }
}

async function build() {
  const started = Date.now();
  await fs.rm(OUT, { recursive: true, force: true });
  await fs.mkdir(OUT, { recursive: true });

  const sources = new Map();
  for (const lesson of allLessons) sources.set(lessonId(lesson), await readLesson(lesson));
  const present = new Set([...sources.entries()].filter(([, v]) => v).map(([k]) => k));

  const index = [];
  const pools = new Map(tracks.map((t) => [t.id, []]));
  let drafts = 0;
  let checkCount = 0;

  for (const lesson of allLessons) {
    const raw = sources.get(lessonId(lesson));
    const { html, plain, summary, isDraft, checks } = await lessonPage(lesson, raw);
    await write(path.join('learn', lesson.trackId, lesson.slug, 'index.html'), html);
    if (isDraft) drafts++;
    checkCount += checks.length;
    for (const check of checks) {
      pools.get(lesson.trackId).push({ number: lesson.number, check });
    }
    index.push({
      n: lesson.number,
      t: lesson.title,
      s: summary,
      b: plain.slice(0, 1400),
      k: lesson.trackTitle,
      u: lesson.href,
      ...(isDraft ? { d: 1 } : {}),
    });
  }

  for (const track of tracks) {
    const pool = pools.get(track.id);
    const hasTrial = pool.length >= TRIAL_SIZE;
    await write(path.join('learn', track.id, 'index.html'), trackPage(track, present, hasTrial));
    // A trial needs enough questions to be a trial. Below that the track has
    // no trial page and nothing links to one.
    if (hasTrial) {
      await write(path.join('learn', track.id, 'trial', 'index.html'), trialPage(track, pool));
    }
  }

  await write('index.html', landingPage(present));
  const trialTracks = new Set(
    tracks.filter((t) => (pools.get(t.id) || []).length >= TRIAL_SIZE).map((t) => t.id)
  );
  await write(path.join('curriculum', 'index.html'), curriculumPage(present, trialTracks));
  await write(path.join('about', 'index.html'), aboutPage());
  await write(path.join('progress', 'index.html'), progressPage());
  await write('404.html', notFoundPage());
  await write('search-index.json', JSON.stringify(index));
  await write('favicon.svg', FAVICON);
  await write('.nojekyll', '');

  const origin = 'https://gauransh18.github.io' + BASE;
  const urls = [
    '/',
    '/curriculum/',
    '/about/',
    '/progress/',
    ...tracks.map((t) => `/learn/${t.id}/`),
    ...tracks
      .filter((t) => (pools.get(t.id) || []).length >= TRIAL_SIZE)
      .map((t) => `/learn/${t.id}/trial/`),
    ...allLessons.map((l) => l.href),
  ];
  await write(
    'sitemap.xml',
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
      urls.map((p) => `  <url><loc>${origin}${p}</loc></url>`).join('\n') +
      `\n</urlset>\n`
  );
  await write('robots.txt', `User-agent: *\nAllow: /\nSitemap: ${origin}/sitemap.xml\n`);

  await copyDir(path.join(HERE, 'assets'), path.join(OUT, 'assets'));

  const trialCount = tracks.filter((t) => (pools.get(t.id) || []).length >= TRIAL_SIZE).length;
  const written = allLessons.length + tracks.length + trialCount + 5;
  console.log(
    `built ${written} pages in ${Date.now() - started}ms  ·  ` +
      `${allLessons.length - drafts}/${allLessons.length} lessons written` +
      (drafts ? `  ·  ${plural(drafts, 'draft')} remaining` : '  ·  complete') +
      `  ·  ${plural(checkCount, 'check')}` +
      `  ·  ${plural(trialCount, 'trial')}` +
      (BASE ? `  ·  base ${BASE}` : '')
  );
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
