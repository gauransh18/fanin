// Site integrity check: every internal link resolves, every prereq exists,
// every lesson has frontmatter, and no page 404s against its own output.
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { tracks, allLessons } from '../content/curriculum.mjs';
import { parseFrontmatter, parseCheck } from './markdown.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const errors = [];
const warnings = [];
const checkCounts = new Map();

async function walk(dir) {
  const out = [];
  for (const e of await fs.readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...(await walk(p)));
    else out.push(p);
  }
  return out;
}

// --- 1. Every manifest lesson has a Markdown file with required frontmatter ---
const slugs = new Set(allLessons.map((l) => l.slug));
for (const lesson of allLessons) {
  const file = path.join(ROOT, 'content', 'lessons', lesson.trackId, `${lesson.slug}.md`);
  let raw;
  try {
    raw = await fs.readFile(file, 'utf8');
  } catch {
    errors.push(`missing lesson file: ${lesson.trackId}/${lesson.slug}.md`);
    continue;
  }
  const { data, body } = parseFrontmatter(raw);
  if (!data.summary) errors.push(`${lesson.slug}: no summary in frontmatter`);
  if (body.trim().length < 500) warnings.push(`${lesson.slug}: body under 500 chars`);

  // Every ::: check block must parse, and a lesson is expected to have some.
  const blocks = body.match(/^:::[ \t]*check[ \t]*$[\s\S]*?^:::[ \t]*$/gm) || [];
  for (const [n, block] of blocks.entries()) {
    const inner = block.split('\n').slice(1, -1).join('\n');
    try {
      parseCheck(inner);
    } catch (err) {
      errors.push(`${lesson.slug}: check ${n + 1}: ${err.message}`);
    }
  }
  checkCounts.set(lesson.slug, blocks.length);

  // A wrapped line whose next line begins "1. " turns that sentence into a
  // one-item ordered list. It happens silently and only shows up in the render,
  // so it is caught here. A list opening at 2 or higher after prose is a
  // genuine continuation and is left alone.
  {
    const lines = body.split('\n');
    for (let i = 1; i < lines.length; i++) {
      if (!/^1\.\s+\S/.test(lines[i])) continue;
      const prev = lines[i - 1];
      if (!prev.trim()) continue;
      if (/^\s*(\d+\.|[-*>]|\||:::|```)/.test(prev)) continue;
      errors.push(
        `${lesson.slug}:${i + 1}: "${lines[i].slice(0, 40)}" reads as a list item ` +
        `because the line above wraps — reflow it`
      );
    }
  }

  for (const p of [].concat(data.prereqs || [])) {
    if (!slugs.has(p)) errors.push(`${lesson.slug}: prereq "${p}" is not a lesson slug`);
    if (p === lesson.slug) errors.push(`${lesson.slug}: lists itself as a prereq`);
  }

  // A prereq must come EARLIER; a `seealso` must come LATER. Getting either
  // backwards means the page is telling the reader something untrue about
  // what they were meant to have read.
  for (const p of [].concat(data.prereqs || [])) {
    const target = allLessons.find((l) => l.slug === p);
    if (target && allLessons.indexOf(target) > allLessons.indexOf(lesson)) {
      errors.push(`${lesson.slug} (${lesson.number}) lists prereq ${p} (${target.number}), which comes later — use seealso`);
    }
  }
  for (const p of [].concat(data.seealso || [])) {
    if (!slugs.has(p)) errors.push(`${lesson.slug}: seealso "${p}" is not a lesson slug`);
    const target = allLessons.find((l) => l.slug === p);
    if (target && allLessons.indexOf(target) < allLessons.indexOf(lesson)) {
      errors.push(`${lesson.slug} (${lesson.number}) lists seealso ${p} (${target.number}), which comes earlier — use prereqs`);
    }
  }

  // Unclosed callouts silently swallow the rest of a lesson.
  const opens = (body.match(/^:::\s*[a-z]+/gm) || []).length;
  const closes = (body.match(/^:::\s*$/gm) || []).length;
  if (opens !== closes) errors.push(`${lesson.slug}: ${opens} callout opens, ${closes} closes`);
}

// --- 2. Every internal href in the output resolves to a real file ---
const files = (await walk(DIST)).filter((f) => f.endsWith('.html'));
const pages = new Set(files.map((f) => '/' + path.relative(DIST, f).replace(/\\/g, '/')));

let linkCount = 0;
for (const file of files) {
  const html = await fs.readFile(file, 'utf8');
  const from = '/' + path.relative(DIST, file).replace(/\\/g, '/');
  for (const m of html.matchAll(/href="([^"]+)"/g)) {
    const href = m[1];
    if (/^(https?:|mailto:|#)/.test(href)) continue;
    linkCount++;
    const clean = href.split('#')[0];
    if (!clean) continue;
    const target = clean.endsWith('/') ? clean + 'index.html' : clean;
    const exists = pages.has(target) ||
      await fs.access(path.join(DIST, target)).then(() => true).catch(() => false);
    if (!exists) errors.push(`${from} -> broken link ${href}`);
  }
}

// --- 3. The search index covers every lesson and carries real content ---
const index = JSON.parse(await fs.readFile(path.join(DIST, 'search-index.json'), 'utf8'));
if (index.length !== allLessons.length) {
  errors.push(`search index has ${index.length} entries, expected ${allLessons.length}`);
}
for (const entry of index) {
  if (!entry.s) errors.push(`search index: ${entry.t} has no summary`);
  if (!entry.b || entry.b.length < 200) errors.push(`search index: ${entry.t} has thin body text`);
  if (entry.d) errors.push(`search index: ${entry.t} is still marked draft`);
}

// --- 4. Report ---
const totalChecks = [...checkCounts.values()].reduce((a, b) => a + b, 0);
const withoutChecks = [...checkCounts.entries()].filter(([, n]) => n === 0);
if (withoutChecks.length) {
  warnings.push(
    `${withoutChecks.length} lesson(s) have no checks: ` +
    withoutChecks.slice(0, 6).map(([slug]) => slug).join(', ') +
    (withoutChecks.length > 6 ? ', …' : '')
  );
}

const stats = {
  tracks: tracks.length,
  lessons: allLessons.length,
  pages: files.length,
  internalLinks: linkCount,
  checks: totalChecks,
};
console.log(Object.entries(stats).map(([k, v]) => `${k}: ${v}`).join('  ·  '));
if (warnings.length) {
  console.log(`\n${warnings.length} warning(s):`);
  warnings.slice(0, 10).forEach((w) => console.log('  ! ' + w));
}
if (errors.length) {
  console.log(`\n${errors.length} error(s):`);
  errors.slice(0, 40).forEach((e) => console.log('  x ' + e));
  process.exit(1);
}
console.log('\nall checks passed');
