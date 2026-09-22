// A small, predictable Markdown subset for lesson content.
// Supports: frontmatter, headings, paragraphs, fenced code, ordered/unordered
// lists (nested), blockquotes, tables, rules, callout blocks, and TeX math.
//
// Math is never touched by the inline pass. It is extracted first, replaced by
// a placeholder, and emitted inside an element KaTeX renders on the client.

import { highlight } from './highlight.mjs';

const escHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
   .replace(/"/g, '&quot;');

export function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[`*_$\\]/g, '')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-');
}

export function parseFrontmatter(raw) {
  const text = raw.replace(/^﻿/, '');
  if (!text.startsWith('---')) return { data: {}, body: text };
  const end = text.indexOf('\n---', 3);
  if (end === -1) return { data: {}, body: text };
  const head = text.slice(3, end).trim();
  const body = text.slice(text.indexOf('\n', end + 1) + 1);
  const data = {};
  for (const line of head.split('\n')) {
    const m = /^([A-Za-z_][\w-]*)\s*:\s*(.*)$/.exec(line.trim());
    if (!m) continue;
    let value = m[2].trim();
    if (/^\[.*\]$/.test(value)) {
      value = value
        .slice(1, -1)
        .split(',')
        .map((s) => s.trim().replace(/^["']|["']$/g, ''))
        .filter(Boolean);
    } else {
      value = value.replace(/^["']|["']$/g, '');
    }
    data[m[1]] = value;
  }
  return { data, body };
}

const CALLOUTS = {
  note: 'Note',
  tip: 'Tip',
  warning: 'Watch out',
  insight: 'The idea',
  key: 'Key result',
  exercise: 'Exercise',
  solution: 'Show solution',
  history: 'Where this came from',
};


// ----------------------------------------------------------------- checks ---
// A check is a question the reader answers on the page. There is no server, so
// the answer ships with the page. It travels base64'd, which is enough to stop
// a stray ctrl-F spoiling it and is not pretending to be anything more.

const encodeKey = (s) => Buffer.from('fanin:' + s, 'utf8').toString('base64');

// Deterministic shuffle: the correct option is not left wherever the author
// happened to write it, but stays in the same place across builds so a reader
// who reloads does not see the options move.
function shuffleOrder(n, seed) {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    h = Math.imul(h ^ (h >>> 15), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    const j = (h >>> 0) % (i + 1);
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  return order;
}

// Parses the body of a ::: check block into a plain object, or throws with a
// message naming what is wrong. The build turns that into a failed build
// rather than a page with a broken question on it.
export function parseCheck(body) {
  const lines = body.replace(/\r\n/g, '\n').split('\n');
  const prompt = [];
  const options = [];
  let numeric = null;
  let cursor = null; // where a following "> ..." explanation attaches

  for (const line of lines) {
    const opt = /^\s*-\s*\[([ xX])\]\s+(.*)$/.exec(line);
    if (opt) {
      cursor = { text: opt[2].trim(), correct: opt[1].toLowerCase() === 'x', why: [] };
      options.push(cursor);
      continue;
    }
    const num = /^\s*=\s*(-?[\d.eE+-]+)\s*(?:(?:\+\/-|±)\s*([\d.eE+-]+))?\s*$/.exec(line);
    if (num) {
      numeric = { value: Number(num[1]), tol: num[2] ? Number(num[2]) : 0, why: [] };
      cursor = numeric;
      continue;
    }
    const why = /^\s*>\s?(.*)$/.exec(line);
    if (why && cursor) { cursor.why.push(why[1]); continue; }
    if (options.length || numeric) {
      if (line.trim()) throw new Error(`stray line inside a check: ${line.trim()}`);
      continue;
    }
    prompt.push(line);
  }

  if (!prompt.join('').trim()) throw new Error('check has no question');
  if (numeric && options.length) throw new Error('check mixes a numeric answer with options');

  if (numeric) {
    if (!Number.isFinite(numeric.value)) throw new Error('numeric check has no finite answer');
    if (!numeric.why.join('').trim()) throw new Error('numeric check has no explanation');
    return { kind: 'numeric', prompt: prompt.join('\n').trim(), numeric };
  }

  if (options.length < 2) throw new Error('check needs at least two options');
  const right = options.filter((o) => o.correct).length;
  if (!right) throw new Error('check has no correct option');
  options.forEach((o, i) => {
    if (!o.why.join('').trim()) throw new Error(`option ${i + 1} has no explanation`);
  });
  return { kind: right > 1 ? 'multi' : 'choice', prompt: prompt.join('\n').trim(), options };
}

function checkBlock(body, ctx, seed) {
  let parsed;
  try {
    parsed = parseCheck(body);
  } catch (err) {
    throw new Error(`${ctx.where || 'check'}: ${err.message}`);
  }
  if (ctx.checks) ctx.checks.push(parsed);
  const promptHtml = render(parsed.prompt, ctx);

  if (parsed.kind === 'numeric') {
    const n = parsed.numeric;
    return (
      `<form class="check" data-kind="numeric" data-k="${encodeKey(n.value + '|' + n.tol)}">` +
      `<p class="check-tag">Your turn</p>` +
      `<div class="check-q">${promptHtml}</div>` +
      `<div class="check-num">` +
      `<input type="text" inputmode="decimal" autocomplete="off" spellcheck="false"` +
      ` aria-label="Your answer" placeholder="Your answer">` +
      `<button class="btn btn-primary" type="submit">Check</button>` +
      `</div>` +
      `<p class="opt-why" data-why>${inline(n.why.join(' ').trim(), ctx)}</p>` +
      `<p class="check-result" role="status" aria-live="polite"></p>` +
      `</form>`
    );
  }

  const order = shuffleOrder(parsed.options.length, seed);
  const shown = order.map((i) => parsed.options[i]);
  const key = shown.map((o, i) => (o.correct ? i : -1)).filter((i) => i >= 0).join(',');

  const opts = shown
    .map(
      (o, i) =>
        `<li class="opt-row"><button class="opt" type="button" data-i="${i}">` +
        `<span class="opt-mark" aria-hidden="true"></span>` +
        `<span class="opt-text">${inline(o.text, ctx)}</span></button>` +
        `<p class="opt-why" data-why>${inline(o.why.join(' ').trim(), ctx)}</p></li>`
    )
    .join('');

  return (
    `<div class="check" data-kind="${parsed.kind}" data-k="${encodeKey(key)}">` +
    `<p class="check-tag">Your turn${
      parsed.kind === 'multi' ? ' <span class="check-hint">· select all that apply</span>' : ''
    }</p>` +
    `<div class="check-q">${promptHtml}</div>` +
    `<ul class="check-opts">${opts}</ul>` +
    (parsed.kind === 'multi'
      ? `<div class="check-actions"><button class="btn btn-primary" type="button" data-submit>Check answer</button></div>`
      : '') +
    `<p class="check-result" role="status" aria-live="polite"></p>` +
    `</div>`
  );
}

// ---------------------------------------------------------------- inline ----

function inline(src, ctx) {
  let s = src;

  // Protect inline code and math from every other transform.
  const shelf = [];
  const park = (html) => {
    shelf.push(html);
    return `\u0000${shelf.length - 1}\u0000`;
  };

  s = s.replace(/`([^`\n]+)`/g, (_, code) => park(`<code>${escHtml(code)}</code>`));
  s = s.replace(/\$([^$\n]+?)\$/g, (_, tex) =>
    park(`<span class="tex" data-display="0">${escHtml(tex)}</span>`)
  );

  s = escHtml(s);

  // Links, then images are not used in lesson prose — links only.
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, text, href) => {
    const external = /^https?:/.test(href);
    const url = external ? href : ctx.url(href);
    const attrs = external ? ' target="_blank" rel="noopener noreferrer"' : '';
    return `<a href="${url}"${attrs}>${text}</a>`;
  });

  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  s = s.replace(/(^|[\s(])_([^_\n]+)_/g, '$1<em>$2</em>');
  s = s.replace(/ -- /g, ' — ');

  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => shelf[Number(i)]);
}

// ----------------------------------------------------------------- blocks ---

function codeBlock(lang, code) {
  const label = lang || 'text';
  const body = highlight(code.replace(/\n$/, ''), label);
  return (
    `<figure class="code" data-lang="${escHtml(label)}">` +
    `<figcaption><span class="code-lang">${escHtml(label)}</span>` +
    `<button class="code-copy" type="button" data-copy>Copy</button></figcaption>` +
    `<pre><code>${body}</code></pre></figure>`
  );
}

function tableBlock(lines, ctx) {
  const cells = (row) =>
    row.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  const header = cells(lines[0]);
  const aligns = cells(lines[1]).map((spec) => {
    const left = spec.startsWith(':');
    const right = spec.endsWith(':');
    if (left && right) return 'center';
    if (right) return 'right';
    return 'left';
  });
  const body = lines.slice(2).map(cells);

  const th = header
    .map((c, i) => `<th style="text-align:${aligns[i] || 'left'}">${inline(c, ctx)}</th>`)
    .join('');
  const rows = body
    .map(
      (row) =>
        '<tr>' +
        row
          .map(
            (c, i) =>
              `<td style="text-align:${aligns[i] || 'left'}">${inline(c, ctx)}</td>`
          )
          .join('') +
        '</tr>'
    )
    .join('');

  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

const indentOf = (line) => /^\s*/.exec(line)[0].replace(/\t/g, '    ').length;
const BULLET = /^\s*[-*]\s+(.*)$/;
const NUMBER = /^\s*(\d+)[.)]\s+(.*)$/;

function listBlock(lines, ctx) {
  const ordered = NUMBER.test(lines[0]);
  const base = indentOf(lines[0]);
  const items = [];

  for (const line of lines) {
    const m = ordered ? NUMBER.exec(line) : BULLET.exec(line);
    if (m && indentOf(line) <= base) {
      items.push({ text: ordered ? m[2] : m[1], children: [] });
    } else if (items.length) {
      items[items.length - 1].children.push(line.slice(Math.min(indentOf(line), base + 2)));
    }
  }

  const html = items
    .map((item) => {
      const nested = item.children.filter((l) => l.trim().length);
      const inner = nested.length ? render(nested.join('\n'), ctx) : '';
      return `<li>${inline(item.text, ctx)}${inner}</li>`;
    })
    .join('');

  const start = ordered ? Number(NUMBER.exec(lines[0])[1]) : 1;
  return ordered
    ? `<ol${start !== 1 ? ` start="${start}"` : ''}>${html}</ol>`
    : `<ul>${html}</ul>`;
}

export function render(markdown, ctx = {}) {
  const context = {
    url: ctx.url || ((h) => h),
    headings: ctx.headings || [],
    // Shared across nested renders so every check on a page gets its own
    // shuffle seed, and the build can report which one failed.
    seed: ctx.seed || 'fanin',
    where: ctx.where || '',
    counter: ctx.counter || { n: 0 },
    checks: ctx.checks || [],
  };
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i++; continue; }

    // Fenced code ------------------------------------------------------
    const fence = /^\s*```+\s*([\w+-]*)\s*$/.exec(line);
    if (fence) {
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```+\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(codeBlock(fence[1], body.join('\n')));
      continue;
    }

    // Display math -----------------------------------------------------
    if (/^\s*\$\$\s*$/.test(line)) {
      const body = [];
      i++;
      while (i < lines.length && !/^\s*\$\$\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(
        `<div class="tex-display"><span class="tex" data-display="1">${escHtml(
          body.join('\n').trim()
        )}</span></div>`
      );
      continue;
    }

    // Callouts ---------------------------------------------------------
    const callout = /^:::\s*([a-z]+)\s*(.*)$/.exec(line);
    if (callout) {
      const kind = callout[1];
      const title = callout[2].trim();
      const body = [];
      i++;
      let depth = 1;
      while (i < lines.length) {
        if (/^:::\s*[a-z]+/.test(lines[i])) depth++;
        else if (/^:::\s*$/.test(lines[i])) { depth--; if (!depth) break; }
        body.push(lines[i++]);
      }
      i++;
      if (kind === 'check') {
        const n = context.counter.n++;
        out.push(
          checkBlock(body.join('\n'), { ...context, where: `${context.where} check ${n + 1}`.trim() },
                     context.seed + '#' + n)
        );
        continue;
      }
      const inner = render(body.join('\n'), context);
      const label = title || CALLOUTS[kind] || kind;
      if (kind === 'solution') {
        out.push(
          `<details class="callout callout-solution"><summary>${escHtml(label)}</summary>` +
          `<div class="callout-body">${inner}</div></details>`
        );
      } else {
        out.push(
          `<aside class="callout callout-${escHtml(kind)}">` +
          `<p class="callout-label">${escHtml(label)}</p>` +
          `<div class="callout-body">${inner}</div></aside>`
        );
      }
      continue;
    }

    // Headings ---------------------------------------------------------
    const heading = /^(#{2,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      // Keep the inline-rendered form too: a heading may contain maths or code,
      // and the table of contents has to show it the same way the heading does.
      const rendered = inline(text, context);
      context.headings.push({ level, text, id, html: rendered });
      out.push(
        `<h${level} id="${id}">${rendered}` +
        `<a class="anchor" href="#${id}" aria-label="Link to this section">#</a></h${level}>`
      );
      i++;
      continue;
    }

    // Horizontal rule --------------------------------------------------
    if (/^\s*(---|\*\*\*)\s*$/.test(line)) { out.push('<hr />'); i++; continue; }

    // Table -------------------------------------------------------------
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|/.test(lines[i + 1])) {
      const block = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) block.push(lines[i++]);
      out.push(tableBlock(block, context));
      continue;
    }

    // Blockquote ---------------------------------------------------------
    if (/^\s*>\s?/.test(line)) {
      const block = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        block.push(lines[i++].replace(/^\s*>\s?/, ''));
      }
      out.push(`<blockquote>${render(block.join('\n'), context)}</blockquote>`);
      continue;
    }

    // List ----------------------------------------------------------------
    if (BULLET.test(line) || NUMBER.test(line)) {
      const block = [];
      while (i < lines.length) {
        const l = lines[i];
        const isItem = BULLET.test(l) || NUMBER.test(l);
        const isContinuation = l.trim() && indentOf(l) > indentOf(line);
        if (!isItem && !isContinuation) break;
        block.push(lines[i++]);
      }
      out.push(listBlock(block, context));
      continue;
    }

    // Paragraph -------------------------------------------------------------
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{2,4}\s|:::|\s*```|\s*>|\s*\|)/.test(lines[i]) &&
      !/^\s*\$\$\s*$/.test(lines[i]) &&
      !BULLET.test(lines[i]) &&
      !NUMBER.test(lines[i]) &&
      !/^\s*(---|\*\*\*)\s*$/.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    if (para.length) out.push(`<p>${inline(para.join(' ').trim(), context)}</p>`);
    else i++;
  }

  return out.join('\n');
}

// Plain text for search indexing and meta descriptions.
export function toPlainText(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    // Checks are stripped whole: their explanations would otherwise turn up
    // in search results and spoil the question.
    .replace(/^:::\s*check[\s\S]*?^:::\s*$/gm, ' ')
    .replace(/^:::.*$/gm, ' ')
    .replace(/[#>*_`|]/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}
