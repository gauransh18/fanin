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
  const context = { url: ctx.url || ((h) => h), headings: ctx.headings || [] };
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
      context.headings.push({ level, text, id });
      out.push(
        `<h${level} id="${id}">${inline(text, context)}` +
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
    .replace(/^:::.*$/gm, ' ')
    .replace(/[#>*_`|]/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}
