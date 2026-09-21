import { site, tracks, totals } from '../content/curriculum.mjs';

export const esc = (s = '') =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

export function makeUrl(base) {
  const relative = base === '' ? false : !base.startsWith('/');
  return (path) => {
    if (/^(https?:|mailto:|#)/.test(path)) return path;
    if (relative) {
      const clean = path.replace(/^\//, '');
      const file = clean === '' || clean.endsWith('/') ? clean + 'index.html' : clean;
      return base + file;
    }
    return base + (path.startsWith('/') ? path : '/' + path);
  };
}

const ICON = {
  search:
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">' +
    '<circle cx="9" cy="9" r="5.5"/><path d="M13.2 13.2L17 17"/></svg>',
  theme:
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">' +
    '<circle cx="10" cy="10" r="3.6"/><path d="M10 2.2v1.6M10 16.2v1.6M17.8 10h-1.6M3.8 10H2.2' +
    'M15.5 4.5l-1.1 1.1M5.6 14.4l-1.1 1.1M15.5 15.5l-1.1-1.1M5.6 5.6L4.5 4.5"/></svg>',
  arrow:
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M3 8h10M9 4l4 4-4 4"/></svg>',
};

/**
 * @param {object} opts
 * @param {string} opts.title     Document title, without the site suffix.
 * @param {string} opts.description Meta description.
 * @param {string} opts.body      Page HTML.
 * @param {string} opts.base      Base path, e.g. "" or "/fanin".
 * @param {string} [opts.nav]     Which nav item is current.
 * @param {boolean} [opts.math]   Load KaTeX.
 * @param {string} [opts.canonical] Path of this page.
 * @param {string} [opts.bodyClass]
 */
export function layout(opts) {
  const { title, description, body, base, nav = '', math = false, canonical = '/' } = opts;
  const u = makeUrl(base);
  const fullTitle = canonical === '/' ? `${site.name} — ${site.tagline}` : `${title} · ${site.name}`;

  const navLink = (href, label, key, hideSm = false) =>
    `<a class="nav-link${hideSm ? ' nav-hide-sm' : ''}" href="${u(href)}"` +
    `${nav === key ? ' aria-current="page"' : ''}>${label}</a>`;

  // KaTeX is vendored, not pulled from a CDN. The site promises no tracking,
  // and every CDN request hands a third party the visitor's IP and referrer.
  // Self-hosting also means maths still renders offline and on a bad network.
  const katex = math
    ? `<link rel="stylesheet" href="${u('/assets/katex/katex.min.css')}">
  <script defer src="${u('/assets/katex/katex.min.js')}" onload="window.dispatchEvent(new Event('fanin:katex'))"></script>`
    : '';

  return `<!doctype html>
<html lang="en" data-base="${esc(base)}" data-repo="${esc(site.repo)}" data-tracks="${tracks.map((t) => `${t.id}:${t.lessons.length}:${t.short}`).join(',')}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>${esc(fullTitle)}</title>
<meta name="description" content="${esc(description)}">
<meta name="color-scheme" content="light dark">
<meta property="og:title" content="${esc(fullTitle)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="${u('/favicon.svg')}" type="image/svg+xml">
<link rel="stylesheet" href="${u('/assets/fonts.css')}">
<link rel="stylesheet" href="${u('/assets/styles.css')}">
${katex}
<script>
  // Apply the stored theme before first paint so the page never flashes.
  try {
    var t = localStorage.getItem('fanin:theme');
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  } catch (e) {}
</script>
</head>
<body${opts.bodyClass ? ` class="${opts.bodyClass}"` : ''}>
<a class="skip-link" href="#main">Skip to content</a>

<header class="site-header">
  <div class="wrap header-inner">
    <a class="brand" href="${u('/')}">
      <span class="brand-mark" aria-hidden="true"></span>
      <span>${site.name}</span>
    </a>
    <nav class="site-nav" aria-label="Main">
      ${navLink('/curriculum/', 'Curriculum', 'curriculum')}
      ${navLink('/progress/', 'Progress', 'progress', true)}
      ${navLink('/about/', 'About', 'about', true)}
      <button class="search-trigger" type="button" data-search-open aria-label="Search lessons">
        ${ICON.search}<span class="search-label">Search</span><kbd>⌘K</kbd>
      </button>
      <button class="icon-btn" type="button" id="theme-toggle" aria-label="Switch theme">${ICON.theme}</button>
    </nav>
  </div>
</header>

<main id="main">
${body}
</main>

<footer class="site-footer">
  <div class="wrap">
    <div class="footer-grid">
      <div class="footer-brand">
        <a class="brand" href="${u('/')}"><span class="brand-mark" aria-hidden="true"></span><span>${site.name}</span></a>
        <p>${esc(site.description)}</p>
      </div>
      <div class="footer-col">
        <h4>Curriculum</h4>
        <ul>
          ${tracks
            .slice(0, 4)
            .map((t) => `<li><a href="${u(`/learn/${t.id}/`)}">${esc(t.title)}</a></li>`)
            .join('\n          ')}
        </ul>
      </div>
      <div class="footer-col">
        <h4>More</h4>
        <ul>
          ${tracks
            .slice(4)
            .map((t) => `<li><a href="${u(`/learn/${t.id}/`)}">${esc(t.title)}</a></li>`)
            .join('\n          ')}
          <li><a href="${u('/about/')}">About</a></li>
          <li><a href="${site.repo}" target="_blank" rel="noopener noreferrer">Source on GitHub</a></li>
        </ul>
      </div>
    </div>
    <div class="footer-bottom">
      <span>${totals.lessons} lessons · ${totals.tracks} tracks · free forever</span>
      <span>Content under ${site.license}. Code under MIT.</span>
    </div>
  </div>
</footer>

<script src="${u('/assets/app.js')}" defer></script>
</body>
</html>`;
}

export { ICON };
