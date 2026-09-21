/* Fanin client runtime. No framework, no build step.
   Everything degrades: with JS off the site is still fully readable. */
(function () {
  'use strict';

  var BASE = document.documentElement.dataset.base || '';
  var PROGRESS_KEY = 'fanin:progress:v1';
  var THEME_KEY = 'fanin:theme';

  /* ---------------------------------------------------------- storage --- */
  // Site data lives in this browser only. Private windows and blocked site
  // data make these throw, so every access is guarded and the page renders
  // correctly with nothing stored.

  function readJSON(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }
  function writeJSON(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      return false;
    }
  }

  var progress = readJSON(PROGRESS_KEY, {});
  if (!progress || typeof progress !== 'object' || Array.isArray(progress)) progress = {};

  function isDone(id) { return progress[id] === 1; }
  function setDone(id, done) {
    if (done) progress[id] = 1;
    else delete progress[id];
    writeJSON(PROGRESS_KEY, progress);
    paintProgress();
  }

  /* ------------------------------------------------------------ theme --- */

  var root = document.documentElement;

  function currentTheme() {
    var stored;
    try { stored = localStorage.getItem(THEME_KEY); } catch (e) { stored = null; }
    if (stored === 'light' || stored === 'dark') return stored;
    return 'system';
  }

  function applyTheme(mode) {
    if (mode === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', mode);
    try {
      if (mode === 'system') localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, mode);
    } catch (e) { /* storage unavailable; the attribute still applies */ }
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      var resolved = mode === 'system'
        ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
        : mode;
      btn.setAttribute('aria-label', 'Switch to ' + (resolved === 'dark' ? 'light' : 'dark') + ' theme');
      btn.dataset.resolved = resolved;
    }
  }

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    applyTheme(currentTheme());
    toggle.addEventListener('click', function () {
      var resolved = toggle.dataset.resolved === 'dark' ? 'dark' : 'light';
      applyTheme(resolved === 'dark' ? 'light' : 'dark');
    });
  }

  /* --------------------------------------------------------- progress --- */

  function paintProgress() {
    document.querySelectorAll('[data-lesson-id]').forEach(function (el) {
      el.dataset.done = isDone(el.dataset.lessonId) ? '1' : '0';
    });

    document.querySelectorAll('[data-track-progress]').forEach(function (el) {
      var ids = (el.dataset.trackLessons || '').split(' ').filter(Boolean);
      var done = ids.filter(isDone).length;
      var pct = ids.length ? Math.round((done / ids.length) * 100) : 0;
      var bar = el.querySelector('.progress > i');
      if (bar) bar.style.width = pct + '%';
      var label = el.querySelector('[data-progress-label]');
      if (label) label.textContent = done + ' / ' + ids.length;
    });

    var overall = document.querySelector('[data-overall-progress]');
    if (overall) {
      var all = (overall.dataset.allLessons || '').split(' ').filter(Boolean);
      var n = all.filter(isDone).length;
      overall.textContent = String(n);
      var pctEl = document.querySelector('[data-overall-pct]');
      if (pctEl && all.length) pctEl.textContent = Math.round((n / all.length) * 100) + '%';
      var bar2 = document.querySelector('[data-overall-bar]');
      if (bar2 && all.length) bar2.style.width = Math.round((n / all.length) * 100) + '%';
    }

    var btn = document.getElementById('complete-btn');
    if (btn) {
      var done2 = isDone(btn.dataset.lesson);
      btn.setAttribute('aria-pressed', done2 ? 'true' : 'false');
      var lbl = btn.querySelector('.label');
      if (lbl) lbl.textContent = done2 ? 'Completed' : 'Mark complete';
    }
  }

  var completeBtn = document.getElementById('complete-btn');
  if (completeBtn) {
    completeBtn.addEventListener('click', function () {
      setDone(completeBtn.dataset.lesson, !isDone(completeBtn.dataset.lesson));
    });
  }

  var resetBtn = document.getElementById('reset-progress');
  if (resetBtn) {
    resetBtn.addEventListener('click', function () {
      if (!confirm('Clear your progress on all lessons? This cannot be undone.')) return;
      progress = {};
      writeJSON(PROGRESS_KEY, progress);
      paintProgress();
    });
  }

  paintProgress();

  /* ------------------------------------------------------- code copy ---- */

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-copy]');
    if (!btn) return;
    var fig = btn.closest('.code');
    var code = fig && fig.querySelector('code');
    if (!code) return;
    var text = code.innerText;
    var done = function () {
      btn.textContent = 'Copied';
      btn.dataset.copied = '1';
      setTimeout(function () {
        btn.textContent = 'Copy';
        delete btn.dataset.copied;
      }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { btn.textContent = 'Press ⌘C'; });
    } else {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); done(); } catch (err) { btn.textContent = 'Press ⌘C'; }
      document.body.removeChild(ta);
    }
  });

  /* -------------------------------------------------------- scrollspy --- */

  var railLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-rail a'));
  if (railLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    railLinks.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var visible = new Set();
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      });
      var first = railLinks.find(function (a) { return visible.has(a.getAttribute('href').slice(1)); });
      railLinks.forEach(function (a) { a.classList.remove('active'); });
      if (first) first.classList.add('active');
    }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });
    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) io.observe(el);
    });
  }

  /* ----------------------------------------------------------- drawer --- */

  var drawerTrigger = document.getElementById('drawer-trigger');
  if (drawerTrigger) {
    drawerTrigger.addEventListener('click', function () {
      var tpl = document.getElementById('drawer-content');
      if (!tpl) return;
      var backdrop = document.createElement('div');
      backdrop.className = 'drawer-backdrop';
      backdrop.innerHTML =
        '<div class="drawer" role="dialog" aria-modal="true" aria-label="Track contents">' +
        '<div class="drawer-head"><strong>Contents</strong>' +
        '<button class="icon-btn" type="button" data-close aria-label="Close">' +
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">' +
        '<path d="M5 5l10 10M15 5L5 15"/></svg></button></div>' +
        tpl.innerHTML + '</div>';
      document.body.appendChild(backdrop);
      document.body.style.overflow = 'hidden';
      var close = function () {
        backdrop.remove();
        document.body.style.overflow = '';
        drawerTrigger.focus();
      };
      backdrop.addEventListener('click', function (e) {
        if (e.target === backdrop || e.target.closest('[data-close]') || e.target.closest('a')) close();
      });
      document.addEventListener('keydown', function esc(e) {
        if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); }
      });
      var firstLink = backdrop.querySelector('a, button');
      if (firstLink) firstLink.focus();
    });
  }

  /* ------------------------------------------------------ search ------- */

  var index = null;
  var indexPromise = null;

  function loadIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(BASE + '/search-index.json')
      .then(function (r) { return r.json(); })
      .then(function (data) { index = data; return data; })
      .catch(function () { index = []; return []; });
    return indexPromise;
  }

  function scoreEntry(entry, terms) {
    var title = entry.t.toLowerCase();
    var summary = (entry.s || '').toLowerCase();
    var body = (entry.b || '').toLowerCase();
    var track = (entry.k || '').toLowerCase();
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var term = terms[i];
      var hit = 0;
      if (title.indexOf(term) === 0) hit += 120;
      else if (title.indexOf(term) > -1) hit += 70;
      if (track.indexOf(term) > -1) hit += 20;
      if (summary.indexOf(term) > -1) hit += 18;
      if (body.indexOf(term) > -1) hit += 8;
      if (!hit) return 0;
      total += hit;
    }
    return total;
  }

  function excerpt(entry, terms) {
    var source = entry.b || entry.s || '';
    var lower = source.toLowerCase();
    var at = -1;
    for (var i = 0; i < terms.length && at < 0; i++) at = lower.indexOf(terms[i]);
    if (at < 0) return entry.s || '';
    var start = Math.max(0, at - 60);
    var slice = source.slice(start, start + 170);
    return (start > 0 ? '…' : '') + slice + '…';
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function mark(text, terms) {
    var out = escapeHtml(text);
    terms.forEach(function (term) {
      if (term.length < 2) return;
      var re = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      out = out.replace(re, '<mark>$1</mark>');
    });
    return out;
  }

  function openPalette() {
    if (document.querySelector('.palette-backdrop')) return;
    var opener = document.activeElement;

    var backdrop = document.createElement('div');
    backdrop.className = 'palette-backdrop';
    backdrop.innerHTML =
      '<div class="palette" role="dialog" aria-modal="true" aria-label="Search lessons">' +
      '<div class="palette-input-row">' +
      '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round">' +
      '<circle cx="9" cy="9" r="5.5"/><path d="M13.2 13.2L17 17"/></svg>' +
      '<input id="palette-input" type="search" autocomplete="off" spellcheck="false" ' +
      'placeholder="Search 108 lessons — try &quot;rope&quot;, &quot;kl&quot;, &quot;fsdp&quot;" aria-label="Search lessons">' +
      '<span class="palette-esc">esc</span></div>' +
      '<div class="palette-results" id="palette-results"></div>' +
      '<div class="palette-foot"><span>↑↓ navigate</span><span>↵ open</span><span>esc close</span></div>' +
      '</div>';
    document.body.appendChild(backdrop);
    document.body.style.overflow = 'hidden';

    var input = backdrop.querySelector('#palette-input');
    var results = backdrop.querySelector('#palette-results');
    var hits = [];
    var active = 0;

    function close() {
      backdrop.remove();
      document.body.style.overflow = '';
      if (opener && opener.focus) opener.focus();
    }

    function paint(list, terms) {
      hits = list;
      active = 0;
      if (!list.length) {
        results.innerHTML =
          '<p class="palette-empty">' +
          (input.value.trim() ? 'No lesson matches that yet.' : 'Type to search the curriculum.') +
          '</p>';
        return;
      }
      results.innerHTML = list
        .map(function (entry, i) {
          return (
            '<a class="palette-hit" href="' + BASE + entry.u + '"' + (i === 0 ? ' data-active' : '') + '>' +
            '<span class="n">' + entry.n + '</span>' +
            '<span><span class="t">' + mark(entry.t, terms) + '</span>' +
            '<span class="s">' + mark(excerpt(entry, terms), terms) + '</span></span></a>'
          );
        })
        .join('');
    }

    function setActive(next) {
      var nodes = results.querySelectorAll('.palette-hit');
      if (!nodes.length) return;
      active = (next + nodes.length) % nodes.length;
      nodes.forEach(function (n) { delete n.dataset.active; });
      nodes[active].dataset.active = '1';
      nodes[active].scrollIntoView({ block: 'nearest' });
    }

    function run() {
      var query = input.value.trim().toLowerCase();
      if (!query) { paint([], []); return; }
      var terms = query.split(/\s+/).filter(Boolean);
      var list = (index || [])
        .map(function (e) { return { e: e, score: scoreEntry(e, terms) }; })
        .filter(function (r) { return r.score > 0; })
        .sort(function (a, b) { return b.score - a.score; })
        .slice(0, 24)
        .map(function (r) { return r.e; });
      paint(list, terms);
    }

    paint([], []);
    loadIndex().then(function () { if (input.value.trim()) run(); });

    input.addEventListener('input', run);
    backdrop.addEventListener('click', function (e) { if (e.target === backdrop) close(); });
    backdrop.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(active - 1); }
      else if (e.key === 'Enter') {
        var node = results.querySelectorAll('.palette-hit')[active];
        if (node) { e.preventDefault(); window.location.href = node.getAttribute('href'); }
      }
    });
    input.focus();
  }

  document.querySelectorAll('[data-search-open]').forEach(function (el) {
    el.addEventListener('click', openPalette);
  });

  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    var typing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
    if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      openPalette();
    } else if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      openPalette();
    }
  });

  // Warm the index once the page is otherwise idle.
  if ('requestIdleCallback' in window) requestIdleCallback(loadIndex, { timeout: 4000 });
  else setTimeout(loadIndex, 2500);

  /* --------------------------------------------------------- KaTeX ------ */

  function renderMath() {
    if (!window.katex) return;
    document.querySelectorAll('.tex').forEach(function (el) {
      if (el.dataset.rendered) return;
      try {
        window.katex.render(el.textContent, el, {
          displayMode: el.dataset.display === '1',
          throwOnError: false,
          strict: false,
                });
      } catch (err) { /* leave the raw TeX visible rather than blanking it */ }
      el.dataset.rendered = '1';
    });
  }
  if (document.querySelector('.tex')) {
    if (window.katex) renderMath();
    else window.addEventListener('fanin:katex', renderMath);
  }
})();
