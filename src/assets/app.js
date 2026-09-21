/* Fanin client runtime. No framework, no build step.
   Everything degrades: with JS off the site is still fully readable. */
(function () {
  'use strict';

  var BASE = document.documentElement.dataset.base || '';
  // BASE is either a server path ("" or "/fanin") or a relative climb
  // ("../../"). Build links the same way the generator did.
  var RELATIVE = BASE !== '' && BASE.charAt(0) !== '/';

  function url(path) {
    var clean = String(path).replace(/^\//, '');
    if (RELATIVE) {
      if (clean === '' || clean.charAt(clean.length - 1) === '/') clean += 'index.html';
      return BASE + clean;
    }
    return BASE + '/' + clean;
  }
  var STORE_KEY = 'fanin:progress:v2';
  var LEGACY_KEY = 'fanin:progress:v1';
  var THEME_KEY = 'fanin:theme';
  var XP_PER_MINUTE = 10;

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

  // One record per completed lesson: { d: "YYYY-MM-DD", x: xp earned }.
  // XP is stored at completion rather than recomputed, so it stays yours even
  // if a lesson is later re-timed, and no metadata has to be fetched to total
  // it up.
  var store = readJSON(STORE_KEY, null);

  if (!store || typeof store !== 'object' || Array.isArray(store)) {
    store = {};
    // v1 was a flat { id: 1 } with no dates. Preserve the completions; their
    // XP and dates are genuinely unknown, so they are left empty rather than
    // invented.
    var legacy = readJSON(LEGACY_KEY, null);
    if (legacy && typeof legacy === 'object' && !Array.isArray(legacy)) {
      Object.keys(legacy).forEach(function (id) {
        if (legacy[id]) store[id] = { d: null, x: 0 };
      });
      writeJSON(STORE_KEY, store);
    }
  }

  function today() {
    var d = new Date();
    return d.getFullYear() + '-' +
      String(d.getMonth() + 1).padStart(2, '0') + '-' +
      String(d.getDate()).padStart(2, '0');
  }

  function isDone(id) { return !!store[id]; }

  function setDone(id, done, minutes) {
    if (done) {
      store[id] = { d: today(), x: (minutes || 0) * XP_PER_MINUTE };
    } else {
      delete store[id];
    }
    writeJSON(STORE_KEY, store);
    return stats();
  }

  /* ------------------------------------------------- derived statistics -- */

  var TRACKS = (document.documentElement.dataset.tracks || '')
    .split(',').filter(Boolean)
    .map(function (part) {
      var bits = part.split(':');
      return { id: bits[0], size: Number(bits[1]) || 0, short: bits[2] || bits[0] };
    });

  var LEVELS = [
    { at: 0,     name: 'Randomly Initialized' },
    { at: 600,   name: 'First Backward Pass' },
    { at: 1800,  name: 'Gradient Descending' },
    { at: 3600,  name: 'Learning Rate Tuned' },
    { at: 6000,  name: 'Attention Is Yours' },
    { at: 9000,  name: 'Residual Connected' },
    { at: 12000, name: 'Scaling Laws Obeyed' },
    { at: 15000, name: 'Policy Optimized' },
    { at: 17500, name: 'Kernel Fused' },
    { at: 20110, name: 'Compute Optimal' },
  ];

  var TRACK_BADGE = {
    math: 'Foundations Laid',
    pytorch: 'Fluent in Tensors',
    'deep-learning': 'Trainable',
    transformers: 'Attention Mastered',
    llms: 'Full Lifecycle',
    rl: 'Policy Converged',
    systems: 'Shipped It',
  };

  function dayBefore(iso) {
    var parts = iso.split('-');
    var d = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]) - 1);
    return d.getFullYear() + '-' +
      String(d.getMonth() + 1).padStart(2, '0') + '-' +
      String(d.getDate()).padStart(2, '0');
  }

  function streakFrom(days) {
    // days: sorted-descending array of unique "YYYY-MM-DD".
    if (!days.length) return { current: 0, longest: 0 };
    var set = {};
    days.forEach(function (d) { set[d] = true; });

    var now = today();
    var current = 0;
    // A streak survives until the end of the next day, so yesterday still counts.
    var cursor = set[now] ? now : (set[dayBefore(now)] ? dayBefore(now) : null);
    while (cursor && set[cursor]) { current++; cursor = dayBefore(cursor); }

    var longest = 0, run = 0, prev = null;
    days.slice().reverse().forEach(function (d) {
      run = prev && dayBefore(d) === prev ? run + 1 : 1;
      if (run > longest) longest = run;
      prev = d;
    });
    return { current: current, longest: longest };
  }

  function stats() {
    var ids = Object.keys(store);
    var xp = 0, byDay = {}, byTrack = {}, earliest = null;

    ids.forEach(function (id) {
      var rec = store[id] || {};
      xp += rec.x || 0;
      var track = id.split('/')[0];
      byTrack[track] = (byTrack[track] || 0) + 1;
      if (rec.d) {
        byDay[rec.d] = (byDay[rec.d] || 0) + 1;
        if (!earliest || rec.d < earliest) earliest = rec.d;
      }
    });

    var days = Object.keys(byDay).sort().reverse();
    var streak = streakFrom(days);

    var level = LEVELS[0], next = LEVELS[1] || null;
    for (var i = 0; i < LEVELS.length; i++) {
      if (xp >= LEVELS[i].at) { level = LEVELS[i]; next = LEVELS[i + 1] || null; }
    }

    var total = TRACKS.reduce(function (s, t) { return s + t.size; }, 0);
    var busiest = days.reduce(function (m, d) { return Math.max(m, byDay[d]); }, 0);

    return {
      count: ids.length,
      total: total,
      xp: xp,
      level: level,
      levelIndex: LEVELS.indexOf(level),
      next: next,
      streak: streak.current,
      longest: streak.longest,
      byDay: byDay,
      byTrack: byTrack,
      days: days,
      busiestDay: busiest,
      badges: badges({ count: ids.length, total: total, byTrack: byTrack,
                       streak: streak, busiest: busiest }),
    };
  }

  function badges(s) {
    var out = [];
    function add(id, name, hint, earned) {
      out.push({ id: id, name: name, hint: hint, earned: !!earned });
    }

    add('first', 'First Light', 'Complete your first lesson', s.count >= 1);
    add('ten', 'Warmed Up', 'Complete ten lessons', s.count >= 10);
    add('half', 'Past the Ridge Point', 'Complete half the curriculum', s.count >= Math.ceil(s.total / 2));
    add('all', 'Compute Optimal', 'Complete all ' + s.total + ' lessons', s.total > 0 && s.count >= s.total);

    add('deep-work', 'Deep Work', 'Five lessons in one day', s.busiest >= 5);
    add('streak-3', 'Three Days Running', 'A three-day streak', s.streak.longest >= 3);
    add('streak-7', 'A Full Week', 'A seven-day streak', s.streak.longest >= 7);
    add('streak-30', 'Converged', 'A thirty-day streak', s.streak.longest >= 30);

    var touched = TRACKS.filter(function (t) { return s.byTrack[t.id]; }).length;
    add('breadth', 'Polymath', 'A lesson in every track', TRACKS.length > 0 && touched >= TRACKS.length);

    TRACKS.forEach(function (t) {
      add('track-' + t.id, TRACK_BADGE[t.id] || ('Cleared ' + t.short),
          'Finish ' + t.short, t.size > 0 && (s.byTrack[t.id] || 0) >= t.size);
    });

    return out;
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
    var s = stats();

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
      var ring = el.querySelector('[data-ring]');
      if (ring) setRing(ring, done / (ids.length || 1));
      var ringLabel = el.querySelector('[data-ring-pct]');
      if (ringLabel) ringLabel.textContent = pct + '%';
      el.dataset.complete = done && done === ids.length ? '1' : '0';
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
      var xpHint = btn.querySelector('.xp-hint');
      if (xpHint) {
        xpHint.textContent = done2 ? '' : '+' + (Number(btn.dataset.minutes) || 0) * XP_PER_MINUTE + ' XP';
      }
    }

    paintLevel(s);
    paintDashboard(s);
    paintResume(s);
    return s;
  }

  /* -------------------------------------------------- level & streak ---- */

  function paintLevel(s) {
    document.querySelectorAll('[data-level-name]').forEach(function (el) {
      el.textContent = s.level.name;
    });
    document.querySelectorAll('[data-level-num]').forEach(function (el) {
      el.textContent = String(s.levelIndex + 1);
    });
    document.querySelectorAll('[data-xp]').forEach(function (el) {
      el.textContent = s.xp.toLocaleString();
    });
    document.querySelectorAll('[data-streak]').forEach(function (el) {
      el.textContent = String(s.streak);
    });
    document.querySelectorAll('[data-longest-streak]').forEach(function (el) {
      el.textContent = String(s.longest);
    });

    // Progress toward the next level, measured from the current threshold so
    // the bar starts empty on arrival rather than part-filled.
    document.querySelectorAll('[data-level-bar]').forEach(function (el) {
      var span = s.next ? s.next.at - s.level.at : 0;
      var into = s.xp - s.level.at;
      el.style.width = (s.next ? Math.min(100, (into / span) * 100) : 100) + '%';
    });
    document.querySelectorAll('[data-next-level]').forEach(function (el) {
      el.textContent = s.next
        ? (s.next.at - s.xp).toLocaleString() + ' XP to ' + s.next.name
        : 'Every lesson complete.';
    });

    var chip = document.getElementById('level-chip');
    if (chip) {
      chip.hidden = s.count === 0;
      chip.setAttribute('title',
        s.level.name + ' · ' + s.xp.toLocaleString() + ' XP' +
        (s.streak ? ' · ' + s.streak + '-day streak' : ''));
    }
    var flame = document.getElementById('chip-streak');
    if (flame) flame.hidden = s.streak < 2;
  }

  /* --------------------------------------------------------- dashboard -- */

  function setRing(ring, fraction) {
    var r = Number(ring.getAttribute('r')) || 26;
    var c = 2 * Math.PI * r;
    ring.style.strokeDasharray = c + ' ' + c;
    ring.style.strokeDashoffset = String(c * (1 - Math.max(0, Math.min(1, fraction))));
  }

  function paintDashboard(s) {
    var grid = document.getElementById('activity-grid');
    if (grid) paintHeatmap(grid, s);

    var badgeGrid = document.getElementById('badge-grid');
    if (badgeGrid) {
      var earned = s.badges.filter(function (b) { return b.earned; }).length;
      badgeGrid.innerHTML = s.badges.map(function (b) {
        return '<li class="badge" data-earned="' + (b.earned ? '1' : '0') + '">' +
          '<span class="badge-mark" aria-hidden="true"></span>' +
          '<span class="badge-name">' + escapeHtml(b.name) + '</span>' +
          '<span class="badge-hint">' + escapeHtml(b.hint) + '</span></li>';
      }).join('');
      // There is more than one of these on the page.
      document.querySelectorAll('[data-badge-count]').forEach(function (el) {
        el.textContent = earned + ' / ' + s.badges.length;
      });
    }
  }

  // Activity calendar. Magnitude is lessons-per-day, so the ramp is a single
  // hue light to dark; the empty cell sits outside the ramp as a neutral.
  function paintHeatmap(grid, s) {
    var WEEKS = Number(grid.dataset.weeks) || 18;
    var cells = [];
    var now = new Date();
    // Start on the Sunday that begins the window, so columns are whole weeks.
    var start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - (WEEKS * 7 - 1));
    start.setDate(start.getDate() - start.getDay());

    var max = Math.max(1, s.busiestDay);
    var months = [];
    var lastMonth = -1;

    for (var w = 0; w < WEEKS + 1; w++) {
      var col = [];
      for (var d = 0; d < 7; d++) {
        var day = new Date(start.getFullYear(), start.getMonth(), start.getDate() + w * 7 + d);
        if (day > now) { col.push(null); continue; }
        var iso = day.getFullYear() + '-' +
          String(day.getMonth() + 1).padStart(2, '0') + '-' +
          String(day.getDate()).padStart(2, '0');
        var n = s.byDay[iso] || 0;
        // Four filled steps; 0 is the neutral empty cell, not a ramp step.
        var step = n === 0 ? 0 : Math.min(4, Math.ceil((n / max) * 4));
        col.push({ iso: iso, n: n, step: step });
      }
      var firstOfCol = new Date(start.getFullYear(), start.getMonth(), start.getDate() + w * 7);
      if (firstOfCol.getMonth() !== lastMonth && firstOfCol <= now) {
        lastMonth = firstOfCol.getMonth();
        months.push({ col: w, label: firstOfCol.toLocaleString(undefined, { month: 'short' }) });
      }
      cells.push(col);
    }

    grid.innerHTML =
      '<div class="cal-months">' +
      months.map(function (m) {
        return '<span style="grid-column:' + (m.col + 1) + '">' + m.label + '</span>';
      }).join('') +
      '</div>' +
      '<div class="cal-body">' +
      cells.map(function (col) {
        return '<div class="cal-week">' + col.map(function (c) {
          if (!c) return '<span class="cal-cell" data-step="-1"></span>';
          var label = c.n === 0
            ? 'No lessons on ' + c.iso
            : c.n + (c.n === 1 ? ' lesson' : ' lessons') + ' on ' + c.iso;
          return '<span class="cal-cell" data-step="' + c.step + '" title="' + label +
                 '" role="img" aria-label="' + label + '"></span>';
        }).join('') + '</div>';
      }).join('') +
      '</div>';
  }

  /* ------------------------------------------------------------ resume -- */
  // The landing page carries the ordered lesson list, so the next unfinished
  // lesson can be found without a fetch. It stays hidden until something has
  // actually been completed — a first-time reader should see "Start 1.01".

  function paintResume(s) {
    var card = document.getElementById('resume-card');
    if (!card) return;

    var raw = document.getElementById('lesson-order');
    if (!raw) { card.hidden = true; return; }

    var order;
    try { order = JSON.parse(raw.textContent); } catch (e) { card.hidden = true; return; }

    if (!s.count) { card.hidden = true; return; }

    var next = null;
    for (var i = 0; i < order.length; i++) {
      if (!isDone(order[i][0])) { next = order[i]; break; }
    }
    if (!next) {
      // Everything is done. Offer the last lesson rather than hiding the card.
      next = order[order.length - 1];
      card.dataset.finished = '1';
    }

    card.href = url('learn/' + next[0] + '/');
    var num = card.querySelector('[data-resume-number]');
    var title = card.querySelector('[data-resume-title]');
    var done = card.querySelector('[data-resume-done]');
    if (num) num.textContent = next[1];
    if (title) title.textContent = next[2];
    if (done) done.textContent = String(s.count);
    card.hidden = false;
  }

  /* -------------------------------------------------- lesson controls --- */

  var completeBtn = document.getElementById('complete-btn');
  if (completeBtn) {
    completeBtn.addEventListener('click', function () {
      var wasDone = isDone(completeBtn.dataset.lesson);
      var minutes = Number(completeBtn.dataset.minutes) || 0;
      var before = stats();
      setDone(completeBtn.dataset.lesson, !wasDone, minutes);
      var after = paintProgress();
      if (!wasDone) celebrate(before, after, minutes);
    });
  }

  var resetBtn = document.getElementById('reset-progress');
  if (resetBtn) {
    resetBtn.addEventListener('click', function () {
      if (!confirm('Clear your progress, XP, streak and badges? This cannot be undone.')) return;
      store = {};
      writeJSON(STORE_KEY, store);
      try { localStorage.removeItem(LEGACY_KEY); } catch (e) {}
      paintProgress();
    });
  }

  /* ------------------------------------------------------- celebration -- */
  // Marking a lesson complete is the only moment the site can reward, so it
  // gets one: the XP earned, and any badge or level that just unlocked.

  function celebrate(before, after, minutes) {
    var gained = minutes * XP_PER_MINUTE;
    var events = [];

    if (after.levelIndex > before.levelIndex) {
      events.push({ kind: 'level', title: 'Level ' + (after.levelIndex + 1),
                    body: after.level.name });
    }
    var had = {};
    before.badges.forEach(function (b) { if (b.earned) had[b.id] = true; });
    after.badges.forEach(function (b) {
      if (b.earned && !had[b.id]) {
        events.push({ kind: 'badge', title: b.name, body: b.hint });
      }
    });
    if (after.streak > before.streak && after.streak >= 2) {
      events.push({ kind: 'streak', title: after.streak + '-day streak',
                    body: 'Keep it going tomorrow.' });
    }

    showToast('+' + gained + ' XP', after.xp.toLocaleString() + ' total', 'xp');
    events.forEach(function (e, i) {
      setTimeout(function () { showToast(e.title, e.body, e.kind); }, 450 * (i + 1));
    });
  }

  function showToast(title, body, kind) {
    var stack = document.getElementById('toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'toast-stack';
      stack.className = 'toast-stack';
      stack.setAttribute('role', 'status');
      stack.setAttribute('aria-live', 'polite');
      document.body.appendChild(stack);
    }
    var toast = document.createElement('div');
    toast.className = 'toast';
    toast.dataset.kind = kind || 'xp';
    toast.innerHTML =
      '<span class="toast-mark" aria-hidden="true"></span>' +
      '<span class="toast-text"><b>' + escapeHtml(title) + '</b>' +
      (body ? '<span>' + escapeHtml(body) + '</span>' : '') + '</span>';
    stack.appendChild(toast);
    setTimeout(function () {
      toast.dataset.leaving = '1';
      setTimeout(function () { toast.remove(); }, 300);
    }, kind === 'xp' ? 2200 : 3600);
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
    indexPromise = fetch(url('search-index.json'))
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
    // A lesson still being written is a worse answer than one you can read.
    return entry.d ? total * 0.35 : total;
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
            '<a class="palette-hit" href="' + url(entry.u) + '"' + (i === 0 ? ' data-active' : '') + '>' +
            '<span class="n">' + entry.n + '</span>' +
            '<span><span class="t">' + mark(entry.t, terms) +
            (entry.d ? '<span class="hit-draft">drafting</span>' : '') + '</span>' +
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
