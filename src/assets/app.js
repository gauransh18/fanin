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
  var STORE_KEY = 'fanin:progress:v3';
  var LEGACY_V2 = 'fanin:progress:v2';
  var LEGACY_V1 = 'fanin:progress:v1';
  var META_KEY = 'fanin:meta:v1';
  var TRIALS_KEY = 'fanin:trials:v1';
  var THEME_KEY = 'fanin:theme';

  // What an answer is worth. Getting it right the first time pays more than
  // getting there after a miss, but a miss still pays -- the explanation is the
  // point, and a reader who works out why they were wrong has learnt the thing.
  var XP_FIRST = 20;
  var XP_RETRY = 8;
  var XP_PER_MINUTE = 5;   // the bonus for clearing a lesson
  var XP_COMBO = 25;       // every COMBO_STEP first-try answers in a row
  var COMBO_STEP = 5;
  var TRIAL_PASS = 0.8;    // share of a trial you must get right to pass it

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

  // One record per lesson you have touched:
  //   { c: 1 when cleared, d: "YYYY-MM-DD" | null, x: xp banked,
  //     a: { checkIndex: 1 | 2 } }
  // `c` and `d` are separate because a lesson can be cleared without a known
  // date -- a v1 record carries no date, and inventing one would put a day on
  // the calendar that never happened. `a` records how each check went, 1 for
  // right the first time and 2 for right after a miss. XP is stored rather
  // than recomputed, so it stays yours if a lesson is later re-timed or
  // re-questioned.
  var store = readJSON(STORE_KEY, null);

  if (!store || typeof store !== 'object' || Array.isArray(store)) {
    store = {};
    var v2 = readJSON(LEGACY_V2, null);
    if (v2 && typeof v2 === 'object' && !Array.isArray(v2)) {
      // v2 had no checks. The lesson stays cleared and keeps its date; its XP
      // is rescaled to the new per-minute rate rather than left to tower over
      // freshly earned XP.
      Object.keys(v2).forEach(function (id) {
        var rec = v2[id] || {};
        store[id] = { c: 1, d: rec.d || null, x: Math.round((rec.x || 0) / 2), a: {} };
      });
    } else {
      // v1 was a flat { id: 1 } with no dates. Preserve the completions; their
      // XP and dates are genuinely unknown, so they are left empty rather than
      // invented.
      var v1 = readJSON(LEGACY_V1, null);
      if (v1 && typeof v1 === 'object' && !Array.isArray(v1)) {
        Object.keys(v1).forEach(function (id) {
          if (v1[id]) store[id] = { c: 1, d: null, x: 0, a: {} };
        });
      }
    }
    if (Object.keys(store).length) writeJSON(STORE_KEY, store);
  }

  // Running state that is not per-lesson: the answer combo, and today's tally
  // for the daily quests.
  var meta = readJSON(META_KEY, null);
  if (!meta || typeof meta !== 'object') meta = {};
  if (typeof meta.combo !== 'number') meta.combo = 0;
  if (typeof meta.bestCombo !== 'number') meta.bestCombo = 0;

  var trials = readJSON(TRIALS_KEY, null);
  if (!trials || typeof trials !== 'object') trials = {};

  function today() {
    var d = new Date();
    return d.getFullYear() + '-' +
      String(d.getMonth() + 1).padStart(2, '0') + '-' +
      String(d.getDate()).padStart(2, '0');
  }

  // Today's counters, reset the moment the date rolls over.
  function day() {
    if (!meta.day || meta.day.d !== today()) {
      meta.day = { d: today(), checks: 0, first: 0, cleared: 0 };
    }
    return meta.day;
  }

  function saveMeta() { writeJSON(META_KEY, meta); }

  function rec(id) {
    if (!store[id]) store[id] = { c: 0, d: null, x: 0, a: {} };
    if (!store[id].a) store[id].a = {};
    return store[id];
  }

  function isDone(id) { return !!(store[id] && store[id].c); }

  function answered(id) { return store[id] && store[id].a ? store[id].a : {}; }

  // Records one check result and returns what it was worth, so the page can
  // show the number that just landed.
  function recordCheck(id, index, firstTry) {
    var r = rec(id);
    var key = String(index);
    if (r.a[key]) return { xp: 0, repeat: true, combo: meta.combo };

    r.a[key] = firstTry ? 1 : 2;
    var gained = firstTry ? XP_FIRST : XP_RETRY;

    var d = day();
    d.checks++;
    if (firstTry) {
      d.first++;
      meta.combo++;
      if (meta.combo > meta.bestCombo) meta.bestCombo = meta.combo;
    } else {
      meta.combo = 0;
    }

    var bonus = 0;
    if (firstTry && meta.combo > 0 && meta.combo % COMBO_STEP === 0) {
      bonus = XP_COMBO;
      gained += bonus;
    }

    r.x += gained;
    writeJSON(STORE_KEY, store);
    saveMeta();
    return { xp: gained, bonus: bonus, combo: meta.combo, repeat: false };
  }

  // A miss costs the combo immediately, before the reader picks again.
  function breakCombo() {
    if (!meta.combo) return 0;
    meta.combo = 0;
    saveMeta();
    return 0;
  }

  function setDone(id, done, minutes) {
    var r = rec(id);
    if (done) {
      if (!r.c) {
        r.c = 1;
        r.d = today();
        r.x += (minutes || 0) * XP_PER_MINUTE;
        day().cleared++;
        saveMeta();
      }
    } else {
      // Un-clearing gives back the clear bonus but keeps answered checks:
      // you did answer them, and re-answering them would pay twice.
      if (r.c) r.x = Math.max(0, r.x - (minutes || 0) * XP_PER_MINUTE);
      r.c = 0;
      r.d = null;
      if (!r.x && !Object.keys(r.a).length) delete store[id];
    }
    writeJSON(STORE_KEY, store);
    return stats();
  }

  function recordTrial(trackId, correct, total) {
    var pct = total ? correct / total : 0;
    var passed = pct >= TRIAL_PASS;
    var prev = trials[trackId] || { best: 0, passed: 0, x: 0 };
    var xp = 0;
    // A trial pays once, the first time it is passed. Retries are for the
    // score on the board, not for farming XP.
    if (passed && !prev.passed) xp = 400;
    trials[trackId] = {
      best: Math.max(prev.best || 0, Math.round(pct * 100)),
      passed: passed || prev.passed ? 1 : 0,
      d: passed && !prev.passed ? today() : prev.d || null,
      x: (prev.x || 0) + xp,
    };
    writeJSON(TRIALS_KEY, trials);
    return { passed: passed, pct: pct, xp: xp, best: trials[trackId].best };
  }

  /* ------------------------------------------------- derived statistics -- */

  var TRACKS = (document.documentElement.dataset.tracks || '')
    .split(',').filter(Boolean)
    .map(function (part) {
      var bits = part.split(':');
      return {
        id: bits[0],
        size: Number(bits[1]) || 0,
        short: bits[2] || bits[0],
        badge: bits[3] || ('Cleared ' + (bits[2] || bits[0])),
      };
    });

  // The ladder tops out at what finishing the curriculum is actually worth:
  // every lesson cleared, every check answered, every trial passed, without
  // needing a perfect first-try record.
  var LEVELS = [
    { at: 0,     name: 'Randomly Initialized' },
    { at: 500,   name: 'First Backward Pass' },
    { at: 1400,  name: 'Gradient Descending' },
    { at: 2600,  name: 'Learning Rate Tuned' },
    { at: 4200,  name: 'Attention Is Yours' },
    { at: 6100,  name: 'Residual Connected' },
    { at: 8300,  name: 'Scaling Laws Obeyed' },
    { at: 10800, name: 'Policy Optimized' },
    { at: 13500, name: 'Kernel Fused' },
    { at: 16500, name: 'Compute Optimal' },
  ];

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

  /* ------------------------------------------------------ daily quests -- */
  // Three goals a day, picked from the date so they are the same all day and
  // different tomorrow. Small, finishable in one sitting.

  var QUESTS = [
    { id: 'answer-5',  goal: 5,  of: 'checks',  name: 'Answer five checks' },
    { id: 'answer-10', goal: 10, of: 'checks',  name: 'Answer ten checks' },
    { id: 'first-3',   goal: 3,  of: 'first',   name: 'Three right first try' },
    { id: 'first-6',   goal: 6,  of: 'first',   name: 'Six right first try' },
    { id: 'clear-1',   goal: 1,  of: 'cleared', name: 'Clear a lesson' },
    { id: 'clear-2',   goal: 2,  of: 'cleared', name: 'Clear two lessons' },
    { id: 'clear-3',   goal: 3,  of: 'cleared', name: 'Clear three lessons' },
  ];

  function questsToday() {
    var iso = today();
    var h = 0;
    for (var i = 0; i < iso.length; i++) h = (h * 31 + iso.charCodeAt(i)) >>> 0;
    var pool = QUESTS.slice();
    var picked = [];
    for (var k = 0; k < 3 && pool.length; k++) {
      h = (h * 1103515245 + 12345) >>> 0;
      picked.push(pool.splice(h % pool.length, 1)[0]);
    }
    var d = day();
    return picked.map(function (q) {
      var have = d[q.of] || 0;
      return {
        id: q.id, name: q.name, goal: q.goal, have: Math.min(have, q.goal),
        done: have >= q.goal,
      };
    });
  }

  function stats() {
    var ids = Object.keys(store);
    var xp = 0, byDay = {}, byTrack = {}, checksRight = 0, firstTry = 0;

    ids.forEach(function (id) {
      var r = store[id] || {};
      xp += r.x || 0;
      Object.keys(r.a || {}).forEach(function (k) {
        checksRight++;
        if (r.a[k] === 1) firstTry++;
      });
      if (r.c) {
        var track = id.split('/')[0];
        byTrack[track] = (byTrack[track] || 0) + 1;
        // Only a dated clear reaches the calendar and the streak.
        if (r.d) byDay[r.d] = (byDay[r.d] || 0) + 1;
      }
    });

    Object.keys(trials).forEach(function (t) { xp += trials[t].x || 0; });

    var cleared = ids.filter(function (id) { return store[id] && store[id].c; });
    var days = Object.keys(byDay).sort().reverse();
    var streak = streakFrom(days);

    var level = LEVELS[0], next = LEVELS[1] || null;
    for (var i = 0; i < LEVELS.length; i++) {
      if (xp >= LEVELS[i].at) { level = LEVELS[i]; next = LEVELS[i + 1] || null; }
    }

    var total = TRACKS.reduce(function (s, t) { return s + t.size; }, 0);
    var busiest = days.reduce(function (m, d) { return Math.max(m, byDay[d]); }, 0);
    var trialsPassed = TRACKS.filter(function (t) {
      return trials[t.id] && trials[t.id].passed;
    }).length;

    return {
      count: cleared.length,
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
      checksRight: checksRight,
      firstTry: firstTry,
      accuracy: checksRight ? Math.round((firstTry / checksRight) * 100) : 0,
      combo: meta.combo,
      bestCombo: meta.bestCombo,
      trials: trials,
      trialsPassed: trialsPassed,
      quests: questsToday(),
      badges: badges({
        count: cleared.length, total: total, byTrack: byTrack, streak: streak,
        busiest: busiest, bestCombo: meta.bestCombo, trialsPassed: trialsPassed,
        checksRight: checksRight, firstTry: firstTry,
      }),
    };
  }

  function badges(s) {
    var out = [];
    function add(id, name, hint, earned) {
      out.push({ id: id, name: name, hint: hint, earned: !!earned });
    }

    add('first', 'First Light', 'Clear your first lesson', s.count >= 1);
    add('ten', 'Warmed Up', 'Clear ten lessons', s.count >= 10);
    add('half', 'Past the Ridge Point', 'Clear half the curriculum', s.count >= Math.ceil(s.total / 2));
    add('all', 'Compute Optimal', 'Clear all ' + s.total + ' lessons', s.total > 0 && s.count >= s.total);

    add('deep-work', 'Deep Work', 'Five lessons in one day', s.busiest >= 5);
    add('streak-3', 'Three Days Running', 'A three-day streak', s.streak.longest >= 3);
    add('streak-7', 'A Full Week', 'A seven-day streak', s.streak.longest >= 7);
    add('streak-30', 'Converged', 'A thirty-day streak', s.streak.longest >= 30);

    add('combo-5', 'On a Roll', 'Five right in a row, first try', s.bestCombo >= 5);
    add('combo-15', 'Low Perplexity', 'Fifteen right in a row, first try', s.bestCombo >= 15);
    add('sharp', 'Well Calibrated', 'A hundred answers right first try', s.firstTry >= 100);
    add('trials', 'Benchmarked', 'Pass every track trial',
        TRACKS.length > 0 && s.trialsPassed >= TRACKS.length);

    var touched = TRACKS.filter(function (t) { return s.byTrack[t.id]; }).length;
    add('breadth', 'Polymath', 'A lesson in every track', TRACKS.length > 0 && touched >= TRACKS.length);

    // A track badge takes both: every lesson cleared and the trial passed.
    TRACKS.forEach(function (t) {
      add('track-' + t.id, t.badge,
          'Finish ' + t.short + ' and pass its trial',
          t.size > 0 && (s.byTrack[t.id] || 0) >= t.size &&
          !!(trials[t.id] && trials[t.id].passed));
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
    paintTrialCards();

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

  // The trial card on a track page carries its own state.
  function paintTrialCards() {
    document.querySelectorAll('[data-trial-card]').forEach(function (card) {
      var t = trials[card.dataset.trialCard];
      var state = card.querySelector('[data-trial-state]');
      if (!state) return;
      if (t && t.passed) {
        state.dataset.passed = '1';
        state.textContent = 'Passed · best ' + t.best + '%';
      } else if (t && t.best) {
        state.dataset.passed = '0';
        state.textContent = 'Best ' + t.best + '% · not passed';
      } else {
        state.dataset.passed = '0';
        state.textContent = 'Not attempted';
      }
    });
  }

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

  /* ------------------------------------------------------------ checks -- */
  // The question loop: attempt, immediate feedback, consequence. A miss shows
  // why that option is wrong and leaves the question open, so the reader who
  // gets it wrong still ends up knowing the answer -- they just earn less for
  // it, and their combo resets.

  var checkEls = [].slice.call(document.querySelectorAll('.check:not(.check-trial)'));
  var completeBtn = document.getElementById('complete-btn');
  var LESSON = completeBtn ? completeBtn.dataset.lesson || '' : '';
  var NEED = completeBtn ? Number(completeBtn.dataset.checks) || 0 : 0;

  function keyOf(el) {
    try {
      var raw = atob(el.dataset.k || '');
      return raw.indexOf('fanin:') === 0 ? raw.slice(6) : '';
    } catch (e) {
      return '';
    }
  }

  function correctSet(el) {
    var out = {};
    keyOf(el).split(',').forEach(function (n) {
      if (n !== '') out[Number(n)] = true;
    });
    return out;
  }

  // A number answer is right within its stated tolerance, and exactly right
  // when none was given -- allowing for the float error of typing "0.1".
  function numericOk(el, typed) {
    var bits = keyOf(el).split('|');
    var want = Number(bits[0]);
    var tol = Number(bits[1]) || 0;
    var got = Number(String(typed).trim().replace(/,/g, ''));
    if (!isFinite(got)) return false;
    return Math.abs(got - want) <= (tol || Math.abs(want) * 1e-9 + 1e-9);
  }

  function xpPop(near, text) {
    if (!near) return;
    var pop = document.createElement('span');
    pop.className = 'xp-pop';
    pop.textContent = text;
    near.appendChild(pop);
    setTimeout(function () { pop.remove(); }, 1400);
  }

  // Paints a check that is finished with: the right answer marked, its
  // explanation open, and nothing left to click.
  function lockSolved(el, how) {
    el.dataset.state = 'solved';
    el.dataset.how = how === 1 ? 'first' : 'retry';
    var right = correctSet(el);
    el.querySelectorAll('.opt').forEach(function (btn) {
      var i = Number(btn.dataset.i);
      btn.disabled = true;
      if (right[i]) {
        btn.dataset.mark = 'right';
        var row = btn.closest('.opt-row');
        if (row) row.dataset.show = '1';
      }
    });
    var input = el.querySelector('.check-num input');
    if (input) {
      input.disabled = true;
      input.value = input.value || keyOf(el).split('|')[0];
      var go = el.querySelector('.check-num button');
      if (go) go.disabled = true;
      var why = el.querySelector('[data-why]');
      if (why) why.dataset.show = '1';
    }
    var submit = el.querySelector('[data-submit]');
    if (submit) submit.disabled = true;
    var res = el.querySelector('.check-result');
    if (res && !res.textContent) {
      res.dataset.tone = 'right';
      res.textContent = how === 1 ? 'Right first time.' : 'Answered.';
    }
  }

  function answeredCount() {
    var a = answered(LESSON);
    return checkEls.filter(function (el, i) { return a[String(i)]; }).length;
  }

  function paintChecks() {
    var n = answeredCount();
    document.querySelectorAll('[data-tally] .tally-n').forEach(function (el) {
      el.textContent = String(n);
    });
    var tally = document.querySelector('[data-tally]');
    if (tally) tally.dataset.full = NEED && n >= NEED ? '1' : '0';
    if (completeBtn && NEED) {
      var ready = n >= NEED;
      completeBtn.dataset.locked = ready || isDone(LESSON) ? '0' : '1';
      completeBtn.setAttribute('aria-disabled', ready || isDone(LESSON) ? 'false' : 'true');
    }
    return n;
  }

  // Scores one attempt. Returns true when the check is now finished.
  function judge(el, index, isRight, chosenRows) {
    var res = el.querySelector('.check-result');

    if (!isRight) {
      el.dataset.missed = '1';
      breakCombo();
      (chosenRows || []).forEach(function (row) {
        row.dataset.show = '1';
        var b = row.querySelector('.opt');
        if (b) b.dataset.mark = 'wrong';
      });
      if (res) {
        res.dataset.tone = 'wrong';
        res.textContent = 'Not this one — read why, then try again.';
      }
      paintLevel(stats());
      return false;
    }

    var firstTry = el.dataset.missed !== '1';
    var got = recordCheck(LESSON, index, firstTry);
    lockSolved(el, firstTry ? 1 : 2);

    if (res) {
      res.dataset.tone = 'right';
      res.textContent = firstTry ? 'Right first time.' : 'That is the one.';
      if (got.combo >= 2) res.textContent += '  ' + got.combo + ' in a row.';
    }
    if (got.xp) xpPop(el.querySelector('.check-tag'), '+' + got.xp + ' XP');
    if (got.bonus) {
      showToast(got.combo + ' in a row', '+' + got.bonus + ' XP combo bonus', 'combo');
    }

    var n = paintChecks();
    paintLevel(stats());
    if (NEED && n >= NEED && !isDone(LESSON) && completeBtn) {
      completeBtn.dataset.ready = '1';
    }
    return true;
  }

  checkEls.forEach(function (el, index) {
    var prior = answered(LESSON)[String(index)];
    if (prior) { lockSolved(el, prior); return; }

    var kind = el.dataset.kind;

    if (kind === 'numeric') {
      el.addEventListener('submit', function (e) {
        e.preventDefault();
        if (el.dataset.state === 'solved') return;
        var input = el.querySelector('input');
        var typed = input ? input.value : '';
        if (!String(typed).trim()) return;
        if (numericOk(el, typed)) {
          judge(el, index, true, []);
        } else {
          judge(el, index, false, []);
          if (input) { input.select(); }
        }
      });
      return;
    }

    if (kind === 'multi') {
      el.querySelectorAll('.opt').forEach(function (btn) {
        btn.addEventListener('click', function () {
          if (el.dataset.state === 'solved') return;
          btn.dataset.picked = btn.dataset.picked === '1' ? '0' : '1';
        });
      });
      var submit = el.querySelector('[data-submit]');
      if (submit) {
        submit.addEventListener('click', function () {
          if (el.dataset.state === 'solved') return;
          var right = correctSet(el);
          var picked = [].slice.call(el.querySelectorAll('.opt'))
            .filter(function (b) { return b.dataset.picked === '1'; });
          if (!picked.length) return;
          var ok = picked.length === Object.keys(right).length &&
                   picked.every(function (b) { return right[Number(b.dataset.i)]; });
          if (ok) {
            judge(el, index, true, []);
          } else {
            var wrongRows = picked
              .filter(function (b) { return !right[Number(b.dataset.i)]; })
              .map(function (b) { return b.closest('.opt-row'); });
            // Nothing wrong was picked, so the set is merely incomplete: say so
            // rather than marking a correct option as a mistake.
            judge(el, index, false, wrongRows);
            if (!wrongRows.length) {
              var res = el.querySelector('.check-result');
              if (res) res.textContent = 'Everything you picked is right, but not all of it is there.';
            }
            picked.forEach(function (b) { b.dataset.picked = '0'; });
          }
        });
      }
      return;
    }

    el.querySelectorAll('.opt').forEach(function (btn) {
      btn.addEventListener('click', function () {
        if (el.dataset.state === 'solved' || btn.disabled) return;
        var right = correctSet(el);
        var i = Number(btn.dataset.i);
        if (right[i]) {
          judge(el, index, true, []);
        } else {
          btn.disabled = true;
          judge(el, index, false, [btn.closest('.opt-row')]);
        }
      });
    });
  });

  paintChecks();

  if (completeBtn) {
    completeBtn.addEventListener('click', function () {
      var wasDone = isDone(LESSON);
      // A lesson with checks is cleared by answering them, not by asserting it.
      if (!wasDone && NEED && answeredCount() < NEED) {
        completeBtn.dataset.nudge = '1';
        setTimeout(function () { completeBtn.dataset.nudge = '0'; }, 700);
        var first = checkEls.filter(function (el) {
          return el.dataset.state !== 'solved';
        })[0];
        if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      var minutes = Number(completeBtn.dataset.minutes) || 0;
      var before = stats();
      setDone(LESSON, !wasDone, minutes);
      completeBtn.dataset.ready = '0';
      var after = paintProgress();
      paintChecks();
      if (!wasDone) celebrate(before, after, minutes);
    });
  }

  var resetBtn = document.getElementById('reset-progress');
  if (resetBtn) {
    resetBtn.addEventListener('click', function () {
      if (!confirm('Clear your progress, XP, streak and badges? This cannot be undone.')) return;
      store = {};
      meta = { combo: 0, bestCombo: 0 };
      trials = {};
      writeJSON(STORE_KEY, store);
      writeJSON(META_KEY, meta);
      writeJSON(TRIALS_KEY, trials);
      try {
        localStorage.removeItem(LEGACY_V1);
        localStorage.removeItem(LEGACY_V2);
      } catch (e) { /* nothing stored to remove */ }
      paintProgress();
    });
  }

  /* ------------------------------------------------------------- trial -- */
  // The boss round. Questions come from the whole track, the pool is bigger
  // than the round, and nothing is explained until the end -- so a trial tests
  // what you carried out of the lessons rather than what you can work out from
  // the feedback.

  var trialEl = document.querySelector('[data-trial]');
  if (trialEl) (function () {
    var TRACK = trialEl.dataset.trial;
    var SIZE = Number(trialEl.dataset.size) || 10;
    var pool = [].slice.call(trialEl.querySelectorAll('.check-trial'));
    var intro = trialEl.querySelector('[data-intro]');
    var runEl = trialEl.querySelector('[data-run]');
    var doneEl = trialEl.querySelector('[data-done]');
    var nextBtn = trialEl.querySelector('[data-next]');
    var pickedNote = trialEl.querySelector('[data-picked]');
    var barEl = trialEl.querySelector('[data-bar]');
    var atEl = trialEl.querySelector('[data-at]');

    var round = [];
    var at = 0;
    var results = [];

    function shuffle(list) {
      var a = list.slice();
      for (var i = a.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var t = a[i]; a[i] = a[j]; a[j] = t;
      }
      return a;
    }

    // Paints the intro: best score so far, and whether the track is finished.
    function paintIntro() {
      var t = trials[TRACK];
      var best = trialEl.querySelector('[data-trial-best]');
      if (best) best.textContent = t && t.best ? t.best + '%' : '—';
      var ids = (trialEl.dataset.trackLessons || '').split(' ').filter(Boolean);
      var left = ids.filter(function (id) { return !isDone(id); }).length;
      var warn = trialEl.querySelector('[data-warn]');
      if (warn) warn.hidden = left === 0;
    }

    function show(i) {
      round.forEach(function (el, n) { el.hidden = n !== i; });
      if (atEl) atEl.textContent = String(i + 1);
      if (barEl) barEl.style.width = Math.round((i / round.length) * 100) + '%';
      if (nextBtn) {
        nextBtn.disabled = true;
        nextBtn.textContent = i === round.length - 1 ? 'Finish' : 'Next question';
      }
      if (pickedNote) {
        pickedNote.textContent = round[i] && round[i].dataset.kind === 'numeric'
          ? 'Type an answer to continue.'
          : 'Pick an answer to continue.';
      }
    }

    // One answer, recorded but not judged out loud.
    function arm(el) {
      if (el.dataset.kind === 'numeric') return armNumeric(el);

      var right = {};
      try {
        var raw = atob(el.dataset.k || '');
        (raw.indexOf('fanin:') === 0 ? raw.slice(6) : '').split(',').forEach(function (n) {
          if (n !== '') right[Number(n)] = true;
        });
      } catch (e) { /* an unreadable key marks the question wrong, not broken */ }

      var multi = el.dataset.kind === 'multi';
      el.querySelectorAll('.opt').forEach(function (btn) {
        btn.addEventListener('click', function () {
          if (!multi) {
            el.querySelectorAll('.opt').forEach(function (b) { b.dataset.picked = '0'; });
          }
          btn.dataset.picked = multi && btn.dataset.picked === '1' ? '0' : '1';
          var any = el.querySelector('.opt[data-picked="1"]');
          if (nextBtn) nextBtn.disabled = !any;
          if (pickedNote) pickedNote.textContent = any ? '' : 'Pick an answer to continue.';
        });
      });

      el.judge = function () {
        var picked = [].slice.call(el.querySelectorAll('.opt[data-picked="1"]'));
        var ok = picked.length === Object.keys(right).length &&
                 picked.every(function (b) { return right[Number(b.dataset.i)]; });
        // Mark every option so the review at the end reads on its own.
        el.querySelectorAll('.opt').forEach(function (b) {
          var i = Number(b.dataset.i);
          var chose = b.dataset.picked === '1';
          b.disabled = true;
          if (right[i]) b.dataset.mark = 'right';
          else if (chose) b.dataset.mark = 'wrong';
          if (right[i] || chose) {
            var row = b.closest('.opt-row');
            if (row) row.dataset.show = '1';
          }
        });
        el.dataset.state = 'solved';
        return ok;
      };
    }

    // A number answer in a trial: typing enables Next, and the input reports
    // right or wrong only once the round is over.
    function armNumeric(el) {
      var input = el.querySelector('input');
      var go = el.querySelector('.check-num button');
      if (go) go.hidden = true;           // Next is the only way forward here
      el.addEventListener('submit', function (e) { e.preventDefault(); });
      if (input) {
        input.addEventListener('input', function () {
          var any = !!input.value.trim();
          if (nextBtn) nextBtn.disabled = !any;
          if (pickedNote) pickedNote.textContent = any ? '' : 'Type an answer to continue.';
        });
      }
      el.judge = function () {
        var ok = input ? numericOk(el, input.value) : false;
        if (input) {
          input.disabled = true;
          input.dataset.mark = ok ? 'right' : 'wrong';
          if (!ok && !input.value.trim()) input.value = '(no answer)';
        }
        var why = el.querySelector('[data-why]');
        if (why) why.dataset.show = '1';
        el.dataset.state = 'solved';
        return ok;
      };
    }

    function begin() {
      round = shuffle(pool).slice(0, SIZE);
      at = 0;
      results = [];
      pool.forEach(function (el) {
        el.hidden = true;
        el.dataset.state = '';
        delete el.dataset.how;
        el.querySelectorAll('.opt').forEach(function (b) {
          b.disabled = false;
          b.dataset.picked = '0';
          delete b.dataset.mark;
        });
        el.querySelectorAll('.opt-row').forEach(function (r) { r.dataset.show = '0'; });
        el.querySelectorAll('input').forEach(function (inp) {
          inp.disabled = false;
          inp.value = '';
          delete inp.dataset.mark;
        });
        el.querySelectorAll('[data-why]').forEach(function (w) { w.dataset.show = '0'; });
      });
      round.forEach(function (el, n) {
        var tag = el.querySelector('.check-tag');
        if (tag) {
          tag.textContent = 'Question ' + (n + 1) +
            (el.dataset.kind === 'multi' ? ' · select all that apply' : '');
        }
      });
      intro.hidden = true;
      doneEl.hidden = true;
      runEl.hidden = false;
      show(0);
      runEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function finish() {
      var right = results.filter(Boolean).length;
      var outcome = recordTrial(TRACK, right, round.length);

      runEl.hidden = true;
      doneEl.hidden = false;

      var scoreEl = doneEl.querySelector('[data-score]');
      if (scoreEl) scoreEl.textContent = String(right);
      var verdict = doneEl.querySelector('[data-verdict]');
      if (verdict) {
        verdict.dataset.tone = outcome.passed ? 'pass' : 'fail';
        verdict.textContent = outcome.passed
          ? 'Passed — ' + trialEl.dataset.badge + ' is yours.'
          : 'Not this time. ' + Math.ceil(round.length * 0.8) + ' of ' + round.length +
            ' to pass — the answers are below.';
      }

      // The review is the round itself, judged, with every explanation open.
      var review = doneEl.querySelector('[data-review]');
      if (review) {
        review.innerHTML = '';
        round.forEach(function (el, n) {
          if (results[n]) return;   // only what went wrong is worth re-reading
          el.hidden = false;
          review.appendChild(el);
        });
        if (!review.children.length) {
          var p = document.createElement('p');
          p.className = 'trial-note';
          p.textContent = 'Every question right. Nothing to review.';
          review.appendChild(p);
        }
      }

      if (outcome.xp) {
        showToast('Trial passed', '+' + outcome.xp + ' XP · ' + trialEl.dataset.badge, 'badge');
      }
      paintProgress();
      paintIntro();
      doneEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    pool.forEach(arm);
    paintIntro();

    var startBtn = trialEl.querySelector('[data-start]');
    if (startBtn) startBtn.addEventListener('click', begin);
    var retryBtn = trialEl.querySelector('[data-retry]');
    if (retryBtn) retryBtn.addEventListener('click', begin);

    if (nextBtn) {
      nextBtn.addEventListener('click', function () {
        var el = round[at];
        if (!el) return;
        results[at] = el.judge();
        at++;
        if (at >= round.length) finish();
        else show(at);
      });
    }
  })();

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

  /* --------------------------------------------- suggest a change ------ */
  // The site is static, so a change request is a prefilled GitHub issue form.
  // Nothing is posted from the page; the reader lands on GitHub with the
  // lesson, the URL and any passage they selected already filled in.

  var REPO = document.documentElement.dataset.repo || '';

  var SUGGEST_KINDS = [
    { id: 'correction', label: 'Something is wrong',
      hint: 'A formula, a claim, or code that does not run.' },
    { id: 'addition', label: 'Something is missing',
      hint: 'A caveat, an example, a section, or a whole lesson.' },
    { id: 'deletion', label: 'Something should go',
      hint: 'Outdated, duplicated, or misleading.' },
  ];

  function lessonContext() {
    var btn = document.querySelector('[data-suggest]');
    if (!btn) return null;
    return {
      number: btn.dataset.lessonNumber || '',
      title: btn.dataset.lessonTitle || '',
      url: window.location.href.split('#')[0],
    };
  }

  function issueUrl(kind, quote) {
    var ctx = lessonContext();
    if (!REPO || !ctx) return REPO;
    var lesson = (ctx.number + ' ' + ctx.title).trim();
    var params = new URLSearchParams();
    params.set('template', kind + '.yml');
    params.set('title', kind.charAt(0).toUpperCase() + kind.slice(1) + ': ' + lesson);
    params.set('lesson', lesson);
    params.set('url', ctx.url);
    if (quote) {
      // The quote fields render as markdown, so a blockquote keeps it readable.
      var trimmed = quote.replace(/\s+/g, ' ').trim().slice(0, 1500);
      params.set(kind === 'addition' ? 'addition' : 'quote', '> ' + trimmed);
    }
    return REPO + '/issues/new?' + params.toString();
  }

  function openSuggestDialog(quote) {
    if (document.querySelector('.suggest-backdrop')) return;
    var opener = document.activeElement;

    var backdrop = document.createElement('div');
    backdrop.className = 'suggest-backdrop';
    backdrop.innerHTML =
      '<div class="suggest-dialog" role="dialog" aria-modal="true" aria-labelledby="suggest-h">' +
      '<div class="suggest-head"><h2 id="suggest-h">Request a change</h2>' +
      '<button class="icon-btn" type="button" data-close aria-label="Close">' +
      '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">' +
      '<path d="M5 5l10 10M15 5L5 15"/></svg></button></div>' +
      (quote
        ? '<blockquote class="suggest-quote">' + escapeHtml(quote.slice(0, 280)) +
          (quote.length > 280 ? '…' : '') + '</blockquote>'
        : '') +
      '<div class="suggest-kinds">' +
      SUGGEST_KINDS.map(function (k) {
        return '<a class="suggest-kind" href="' + issueUrl(k.id, quote) + '"' +
               ' target="_blank" rel="noopener noreferrer">' +
               '<span class="k-label">' + k.label + '</span>' +
               '<span class="k-hint">' + k.hint + '</span></a>';
      }).join('') +
      '</div>' +
      '<p class="suggest-foot">Opens a prefilled issue on GitHub. ' +
      'You need an account there; nothing is sent from this page.</p>' +
      '</div>';

    document.body.appendChild(backdrop);
    document.body.style.overflow = 'hidden';

    function close() {
      backdrop.remove();
      document.body.style.overflow = '';
      if (opener && opener.focus) opener.focus();
    }
    backdrop.addEventListener('click', function (e) {
      if (e.target === backdrop || e.target.closest('[data-close]') || e.target.closest('a')) close();
    });
    backdrop.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
    });
    document.addEventListener('keydown', function esc(e) {
      if (e.key === 'Escape' && document.body.contains(backdrop)) {
        close();
        document.removeEventListener('keydown', esc);
      }
    });
    var first = backdrop.querySelector('.suggest-kind');
    if (first) first.focus();
  }

  document.querySelectorAll('[data-suggest]').forEach(function (el) {
    el.addEventListener('click', function () { openSuggestDialog(''); });
  });

  // Selecting a passage inside a lesson offers to quote it into the request.
  var quoteChip = null;
  var quoteTimer = null;

  function hideQuoteChip() {
    if (quoteChip) { quoteChip.remove(); quoteChip = null; }
  }

  function showQuoteChip(text, rect) {
    hideQuoteChip();
    quoteChip = document.createElement('button');
    quoteChip.type = 'button';
    quoteChip.className = 'quote-chip';
    quoteChip.textContent = 'Suggest a change to this';
    quoteChip.style.top = (window.scrollY + rect.top - 42) + 'px';
    quoteChip.style.left = (window.scrollX + rect.left + rect.width / 2) + 'px';
    quoteChip.addEventListener('mousedown', function (e) { e.preventDefault(); });
    quoteChip.addEventListener('click', function () {
      var t = text;
      hideQuoteChip();
      openSuggestDialog(t);
    });
    document.body.appendChild(quoteChip);
  }

  var proseEl = document.querySelector('.lesson-main .prose');
  if (proseEl && REPO) {
    document.addEventListener('selectionchange', function () {
      // Settle at the end of the drag rather than firing per character.
      clearTimeout(quoteTimer);
      quoteTimer = setTimeout(function () {
        var sel = window.getSelection();
        if (!sel || sel.isCollapsed || !sel.rangeCount) { hideQuoteChip(); return; }
        var text = sel.toString().trim();
        if (text.length < 12) { hideQuoteChip(); return; }
        var node = sel.anchorNode;
        if (!node || !proseEl.contains(node.nodeType === 1 ? node : node.parentNode)) {
          hideQuoteChip();
          return;
        }
        var rect = sel.getRangeAt(0).getBoundingClientRect();
        if (rect.width || rect.height) showQuoteChip(text, rect);
      }, 250);
    });
    window.addEventListener('scroll', hideQuoteChip, { passive: true });
  }

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
