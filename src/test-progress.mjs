// Tests for the progress engine in assets/app.js.
//
// app.js is a browser IIFE with no module boundary, so it is loaded into a
// minimal DOM shim and its internals handed back through a capture hook.
// Streak arithmetic and XP accounting fail silently in the browser -- a wrong
// answer still renders -- so they are pinned down here.
//
// Run: node src/test-progress.mjs
import fs from 'node:fs';

const TRACKS = 'math:3:Math,pytorch:2:PyTorch';

// Boots app.js over a seeded localStorage and returns its progress API.
function boot(seed, todayISO, trackSpec = TRACKS) {
  const localStore = {};
  for (const [k, v] of Object.entries(seed || {})) localStore[k] = JSON.stringify(v);

  const noop = () => {};
  const el = () => ({ dataset: {}, style: {}, hidden: false, textContent: '',
    setAttribute: noop, getAttribute: () => null, querySelector: () => null,
    querySelectorAll: () => [], addEventListener: noop, appendChild: noop,
    remove: noop, closest: () => null, focus: noop, innerHTML: '', disabled: false });

  const sandbox = {
    localStorage: {
      getItem: k => (k in localStore ? localStore[k] : null),
      setItem: (k, v) => { localStore[k] = v; },
      removeItem: k => { delete localStore[k]; },
    },
    document: {
      documentElement: { dataset: { tracks: trackSpec }, setAttribute: noop,
                         removeAttribute: noop },
      querySelector: () => null,
      querySelectorAll: () => [],
      getElementById: () => null,
      addEventListener: noop,
      createElement: el,
      body: { appendChild: noop, style: {} },
      fonts: { size: 0 },
    },
    window: { addEventListener: noop, location: { href: 'x' }, matchMedia: () => ({ matches: false }) },
    navigator: {}, matchMedia: () => ({ matches: false }),
    setTimeout: noop, clearTimeout: noop, requestIdleCallback: noop,
    atob: s => Buffer.from(s, 'base64').toString('binary'),
    isFinite, parseInt, parseFloat,
    fetch: () => Promise.resolve({ json: () => Promise.resolve([]) }),
    Date, Math, JSON, Object, Array, String, Number, RegExp, Set, URLSearchParams, Boolean,
    console,
  };

  // Freeze "today" so streak assertions are deterministic.
  const RealDate = Date;
  sandbox.Date = class extends RealDate {
    constructor(...a) { return a.length ? new RealDate(...a) : new RealDate(todayISO + 'T12:00:00'); }
    static now() { return new RealDate(todayISO + 'T12:00:00').getTime(); }
  };

  let api = null;
  const HOOK = '\n  paintProgress();\n';
  const src = fs.readFileSync('src/assets/app.js', 'utf8').replace(
    HOOK,
    '\n  __capture({ stats: stats, setDone: setDone, recordCheck: recordCheck,' +
    ' recordTrial: recordTrial, breakCombo: breakCombo, raw: function () { return store; } });\n'
  );
  if (!src.includes('__capture')) throw new Error('capture hook did not match app.js');
  sandbox.__capture = a => { api = a; };

  const keys = Object.keys(sandbox);
  new Function(...keys, src)(...keys.map(k => sandbox[k]));
  return api;
}

const V3 = (pairs) => ({ 'fanin:progress:v3': Object.fromEntries(pairs) });
const L = (id, d, x, a = {}) => [id, { c: 1, d, x, a }];

let pass = 0, fail = 0;
function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}` +
    (ok ? '' : `  got ${JSON.stringify(actual)} want ${JSON.stringify(expected)}`));
  ok ? pass++ : fail++;
}

console.log('streak');
let s = boot(V3([L('math/a','2026-09-21',100), L('math/b','2026-09-20',100), L('math/c','2026-09-19',100)]), '2026-09-21').stats();
check('three consecutive days ending today -> 3', s.streak, 3);

s = boot(V3([L('math/a','2026-09-20',100), L('math/b','2026-09-19',100)]), '2026-09-21').stats();
check('ending yesterday still counts -> 2', s.streak, 2);

s = boot(V3([L('math/a','2026-09-18',100), L('math/b','2026-09-17',100)]), '2026-09-21').stats();
check('ending two days ago -> 0', s.streak, 0);

s = boot(V3([L('math/a','2026-09-18',100), L('math/b','2026-09-17',100), L('math/c','2026-09-16',100)]), '2026-09-21').stats();
check('broken streak keeps longest -> 3', s.longest, 3);

s = boot(V3([L('math/a','2026-09-21',100), L('math/b','2026-09-21',100)]), '2026-09-21').stats();
check('two lessons same day -> streak 1', s.streak, 1);
check('two lessons same day -> busiest 2', s.busiestDay, 2);

// A lesson with answered checks but not yet cleared must not put a day on the
// calendar -- otherwise a half-finished lesson silently props up a streak.
s = boot({ 'fanin:progress:v3': { 'math/a': { c: 0, d: null, x: 40, a: { 0: 1, 1: 1 } } } }, '2026-09-21').stats();
check('answers without a clear -> streak 0', s.streak, 0);
check('answers without a clear -> count 0', s.count, 0);
check('answers without a clear -> xp still banked', s.xp, 40);

console.log('\nchecks and xp');
let api = boot({}, '2026-09-21');
check('first try pays 20', api.recordCheck('math/a', 0, true).xp, 20);
check('after a miss pays 8', api.recordCheck('math/a', 1, false).xp, 8);
check('re-answering pays nothing', api.recordCheck('math/a', 0, true).xp, 0);
check('xp banked on the lesson', api.stats().xp, 28);
check('checks counted', api.stats().checksRight, 2);
check('accuracy is first-try share', api.stats().accuracy, 50);

api = boot({}, '2026-09-21');
for (let i = 0; i < 4; i++) api.recordCheck('math/a', i, true);
check('combo builds', api.stats().combo, 4);
check('no bonus before the fifth', api.stats().xp, 80);
const fifth = api.recordCheck('math/a', 4, true);
check('fifth in a row pays the combo bonus', fifth.bonus, 25);
check('bonus lands in the total', api.stats().xp, 80 + 20 + 25);
api.recordCheck('math/a', 5, false);
check('a miss resets the combo', api.stats().combo, 0);
check('best combo is kept', api.stats().bestCombo, 5);

api = boot({}, '2026-09-21');
api.recordCheck('math/a', 0, true);
api.setDone('math/a', true, 12);
check('clearing adds minutes x 5', api.stats().xp, 20 + 60);
check('clearing counts the lesson', api.stats().count, 1);
api.setDone('math/a', false, 12);
check('un-clearing returns only the clear bonus', api.stats().xp, 20);
check('un-clearing keeps the answered check', api.stats().checksRight, 1);

console.log('\nlevels');
s = boot({}, '2026-09-21').stats();
check('empty -> level 1', s.levelIndex, 0);
check('empty -> xp 0', s.xp, 0);
s = boot(V3([L('math/a','2026-09-21',700)]), '2026-09-21').stats();
check('700 xp -> level 2', s.levelIndex, 1);
check('700 xp -> next at 1400', s.next.at, 1400);
s = boot(V3([L('math/a','2026-09-21',16500)]), '2026-09-21').stats();
check('16500 xp -> top level', s.level.name, 'Compute Optimal');
check('top level has no next', s.next, null);

console.log('\ntrials');
api = boot(V3([L('math/a','2026-09-21',10), L('math/b','2026-09-21',10), L('math/c','2026-09-21',10)]), '2026-09-21');
check('track badge withheld until the trial is passed',
  api.stats().badges.find(b => b.id === 'track-math').earned, false);
check('failing a trial pays nothing', api.recordTrial('math', 5, 10).xp, 0);
check('passing a trial pays 400', api.recordTrial('math', 9, 10).xp, 400);
check('a passed trial is in the xp total', api.stats().xp, 30 + 400);
check('track badge earned once both are done',
  api.stats().badges.find(b => b.id === 'track-math').earned, true);
check('passing again pays nothing', api.recordTrial('math', 10, 10).xp, 0);
check('best score is kept', api.stats().trials.math.best, 100);

console.log('\nbadges');
s = boot(V3([L('math/a','2026-09-21',100), L('pytorch/a','2026-09-21',100)]), '2026-09-21').stats();
check('one per track -> polymath', s.badges.find(b => b.id === 'breadth').earned, true);
check('two tracks is not all lessons', s.badges.find(b => b.id === 'all').earned, false);

console.log('\ndaily quests');
s = boot({}, '2026-09-21').stats();
check('three quests a day', s.quests.length, 3);
check('quests start unfinished', s.quests.every(q => !q.done), true);
const sameDay = boot({}, '2026-09-21').stats().quests.map(q => q.id);
const nextDay = boot({}, '2026-09-22').stats().quests.map(q => q.id);
check('quests are stable within a day', boot({}, '2026-09-21').stats().quests.map(q => q.id), sameDay);
check('quests differ the next day', JSON.stringify(nextDay) !== JSON.stringify(sameDay), true);

console.log('\nmigration');
check('no store at all -> count 0', boot({}, '2026-09-21').stats().count, 0);
api = boot({ 'fanin:progress:v2': { 'math/a': { d: '2026-09-20', x: 120 } } }, '2026-09-21');
check('v2 lesson survives', api.stats().count, 1);
check('v2 xp is rescaled to the new rate', api.stats().xp, 60);
check('v2 streak survives', api.stats().streak, 1);
api = boot({ 'fanin:progress:v1': { 'math/a': 1, 'math/b': 1 } }, '2026-09-21');
check('v1 completions survive without dates', api.stats().count, 2);
check('v1 completions add nothing to the streak', api.stats().streak, 0);
check('v1 lessons keep no invented xp', api.stats().xp, 0);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
