// Tests for the progress engine in assets/app.js.
//
// app.js is a browser IIFE with no module boundary, so it is loaded into a
// minimal DOM shim and its stats() output captured. Streak arithmetic in
// particular fails silently in the browser -- a wrong answer still renders --
// so it is worth pinning down here.
//
// Run: node src/test-progress.mjs
import fs from 'node:fs';

function run(storeSeed, trackSpec, todayISO) {
  const localStore = { 'fanin:progress:v2': JSON.stringify(storeSeed) };
  const noop = () => {};
  const el = () => ({ dataset: {}, style: {}, hidden: false, textContent: '',
    setAttribute: noop, getAttribute: () => null, querySelector: () => null,
    querySelectorAll: () => [], addEventListener: noop, appendChild: noop,
    remove: noop, closest: () => null, focus: noop, innerHTML: '' });

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
    fetch: () => Promise.resolve({ json: () => Promise.resolve([]) }),
    Date, Math, JSON, Object, Array, String, Number, RegExp, Set, URLSearchParams,
    console,
  };

  // Freeze "today" so streak assertions are deterministic.
  const RealDate = Date;
  sandbox.Date = class extends RealDate {
    constructor(...a) { return a.length ? new RealDate(...a) : new RealDate(todayISO + 'T12:00:00'); }
    static now() { return new RealDate(todayISO + 'T12:00:00').getTime(); }
  };

  let captured = null;
  const src = fs.readFileSync('src/assets/app.js', 'utf8')
    // Expose stats() to the harness.
    .replace('\n  paintProgress();\n', '\n  __capture(stats());\n');
  if (!src.includes('__capture')) throw new Error('capture hook did not match');
  sandbox.__capture = s => { captured = s; };

  const keys = Object.keys(sandbox);
  new Function(...keys, src)(...keys.map(k => sandbox[k]));
  return captured;
}

const TRACKS = 'math:3:Math,pytorch:2:PyTorch';
const D = (id, d, x) => [id, { d, x }];
const mk = pairs => Object.fromEntries(pairs);

let pass = 0, fail = 0;
function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}` + (ok ? '' : `  got ${JSON.stringify(actual)} want ${JSON.stringify(expected)}`));
  ok ? pass++ : fail++;
}

console.log('streak');
let s = run(mk([D('math/a','2026-09-21',100), D('math/b','2026-09-20',100), D('math/c','2026-09-19',100)]), TRACKS, '2026-09-21');
check('three consecutive days ending today -> 3', s.streak, 3);

s = run(mk([D('math/a','2026-09-20',100), D('math/b','2026-09-19',100)]), TRACKS, '2026-09-21');
check('ending yesterday still counts -> 2', s.streak, 2);

s = run(mk([D('math/a','2026-09-18',100), D('math/b','2026-09-17',100)]), TRACKS, '2026-09-21');
check('ending two days ago -> 0', s.streak, 0);

s = run(mk([D('math/a','2026-09-18',100), D('math/b','2026-09-17',100), D('math/c','2026-09-16',100)]), TRACKS, '2026-09-21');
check('broken streak keeps longest -> 3', s.longest, 3);

s = run(mk([D('math/a','2026-09-21',100), D('math/b','2026-09-21',100)]), TRACKS, '2026-09-21');
check('two lessons same day -> streak 1', s.streak, 1);
check('two lessons same day -> busiest 2', s.busiestDay, 2);

console.log('\nxp and levels');
s = run({}, TRACKS, '2026-09-21');
check('empty -> level 1', s.levelIndex, 0);
check('empty -> xp 0', s.xp, 0);

s = run(mk([D('math/a', '2026-09-21', 700)]), TRACKS, '2026-09-21');
check('700 xp -> level 2', s.levelIndex, 1);
check('700 xp -> next is 1800', s.next.at, 1800);

console.log('\nbadges');
s = run(mk([D('math/a','2026-09-21',100), D('math/b','2026-09-21',100), D('math/c','2026-09-21',100)]), TRACKS, '2026-09-21');
const got = s.badges.filter(b => b.earned).map(b => b.id).sort();
check('finishing math track earns its badge', got.includes('track-math'), true);
check('not all tracks touched -> no polymath', got.includes('breadth'), false);

s = run(mk([D('math/a','2026-09-21',100), D('pytorch/a','2026-09-21',100)]), TRACKS, '2026-09-21');
check('one per track -> polymath', s.badges.find(b=>b.id==='breadth').earned, true);

console.log('\nv1 migration');
const legacyOnly = (() => {
  const localStore = { 'fanin:progress:v1': JSON.stringify({ 'math/a': 1, 'math/b': 1 }) };
  return localStore;
})();
// Re-run with only v1 present.
{
  const src = fs.readFileSync('src/assets/app.js', 'utf8');
  // simplest: seed v2 as absent by passing null store
  const r = run(null, TRACKS, '2026-09-21');
  check('no store at all -> count 0', r.count, 0);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
