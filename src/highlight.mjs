// Build-time syntax highlighting. No client JS, no CDN, no dependency.
// Covers the languages this curriculum actually uses.

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const PY_KEYWORDS = new Set(
  ('False None True and as assert async await break class continue def del elif ' +
   'else except finally for from global if import in is lambda nonlocal not or ' +
   'pass raise return try while with yield match case').split(' ')
);
const PY_BUILTINS = new Set(
  ('abs all any bool dict enumerate float format getattr hasattr int isinstance ' +
   'len list map max min next object open print range repr res reversed round set ' +
   'setattr slice sorted str sum super tuple type zip staticmethod classmethod ' +
   'property NotImplementedError ValueError TypeError RuntimeError IndexError ' +
   'KeyError AssertionError Exception self cls').split(' ')
);
const JS_KEYWORDS = new Set(
  ('async await break case catch class const continue default delete do else ' +
   'export extends finally for from function if import in instanceof let new of ' +
   'return static super switch this throw try typeof var void while yield true ' +
   'false null undefined').split(' ')
);
const SH_KEYWORDS = new Set(
  ('if then else elif fi for while do done case esac function return export ' +
   'local source cd echo set unset').split(' ')
);

// Ordered rules. The first pattern that matches at the cursor wins, so more
// specific patterns (triple-quoted strings, decorators) must come first.
const RULES = {
  python: [
    ['comment', /^#[^\n]*/],
    ['str', /^[fFrRbBuU]{0,2}"""[\s\S]*?"""/],
    ['str', /^[fFrRbBuU]{0,2}'''[\s\S]*?'''/],
    ['str', /^[fFrRbBuU]{0,2}"(?:\\.|[^"\\\n])*"/],
    ['str', /^[fFrRbBuU]{0,2}'(?:\\.|[^'\\\n])*'/],
    ['decorator', /^@[A-Za-z_][\w.]*/],
    ['num', /^0[xXbBoO][0-9a-fA-F_]+|^\d[\d_]*\.?[\d_]*(?:[eE][-+]?\d+)?j?/],
    ['word', /^[A-Za-z_]\w*/],
    ['op', /^(?:\*\*=?|\/\/=?|[-+*/%@&|^~<>=!]=?|<<=?|>>=?|->|:=)/],
    ['punct', /^[()[\]{},;:.]/],
  ],
  javascript: [
    ['comment', /^\/\/[^\n]*/],
    ['comment', /^\/\*[\s\S]*?\*\//],
    ['str', /^`(?:\\.|[^`\\])*`/],
    ['str', /^"(?:\\.|[^"\\\n])*"/],
    ['str', /^'(?:\\.|[^'\\\n])*'/],
    ['num', /^0[xXbBoO][0-9a-fA-F_]+|^\d[\d_]*\.?[\d_]*(?:[eE][-+]?\d+)?n?/],
    ['word', /^[A-Za-z_$][\w$]*/],
    ['op', /^(?:=>|\.{3}|\?\?=?|\?\.|&&=?|\|\|=?|\*\*=?|[-+*/%&|^~<>=!]=?)/],
    ['punct', /^[()[\]{},;:.]/],
  ],
  bash: [
    ['comment', /^#[^\n]*/],
    ['str', /^"(?:\\.|[^"\\])*"/],
    ['str', /^'[^']*'/],
    ['var', /^\$\{[^}]*\}|^\$[A-Za-z_]\w*|^\$[0-9@*#?]/],
    ['flag', /^(?<=\s)--?[A-Za-z][\w-]*/],
    ['num', /^\d+/],
    ['word', /^[A-Za-z_][\w.-]*/],
    ['op', /^(?:&&|\|\||[|&<>=!]=?)/],
    ['punct', /^[()[\]{},;:]/],
  ],
  json: [
    ['key', /^"(?:\\.|[^"\\])*"(?=\s*:)/],
    ['str', /^"(?:\\.|[^"\\])*"/],
    ['num', /^-?\d+\.?\d*(?:[eE][-+]?\d+)?/],
    ['kw', /^(?:true|false|null)\b/],
    ['punct', /^[()[\]{},:]/],
  ],
};

RULES.py = RULES.python;
RULES.js = RULES.javascript;
RULES.ts = RULES.javascript;
RULES.sh = RULES.bash;
RULES.shell = RULES.bash;
RULES.console = RULES.bash;

function classify(lang, kind, text, rest) {
  if (kind !== 'word') return kind;
  if (lang === 'python' || lang === 'py') {
    if (PY_KEYWORDS.has(text)) return 'kw';
    if (/^\s*\(/.test(rest)) return 'fn';
    if (PY_BUILTINS.has(text)) return 'builtin';
    if (/^[A-Z]/.test(text)) return 'type';
    return null;
  }
  if (RULES[lang] === RULES.javascript) {
    if (JS_KEYWORDS.has(text)) return 'kw';
    if (/^\s*\(/.test(rest)) return 'fn';
    if (/^[A-Z]/.test(text)) return 'type';
    return null;
  }
  if (RULES[lang] === RULES.bash) {
    if (SH_KEYWORDS.has(text)) return 'kw';
    return null;
  }
  return null;
}

export function highlight(code, lang) {
  const key = (lang || '').toLowerCase();
  const rules = RULES[key];
  if (!rules) return esc(code);

  let out = '';
  let src = code;
  let guard = 0;

  while (src.length && guard++ < 200000) {
    // Whitespace passes through untouched so indentation survives.
    const ws = /^\s+/.exec(src);
    if (ws) {
      out += esc(ws[0]);
      src = src.slice(ws[0].length);
      continue;
    }

    let matched = false;
    for (const [kind, re] of rules) {
      const m = re.exec(src);
      if (!m || !m[0].length) continue;
      const text = m[0];
      const rest = src.slice(text.length);
      const cls = classify(key, kind, text, rest);
      out += cls ? `<span class="t-${cls}">${esc(text)}</span>` : esc(text);
      src = rest;
      matched = true;
      break;
    }
    if (!matched) {
      out += esc(src[0]);
      src = src.slice(1);
    }
  }
  return out + esc(src);
}
