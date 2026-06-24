"""
data.py — local SEDRA IV reader + Syriac morphological parser

Reads the SEDRA IV records already mirrored to the shared ``corpora`` package
(``corpora/sedra_cache/api/{root,lexeme,word}``) — no network needed — and parses
each inflected form into a morphological triple:

  (root_consonants: tuple[str], template: tuple[str], surface: str)

Field map (SEDRA IV ``word`` record):
  ``syriac``   consonantal skeleton -> kept as the graph's surface target
  ``western``  vocalised form (vowels in U+0730–U+074A); the template is read
               from here, since the consonantal ``syriac`` carries no vowels
A word's root consonants come from its lexeme's root, resolved
word -> lexeme.id -> root via the small root index (see _load_lexeme_to_root).

Syriac consonants (Estrangela): ܐ ܒ ܓ ܕ ܗ ܘ ܙ ܚ ܛ ܝ ܟ ܠ ܡ ܢ ܣ ܥ ܦ ܨ ܩ ܪ ܫ ܬ
(22 consonants, U+0710–U+072F); vowel/diacritic marks are U+0730–U+074A.

SEDRA is license-restricted (cite Kiraz); the cache is git-ignored and never
committed. Rebuild it with ``.venv/bin/python -m corpora.sedra_scrape``.

``build_synthetic_dataset`` still fabricates Syriac-like data from hand-written
templates as an offline fallback when the SEDRA cache is absent.
"""

import json
import os
import random
from typing import Optional
from collections import defaultdict

try:
    from corpora import SEDRA_WORD_DIR, SEDRA_ROOT_DIR, sedra_word_dir
except Exception:  # corpora not importable (e.g. run outside the repo root)
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SEDRA_WORD_DIR = os.path.join(_REPO, "corpora", "sedra_cache", "api", "word")
    SEDRA_ROOT_DIR = os.path.join(_REPO, "corpora", "sedra_cache", "api", "root")

    def sedra_word_dir(require: bool = False) -> str:
        if require and not os.path.isdir(SEDRA_WORD_DIR):
            raise FileNotFoundError(
                f"No SEDRA IV word records at {SEDRA_WORD_DIR}. SEDRA is "
                "license-restricted and not shipped; rebuild with "
                ".venv/bin/python -m corpora.sedra_scrape (see neural/docs/DATA.md).")
        return SEDRA_WORD_DIR

# ── Syriac Unicode ──────────────────────────────────────────────────────────

# Syriac consonant codepoints (U+0710–U+072F active range)
SYRIAC_CONSONANTS = set(chr(c) for c in range(0x0710, 0x0730))

# The 22 Estrangela consonants as an ordered list (re-exported for train.py and
# mirrored by graph.SYRIAC_CONSONANT_LIST).
SYRIAC_CONSONANT_LIST = [
    'ܐ', 'ܒ', 'ܓ', 'ܕ', 'ܗ', 'ܘ', 'ܙ', 'ܚ', 'ܛ', 'ܝ',
    'ܟ', 'ܠ', 'ܡ', 'ܢ', 'ܣ', 'ܥ', 'ܦ', 'ܨ', 'ܩ', 'ܪ', 'ܫ', 'ܬ',
]

# Syriac vowel/diacritic marks (U+0730–U+074A)  
SYRIAC_DIACRITICS = set(chr(c) for c in range(0x0730, 0x074B))

# Common word-internal patterns to strip (particles, enclitics)
SYRIAC_PREFIX_CHARS = {'ܒ', 'ܕ', 'ܘ', 'ܠ', 'ܡ', 'ܢ', 'ܦ', 'ܟ'}

def consonants_only(word: str) -> str:
    """Strip all vowel marks/diacritics from a Syriac string."""
    return ''.join(c for c in word if c not in SYRIAC_DIACRITICS)

def extract_root_consonants(word: str) -> list[str]:
    """Return just the consonants of a word as an ordered list."""
    return [c for c in consonants_only(word) if c in SYRIAC_CONSONANTS]

def extract_template(word: str) -> list[str]:
    """
    Extract a template pattern from a vocalized Syriac word.

    Each consonant slot is labeled by what follows it:
      'C'  — consonant with no vowel (shewa or zero)
      'Ca' — consonant followed by patah (a-vowel)
      'Ci' — consonant followed by hbasa (i-vowel)
      'Cu' — consonant followed by rwaha (u-vowel)
      'Ce' — consonant followed by zqapa (e-vowel)
      'Co' — consonant followed by esasa (o-vowel)
      'C:' — consonant with diaeresis (geminate mark)

    Vowel diacritic assignments (approximate Syriac Eastern tradition):
      U+0730 — Pthaha (a)
      U+0731 — Pthaha raised (a variant)
      U+0732 — Pthaha dotted
      U+0733 — Zqapha (a/e)
      U+0734 — Zqapha raised
      U+0735 — Zqapha dotted
      U+0736 — Rbasa (i/e)
      U+0737 — Rbasa dotted
      U+0738 — Hbasa (i)
      U+0739 — Hbasa dotted
      U+073A — Hbasa-Esasa (compound)
      U+073B — Esasa reversed (u)
      U+073C — Esasa (u)
      U+073D — Zqapha-Esasa
      U+073E — Rwaha (o)
      U+073F — Dotted Zlama horizontal (i-variant)
    """
    VOWEL_CLASSES = {
        '\u0730': 'a', '\u0731': 'a', '\u0732': 'a',  # pthaha variants
        '\u0733': 'e', '\u0734': 'e', '\u0735': 'e',  # zqapha variants
        '\u0736': 'i', '\u0737': 'i', '\u0738': 'i', '\u0739': 'i',  # rbasa/hbasa
        '\u073A': 'iu', '\u073B': 'u', '\u073C': 'u', '\u073D': 'eu',
        '\u073E': 'o', '\u073F': 'i',
    }

    slots = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in SYRIAC_CONSONANTS:
            # Collect any diacritics following this consonant
            j = i + 1
            vowel_label = ''
            while j < len(chars) and chars[j] in SYRIAC_DIACRITICS:
                diac = chars[j]
                if diac in VOWEL_CLASSES and not vowel_label:
                    vowel_label = VOWEL_CLASSES[diac]
                j += 1
            slots.append('C' + vowel_label if vowel_label else 'C')
            i = j
        else:
            i += 1
    return slots


# ── SEDRA IV API ─────────────────────────────────────────────────────────────

# The SEDRA IV API has already been mirrored to one ``<id>.json`` per record
# under the shared ``corpora`` package (``corpora/sedra_cache/api/{root,lexeme,
# word}``). We read those files directly — no network, no duplicate cache.

# Processed (root, template, surface) triples are memoised here so repeat runs
# skip the full scan of the word records (git-ignored; see defog/.gitignore).
PROCESSED_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_cache", "sedra_iv_triples.json")


def _json_records(path: str) -> list[dict]:
    """Load a cached ``<id>.json`` (a JSON array of one object) as a list."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else [data]


def _id_from_uri(uri: str) -> Optional[int]:
    """``https://.../lexeme/123.json`` -> ``123``."""
    tail = os.path.basename(uri or "").split(".")[0]
    return int(tail) if tail.isdigit() else None


def _load_lexeme_to_root(root_dir: str = SEDRA_ROOT_DIR) -> dict:
    """Map ``lexeme_id -> (root_id, root_syriac)`` from the small root index.

    Each root record lists its member lexemes as API URIs, so one pass over the
    ~3.5k root files resolves every word's root via its lexeme — far cheaper than
    reading the ~36k lexeme files.
    """
    lex2root: dict = {}
    if not os.path.isdir(root_dir):
        return lex2root
    for fn in os.listdir(root_dir):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        for rec in _json_records(os.path.join(root_dir, fn)):
            try:
                root_id = int(rec.get("id"))
            except (TypeError, ValueError):
                continue
            root_syriac = (rec.get("syriac") or "").strip()
            if not root_syriac:
                continue
            for uri in rec.get("lexemes") or []:
                lex_id = _id_from_uri(uri)
                if lex_id is not None:
                    lex2root[lex_id] = (root_id, root_syriac)
    return lex2root


def word_to_triple(root_syriac: str, rec: dict) -> Optional[tuple]:
    """Convert a SEDRA IV word record + its root string into a morphological triple.

    Returns ``(root_consonants, template, surface)`` or ``None`` if the form
    cannot be parsed reliably. The template is read from the **vocalised**
    ``western`` field (the consonantal ``syriac`` field carries no vowels); the
    surface kept for the graph is the consonantal ``syriac`` skeleton.
    """
    surface = (rec.get("syriac") or "").strip()            # consonantal skeleton
    vocalised = (rec.get("western") or rec.get("eastern") or "").strip()
    if not surface or not vocalised:
        return None

    root_cons = extract_root_consonants(root_syriac)
    surface_cons = extract_root_consonants(surface)
    # Every root consonant should appear in the surface; extra surface consonants
    # are affixes (handled as 'affix' slots when the graph is built).
    if len(root_cons) < 2 or len(surface_cons) < len(root_cons):
        return None

    template = extract_template(vocalised)
    if len(template) < 2:
        return None

    return (tuple(root_cons), tuple(template), surface)


def _morph_tag(rec: dict) -> str:
    """A short readable label from the word's morphology (for qualitative output)."""
    parts = [rec.get(k) for k in ("category", "number", "gender", "state")]
    return ".".join(p for p in parts if p) or "word"


def _build_all(verbose: bool = True, limit_words: Optional[int] = None) -> list[dict]:
    """Scan the local SEDRA IV word records into morphological-triple items."""
    word_dir = sedra_word_dir(require=True)
    lex2root = _load_lexeme_to_root()
    if not lex2root:
        raise FileNotFoundError(
            f"No SEDRA root records under {SEDRA_ROOT_DIR}; rebuild the cache with "
            ".venv/bin/python -m corpora.sedra_scrape --endpoint root")

    files = sorted(fn for fn in os.listdir(word_dir)
                   if fn.endswith(".json") and not fn.startswith("_"))
    if limit_words:
        files = files[:limit_words]

    dataset: list[dict] = []
    for i, fn in enumerate(files):
        if verbose and i and i % 10000 == 0:
            print(f"  ...scanned {i:,}/{len(files):,} word records, "
                  f"{len(dataset):,} triples")
        for rec in _json_records(os.path.join(word_dir, fn)):
            lex_id = (rec.get("lexeme") or {}).get("id")
            info = lex2root.get(lex_id) if lex_id is not None else None
            if info is None:
                continue
            root_id, root_syriac = info
            triple = word_to_triple(root_syriac, rec)
            if triple is None:
                continue
            root_cons, template, surface = triple
            dataset.append({
                "root_id": root_id,
                "root_syriac": root_syriac,
                "root_consonants": list(root_cons),
                "template": list(template),
                "surface": surface,
                "template_name": _morph_tag(rec),
                "lexeme_id": lex_id,
                "morphology": {
                    k: rec[k] for k in
                    ("category", "number", "gender", "state", "person", "tense")
                    if rec.get(k)
                },
            })
    if verbose:
        n_roots = len({d["root_id"] for d in dataset})
        print(f"  Parsed {len(dataset):,} triples from {n_roots:,} roots")
    return dataset


def _load_or_build(rebuild: bool, verbose: bool) -> list[dict]:
    if not rebuild and os.path.exists(PROCESSED_CACHE):
        if verbose:
            print(f"  Loading cached triples: {PROCESSED_CACHE}")
        with open(PROCESSED_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    dataset = _build_all(verbose=verbose)
    os.makedirs(os.path.dirname(PROCESSED_CACHE), exist_ok=True)
    with open(PROCESSED_CACHE, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False)
    return dataset


def build_dataset(
    max_roots: Optional[int] = 200,
    min_words_per_root: int = 3,
    verbose: bool = True,
    rebuild: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Build morphological triples from the local SEDRA IV cache.

    Each item is ``{root_id, root_syriac, root_consonants, template, surface,
    template_name, lexeme_id, morphology}``. The full scan is memoised; this then
    keeps roots with at least ``min_words_per_root`` forms and, if ``max_roots``
    is given, deterministically subsamples that many roots.

    Raises ``FileNotFoundError`` if the (license-restricted, git-ignored) SEDRA
    cache is absent — rebuild it with ``-m corpora.sedra_scrape``.
    """
    if verbose:
        print(f"Building dataset from local SEDRA IV cache (max_roots={max_roots})...")
    items = _load_or_build(rebuild, verbose)

    by_root: dict = defaultdict(list)
    for it in items:
        by_root[it["root_id"]].append(it)
    roots = sorted(r for r, v in by_root.items() if len(v) >= min_words_per_root)

    if max_roots and len(roots) > max_roots:
        roots = sorted(random.Random(seed).sample(roots, max_roots))

    dataset = [it for r in roots for it in by_root[r]]
    if verbose:
        print(f"  Dataset: {len(dataset):,} triples from {len(roots):,} roots "
              f"(>= {min_words_per_root} forms each)")
    return dataset


def build_synthetic_dataset(n_samples: int = 2000) -> list[dict]:
    """
    Build a synthetic dataset that mimics Syriac morphological structure,
    for testing the architecture without needing API access.

    Uses real Syriac consonant inventory and realistic template patterns
    derived from Classical Syriac verbal and nominal binyanim.

    Synthetic roots: random triples/quadruples from Syriac consonant inventory.
    Synthetic templates: drawn from a set of attested Syriac patterns.
    Surface: generated by interdigitating root into template.
    """
    import random
    random.seed(42)

    # Real Syriac consonants (Estrangela)
    CONSONANTS = [
        'ܐ', 'ܒ', 'ܓ', 'ܕ', 'ܗ', 'ܘ', 'ܙ', 'ܚ', 'ܛ', 'ܝ',
        'ܟ', 'ܠ', 'ܡ', 'ܢ', 'ܣ', 'ܥ', 'ܦ', 'ܨ', 'ܩ', 'ܪ', 'ܫ', 'ܬ'
    ]

    # Template patterns: (slot_labels, surface_template_fn)
    # Slot labels describe the vocalic environment of each root consonant
    # Patterns derived from Syriac verbal paradigms (Peal, Pael, Aphel, etc.)
    TEMPLATES = [
        # Peal perfect (C1eC2eC3):   ܟܬܒ → CeCeC
        ('peal_perf',   ['Ce', 'Ce', 'C'],   lambda r: f"{r[0]}\u0736{r[1]}\u0736{r[2]}"),
        # Peal imperfect (niC1C2uC3): ܟܬܒ → niC1C2uC3
        ('peal_impf',   ['C', 'Cu', 'C'],    lambda r: f"ܢ{r[0]}{r[1]}\u073C{r[2]}"),
        # Pael perfect (C1aCC2eC3):  doubled middle
        ('pael_perf',   ['Ca', 'C:', 'Ce', 'C'], lambda r: f"{r[0]}\u0730{r[1]}{r[1]}\u0736{r[2]}" if len(r)>=3 else ""),
        # Aphel perfect (aC1C2iC3):
        ('aphel_perf',  ['C', 'Ci', 'C'],   lambda r: f"ܐ{r[0]}{r[1]}\u0738{r[2]}"),
        # Active participle (C1aC2eC3):
        ('part_act',    ['Ca', 'Ce', 'C'],   lambda r: f"{r[0]}\u0730{r[1]}\u0736{r[2]}"),
        # Passive participle (C1C2iC3):
        ('part_pass',   ['C', 'Ci', 'C'],    lambda r: f"{r[0]}{r[1]}\u0738{r[2]}"),
        # Verbal noun (C1aC2C3a):
        ('verb_noun',   ['Ca', 'C', 'Ca'],   lambda r: f"{r[0]}\u0730{r[1]}{r[2]}\u0730"),
        # Absolute state noun (C1iC2C3a):
        ('abs_noun',    ['Ci', 'C', 'Ca'],   lambda r: f"{r[0]}\u0738{r[1]}{r[2]}\u0730"),
        # Construct state (C1eC2C3at):
        ('cst_noun',    ['Ce', 'C', 'Cat'],  lambda r: f"{r[0]}\u0736{r[1]}{r[2]}\u0730ܬ"),
        # Place noun (maC1C2aC3a):
        ('place_noun',  ['C', 'Ca', 'Ca'],   lambda r: f"ܡ{r[0]}{r[1]}\u0730{r[2]}\u0730"),
    ]

    # Generate synthetic roots (unique triliteral consonant sets)
    all_roots = []
    seen = set()
    attempts = 0
    while len(all_roots) < 150 and attempts < 10000:
        attempts += 1
        n_radicals = random.choice([3, 3, 3, 4])  # mostly triliteral
        root = tuple(random.sample(CONSONANTS, n_radicals))
        key = frozenset(root)
        if key not in seen and len(root) == len(set(root)):
            seen.add(key)
            all_roots.append(root)

    dataset = []
    sample_id = 0

    for root_idx, root in enumerate(all_roots):
        n_triliteral = len(root) == 3
        applicable_templates = [t for t in TEMPLATES if len(t[1]) <= len(root) + 1]

        for templ_name, templ_slots, surface_fn in applicable_templates:
            if len(root) < 3:
                continue
            try:
                surface = surface_fn(root)
                if not surface:
                    continue
            except Exception:
                continue

            dataset.append({
                'root_id': root_idx,
                'root_syriac': ''.join(root),
                'root_consonants': list(root),
                'template': templ_slots,
                'template_name': templ_name,
                'surface': surface,
                'lexeme_id': sample_id,
                'morphology': {'template_name': templ_name, 'synthetic': True}
            })
            sample_id += 1

    random.shuffle(dataset)
    print(f"Built synthetic dataset: {len(dataset)} triples, {len(all_roots)} roots")
    print(f"Template distribution: { {t[0]: sum(1 for d in dataset if d.get('template_name')==t[0]) for t in TEMPLATES} }")
    return dataset
