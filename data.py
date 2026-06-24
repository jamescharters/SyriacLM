"""
data.py — SEDRA IV API fetcher + Syriac morphological parser

SEDRA IV API (sedra.bethmardutho.org) contains:
  - 3284 Syriac roots
  - 32336 lexemes
  - 61445 words with full morphological tagging

We fetch root→lexeme→word chains and parse each word into the triple:
  (root_consonants: frozenset, template: tuple[str], surface: str)

Syriac consonants (Eastern Syriac, Estrangela script):
  ܐ ܒ ܓ ܕ ܗ ܘ ܙ ܚ ܛ ܝ ܟ ܠ ܡ ܢ ܣ ܥ ܦ ܨ ܩ ܪ ܫ ܬ
  (22 consonants, Unicode range U+0710–U+072F)

Vowel marks (diacritics) are in U+0730–U+074A range.

Template extraction: strip vowels from surface → consonantal skeleton.
Then align root consonants into template slots by position.
Template slots are labeled by their vocalic environment.
"""

import json
import os
import time
import unicodedata
import requests
from pathlib import Path
from typing import Optional
from collections import defaultdict

# ── Syriac Unicode ──────────────────────────────────────────────────────────

# Syriac consonant codepoints (U+0710–U+072F active range)
SYRIAC_CONSONANTS = set(chr(c) for c in range(0x0710, 0x0730))

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

BASE_URL = "https://sedra.bethmardutho.org/api"
CACHE_DIR = Path("sedra_cache")


def _get(endpoint: str, cache_key: str, timeout: int = 10) -> Optional[dict]:
    """Fetch JSON from SEDRA API with disk caching."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"  [WARN] API error for {url}: {e}")
        return None


def fetch_roots(max_roots: Optional[int] = None) -> list[dict]:
    """
    Fetch root list from SEDRA.
    SEDRA root IDs appear to be sequential; we probe 1..N.
    Returns list of {id, syriac, transliteration, lexemes: [...]}
    """
    # First try the root listing endpoint
    data = _get("roots.json", "roots_index")
    if data and isinstance(data, list):
        roots = data
    else:
        # Fallback: probe individual roots
        print("Root index not available, probing individual roots...")
        roots = []
        for rid in range(1, 500):  # Start with first 500 for toy
            r = _get(f"root/{rid}.json", f"root_{rid}")
            if r:
                roots.append(r)
            time.sleep(0.05)  # polite rate limiting
            if max_roots and len(roots) >= max_roots:
                break

    if max_roots:
        roots = roots[:max_roots]

    print(f"Fetched {len(roots)} roots")
    return roots


def fetch_lexemes_for_root(root_id: int) -> list[dict]:
    """Fetch all lexemes for a given root ID."""
    data = _get(f"root/{root_id}/lexemes.json", f"root_{root_id}_lexemes")
    if data and isinstance(data, list):
        return data
    return []


def fetch_words_for_lexeme(lexeme_id: int) -> list[dict]:
    """Fetch all word forms for a given lexeme ID."""
    data = _get(f"lexeme/{lexeme_id}/words.json", f"lexeme_{lexeme_id}_words")
    if data and isinstance(data, list):
        return data
    return []


# ── Morphological Triple Extraction ─────────────────────────────────────────

def word_to_triple(root_syriac: str, word_data: dict) -> Optional[tuple]:
    """
    Convert a SEDRA word record + root string into a morphological triple:
      (root_consonants: tuple, template: tuple, surface: str)

    root_consonants is a TUPLE (not frozenset) of consonants in canonical order.
    template is a TUPLE of slot labels.
    surface is the consonantal-only surface string.

    Returns None if word can't be parsed reliably.
    """
    # SEDRA word records have various surface fields
    # Try 'syriac', 'word', or 'form' keys
    surface = (word_data.get('syriac') or
               word_data.get('word') or
               word_data.get('form') or '')

    if not surface:
        return None

    root_cons = extract_root_consonants(root_syriac)
    surface_cons = extract_root_consonants(surface)

    # Sanity check: surface consonants must be a superset of root consonants
    # (root consonants should all appear in surface; other consonants are affixes)
    if len(root_cons) < 2 or len(surface_cons) < len(root_cons):
        return None

    template = extract_template(surface)
    if len(template) < 2:
        return None

    return (tuple(root_cons), tuple(template), surface)


def build_dataset(
    max_roots: int = 200,
    min_words_per_root: int = 3,
    verbose: bool = True
) -> list[dict]:
    """
    Build a dataset of morphological triples from SEDRA.

    Returns list of:
      {
        'root_id': int,
        'root_syriac': str,
        'root_consonants': tuple[str],    # ordered set of root consonants
        'template': tuple[str],           # slot labels
        'surface': str,                   # consonantal surface
        'lexeme_id': int,
        'morphology': dict,               # raw SEDRA morphology tags
      }
    """
    if verbose:
        print(f"Building dataset (max_roots={max_roots})...")

    # Check for existing processed dataset
    processed_path = CACHE_DIR / "processed_dataset.json"
    if processed_path.exists():
        print("  Loading cached processed dataset")
        with open(processed_path) as f:
            return json.load(f)

    dataset = []
    roots = fetch_roots(max_roots=max_roots)

    for root in roots:
        root_id = root.get('id') or root.get('rootId')
        root_syriac = root.get('syriac') or root.get('root') or ''
        if not root_id or not root_syriac:
            continue

        if verbose:
            print(f"  Root {root_id}: {root_syriac}")

        lexemes = fetch_lexemes_for_root(root_id)
        root_triples = []

        for lex in lexemes:
            lex_id = lex.get('id') or lex.get('lexemeId')
            if not lex_id:
                continue

            words = fetch_words_for_lexeme(lex_id)
            for word in words:
                triple = word_to_triple(root_syriac, word)
                if triple is None:
                    continue
                root_cons, template, surface = triple
                root_triples.append({
                    'root_id': root_id,
                    'root_syriac': root_syriac,
                    'root_consonants': list(root_cons),
                    'template': list(template),
                    'surface': surface,
                    'lexeme_id': lex_id,
                    'morphology': {
                        k: v for k, v in word.items()
                        if k not in ('syriac', 'word', 'form')
                    }
                })
            time.sleep(0.02)  # polite

        if len(root_triples) >= min_words_per_root:
            dataset.extend(root_triples)

    print(f"  Total triples: {len(dataset)} from {len(set(d['root_id'] for d in dataset))} roots")

    # Cache
    with open(processed_path, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

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
