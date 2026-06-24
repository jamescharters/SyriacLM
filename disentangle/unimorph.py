#!/usr/bin/env python3
"""Cross-language balanced grids from UniMorph (Arabic and Hebrew).

The disentanglement instrument is not specific to Syriac: any language with
inflectional morphology and a labelled lemma/feature/form resource gives a
balanced, decorrelated grid for the same cross-erasure test. This module builds
that grid for **Arabic** and **Hebrew** from UniMorph, so the Syriac result can be
replicated on two further Semitic languages with native encoders.

UniMorph lines are ``lemma <TAB> form <TAB> features`` where ``features`` is a
semicolon-separated bundle in the UniMorph schema (e.g. ``N;PL;NDEF;NOM``). We use
two factors that vary within a lemma:

* number  -- ``SG`` vs ``PL``
* gender  -- ``MASC`` vs ``FEM``

For each lemma we take **minimal pairs**: two forms whose feature bundles are
identical except for the factor value (e.g. ``N;SG;NDEF;NOM`` vs
``N;PL;NDEF;NOM``). This holds every other morphosyntactic dimension fixed, so the
only systematic difference between the two forms is the factor -- a tighter
decorrelation of lexical identity from the factor than the lexicon grid. One pair
per lemma keeps identity independent of the factor.

Arabic UniMorph forms are **vocalised** (they carry harakat), so stripping the
diacritics gives a consonantal control exactly as for Syriac. Hebrew UniMorph
forms are **unvocalised**, so there the consonantal form equals the surface form
(the control is vacuous, noted in the output).

The data is downloaded on demand into a git-ignored cache and never committed.
UniMorph is released under CC-BY-SA-3.0 (cite McCarthy et al.); see
``disentangle/docs/DATA.md``.

    .venv/bin/python -m disentangle.unimorph --lang ara --factor number --stats
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from collections import defaultdict

try:  # route HuggingFace/GitHub TLS through the OS trust store (corporate proxy)
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

CACHE_DIR = os.path.join(os.path.dirname(__file__), "unimorph_cache")
RAW_URL = "https://raw.githubusercontent.com/unimorph/{lang}/master/{lang}"

LANGS = ("ara", "heb")
FACTORS = {
    "number": ("SG", "PL"),
    "gender": ("MASC", "FEM"),
}

# Encoders that read each language in its native script (subword, mean-pooled).
ENCODER_FOR = {
    "ara": ("CAMeL-Lab/bert-base-arabic-camelbert-ca", "CAMeLBERT-CA (Arabic)"),
    "heb": ("onlplab/alephbert-base", "AlephBERT (Hebrew)"),
}

# Diacritics to strip for the consonantal control.
_ARABIC_DIACRITICS = {chr(c) for c in range(0x064B, 0x0653)} | {
    "\u0670", "\u0640", "\u0653", "\u0654", "\u0655", "\u0656", "\u0670"}
_HEBREW_POINTS = {chr(c) for c in range(0x0591, 0x05C8)
                  if chr(c) not in ("\u05BE", "\u05C0", "\u05C3", "\u05C6")}


def _strip(form: str, lang: str) -> str:
    marks = _ARABIC_DIACRITICS if lang == "ara" else _HEBREW_POINTS
    return "".join(ch for ch in form if ch not in marks)


def _ensure(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"unknown lang {lang!r}; choose from {LANGS}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, lang)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = RAW_URL.format(lang=lang)
    print(f"downloading UniMorph {lang} from {url} ...", file=sys.stderr)
    tmp = path + ".tmp"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, path)
    return path


def _parse(path: str):
    """Yield ``(lemma, form, feature_tuple)`` for non-empty UniMorph lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            lemma, form, feats = parts[0], parts[1], parts[2]
            if not lemma or not form or not feats:
                continue
            yield lemma, form, tuple(feats.split(";"))


def load_grid_unimorph(lang: str, factor: str = "number",
                       max_lines: int | None = None, seed: int = 0,
                       keep_diacritics: bool = True) -> list[dict]:
    """Balanced minimal-pair grid for ``lang`` and ``factor``.

    Each item: ``{lexeme, label(0/1), voc, cons}`` where ``voc`` is the
    model-facing surface and ``cons`` the diacritics-stripped form. Identity
    (lemma) is held independent of the factor by taking one minimal pair per lemma.

    ``keep_diacritics=False`` puts the stripped form in *both* ``voc`` and
    ``cons``. This is required when the encoder's tokenizer does not cover
    diacritised text: CAMeLBERT-CA, trained on undiacritised Classical Arabic,
    maps every fully-vocalised UniMorph form to a single ``<unk>``, so the
    undiacritised form is the real surface for it (and the consonantal control is
    then vacuous, as for unvocalised Hebrew).
    """
    if factor not in FACTORS:
        raise ValueError(f"unknown factor {factor!r}; choose from {list(FACTORS)}")
    v0, v1 = FACTORS[factor]
    path = _ensure(lang)

    import numpy as np
    rng = np.random.default_rng(seed)

    # (lemma, bundle-without-factor) -> {factor value -> set of forms}
    cells: dict[tuple, dict[str, set]] = defaultdict(
        lambda: {v0: set(), v1: set()})
    for i, (lemma, form, feats) in enumerate(_parse(path)):
        if max_lines and i >= max_lines:
            break
        fs = set(feats)
        has0, has1 = v0 in fs, v1 in fs
        if has0 == has1:           # need exactly one of the two factor values
            continue
        val = v0 if has0 else v1
        bundle = tuple(sorted(fs - {v0, v1}))
        cells[(lemma, bundle)][val].add(form)

    # lemma -> list of qualifying minimal-pair (form0, form1)
    by_lemma: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (lemma, _bundle), d in cells.items():
        f0 = [f for f in d[v0] if f not in d[v1]]
        f1 = [f for f in d[v1] if f not in d[v0]]
        if f0 and f1:
            by_lemma[lemma].append((sorted(f0)[0], sorted(f1)[0]))

    grid: list[dict] = []
    lemma_id = {}
    for lemma, pairs in by_lemma.items():
        a, b = pairs[int(rng.integers(len(pairs)))]
        lid = lemma_id.setdefault(lemma, len(lemma_id))
        sa, sb = _strip(a, lang), _strip(b, lang)
        va, vb = (a, b) if keep_diacritics else (sa, sb)
        grid.append({"lexeme": lid, "label": 0, "voc": va, "cons": sa})
        grid.append({"lexeme": lid, "label": 1, "voc": vb, "cons": sb})
    return grid


def build_encoder(lang: str, device):
    """Native-script subword encoder for ``lang`` as a ``wv`` (mean-pooled)."""
    from transformers import AutoModel, AutoTokenizer
    from neural.hf_encoder import HFWordVectors
    mid, label = ENCODER_FOR[lang]
    model = AutoModel.from_pretrained(mid).to(device)
    tok = AutoTokenizer.from_pretrained(mid)
    return HFWordVectors(model, tok, device), label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=LANGS, default="heb")
    ap.add_argument("--factor", choices=list(FACTORS), default="number")
    ap.add_argument("--stats", action="store_true", help="print grid statistics")
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args(argv)

    grid = load_grid_unimorph(args.lang, args.factor, max_lines=args.max_lines)
    n_pairs = len(grid) // 2
    voc_eq_cons = sum(1 for g in grid if g["voc"] == g["cons"])
    print(f"{args.lang} / {args.factor}: {n_pairs:,} lemma pairs "
          f"({len(grid):,} forms)")
    print(f"  vocalised==consonantal: {voc_eq_cons}/{len(grid)} forms "
          f"({'unvocalised script' if voc_eq_cons > len(grid) * 0.9 else 'vocalised'})")
    if args.stats and grid:
        for g in grid[:4]:
            print(f"    lemma {g['lexeme']} label {g['label']}  "
                  f"voc={g['voc']!r}  cons={g['cons']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
