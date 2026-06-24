#!/usr/bin/env python3
"""Harvest a *classical* Syriac vocalisation gold set from the Digital Syriac Corpus.

Why this exists
---------------
The vocaliser ([`vocalizer.py`](vocalizer.py)) is trained on the SEDRA lexicon,
which is New-Testament-scoped. The obvious question is whether it generalises to
*classical* running text. The project long assumed there was no openly vocalised
classical gold to test against -- but that is false: **~56% of Digital Syriac
Corpus tokens carry Syriac vowel points** (U+0730..U+074A: pthaha, zqapha, rbasa,
hbasa, esasa, the East-Syriac dotted zlamas, plus quššaya/rukkaka), across 600 of
632 files. This module turns that pointing into a gold set in the *same* CAL-ASCII
label space the vocaliser predicts, so cross-register accuracy becomes a real
number rather than a guess.

How the Unicode->CAL mapping is obtained (honestly)
---------------------------------------------------
We do **not** hand-assert an East/West vowel correspondence. Instead we *derive*
it from data: align DSC vocalised tokens to SEDRA skeletons that have a **unique**
vocalisation, then tally, per consonant slot, which Unicode mark co-occurs with
which CAL vowel. The dominant correspondence and its *purity* are reported, so the
mapping is transparent and its uncertainty is visible. Empirically the vowel core
is clean (pthaha->a, zqapha->o, rbasa->e, hbasa->i, esasa->u; quššaya->', rukkaka->,)
at 0.7-1.0 purity. The residual impurity is a real philological fact, not a bug:
**DSC points at a lower density than SEDRA** (a classical scribe leaves many slots
bare that the lexicon vocalises), and the U+073C hbasa-esasa dotted mark is
genuinely ambiguous between a vowel and the rukkaka dot. Because of the
partial-pointing asymmetry, the fair cross-register metric is **vowel accuracy on
the slots the scribe actually pointed**, not exact full-word match.

Licensing / containment
-----------------------
Reuses the parent corpus loaders read-only ([`script`](../script.py)) and the
SEDRA tables ([`sedra`](sedra.py)); writes nothing. The DSC is CC-BY (cite the
Digital Syriac Corpus); SEDRA is license-restricted (cite Kiraz, see
[`docs/DATA.md`](docs/DATA.md)). No derived data is committed.

    .venv/bin/python -m neural.dsc_gold --report
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from core.script import DEFAULT_CACHE, ensure_corpus, find_body, iter_words, strip_marks
from neural import sedra
from neural.transliterate import syriac_to_cal

# Syriac vowel-point / diacritic block. A token is "vocalised" if it carries at
# least one mark in this range (seyame U+0308 and other generic combining marks
# are reading/grammatical, not lexical vocalisation, and do not count).
POINT_LO, POINT_HI = 0x0730, 0x074A

CAL_VOWELS = frozenset("aeiou")


def _vowel_of(cal_label: str | None) -> str:
    """The vowel component (a/e/i/o/u) of a CAL pointing label, or '' if none."""
    if not cal_label:
        return ""
    for ch in cal_label:
        if ch in CAL_VOWELS:
            return ch
    return ""


def dsc_slots(token: str) -> list[tuple[str, list[str]]]:
    """Split a raw DSC token into ``(cal_consonant, [unicode_marks])`` per slot.

    Combining marks attach to the preceding consonant slot, mirroring how
    ``sedra.split_skeleton_pointing`` attaches CAL pointing to skeleton slots, so
    the two can be aligned position-for-position.
    """
    slots: list[tuple[str, list[str]]] = []
    for ch in unicodedata.normalize("NFD", token):
        if unicodedata.combining(ch):
            if slots:
                slots[-1][1].append(ch)
        else:
            cal = syriac_to_cal(ch)
            if cal and not cal.isspace():
                slots.append((cal, []))
    return slots


def _mark_name(ch: str) -> str:
    return unicodedata.name(ch, f"U+{ord(ch):04X}").replace("SYRIAC ", "")


def iter_vocalised_tokens(data_dir: Path):
    """Yield raw DSC tokens that carry at least one Syriac vowel point."""
    for path in sorted(data_dir.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        body = find_body(root)
        if body is None:
            continue
        for tok in iter_words(body):
            if any(POINT_LO <= ord(c) <= POINT_HI
                   for c in unicodedata.normalize("NFD", tok)):
                yield tok


def _sedra_skeleton_index() -> tuple[dict[str, str], set[str]]:
    """Return (unambiguous skeleton->vocalised, all skeletons) from SEDRA."""
    src = sedra.find_sedra_source()
    if src is None:
        raise RuntimeError(
            "No SEDRA table found. Build it first (see neural/docs/DATA.md):\n"
            "    .venv/bin/python -m corpora.sedra_build --sedra-dir ~/.cache/sedrajs/sedra")
    skel_to_voc: dict[str, set[str]] = defaultdict(set)
    for f in sedra.load_words(src):
        if not f.vocalised:
            continue
        skel = f.skeleton or sedra.split_skeleton_pointing(f.vocalised)[0]
        skel_to_voc[skel].add(f.vocalised)
    unambiguous = {s: next(iter(v)) for s, v in skel_to_voc.items() if len(v) == 1}
    return unambiguous, set(skel_to_voc)


def derive_vowel_map(data_dir: Path, unambiguous: dict[str, str]) -> dict:
    """Data-driven Unicode-mark -> CAL-vowel map with purities.

    Aligns DSC vocalised tokens to SEDRA skeletons with a unique vocalisation and
    tallies, for single-mark slots, which Unicode mark co-occurs with which CAL
    vowel. Returns the dominant vowel per mark plus the supporting counts/purity.
    """
    mark_to_vowel: dict[str, Counter] = defaultdict(Counter)
    aligned = 0
    for tok in iter_vocalised_tokens(data_dir):
        skel = syriac_to_cal(strip_marks(tok))
        voc = unambiguous.get(skel)
        if voc is None:
            continue
        s_skel, s_pts = sedra.split_skeleton_pointing(voc)
        if s_skel != skel:
            continue
        slots = dsc_slots(tok)
        if len(slots) != len(s_pts):
            continue
        aligned += 1
        for (_c, umarks), cal_pt in zip(slots, s_pts):
            if len(umarks) == 1:        # isolate single marks to learn per-mark
                mark_to_vowel[_mark_name(umarks[0])][_vowel_of(cal_pt)] += 1

    vowel_map: dict[str, str] = {}
    report: list[tuple[str, str, float, int]] = []
    for name, cnt in mark_to_vowel.items():
        total = sum(cnt.values())
        vowel, n = cnt.most_common(1)[0]
        purity = n / total
        report.append((name, vowel, purity, total))
        # accept a mark as a vowel only if it maps to a real vowel with majority
        if vowel and purity >= 0.5:
            vowel_map[name] = vowel
    report.sort(key=lambda r: -r[3])
    return {"vowel_map": vowel_map, "report": report, "aligned_tokens": aligned}


def gold_vowels(token: str, vowel_map: dict[str, str]) -> tuple[str, list[str]]:
    """Map a DSC token to ``(cal_skeleton, [gold_vowel_per_slot])``.

    Each slot's gold vowel is the mapped vowel of its (single) vowel mark, or ''
    if the scribe left the slot unvocalised / marked it only with a non-vowel
    reading mark. Multi-vowel-mark slots (rare) take the first recognised vowel.
    """
    skel_chars: list[str] = []
    vowels: list[str] = []
    for cal_c, umarks in dsc_slots(token):
        skel_chars.append(cal_c)
        v = ""
        for m in umarks:
            mv = vowel_map.get(_mark_name(m), "")
            if mv:
                v = mv
                break
        vowels.append(v)
    return "".join(skel_chars), vowels


def build_gold(data_dir: Path, vowel_map: dict[str, str], sedra_skeletons: set[str]) -> dict:
    """Build the type-level DSC gold set, split by SEDRA-vocabulary membership.

    De-duplicates to unique ``(skeleton, gold-vowel-pattern)`` pairs so the eval
    is not dominated by a few hyper-frequent forms, then partitions by whether the
    skeleton is in SEDRA's vocabulary (seen consonantal form) or out of it (the
    decisive cross-register generalisation case).
    """
    seen: set[tuple[str, tuple[str, ...]]] = set()
    in_vocab: list[tuple[str, list[str]]] = []
    oov: list[tuple[str, list[str]]] = []
    tok_total = tok_marked_slots = 0
    for tok in iter_vocalised_tokens(data_dir):
        skel, vowels = gold_vowels(tok, vowel_map)
        if not skel or not any(vowels):
            continue
        tok_total += 1
        tok_marked_slots += sum(1 for v in vowels if v)
        key = (skel, tuple(vowels))
        if key in seen:
            continue
        seen.add(key)
        (in_vocab if skel in sedra_skeletons else oov).append((skel, vowels))
    return {
        "in_vocab": in_vocab, "oov": oov,
        "n_tokens": tok_total, "n_marked_slots": tok_marked_slots,
        "n_types": len(seen),
    }


def harvest(data_dir: Path | None = None) -> dict:
    """End-to-end: derive the vowel map and build the gold splits."""
    data_dir = data_dir or ensure_corpus(DEFAULT_CACHE)
    unambiguous, sedra_skeletons = _sedra_skeleton_index()
    derived = derive_vowel_map(data_dir, unambiguous)
    gold = build_gold(data_dir, derived["vowel_map"], sedra_skeletons)
    return {"data_dir": data_dir, "sedra_skeletons": sedra_skeletons,
            "derived": derived, "gold": gold}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="derive the Unicode->CAL vowel map and gold stats, then print")
    args = ap.parse_args(argv)
    if not args.report:
        ap.print_help()
        return 1

    try:
        res = harvest()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    d, g = res["derived"], res["gold"]
    print("\n=== derived Unicode-point -> CAL vowel map "
          f"(aligned {d['aligned_tokens']:,} DSC tokens to unambiguous SEDRA) ===")
    print(f"  {'mark':<32} {'->CAL':>6} {'purity':>7} {'count':>9}")
    for name, vowel, purity, total in d["report"][:18]:
        accepted = "*" if d["vowel_map"].get(name) == vowel and vowel else " "
        print(f" {accepted}{name:<32} {vowel or '-':>6} {purity:>7.2f} {total:>9,}")
    print("  (* = accepted as a vowel mark; vowels map cleanly, reading/grammatical")
    print("   marks fall below threshold and are treated as unvocalised)")

    print("\n=== DSC classical vocalisation gold ===")
    print(f"  vocalised tokens harvested : {g['n_tokens']:,}")
    print(f"  marked (pointed) slots     : {g['n_marked_slots']:,}")
    print(f"  unique (skeleton, pattern) : {g['n_types']:,}")
    print(f"    in SEDRA vocabulary      : {len(g['in_vocab']):,}")
    print(f"    out of SEDRA vocab (OOV) : {len(g['oov']):,}")
    print("\nCite the Digital Syriac Corpus (data) and SEDRA/Kiraz (alignment);")
    print("see neural/docs/DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
