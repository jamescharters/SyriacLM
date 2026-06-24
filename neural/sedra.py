#!/usr/bin/env python3
"""SEDRA-derived vocalisation/morphology supervision (license-restricted source).

Why this module exists
----------------------
The non-obvious twist of this sub-project (``docs/DESIGN.md``) treats the
*pointing* (vocalisation) of Syriac as self-supervision: predicting the vowels of
a consonantal form is recovering the root-and-pattern *pattern* morpheme. The
SEDRA database (George A. Kiraz, Beth Mardutho) is the openly-documented source
of aligned consonantal/vocalised word forms with root and full morphology.

Each SEDRA *word* record is, per the published ``sedra-model`` schema,

    makeWord(lexemeId, word, vocalised, morphologicalType, attributes)

e.g. ``word="ABHOH;"`` (consonantal skeleton) and ``vocalised="AaB,oHaOH_;"``.
In SEDRA's CAL-style ASCII, **lowercase a/e/i/o/u are vowels** and a small set of
punctuation marks are **diacritics** (quššaya/rukkaka, etc.); everything else is
the **consonantal skeleton**. That single fact gives us, for free:

* ``(skeleton -> pointing)`` pairs for the Twist-1 vocalisation objective; and
* a ready-made split of each form into a *consonant (root) channel* and a
  *vowel (pattern) channel* for the Twist-2 factored encoder -- no inference
  needed, because the channels are literally different symbol classes.

LICENSE (important)
-------------------
SEDRA 3 is distributed "MIT **with restrictions**": academic/personal use only,
**no redistribution of altered versions**, not for profit, and any publication
using it must cite Kiraz (see ``SEDRA_CITATION``). Therefore:

* this repo ships **no** SEDRA data and no SEDRA-derived training files;
* you point this loader at a SEDRA source you obtained yourself (e.g. the
  ``peshitta/sedrajs`` ``sedra/`` text DB or its generated JSON);
* derived pairs are cached locally under ``sedra_cache/`` (git-ignored) and never
  committed.

The skeleton/pointing logic below needs no SEDRA data to run or be tested; the
``--selftest`` CLI verifies it against the schema's documented example.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Required acknowledgement for any publication using SEDRA (verbatim formula from
# the SEDRA III distribution terms).
SEDRA_CITATION = (
    'This work makes use of the Syriac Electronic Data Retrieval Archive (SEDRA) '
    'by George A. Kiraz, distributed by the Syriac Computing Institute. '
    'See also G. Kiraz, "Automatic Concordance Generation of Syriac Texts", in '
    'VI Symposium Syriacum 1992, ed. R. Lavenant, Orientalia Christiana Analecta '
    '247, Rome, 1994.'
)

# In SEDRA's CAL-style ASCII, vowels are the lowercase aeiou; the consonantal
# skeleton is written with uppercase letters (and a few punctuation letter-signs).
SEDRA_VOWELS: frozenset[str] = frozenset("aeiou")
# Diacritic / pointing marks that are part of the vocalisation but not vowels:
# quššaya (') and rukkaka (,) on bgdkpt, linea occultans (_), and the marker (*).
# Verified empirically against the SEDRA 3 WORDS.TXT vocalised field. These ride
# on top of the skeleton and are predicted as part of the pointing target.
SEDRA_DIACRITICS: frozenset[str] = frozenset("',_*")


@dataclass(frozen=True)
class VocalizedForm:
    """One aligned form: consonantal skeleton + full vocalisation + labels."""

    skeleton: str          # consonants only (SEDRA ASCII), e.g. "ABHOH;"
    vocalised: str         # skeleton + interleaved vowels/diacritics
    root: str = ""         # SEDRA root string (ASCII), if available
    lexeme: str = ""       # SEDRA lexeme string (ASCII), if available
    features: tuple[tuple[str, str], ...] = ()   # (name, value) morphology pairs


def split_skeleton_pointing(vocalised: str) -> tuple[str, list[str | None]]:
    """Separate a vocalised SEDRA form into its skeleton and per-slot pointing.

    Returns ``(skeleton, pointing)`` where ``skeleton`` is the consonant string
    and ``pointing`` has one entry per skeleton character: the vowel/diacritic
    string attached *after* that consonant, or ``None`` if bare. This is exactly
    the supervision the Twist-1 objective restores -- the model sees ``skeleton``
    and must predict ``pointing``.

    Example (from the sedra-model schema docs)::

        >>> skel, pts = split_skeleton_pointing("AaB,oHaOH_;")
        >>> skel
        'ABHOH;'
        >>> pts
        ['a', ',o', 'a', None, '_', None]
    """
    skeleton_chars: list[str] = []
    pointing: list[str | None] = []
    for ch in vocalised:
        if ch in SEDRA_VOWELS or ch in SEDRA_DIACRITICS:
            if not skeleton_chars:
                # Leading vocalisation with no preceding consonant: attach to a
                # virtual slot by prepending; rare, but keep information.
                skeleton_chars.append("")
                pointing.append(ch)
            else:
                prev = pointing[-1]
                pointing[-1] = (prev or "") + ch
        else:
            skeleton_chars.append(ch)
            pointing.append(None)
    return "".join(skeleton_chars), pointing


def consonant_vowel_channels(vocalised: str) -> tuple[str, str]:
    """Factored view for the Twist-2 encoder: (consonant_channel, vowel_channel).

    The consonant channel is the skeleton; the vowel channel is the concatenation
    of vowels/diacritics aligned to skeleton slots (``.`` marks a bare slot), so
    the two strings have equal length and can feed parallel streams.
    """
    skeleton, pointing = split_skeleton_pointing(vocalised)
    vowel_slots = [(p if p is not None else ".") for p in pointing]
    # Collapse multi-char pointing to a single representative slot symbol while
    # keeping alignment 1:1 with the skeleton (full string remains in pointing).
    vowel_channel = "".join(s[0] if s != "." else "." for s in vowel_slots)
    return skeleton, vowel_channel


# --------------------------------------------------------------------------- #
# Loading a user-provided SEDRA source (guarded; no data shipped).
# --------------------------------------------------------------------------- #
def _default_source_candidates() -> list[Path]:
    """Likely locations of a user-provided / extracted SEDRA word table."""
    here = Path(__file__).resolve().parent          # neural/
    repo = here.parent                              # repo root
    home = Path.home()
    return [
        repo / "corpora" / "sedra_cache" / "words.json",     # corpora.sedra_build output
        Path("corpora/sedra_cache/words.json"),              # from repo root
        here / "sedra_cache" / "words.json",                 # legacy in-neural location
        Path("neural/sedra_cache/words.json"),               # legacy, from repo root
        Path("sedra_cache/words.json"),                      # local, pre-extracted
        home / ".cache" / "sedrajs" / "words.json",
    ]


def find_sedra_source(explicit: Path | None = None) -> Path | None:
    """Return a SEDRA source path if one is available, else ``None``."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    for cand in _default_source_candidates():
        if cand.exists():
            return cand
    return None


def load_words(source: Path) -> list[VocalizedForm]:
    """Load ``VocalizedForm`` records from a pre-extracted SEDRA JSON export.

    Expected JSON: a list of objects with at least ``skeleton``/``word`` and
    ``vocalised`` keys, optionally ``root``, ``lexeme``, ``features``. Converting
    the raw SEDRA 3 text DB (or sedrajs modules) into this JSON is left to a
    user-run extraction step so that no SEDRA-derived data is committed here.
    """
    raw = json.loads(source.read_text(encoding="utf-8"))
    forms: list[VocalizedForm] = []
    for rec in raw:
        voc = rec.get("vocalised") or rec.get("vocalized") or ""
        skel = rec.get("skeleton") or rec.get("word") or split_skeleton_pointing(voc)[0]
        feats = tuple((k, str(v)) for k, v in (rec.get("features") or {}).items())
        forms.append(VocalizedForm(
            skeleton=skel, vocalised=voc,
            root=rec.get("root", ""), lexeme=rec.get("lexeme", ""), features=feats,
        ))
    return forms


def pointing_examples(forms: list[VocalizedForm]) -> list[dict]:
    """Build Twist-1 training examples: input skeleton -> target pointing."""
    examples: list[dict] = []
    for f in forms:
        skel, pointing = split_skeleton_pointing(f.vocalised)
        examples.append({
            "skeleton": skel,
            "pointing": pointing,
            "vocalised": f.vocalised,
            "root": f.root,
        })
    return examples


_SELFTEST_CASES = [
    # (vocalised, expected skeleton)  -- the schema's documented example.
    ("AaB,oHaOH_;", "ABHOH;"),
    ("ABA", "ABA"),               # already bare skeleton
    ("MaLoKoA", "MLKA"),          # illustrative malka-style form
]


def _selftest() -> int:
    ok = True
    for voc, expect in _SELFTEST_CASES:
        skel, pts = split_skeleton_pointing(voc)
        status = "OK" if skel == expect else "FAIL"
        if skel != expect:
            ok = False
        print(f"[{status}] {voc!r:>16} -> skeleton {skel!r:<10} pointing {pts}")
    # Round-trip: re-interleave skeleton+pointing must reproduce the vocalised form.
    for voc, _ in _SELFTEST_CASES:
        skel, pts = split_skeleton_pointing(voc)
        rebuilt = "".join((c + (p or "")) for c, p in zip(skel, pts))
        if rebuilt != voc:
            ok = False
            print(f"[FAIL] round-trip {voc!r} != {rebuilt!r}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SEDRA vocalisation/morphology loader.")
    ap.add_argument("--selftest", action="store_true",
                    help="verify skeleton/pointing logic (no SEDRA data needed)")
    ap.add_argument("--source", type=Path, default=None,
                    help="path to a pre-extracted SEDRA words JSON")
    ap.add_argument("--cite", action="store_true", help="print the required SEDRA citation")
    args = ap.parse_args(argv)

    if args.cite:
        print(SEDRA_CITATION)
        return 0
    if args.selftest:
        return _selftest()

    src = find_sedra_source(args.source)
    if src is None:
        print("No SEDRA source found. SEDRA is license-restricted and not shipped; "
              "see neural/docs/DATA.md to obtain and extract it.", file=sys.stderr)
        print("Skeleton/pointing logic still works -- try: "
              ".venv/bin/python -m neural.sedra --selftest", file=sys.stderr)
        return 2
    forms = load_words(src)
    n_voc = sum(1 for f in forms if any(c in SEDRA_VOWELS for c in f.vocalised))
    print(f"Loaded {len(forms):,} SEDRA forms from {src} "
          f"({n_voc:,} carry vocalisation).")
    print("Reminder -- cite SEDRA in any publication:")
    print("  " + SEDRA_CITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
