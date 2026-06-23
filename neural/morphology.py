#!/usr/bin/env python3
"""Morphological supervision for the structure door (ETCBC + SEDRA).

Provides root/lexeme/feature labels that turn the root-and-pattern grammar into
explicit training signal -- the multitask companion to the pointing objective.
Two complementary sources:

* **ETCBC SyrNT** (Text-Fabric) -- per-word part of speech, lexeme, person,
  number, gender, state, tense, prefix/suffix over *running* biblical text. This
  needs the optional ``text-fabric`` package; absent it, this module still
  imports and explains how to install it.
* **SEDRA** (``sedra.py``) -- per-*type* vocalisation + root + morphology over the
  lexicon (license-restricted; see ``docs/DATA.md``).

The output is a list of ``MorphExample`` records consumable by ``modeling.py``'s
token-classification / multitask heads. Running text features come from ETCBC;
the vocalisation channel and root come from SEDRA.

    .venv/bin/python -m neural.morphology --selftest      # no heavy deps
    .venv/bin/python -m neural.morphology --etcbc SyrNT   # needs text-fabric
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

try:
    from tf.app import use as _tf_use      # type: ignore
    _TF = True
except Exception:  # pragma: no cover - text-fabric optional
    _TF = False

from neural import sedra


# ETCBC SyrNT word features we map into a common schema (see the SyrNT tutorial's
# feature list). Keys are our canonical names; values are TF feature names.
ETCBC_FEATURE_MAP: dict[str, str] = {
    "pos": "sp",
    "lexeme": "lexeme",
    "person": "ps",
    "number": "nu",
    "gender": "gn",
    "state": "st",
    "tense": "tense",
    "prefix": "prefix",
    "suffix": "suffix",
}


@dataclass
class MorphExample:
    """A surface form with its morphological labels (source-agnostic)."""

    surface: str                       # consonantal surface form
    source: str                        # "etcbc" or "sedra"
    root: str = ""
    lexeme: str = ""
    vocalised: str = ""                # only SEDRA carries this
    features: dict[str, str] = field(default_factory=dict)


def load_etcbc_morphology(corpus: str = "SyrNT") -> list[MorphExample]:
    """Read per-word morphology from an ETCBC Text-Fabric corpus.

    Requires the ``text-fabric`` package. Returns one ``MorphExample`` per word
    token (running order), so callers can build either token-classification
    sequences or type-level tables.
    """
    if not _TF:
        raise RuntimeError(
            "text-fabric is not installed. Install the neural extras:\n"
            "    .venv/bin/python -m pip install -r neural/requirements-neural.txt")
    A = _tf_use(f"etcbc/{corpus.lower()}", silent="deep")
    F = A.api.F
    examples: list[MorphExample] = []
    for w in F.otype.s("word"):
        feats: dict[str, str] = {}
        for canon, tf_name in ETCBC_FEATURE_MAP.items():
            getter = getattr(F, tf_name, None)
            if getter is not None:
                val = getter.v(w)
                if val not in (None, ""):
                    feats[canon] = str(val)
        surface = str(getattr(F, "lexeme").v(w)) if hasattr(F, "lexeme") else ""
        examples.append(MorphExample(
            surface=surface, source="etcbc",
            lexeme=feats.get("lexeme", ""), features=feats))
    return examples


def load_sedra_morphology(source=None) -> list[MorphExample]:
    """Read per-type vocalisation + root + morphology from a SEDRA source."""
    src = sedra.find_sedra_source(source)
    if src is None:
        raise RuntimeError(
            "No SEDRA source found (license-restricted; not shipped). "
            "See neural/docs/DATA.md.")
    forms = sedra.load_words(src)
    return [
        MorphExample(surface=f.skeleton, source="sedra", root=f.root,
                     lexeme=f.lexeme, vocalised=f.vocalised,
                     features=dict(f.features))
        for f in forms
    ]


def feature_label_space(examples: list[MorphExample]) -> dict[str, list[str]]:
    """Collect the sorted set of values per feature, for building classifier heads."""
    space: dict[str, set[str]] = {}
    for ex in examples:
        for k, v in ex.features.items():
            space.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in space.items()}


def _selftest() -> int:
    """Exercise the SEDRA-backed path with a tiny in-memory stand-in (no deps)."""
    sample = [
        sedra.VocalizedForm(skeleton="MLKA", vocalised="MaLoKoA", root="MLK",
                            lexeme="MLKA", features=(("pos", "noun"), ("state", "emphatic"))),
        sedra.VocalizedForm(skeleton="KTBA", vocalised="KaToBoA", root="KTB",
                            lexeme="KTBA", features=(("pos", "noun"),)),
    ]
    examples = [
        MorphExample(surface=f.skeleton, source="sedra", root=f.root,
                     lexeme=f.lexeme, vocalised=f.vocalised, features=dict(f.features))
        for f in sample
    ]
    space = feature_label_space(examples)
    print(f"built {len(examples)} morph examples from {len(sample)} SEDRA forms")
    print("label space:", space)
    # Channel split (Twist 2) derived from the vocalised field:
    for ex in examples:
        skel, vowels = sedra.consonant_vowel_channels(ex.vocalised)
        print(f"  {ex.vocalised:<10} root={ex.root:<5} skeleton={skel:<6} vowels={vowels}")
    ok = space.get("pos") == ["noun"] and len(examples) == 2
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Morphological supervision loader.")
    ap.add_argument("--selftest", action="store_true",
                    help="run a dependency-free check of the example builder")
    ap.add_argument("--etcbc", metavar="CORPUS",
                    help="load ETCBC morphology (e.g. SyrNT); needs text-fabric")
    ap.add_argument("--sedra", action="store_true",
                    help="load SEDRA morphology; needs a SEDRA source")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.etcbc:
        try:
            ex = load_etcbc_morphology(args.etcbc)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"loaded {len(ex):,} ETCBC word examples from {args.etcbc}")
        print("feature label space:", {k: len(v) for k, v in
                                       feature_label_space(ex).items()})
        return 0
    if args.sedra:
        try:
            ex = load_sedra_morphology()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"loaded {len(ex):,} SEDRA type examples")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
