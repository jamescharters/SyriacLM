#!/usr/bin/env python3
"""Paper 3 -- root/pattern disentanglement in frozen Syriac encoders.

Templatic (root-and-pattern) morphology plus a complete morphological lexicon
(SEDRA) form a naturally-occurring, cleanly-decorrelated benchmark for the
representation-disentanglement question: are lexical identity and morphosyntactic
pattern stored in separable linear subspaces of a frozen encoder?

This package builds on ``neural`` (it reuses the CANINE / Hebrew-transfer encoders
and the transliteration there), which in turn builds on ``core``:

    disentangle/  ->  neural/  ->  core/

    .venv/bin/python -m disentangle.disentangle --demo
"""

__version__ = "0.1.0"
