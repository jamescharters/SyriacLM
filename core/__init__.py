#!/usr/bin/env python3
"""Shared core library for the Syriac research repository.

The foundation reused by every paper in this repo: corpus access and the Syriac
tokenizer (``script``), the stylometric feature/AUC machinery (``stylometry``),
authorship attribution (``authorship``), the supervised AV head (``av_head``), the
FastText encoder (``fasttext_model``), the ETCBC corpus loaders (``etcbc_corpus``),
and the author-cluster bootstrap (``bootstrap``).

Nothing here depends on any paper package; the dependency arrows point inward:

    disentangle/  ->  neural/  ->  core/  <-  stylometry/

Run any module from the repository root, e.g.::

    .venv/bin/python -m core.stylometry
"""

__version__ = "0.1.0"
