#!/usr/bin/env python3
"""Paper 1 -- Classical Syriac stylometry and authorship (FastText + baselines).

Character n-gram FastText embeddings for a templatic, low-resource language, with
an application to authorship analysis: subword ablation, OOV generalisation,
embedding geometry, a representation bake-off (FastText / word2vec / Burrows's
Delta / byte-LM / char-Transformer), a genre confound, and disputed-text case
studies. This package holds the paper-1 experiment drivers and the preprint
source; the shared encoders and metrics live in ``core``.

Modules
-------
* ``paper_experiments`` -- produces every table number (and LaTeX rows).
* ``nn_baselines``      -- from-scratch byte-LM and char-Transformer baselines.
* ``vocab_stats``       -- corpus frequency statistics.

Writeup in ``stylometry/paper/`` (XeLaTeX); lab notebook in ``stylometry/docs``.

    .venv/bin/python -m stylometry.paper_experiments
"""

__version__ = "0.1.0"
