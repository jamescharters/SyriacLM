#!/usr/bin/env python3
"""Neural language modeling for Classical Syriac (self-contained research scaffold).

A transfer-first, structure-aware approach to the gap named in the paper: there is
no large pretrained language model for Classical Syriac, and from-scratch models
on 2.18M tokens are data-starved. This sub-project is a forward-looking research
scaffold; nothing here is required to reproduce the published stylometry pipeline.

Design in one breath
---------------------
The gap is *data scale*, not architecture, so we never pretrain from scratch and
never manufacture "new" Syriac from a model fit on Syriac (information-circular;
risks model collapse). New information enters through three doors:

  1. STRUCTURE  -- the root-and-pattern grammar as explicit supervision, taken
     from the SEDRA vocalised lexicon and ETCBC morphology (``morphology.py``,
     ``sedra.py``).
  2. TRANSFER   -- knowledge from byte/character multilingual and Semitic
     checkpoints via continued pretraining and shared-script transliteration
     (``modeling.py``, ``pretrain.py``, ``transliterate.py``).
  3. REAL TEXT  -- genuinely new tokens via corpus aggregation with provenance
     and a leakage-safe split (``aggregate.py``); HTR/OCR is future work.

The non-obvious twist (see ``docs/DESIGN.md``): Syriac's script is *defective*
(consonants written, vowels/pointing optional and usually absent). The published
pipeline discards the pointing; here we invert that -- predicting the pointing is
recovering the *pattern* morpheme, so vocalisation restoration becomes cheap,
morphology-aligned self-supervision and yields the first neural Syriac vocaliser
as a by-product.

Containment
-----------
Everything for this idea lives under ``neural/``. The only outward dependency is
*read-only* imports of the parent project's tokenizer and corpus loaders
(``script.py``, ``etcbc_corpus.py``) and metric helpers (``stylometry.py``), so
that any comparison against the released FastText baseline uses byte-for-byte the
same tokenization. We never modify parent modules.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.0.1"
