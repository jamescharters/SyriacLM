"""corpora -- shared dataset locations for the Syriac project.

Several sub-projects consume the same source data (the ``neural`` vocaliser, the
``disentangle`` probe and the ``defog`` generator all read SEDRA), so the caches
and the scripts that build them live here, in one place, rather than inside any
single package. Every consumer imports the canonical paths from this module so
there is a single source of truth for *where the data is*.

This package is intentionally dependency-free (only the standard library): it is
imported by packages that may or may not have ``torch``/``transformers``
installed, and -- importantly -- it must never be confused with the unrelated
HuggingFace ``datasets`` library, which is why this directory is ``corpora`` and
not ``datasets``.

Contents
--------
* ``sedra_scrape`` -- mirror the SEDRA IV REST API to ``sedra_cache/api/`` (one
  ``<id>.json`` per record).
* ``sedra_db``     -- build a queryable SQLite ``sedra_cache/sedra.db`` from it.
* ``sedra_build``  -- parse the SEDRA 3 text DB into ``sedra_cache/words.json``
  (the vocaliser's pointing supervision).
* ``sedra_cache/`` -- the git-ignored data caches the scripts above produce.

LICENSE / CITATION (important)
------------------------------
SEDRA (George A. Kiraz, Beth Mardutho / Syriac Computing Institute) is
distributed for academic/personal use with restrictions: no redistribution of
altered versions, and any publication using it must cite Kiraz (``SEDRA_CITATION``
below). This repo commits the SEDRA IV JSON cache and ``words.json`` under
``sedra_cache/`` for reproducibility; observe SEDRA's terms if you redistribute.
The large derived SQLite database (``sedra_cache/sedra.db``) is git-ignored for
size -- rebuild it with ``-m corpora.sedra_db`` (see ``neural/docs/DATA.md``).
"""

from __future__ import annotations

import os

__all__ = [
    "CORPORA_DIR",
    "SEDRA_CACHE",
    "SEDRA_API_DIR",
    "SEDRA_WORD_DIR",
    "SEDRA_LEXEME_DIR",
    "SEDRA_ROOT_DIR",
    "SEDRA_DB",
    "SEDRA_WORDS_JSON",
    "SEDRA_CITATION",
    "sedra_word_dir",
]

# Absolute path to this package (``<repo>/corpora``); all dataset paths hang off
# it so they are independent of the current working directory.
CORPORA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- SEDRA IV: REST API mirror (sedra_scrape) + derived SQLite (sedra_db) ---- #
SEDRA_CACHE = os.path.join(CORPORA_DIR, "sedra_cache")
SEDRA_API_DIR = os.path.join(SEDRA_CACHE, "api")
SEDRA_WORD_DIR = os.path.join(SEDRA_API_DIR, "word")
SEDRA_LEXEME_DIR = os.path.join(SEDRA_API_DIR, "lexeme")
SEDRA_ROOT_DIR = os.path.join(SEDRA_API_DIR, "root")
SEDRA_DB = os.path.join(SEDRA_CACHE, "sedra.db")

# ---- SEDRA 3: text DB -> words.json (sedra_build), the vocaliser supervision -- #
SEDRA_WORDS_JSON = os.path.join(SEDRA_CACHE, "words.json")

# Required acknowledgement for any publication using SEDRA.
SEDRA_CITATION = (
    "This work makes use of the Syriac Electronic Data Retrieval Archive (SEDRA) "
    "by George A. Kiraz, distributed by the Syriac Computing Institute."
)


def sedra_word_dir(require: bool = False) -> str:
    """Return the absolute path to the SEDRA IV per-word JSON records.

    The directory holds one ``<id>.json`` per inflected form (Syriac consonantal
    spelling, ``western``/``eastern`` vocalised pointing, morphology, owning
    lexeme). With ``require=True`` a missing cache raises ``FileNotFoundError``
    with instructions, since the data is license-restricted and not shipped.
    """
    if require and not os.path.isdir(SEDRA_WORD_DIR):
        raise FileNotFoundError(
            f"No SEDRA IV word records at {SEDRA_WORD_DIR}.\n"
            "SEDRA is license-restricted and not shipped with the repo; "
            "rebuild the cache from your own download:\n"
            "    .venv/bin/python -m corpora.sedra_scrape\n"
            "See neural/docs/DATA.md."
        )
    return SEDRA_WORD_DIR
