# Data

Verified data-availability findings (the gate for Twist 1) and the licensing
constraints that shape what this sub-project can release.

## Sources

| Source | Content | Vocalized? | License | Role here |
|---|---|---|---|---|
| **Digital Syriac Corpus** (`srophe/syriac-corpus`) | 632 classical authored TEI texts, 2.18M tokens | No (consonantal) | CC BY 4.0 | Primary running-text corpus (parent cache) |
| **ETCBC SyrNT** (`ETCBC/syrnt`) | Syriac NT, ~109k tokens, Text-Fabric morphology | No (consonantal running text) | MIT | Running text + morphology labels |
| **ETCBC Peshitta** (`ETCBC/peshitta`) | Syriac OT, ~426k tokens | No (consonantal) | MIT | Running text (register/translation reference) |
| **SEDRA 3** (`peshitta/sedrajs`; Beth Mardutho / G. Kiraz) | Vocalized lexicon: word/lexeme/root + full morphology | **Yes** (`vocalised` field) | **MIT *with restrictions*** | Twist 1/2 vocalization + root supervision |
| **CAL** (cal.huc.edu) | ~3M parsed Aramaic words, vocalized | Yes | Access-restricted (search UI) | Reference only, not bulk training |

## The data gate for Twist 1 — CLOSED

The make-or-break question was whether enough **vocalized** Syriac exists. It
does, at the lexicon/type level:

- The SEDRA `Word` record (published `sedra-model` schema) is
  `makeWord(lexemeId, word, vocalised, morphologicalType, attributes)` with an
  explicit **`vocalised`** field, e.g. consonantal `"ABHOH;"` → vocalized
  `"AaB,oHaOH_;"`. Each word chains to its **root** and full morphology
  (state/person/number/gender/tense/form).
- In SEDRA's CAL-style ASCII, **lowercase a/e/i/o/u are vowels** and a few
  punctuation marks are diacritics; everything else is the consonantal skeleton.
  This yields `(skeleton → pointing)` pairs *and* the Twist-2 consonant/vowel
  channel split with **no inference** — verified by `sedra --selftest`.
- The ETCBC SyrNT/Peshitta running text is **consonantal** (its Text-Fabric
  `text-orig-full` format carries no pointing); it supplies morphology, not
  vocalization. So vocalization supervision is **type-level (SEDRA)**, not running
  prose — which is why Twist 1 is framed at the word level first.

## The honest caveat: register skew

SEDRA's vocalized vocabulary is **New-Testament-Peshitta-scoped**; the training
corpus (DSC) is classical authored prose and verse across many genres and
centuries. So vocalization supervision is narrow in register. Rather than bury
this, we make it a measured research question:

> Does morphological vocalization learned on NT vocabulary transfer to
> classical/poetic register?

Reported by splitting vocalization accuracy into **in-SEDRA-vocab** vs.
**OOV-of-SEDRA** forms on held-out DSC text. This is *on-thesis* with the paper's
result that subword representations generalize across the long tail via shared
roots (56% OOV root-NN).

## Licensing constraint (shapes releases)

**SEDRA 3 is "MIT with restrictions"** (per the `peshitta/sedrajs` LICENSE):

- academic/personal use only; **not** for profit;
- **do not redistribute altered versions** of the database files;
- any publication using SEDRA must include the acknowledgement formula and cite
  Kiraz 1994 (see `sedra.SEDRA_CITATION`; print with
  `python -m neural.sedra --cite`).

Consequences for this repo:

1. **No SEDRA data and no SEDRA-derived training files are committed.** You obtain
   SEDRA yourself and point `sedra.py` at it; derived pairs cache under
   `sedra_cache/` (git-ignored).
2. We release the **regeneration code**, not the derived data — the same pattern
   `sedrajs` itself follows.
3. The SEDRA path is **optional and guarded**, so the core sub-project stays
   CC-BY/MIT-clean. The DSC + ETCBC pipeline (`aggregate.py`) needs no SEDRA at
   all.

## Obtaining SEDRA (user-run, not automated here)

SEDRA is intentionally **not** auto-downloaded. To enable Twists 1–2 locally:

1. Obtain the SEDRA 3 database via `peshitta/sedrajs` (Node) or another route you
   are licensed to use.
2. Export word records to a JSON list of
   `{"word"/"skeleton", "vocalised", "root", "lexeme", "features"}` objects.
3. Place it where `sedra.find_sedra_source()` looks (e.g.
   `neural/sedra_cache/words.json`) — it is git-ignored.
4. Verify: `python -m neural.sedra --source neural/sedra_cache/words.json`.
