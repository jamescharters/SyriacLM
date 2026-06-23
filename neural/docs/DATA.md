# Data

Verified data-availability findings (the gate for Twist 1) and the licensing
constraints that shape what this sub-project can release.

## Sources

| Source | Content | Vocalized? | License | Role here |
|---|---|---|---|---|
| **Digital Syriac Corpus** (`srophe/syriac-corpus`) | 632 classical authored TEI texts, 2.18M tokens | **Partly** (~56% of tokens carry vowel points) | CC BY 4.0 | Primary running-text corpus; also **classical vocalization gold** (see below) |
| **ETCBC SyrNT** (`ETCBC/syrnt`) | Syriac NT, ~109k tokens, Text-Fabric morphology | No (consonantal running text) | MIT | Running text + morphology labels |
| **ETCBC Peshitta** (`ETCBC/peshitta`) | Syriac OT, ~426k tokens | No (consonantal) | MIT | Running text (register/translation reference) |
| **SEDRA 3** (`peshitta/sedrajs`; Beth Mardutho / G. Kiraz) — the openly-distributed text DB, *not* the current web-based SEDRA IV (2015, ~65k words) | Vocalized lexicon: word/lexeme/root + full morphology | **Yes** (`vocalised` field) | **MIT *with restrictions*** | Twist 1/2 vocalization + root supervision |
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
  vocalization. So *training* vocalization supervision is **type-level (SEDRA)**,
  which is why Twist 1 is framed at the word level first.

## Classical vocalization gold (the register question — now measured)

The DSC was long assumed to be consonantal, but **~56% of its tokens actually
carry Syriac vowel points** (U+0730–U+074A: pthaha, zqapha, rbasa, hbasa, esasa,
the East-Syriac dotted zlamas, quššaya/rukkaka), across 600 of 632 files. This is
a genuine **classical** vocalization gold set, distinct in register from SEDRA's
NT lexicon. [`dsc_gold.py`](../dsc_gold.py) harvests it: it *derives* the
Unicode-point→CAL vowel mapping from data (aligning ~541K DSC tokens to SEDRA
skeletons with a unique vocalization; vowels map at 0.90–0.98 purity) and builds a
~134K-form gold set, split into **in-SEDRA-vocab** vs. **OOV-of-SEDRA** forms.

SEDRA's vocalized vocabulary is **New-Testament-Peshitta-scoped** while the DSC is
classical authored prose and verse, so the standing research question —

> Does morphological vocalization learned on NT vocabulary transfer to
> classical/poetic register?

— is now answered with a number. Because classical pointing is *partial* (a scribe
vocalizes selectively), the fair metric is vowel accuracy on the slots actually
pointed. The NT-only vocaliser reaches **~0.63** on in-vocab forms and **~0.59**
on OOV-of-SEDRA forms (unseen consonantal skeletons; per-consonant baselines 0.40 /
0.45) — genuine cross-register, cross-vocabulary generalization, *on-thesis* with
the paper's finding that subword/templatic structure generalizes across the long
tail via shared roots. Honest limits: partial pointing, the vowel/rukkaka-ambiguous
U+073C mark, and a few archaic variant letters left unmapped.

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
