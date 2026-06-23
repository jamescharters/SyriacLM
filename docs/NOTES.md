# Project Notes — Syriac Corpus Stylometry

A lab notebook and handoff document. It records **what was built, why, the exact
results, and the open threads**, so the project (or the conversation behind it)
can be resumed in a future session without re-deriving everything.

Last updated: 2026-06-23.

---

## Preprint (paper/)

A full arXiv-style write-up lives in [`paper/`](../paper/) (`main.tex`,
`references.bib`, `tables/`), built with **XeLaTeX** (native Syriac via Noto Sans
Syriac, always paired with transliteration). All table numbers are produced by
[`paper_experiments.py`](../paper_experiments.py) (reproducible, seeded) and the
neural baselines by [`nn_baselines.py`](../nn_baselines.py) (PyTorch; byte-LM +
char-Transformer). Headline additions beyond the sections below:

- **Subword ablation (vs. word2vec):** word2vec's morphology margin *degrades for
  rare forms* (+0.125 → +0.076) while FastText holds (+0.32 / +0.29). FastText
  vectorizes 100% of 9,023 held-out OOV forms (word2vec 0%).
- **Representation bake-off (same/cross AUC, floor 1000):** word2vec 0.946,
  FastText 0.915, Delta 0.907, char-Transformer 0.845, byte-LM 0.762. Honest
  finding: subword's win is *intrinsic* (rare/OOV), not at the document level.
- **Bootstrap + multi-seed:** restricted all-words AUC 0.900 [0.830, 0.950];
  five-seed 0.886 ± 0.007. (Cluster bootstrap over authors; a duplicate-author
  pairing bug was fixed so the point estimate lies inside the CI.)
- **Genre-matched test:** AUC 0.900 → 0.883 when cross-author pairs share genre →
  signal survives. **Neural LM quality:** byte-LM 0.861 bpb, char-Transformer
  0.823 bpb (≪ 8-bpb uniform baseline).
- **Cohort note:** `paper_experiments.py` excludes the disputed ids and
  "Anonymous" from the cohorts, so it reports 20 authors (full) / 11 (restricted),
  vs. 22 / 13 in `stylometry.py`. Keep paper numbers internally consistent with
  `paper_experiments.py`.
- **No local TeX toolchain:** the `.tex` passes a structural check (braces,
  environments, `\input` files, citation keys) but was **not** compiled to PDF
  here. Build with `xelatex main && bibtex main && xelatex main && xelatex main`.

---

## Table of contents

1. [Goal & summary](#1-goal--summary)
2. [Environment & reproducibility](#2-environment--reproducibility)
3. [The corpus](#3-the-corpus)
4. [Pipeline & scripts](#4-pipeline--scripts)
5. [Methodology notes](#5-methodology-notes)
6. [Detailed results](#6-detailed-results)
7. [Decisions made (and why)](#7-decisions-made-and-why)
8. [Gotchas & caveats](#8-gotchas--caveats)
9. [Open questions / future work](#9-open-questions--future-work)
10. [How to reproduce everything](#10-how-to-reproduce-everything)

---

## 1. Goal & summary

**Question:** do character n-gram FastText embeddings of Classical Syriac capture
**authorial style**, and how does that compare to traditional stylometry?

**What we found:**

- The embeddings are **morphologically coherent** (root-sharing word pairs are
  closer than semantic controls).
- Texts **separate by author** with mean-centered AUC up to ~0.90; a
  false-positive control confirms the signal is real (~0.51, i.e. chance, when an
  author is split against himself).
- FastText is **competitive with Burrows's Delta**; Delta slightly wins
  attribution accuracy.
- On **disputed texts**, the pipeline gives historically sensible verdicts (the
  pseudo-Ephrem letter is not Ephrem; the Pseudo-Clementines read as
  translationese; the Chronicle of Zuqnin sits near Eusebius).
- **Genre is a real confound**: within one author, hymns vs prose separate
  strongly once text length is controlled.

The work was built in four layers, each its own script: corpus stats → FastText
model → same/cross-author test → Delta + disputed + genre analyses.

---

## 2. Environment & reproducibility

- **OS:** macOS (Apple Silicon).
- **Python:** **3.13** (in `.venv/`). **Do not use 3.14** — no gensim/scipy wheels
  for CPython 3.14 as of late 2025, so installs fall back to failing source builds.
- **Packages** (`requirements.txt`): `gensim==4.4.0`, `numpy==2.2.6`,
  `scipy==1.18.0`, plus transitive `smart_open==7.6.1`, `wrapt==2.2.2`.
- **Create the env:**
  ```bash
  python3.13 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  ```
- **Corpus cache:** `~/.cache/syriac-corpus` (cloned automatically on first run;
  re-clone with `--refresh`, update with `--update`).
- **Trained model:** `syriac_fasttext.model` (+ three `.npy` sidecars, ~190 MB
  total). **Git-ignored**; regenerate with `fasttext_model.py`. Training takes
  well under a minute on this machine; seed is fixed (42).

Two **benign warnings** were diagnosed and handled (see §8): a gensim+numpy
`our_dot_float` message (suppressed in `fasttext_model.py`) and spurious NumPy
matmul FP-flag warnings on Apple Silicon (silenced with `np.errstate` in
`stylometry.py`).

---

## 3. The corpus

[srophe/syriac-corpus](https://github.com/srophe/syriac-corpus) — TEI/XML, one
file per text under `data/tei/<id>.xml`, the *Digital Syriac Corpus*, CC BY 4.0.

**Headline stats** (normalized = diacritics stripped):

| Metric | Value |
| --- | --- |
| Files parsed | 632 |
| Single-author texts | 631 |
| Distinct authors | 45 (22 with ≥2 texts) |
| Tokens | 2,179,065 |
| Surface forms | 371,922 |
| Normalized forms | 141,595 |
| Hapax (n=1) | 70,870 (50.0%) |
| Rare (n≤5) | 111,911 (79.0%) |

### How metadata is encoded (verified, important)

- **Author:** `teiHeader/fileDesc/titleStmt/author`, usually with
  `ref="http://syriaca.org/person/<N>"`. Some texts name an author **without**
  the `@ref`; those are merged into the URI-identified author by casefolded name
  (logic in `script.py: collect_stats` and reused in `stylometry.load_texts`).
- **Genre:** **NOT** formally encoded — there are **no** `textClass`/`catRef`/
  `keywords` elements. Genre is only recoverable from the **series title**
  (`title level="s"`), e.g. "Hymns on Nativity" (madrāšē) vs "Prose Refutations
  …". `script.extract_series` returns it (ignoring the boilerplate "Digital
  Syriac Corpus" series).
- **Anonymous:** encoded as `<author role="anonymous">Anonymous</author>` with no
  ref → all such texts collapse to one pseudo-author `"anonymous"`. Excluded from
  author-level analyses.

### Key people / texts (referenced throughout)

- **Ephrem the Syrian** = `syriaca.org/person/13`, ~112 texts. Genres in-corpus:
  **101 madrāšē** (73 "Hymns on Nisibis" + 28 "Hymns on Nativity"), **10 prose**
  ("Prose Refutations of Mani, Marcion, and Bardaisan"), **0 mēmrē/homilies**
  (searched — none for person/13), and **1 disputed letter** (file 690). Hymns are
  short (~454 tok mean); prose is long (~5,400 tok mean).
- **Disputed / pseudonymous texts** used in `authorship.py`:
  | id(s) | work | encoded author | note |
  | --- | --- | --- | --- |
  | 690 | Letter to Mar Papa (Papa bar Aggai dossier) | `Ephrem the Syrian` (no ref) | dossier widely held to be a later forgery; ~1,126 tok |
  | 219–227 | Pseudo-Clementine Recognitions/Homilies | `(Ps.-)Clement of Rome` (person/423) | a translation from Greek; 9 texts |
  | 519 | Chronicle of Zuqnin, Part 3 | `Anonymous` | traditionally mis-ascribed to "Pseudo-Dionysius" of Tel-Mahre; ~25,395 tok |

  File **222** is a 99-token fragment — flagged `[!short]` in output and not
  trusted.

---

## 4. Pipeline & scripts

### `script.py` — corpus + shared helpers
Clones/caches the corpus and reports authors, texts-per-author, vocabulary size.
Exposes the reusable core: `ensure_corpus`, `find_body`, `iter_words`,
`strip_marks`, `extract_authors`, `extract_title`, `extract_series`,
`normalize_space`, `DEFAULT_CACHE`. Tokenizer = runs of Syriac letters
(U+0710–074F) + combining marks (U+0300–036F), requiring ≥1 real letter;
invisible joiners/bidi controls stripped.

### `vocab_stats.py` — frequency statistics
Total tokens, unique forms, hapax, rare (≤5), mean frequency. `--normalize` folds
diacritics. Reuses `script.py`.

### `fasttext_model.py` — the embedding model
- gensim **FastText**, character n-grams, **`min_count=1`** (every form counts),
  `vector_size=100`, skip-gram, `min_n=2`, `max_n=5`, `bucket=200_000`,
  `epochs=10`, `seed=42`. One "sentence" per TEI file. Default trains on
  diacritic-stripped tokens (`--no-normalize` to keep them).
- **Morphological coherence test:** for root families, checks that a root-sharing
  pair is closer than a semantically adjacent control pair (king/kingdom vs
  king/father on root m-l-k; write/book vs write/read on k-t-b). Prints
  per-concept neighbors and a PASS/FAIL. Exits non-zero if any case fails (usable
  as a CI check).
- Saves to `syriac_fasttext.model` with `--save`.

### `stylometry.py` — same vs cross-author separation
- Each text → frequency-weighted **mean** of its word vectors, L2-normalized.
- Compares **same-author** vs **cross-author** cosine over all text pairs; reports
  **AUC** (Mann–Whitney), Cohen's d, means — in four variants: {raw, mean-centered}
  × {all words, function words (top-200 by corpus frequency)}.
- Two cohorts side by side: **full** (≥2 texts) and **restricted** (≥3 texts,
  ≥2000 tokens).
- **False-positive control:** split one author (default Ephrem) into two random
  pseudo-authors over K=20 seeded half-splits; expect AUC ≈ 0.5.
- Flags: `--num-function-words`, `--min-texts`, `--min-tokens`, `--control-author`,
  `--control-splits`, `--seed`, `--no-control`.
- **Additive extensions for `authorship.py`:** `Text` now carries `text_id`,
  `series`, `title`; `load_texts(..., exclude_ids=, drop_anonymous=)`; new
  `load_one_text(data_dir, id, normalize)` loads any file regardless of
  attribution. All backward-compatible (the original run is unchanged).

### `authorship.py` — Delta, disputed texts, genre
Three analyses (select with `--analyses delta,attribution,genre`):
1. **fastText vs Burrows's Delta** — both scored by same/cross **AUC** and
   leave-one-out **top-1/top-3** nearest-centroid attribution, on the restricted
   cohort, swept over token floors (`--min-tokens 1000,2000`) and, for Delta, MFW
   sizes (`--delta-mfw 100,200,500`). Delta = z-scored relative frequencies of the
   top-k MFW; distance = mean |Δz| (Manhattan).
2. **Disputed-text attribution** — build known-author centroids (mean-centered,
   cosine), validate with LOO on knowns, then rank each held-out disputed text;
   special read-outs for 690 (percentile within Ephrem's own range) and group
   cohesion vs. external best for the Pseudo-Clementines.
3. **Genre control** — classify Ephrem's genuine texts by series/title
   (`classify_genre`) and measure same- vs cross-genre AUC across a **token-floor
   sweep** (`--genre-min-tokens 0,500,1000`) to expose the length/genre confound.
- The genuine cohort **excludes** the disputed ids and **drops "Anonymous"** so
  they cannot pollute the centroids.

---

## 5. Methodology notes

- **AUC via Mann–Whitney U:** `auc_same_higher(same, cross)` = P(same-author pair
  more similar than cross-author pair). Implemented with an average-rank function
  (`_avg_ranks`) so no scipy dependency for ranking; ties handled.
- **Embedding anisotropy (critical):** averaged document vectors share a large
  common component, so *raw* cosines cluster near 1 (cross-author mean ≈ 0.95) and
  hide the signal. **Mean-centering** (subtract the corpus-mean document vector —
  unsupervised, label-free) removes it and is required. Effect: same/cross AUC
  jumps 0.725 → 0.830 (all words). Both raw and centered are reported.
- **Function words** = top-N (default 200) most frequent forms corpus-wide — the
  traditional topic-independent stylometric markers. Computed once and reused.
- **Burrows's Delta:** per-text relative frequencies over top-k MFW, z-scored by
  corpus mean/SD per word; Delta distance = mean absolute z-difference.
- **Attribution:** nearest **author centroid** (cosine for FastText, mean-|Δz| for
  Delta), strictly **leave-one-out** (the held-out text is removed from its own
  author's centroid before scoring).
- **Cohort design:** restricting to long texts by prolific authors removes
  size-imbalance (a few huge authors like Ephrem dominating); it *raises* AUC,
  i.e. sharpens rather than manufactures the signal.

---

## 6. Detailed results

### 6.1 Morphology (`fasttext_model.py`)

| root | related pair | cos | control pair | cos | margin |
| --- | --- | --- | --- | --- | --- |
| m-l-k | king / kingdom | +0.686 | king / father | +0.375 | +0.311 |
| k-t-b | write / book | +0.751 | write / read | +0.303 | +0.448 |

Vocabulary 141,595 forms; both cases PASS.

### 6.2 Same vs cross-author (`stylometry.py`, seed 42)

Mean-centered AUC (raw in parentheses):

| Cohort | texts / authors | all words | function words |
| --- | --- | --- | --- |
| Full (≥2 texts) | 608 / 22 | 0.830 (0.725) | 0.784 (0.611) |
| Restricted (≥3 texts, ≥2000 tok) | 243 / 13 | 0.898 | 0.885 |

**False-positive control** (Ephrem, 10 texts ≥2000 tok, 20 splits): mean AUC
centered-all **0.511**, func 0.505, raw ≈ 0.50 → **PASS** (no spurious
separation). Raw cross-author cosine mean ≈ 0.95 (anisotropy).

### 6.3 FastText vs Delta (`authorship.py`)

Restricted cohort, **11 authors** (fewer than stylometry's 13 because disputed
texts + Anonymous are excluded here).

Floor ≥1000 tok (256 texts):

| method / feature | AUC | top-1 | top-3 |
| --- | --- | --- | --- |
| fastText all words | 0.915 | 0.930 | 0.973 |
| fastText function words | 0.909 | 0.922 | 0.988 |
| Delta MFW=100 | 0.907 | 0.957 | 0.992 |
| Delta MFW=200 | 0.898 | 0.969 | 0.988 |
| Delta MFW=500 | 0.850 | 0.965 | 0.992 |

Floor ≥2000 tok (201 texts): fastText all 0.900 / 0.930; Delta MFW=100 **0.940 /
0.970**; Delta MFW=500 0.873 / **0.980**. Takeaway: comparable; Delta edges
attribution, large MFW hurts Delta AUC. LOO accuracy on knowns ≈ **0.93**.

### 6.4 Disputed texts (`authorship.py`, reference floor 1000)

- **690 (pseudo-Ephrem letter):** cos-to-Ephrem **−0.454**, **0th percentile** of
  his 13 genuine texts, Ephrem ranks **#11/11**; nearest Paul/Aphrahat →
  **not Ephrem** (the reference centroid is prose-dominated, so it's a genre-fair
  comparison for a prose letter).
- **219–227 (Pseudo-Clementines):** per-text nearest is **Paul / Eusebius** (other
  Greek→Syriac translations); group cohesion 0.696 < best external 0.777 → style
  dominated by **translationese**, not a single author.
- **519 (Chronicle of Zuqnin, Anonymous):** nearest **Eusebius** 0.75 (the
  historiography register).

### 6.5 Genre control within Ephrem (`authorship.py`)

madrāšē (101, ~454 tok) vs prose (10, ~5,434 tok). Same/cross-genre AUC across a
token-floor sweep:

| token floor | sizes | AUC | Cohen d |
| --- | --- | --- | --- |
| ≥0 | madrāšē 101, prose 10 | 0.587 | +0.34 |
| ≥500 | madrāšē 29, prose 10 | 0.860 | +1.47 |
| ≥1000 | madrāšē 3, prose 10 | 0.831 | +1.33 |

**Genre signal is length-dependent**: weak when short noisy hymns dominate, strong
at matched length → genre/register is a genuine confound. Compare like with like.

---

## 7. Decisions made (and why)

- **`min_count=1`** in FastText — per the original request; every form counts.
- **Normalize (strip diacritics) by default** — aligns forms by consonantal root,
  which is what character n-grams should exploit. Model and analyses must agree on
  this (they do).
- **Mean-centering** added after observing inflated raw cosines (anisotropy).
- **Restricted cohort (≥3 texts, ≥2000 tok)** to remove author-size imbalance;
  shown **side by side** with the full cohort.
- **False-positive control** averaged over **K=20 seeded** splits (not one
  arbitrary split) for a reliable null.
- **Token floor shown at both 1000 and 2000** (sensitivity analysis) rather than
  picking one.
- **Delta evaluated on both AUC and LOO top-1/top-3**, MFW swept 100/200/500.
- **Disputed set** = 690 + 219–227 + 519, plus LOO validation on knowns.
- **Genre** = madrāšē vs prose (mēmrē absent for Ephrem); reported as a **floor
  sweep** after discovering the length confound (a single number was misleading).
- **"Anonymous" excluded** from centroids (not a real author); disputed texts
  excluded from the training cohort so they can't leak into centroids.

---

## 8. Gotchas & caveats

- **Python 3.14 has no gensim/scipy wheels** → use 3.13.
- **gensim 4.x + numpy 2.x** prints a benign `Exception ignored in:
  '…our_dot_float'` once per worker during training. It does **not** affect the
  vectors. `sys.unraisablehook` does **not** catch it, but `contextlib.
  redirect_stderr` does (Cython uses `PySys_WriteStderr`); `fasttext_model.py`
  captures and filters it. A numpy downgrade does not help.
- **NumPy matmul on Apple Silicon** can raise spurious `divide/overflow/invalid
  encountered in matmul` RuntimeWarnings even when inputs **and** outputs are
  finite (SIMD kernel trips FP flags). Verified finite, then silenced with
  `np.errstate(...)` around the cosine product in `stylometry.py`.
- **Anisotropy:** always mean-center averaged doc vectors before cosine.
- **Short texts** give noisy vectors (file 222 = 99 tok). Results are reported
  across token floors; treat single short-text verdicts cautiously.
- **Genre ⟂ authorship confound:** within-author genre separation is strong at
  matched length, so headline cross-author numbers partly reflect genre.
- **Translations** (Pseudo-Clementines) reflect the **translator/register**, not a
  native author — expected to look like other translated works.
- **Author identity merging:** name-only authors are merged into their
  syriaca.org URI by casefolded name; imperfect if names vary in spelling.

---

## 9. Open questions / future work

- **Per-stanza / windowed vectors** for madrāšē so short hymns aren't penalized;
  or length-normalized document representations.
- **Genre-matched authorship test:** redo same/cross-author AUC within a single
  genre to factor out the genre confound quantified in §6.5.
- **Supervised verification** (e.g. an SVM / logistic "same-author?" classifier on
  vector pairs) and proper **calibration / significance** (bootstrap CIs on AUC).
- **Hyperparameters:** sweep `vector_size`, n-gram range, epochs; try CBOW;
  compare word-level vs subword contributions.
- **More disputed cases:** other pseudonymous dossiers; the Ephrem Graecus vs
  Syriac question; works of uncertain ascription beyond those tested.
- **Better author normalization** using the syriaca.org person IDs directly.
- **Delta variants:** Cosine Delta / Eder's Delta; quantify FastText vs Delta with
  significance, not just point estimates.
- **Package the model** or ship a smaller quantized version (currently git-ignored
  at ~190 MB).

---

## 10. How to reproduce everything

```bash
# 0. Environment (Python 3.13)
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 1. Corpus overview (clones to ~/.cache/syriac-corpus on first run)
.venv/bin/python script.py --top 10
.venv/bin/python vocab_stats.py --normalize

# 2. Train FastText + morphology test  (writes syriac_fasttext.model*, git-ignored)
.venv/bin/python fasttext_model.py --save syriac_fasttext.model

# 3. Same/cross-author separation + false-positive control
.venv/bin/python stylometry.py

# 4. Delta vs FastText, disputed texts, genre control
.venv/bin/python authorship.py
```

All randomness is seeded (42); reruns reproduce the numbers in §6. Each script
takes `--help`. Use `--refresh` to re-clone or `--update` to git-pull the corpus.

---

*This document plus `README.md` and `requirements.txt` are the committable record.
The matching machine memory lives under `/memories/` (session `plan.md` and the
repo note) for agent continuity, but everything needed by a human is here.*
