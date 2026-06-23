# Roadmap

Phased milestones. Phase 0 runs today with no new dependencies; later phases need
the optional extras in [`../requirements-neural.txt`](../requirements-neural.txt)
and compute. Each phase is gated on the previous one and is validated on the
paper's existing evaluation harness via [`../benchmark.py`](../benchmark.py).

## P0 — Foundation (runnable now, no GPU) ✅

- [x] `config.py` — typed configuration with SOTA defaults.
- [x] `transliterate.py` — deterministic Syriac↔Hebrew / →Latin (cross-script transfer).
- [x] `sedra.py` — license-aware vocalization loader; verified skeleton/pointing split.
- [x] `aggregate.py` — provenance-tagged, deduplicated, **leakage-safe** shards from
  DSC + ETCBC (reuses the parent tokenizer; token counts match the paper).
- [x] `morphology.py` — morphology example builder (SEDRA path + ETCBC TF path).

**Exit check:** `aggregate --dry-run` reports token/type counts, dedup stats, and
a passing leakage assertion; `sedra --selftest`, `morphology --selftest`,
`benchmark --selftest` all PASS. *(All passing.)*

## P1 — First encoder (CANINE-c transfer) ✅ done

- [x] Tokenizer-free CANINE-c wrapped as a **zero-OOV** word/document encoder
  ([`canine_encoder.py`](../canine_encoder.py)), scored on the paper's exact
  authorship cohort and metric.
- [x] **Off-the-shelf transfer baseline** (no Syriac training): centered AUC
  **0.870 / 0.849** at floors 1000 / 2000 — the first demonstration that a
  pretrained multilingual byte model transfers to Syriac authorship, already
  above the from-scratch byte-LM (0.762) and char-Transformer (0.845).
- [x] **LoRA continued-pretraining** ([`canine_pretrain.py`](../canine_pretrain.py)):
  masked-codepoint denoising, 1500 steps, ~1.05M / 133M params adapted (0.79%).
  Held-out masked-codepoint accuracy **0.176 → 0.342** — it demonstrably learned
  Syriac.
- [x] **Honest result:** continued-pretraining *did not* improve document-level
  authorship (AUC **0.857 / 0.838**, marginally below off-the-shelf). This mirrors
  the paper's own finding: the masked-LM objective is *intrinsic* (local
  characters/morphology), and mean-pooling washes it out at the document level, so
  it doesn't transfer to authorship separation. The transfer win is the
  off-the-shelf encoder; the adaptation win is the intrinsic LM metric.
- [x] **Supervised AV head on CANINE vectors** (the paper's leave-one-author-out
  SupCon projection, reused read-only): centered AUC **0.991 [0.979, 0.999]** /
  **0.916 [0.893, 1.000]** at floors 1000 / 2000 (author-cluster bootstrap, B=1000).
  The floor-1000 lower bound (0.979) is **above** the FastText AV head (0.966) — the
  strongest authorship separation in the project, and the clearest demonstration
  that a learned metric on the contextual CANINE encoding beats raw cosine.
  Run: `neural.canine_encoder --av-head`.
- [x] **Intrinsic LM gain isolated** (frozen linear-probe vs. LoRA, identical
  masked *pseudo* bits-per-byte scorer — bidirectional, so CANINE-variants-only,
  not vs. the autoregressive byte-LM): frozen **2.000** bpb / 0.307 acc → LoRA
  **1.859** bpb / 0.342 acc. Adapting 0.79% of params measurably improves the LM.
  Run: `neural.canine_pretrain --freeze-encoder` for the control.
- [x] **SOTA-currency check — Glot500-m** ([`hf_encoder.py`](../hf_encoder.py)):
  tested a newer massively-multilingual base (XLM-R extended to 511 languages,
  2023). **Honest negative:** authorship AUC **0.798 / 0.781**, *below* CANINE-c
  (0.870 / 0.849). The cause is concrete — a tokenizer-coverage check shows
  Glot500 tokenizes Syriac as **pure character fallback** (~7 pieces/word, no
  Syriac subwords), so its "511 languages" does not meaningfully include Syriac.
  The lesson: for a zero-resource abjad the right base is the one with the right
  *inductive bias for the script* (tokenizer-free CANINE), not the one with the
  largest language count. (The newest derivative, EMMA-500 / Llama-3.1-8B, shares
  the Glot lineage's tokenizer, so the same coverage limit is expected; not run.)
- [x] **Twist 1 — the first neural Syriac vocaliser** ✅. Pointing restoration as
  *morphological self-supervision*: the SEDRA vocalised lexicon
  ([`sedra_build.py`](../sedra_build.py), 29,699 words, license-aware/git-ignored)
  supervises a BiLSTM that predicts the vowel/diacritic of each consonant slot
  ([`vocalizer.py`](../vocalizer.py)). Held-out SEDRA words: **per-position
  pointing accuracy 0.811** (majority baseline 0.507) and **full-word exact match
  0.361** (baseline 0.003 — 120×). Honest caveat: SEDRA is New-Testament-scoped,
  so this is held-out *NT-vocabulary*; cross-register transfer to classical text
  is the open question (no openly vocalised classical gold exists).

**Exit check (met):** a pretrained neural encoder plugs into the bake-off and
reports same/cross-author AUC on the identical cohort. This is the tractable next
paper — including the honest negative on continued-pretraining for authorship, the
strong positive from a supervised metric on the contextual encoding, and the
honest negative that a bigger multilingual base does not help when its tokenizer
misses the script.

## P2 — Twist 2: factored root/pattern ✅ ablated (honest negative)

- [x] Built a fair factored-vs-flat ablation ([`factored.py`](../factored.py)):
  both encoders see the same vocalised SEDRA word and are trained with the same
  supervised-contrastive objective on the root; the **factored** model adds two
  parallel BiLSTM streams over the aligned consonant (root) and pointing (pattern)
  tiers, the **flat** model reads the raw character sequence.
- [x] **Result — no benefit.** Root-nearest-neighbour retrieval: flat **0.978**
  (seen roots) / **0.994** (unseen roots) vs. factored **0.972 / 0.994** (Δ −0.007
  / +0.000). Both are near ceiling.
- [x] **Why (reported, not hidden):** root-NN is near-trivial because the consonant
  skeleton is *directly visible* in the input, so surface consonant overlap already
  encodes the root — there is no headroom for the architectural prior. The split is
  easy (vowels are combining marks), so making it explicit does not help. This is
  consistent with the paper's thesis that the intrinsic root signal is easy to
  capture; the hard problem is document-level style, not root recovery.
- The factored encoder remains available for settings where the split is *not*
  free (e.g. unvocalised input, or as a regulariser), but it is not adopted as a
  win here.

## P3 — Semitic transfer curriculum ✅ first transfer win

- [x] **Hebrew transfer via transliteration** ([`hf_encoder.py`](../hf_encoder.py)
  `--transliterate hebrew`): map Syriac into Hebrew script (a ~1:1 abjad
  correspondence, [`transliterate.py`](../transliterate.py)) and encode with a
  Hebrew-pretrained model (AlephBERT). Authorship AUC **0.888 / 0.857** at floors
  1000 / 2000 — **above** off-the-shelf CANINE-c (0.870 / 0.849), the first transfer
  model to beat it.
- [x] **Why it works (tokenizer-confirmed):** AlephBERT segments transliterated
  Syriac into **2.51 real subwords/word** (96.5% covered), vs. Glot500's character
  fallback (7.05/word). A related-language model *does* carry usable Semitic
  morphology once the scripts are aligned. This gives a clean three-way story:
  char-fallback multilingual (Glot500 **0.798**) < tokenizer-free byte
  (CANINE **0.870**) < shared-script Semitic transfer (Hebrew **0.888**).
- [ ] Stack the supervised AV head on the Hebrew-transfer vectors; extend to a
  full curriculum (Arabic too, Aramaic intermediate) and back-translation off the
  biblical parallel texts (real Syriac target side).

## P4 — Twist 3: textual restoration (application) ✅ runnable, working

- [x] Lacuna restoration via a self-contained **causal character Transformer** —
  a genuine from-scratch Syriac character LM ([`restoration.py`](../restoration.py)) —
  evaluated by **synthetic masking** of held-out real text. Needs only `torch` +
  the cached DSC (no `transformers` download, no SEDRA), so it runs today:
  `.venv/bin/python -m neural.restoration --demo`.
- [x] Verified result (2000-step demo, ~620K params, ~70s on MPS): masked
  char-accuracy **0.44** and span exact-match **0.09**, well above the ~0.19
  unigram floor; qualitative fills are morphologically valid Syriac.
- [x] Lesson recorded: a bidirectional masked objective collapses to the unigram
  prior at this model/data scale; the causal objective (dense supervision) is what
  learns. See the module docstring.
- [ ] Scale up (larger model/steps) and add a morphology-aware variant once the
  Phase-1 encoder exists.

## P5 — Community benchmark

- [ ] Package the eval suite (bpb, morphology probe, OOV, authorship AUC) so any
  future Syriac model has a standard to beat.

## Out of scope (separate efforts)

- HTR/OCR digitization — the real lever on token count, a substantial sub-project.
- Instruction-tuned generative LLMs.
- Synthetic Syriac as a *scale* substitute (see `DESIGN.md`, information budget).
