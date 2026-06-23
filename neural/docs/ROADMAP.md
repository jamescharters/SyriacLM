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

## P1 — First encoder + Twist 1 (needs deps + compute)

- [ ] Wire the base tokenizer (CANINE codepoints / ByT5 bytes) into
  `benchmark.EncoderAdapter._encode`.
- [ ] Implement the masking collators in `pretrain.py`: span masking (MLM) on
  running text; **diacritics-only** masking for the pointing objective.
- [ ] LoRA continued-pretraining over the aggregated shards.
- [ ] **Deliverable:** the first neural Syriac vocalizer; report vocalization
  accuracy split by in-SEDRA-vocab vs. OOV-of-SEDRA forms.

**Exit check:** held-out bits-per-byte below the from-scratch byte-LM baseline;
encoder plugs into the bake-off and reports same/cross-author AUC with the
parent's bootstrap CIs. This is the tractable next paper.

## P2 — Twist 2: factored root/pattern

- [ ] Enable `FactoredEncoder`; supervise the root stream from SEDRA roots and the
  pattern stream from the pointing objective.
- [ ] **Ablate** factored vs. flat on the morphology probe, OOV, and authorship.

## P3 — Semitic transfer curriculum

- [ ] Hebrew/Arabic → Aramaic → Syriac via transliteration; PEFT adapters.
- [ ] Back-translation off the biblical parallel texts (real Syriac target side).

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
