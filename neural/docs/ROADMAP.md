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
- [ ] Twist 1 (pointing restoration) and bits-per-byte vs. the byte-LM remain to
  do; gated on a SEDRA source for vocalization supervision.

**Exit check (met):** a pretrained neural encoder plugs into the bake-off and
reports same/cross-author AUC on the identical cohort. This is the tractable next
paper — including the honest negative on continued-pretraining for authorship.

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
