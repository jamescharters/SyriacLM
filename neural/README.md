# Syriac Neural Models (`neural/`)

A self-contained research scaffold for building the **first usable neural
language model for Classical Syriac** — the gap the paper names explicitly
(*"no large pretrained language model… they probe architecture, not scale"*).

Everything for this idea lives under `neural/`. Nothing here is required to
reproduce the published stylometry pipeline; the paper and its FastText model
stand on their own.

---

## The gap is *data scale*, not architecture

A from-scratch monolingual Transformer wants 10⁸–10⁹ tokens. The Digital Syriac
Corpus has **~2.18M**. Even aggregating every openly digitized Syriac text lands
in the low tens of millions — one to two orders of magnitude short. That single
fact dictates the design: **we never pretrain from scratch, and we never
manufacture "new" Syriac from a model fit on Syriac** (information-circular; risks
model collapse). New information enters through exactly three doors:

| Door | What it injects | Where |
|---|---|---|
| **1. Structure** | The root-and-pattern grammar as explicit signal | [`morphology.py`](morphology.py), [`sedra.py`](sedra.py) |
| **2. Transfer** | Knowledge from byte/character + Semitic checkpoints | [`modeling.py`](modeling.py), [`pretrain.py`](pretrain.py), [`transliterate.py`](transliterate.py) |
| **3. Real text** | Genuinely new tokens | [`aggregate.py`](aggregate.py) (HTR/OCR is future work) |

---

## Three novel twists (what makes this more than "continued-pretrain a byte model")

The unifying insight: **Syriac's script is *defective*** — consonants are
written, vowels/pointing are optional and usually absent. The published pipeline
*discards* the pointing ([`strip_marks`](../core/script.py)); we **invert** that.

1. **Pointing restoration as morphological self-supervision** *(non-obvious, lead).*
   A word = root (consonants) + pattern (vocalization). Predicting the pointing is
   recovering the *pattern* morpheme. The vocalized SEDRA lexicon gives aligned
   `(skeleton → pointing)` pairs for free, so vocalization becomes a cheap,
   morphology-aligned pretraining objective — and yields **the first neural
   Syriac vocalizer** as a by-product. Masking *only* the diacritics is structured
   denoising, not random MLM.
2. **Factored root/pattern representation.** A two-stream encoder with explicit
   consonant-skeleton (root) and vocalic-pattern channels, biasing the
   architecture toward Semitic morphology. SEDRA's encoding hands us the split for
   free (`sedra.consonant_vowel_channels`).
3. **Neural textual restoration** of lacunae/emendation ("Ithaca for Syriac") — a
   high-value application demonstrator (honest precedent: restoration exists for
   Greek/Latin; the novelty here is Syriac-first + the morphology framing).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full argument and the honest
prior-art positioning (diacritization itself is not new for Arabic/Hebrew).

---

## Why these architectures (domain fit)

Tokenizer-free **byte/character models are the neural continuation of the paper's
subword thesis**, OOV-free by construction: **CANINE** (codepoints, default) and
**ByT5** (bytes). mBERT/XLM-R barely cover the Syriac script — a concrete reason
to prefer byte/char or cross-script transfer. **PEFT (LoRA)** is the correct
default at tens-of-millions of tokens; **continued pretraining** is the frame.

---

## What runs today vs. what is staged

| File | Status | Needs |
|---|---|---|
| [`config.py`](config.py) | ✅ runnable | stdlib |
| [`transliterate.py`](transliterate.py) | ✅ runnable | stdlib |
| [`sedra.py`](sedra.py) | ✅ runnable (`--selftest`) | stdlib (data is license-gated) |
| [`aggregate.py`](aggregate.py) | ✅ runnable | reuses cached corpora, no new deps |
| [`morphology.py`](morphology.py) | ✅ `--selftest`; ⛓ ETCBC path needs `text-fabric` | optional |
| [`modeling.py`](modeling.py) | ⛓ guarded | `transformers`, `torch`, `peft` |
| [`pretrain.py`](pretrain.py) | ⛓ guarded (`--plan` runs) | the above + compute |
| [`benchmark.py`](benchmark.py) | ✅ `--plan`/`--selftest`; ⛓ checkpoint needs torch | optional |
| [`restoration.py`](restoration.py) | ✅ runnable (`--demo`) | `torch` (ships with the parent) |
| [`sedra_build.py`](sedra_build.py) | ✅ runnable | stdlib + a user-provided SEDRA source (license-gated) |
| [`vocalizer.py`](vocalizer.py) | ✅ runnable (`--demo`, `--cross-register`) | `torch` + the SEDRA table |
| [`dsc_gold.py`](dsc_gold.py) | ✅ runnable (`--report`) | reuses cached DSC + SEDRA, no new deps |
| [`factored.py`](factored.py) | ✅ runnable (`--demo`) | `torch` + the SEDRA table |
| [`canine_encoder.py`](canine_encoder.py) | ✅ runnable | `transformers` + a CANINE download |
| [`canine_pretrain.py`](canine_pretrain.py) | ✅ runnable | `transformers` + `peft` + compute |
| [`hf_encoder.py`](hf_encoder.py) | ✅ runnable | `transformers` + `sentencepiece` |

"Guarded" follows the parent repo's `_TORCH` pattern: the module imports cleanly
and prints precise install instructions if a heavy dependency is absent, so the
scaffold never breaks and the Phase-0 tools always run.

---

## Quick start

```bash
# 0. Phase-0 tools need NO new dependencies (stdlib + what the parent installs).

# Inspect the default configuration
.venv/bin/python -m neural.config

# Deterministic Syriac↔Hebrew / →Latin transliteration (cross-script transfer)
.venv/bin/python -m neural.transliterate --demo

# Verify the SEDRA skeleton/pointing logic (no SEDRA data needed)
.venv/bin/python -m neural.sedra --selftest

# Aggregate the corpora you already have cached into leakage-safe shards
.venv/bin/python -m neural.aggregate --out ~/.cache/syriac-neural

# 1. To train (Phases 1+), install the optional extras into the SAME venv:
.venv/bin/python -m pip install -r neural/requirements-neural.txt
.venv/bin/python -m neural.pretrain --plan          # describe a run
.venv/bin/python -m neural.benchmark --plan         # describe the eval

# Phase 1 — CANINE-c (tokenizer-free, zero-OOV) authorship transfer:
.venv/bin/python -m neural.canine_encoder --floors 1000,2000        # off-the-shelf AUC
.venv/bin/python -m neural.canine_pretrain --steps 1500             # LoRA-adapt to Syriac
.venv/bin/python -m neural.canine_encoder \
    --checkpoint ~/.cache/syriac-neural/checkpoints/canine-lora     # adapted AUC
```

> **Corporate-TLS note.** If `transformers` cannot reach HuggingFace
> (`CERTIFICATE_VERIFY_FAILED`) behind a TLS-intercepting proxy, install
> `truststore` (`pip install truststore`); the CANINE modules inject it at import
> so Python verifies through the OS trust store (the same one `curl` uses). No
> certificate verification is ever disabled.

Checkpoints and aggregated/derived corpora are large and/or license-restricted —
they live in `~/.cache` and are git-ignored ([`.gitignore`](.gitignore)); they are
**never** committed.

---

## Containment & dependencies

- **All files and docs live under `neural/`.** No file outside this folder is
  created or modified.
- The only outward dependency is **read-only imports** of the shared `core`
  package's tokenizer/loaders ([`core/script.py`](../core/script.py),
  [`core/etcbc_corpus.py`](../core/etcbc_corpus.py)) and metric helpers
  ([`core/stylometry.py`](../core/stylometry.py)). This guarantees a
  neural model is compared against the released FastText baseline on
  byte-for-byte the same tokenization. Shared modules are never modified.
- Optional heavy deps are isolated in [`requirements-neural.txt`](requirements-neural.txt);
  the parent `requirements.txt` is untouched.

## Documents

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, the three twists, objectives,
  and the evaluation discipline that keeps the program honest.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased milestones (P0 → P5).
- [`docs/DATA.md`](docs/DATA.md) — corpora, the SEDRA license constraint, and the
  verified data-availability findings.

## Scope boundaries

- **In:** representation/encoder models, transfer + structure injection,
  evaluation reuse, runnable Phase-0 tooling.
- **Out (for now):** HTR/OCR digitization (the real lever on token count — a
  separate sub-project), instruction-tuned generative LLMs, and any use of
  synthetic Syriac as a *scale* substitute.
