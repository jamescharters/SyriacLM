# Design

## Problem statement

The paper establishes that Classical Syriac has **no large pretrained language
model** and that from-scratch neural LMs on 2.18M tokens are *data-starved*
(byte-LM 0.762 / char-Transformer 0.845 separation AUC, trailing the count-based
methods). The gap is therefore **data scale and evaluation**, not architecture.

This sub-project bridges it with a transfer-first, structure-aware encoder, plus
two genuinely novel objectives that exploit a property unique to the script.

## The core insight: a defective script is free supervision

Syriac is written as an **abjad**: consonants are written, vowels and most
diacritics (the *pointing*) are optional and usually omitted. In a
root-and-pattern (templatic) morphology, a word is

$$\text{word} = \underbrace{\text{root}}_{\text{consonants}} \;+\; \underbrace{\text{pattern}}_{\text{vocalization}}.$$

So the pointing a scribe omits **is** the pattern morpheme. The published
pipeline throws it away (`strip_marks`, `normalize=True`). We invert that stance:
**predicting the pointing is learning templatic morphology.**

### Information budget (why we never recycle the corpus)

Let $L$ be the language, $C$ the corpus, $M$ the model. Any generator $G$ fit on
$C$ and sampled to make synthetic text $S$ forms a Markov chain
$L \to C \to G \to S \to M$, so by the data-processing inequality
$I(M;L) \le I(C;L)$ — resampling $C$ cannot add information and risks model
collapse. New information must come from **outside** $C$: structure (the grammar),
transfer (related languages / multilingual checkpoints), or genuinely new real
text. Every technique here opens one of those three doors.

## The three twists

### Twist 1 — pointing restoration as morphological pretraining (lead, non-obvious)

- **Objective.** Given a consonantal skeleton, restore its vocalization. Mask
  *only* the diacritics, keeping the skeleton visible — structured denoising, not
  random MLM. Each vocalized word is one labeled example with known root context.
- **Why non-obvious.** The field reflex for low-resource Semitic is to *strip*
  diacritics to cut sparsity (what the paper does). We turn the discarded signal
  into the supervision. Diacritization is normally an *end task* (reading aid);
  reframing it as a *pretext* for general representations is the move.
- **By-product.** The first neural Syriac vocalizer.
- **Supervision source.** The SEDRA lexicon's `vocalised` field
  ([`sedra.py`](../sedra.py)); `split_skeleton_pointing` produces the exact
  `(skeleton → pointing)` targets.

### Twist 2 — factored root/pattern representation

- A two-stream encoder: a consonant-skeleton (root) channel and a vocalic-pattern
  channel, fused before the task heads, so the inductive bias *matches* Semitic
  morphology instead of hoping a flat model discovers it
  ([`modeling.py`](../modeling.py) `FactoredEncoder`).
- SEDRA's encoding gives the split for free (`consonant_vowel_channels`): vowels
  are lowercase, the skeleton is uppercase. **Ablate** factored vs. flat on the
  morphology probe, OOV, and authorship AUC.

### Twist 3 — neural textual restoration (application demonstrator)

- Fill manuscript lacunae / propose emendations ("Ithaca for Syriac"), evaluated
  via synthetic masking. **Honest framing:** restoration exists for Greek/Latin;
  the novelty here is Syriac-first and the morphology-aware encoder underneath,
  not the task itself.

## Architecture choices (suitability)

- **Tokenizer-free base (CANINE-c default, ByT5 alternative).** Operating on
  codepoints/bytes covers the entire Syriac script with **zero OOV** — the neural
  continuation of the paper's subword/OOV argument. mBERT/XLM-R subword vocabs
  barely cover Syriac, so they are poor bases; prefer byte/char or cross-script
  transfer.
- **PEFT (LoRA).** At tens-of-millions of tokens you cannot re-estimate hundreds
  of millions of parameters; LoRA/adapters are the methodologically correct
  default. Continued pretraining (DAPT) is the overall frame.
- **Semitic transfer via transliteration.** Syriac↔Hebrew is a ~1:1 22-letter
  abjad correspondence ([`transliterate.py`](../transliterate.py)); mapping Syriac
  into Hebrew script lets a Hebrew/Aramaic model's subword space apply — the same
  lever the cited Naaijer et al. (2023) parser uses by borrowing Hebrew data.

## Evaluation discipline (what keeps it honest)

- **Reuse the paper's harness.** Score every model on the *same* tasks and metrics
  as the bake-off via [`benchmark.py`](../benchmark.py), which imports the parent
  `stylometry` metrics read-only: bits-per-byte, the morphology coherence probe
  (T3a), OOV root-NN (T3b), and same/cross-author AUC (T5/T6) with the
  author-cluster bootstrap CIs.
- **Real-only test.** Synthetic/transliterated/vocalized text never enters
  val/test. `aggregate.py` provenance-tags every document and splits **by
  document** with a leakage assertion.
- **Ablate everything.** Report each twist as a *delta* vs. a plain
  continued-pretrain baseline, with CIs — if it doesn't beat the baseline on the
  real held-out tasks, it is noise.
- **Watch for collapse.** Cap any synthetic fraction; monitor distributional
  narrowing (rising perplexity on the real long tail).

## Honest limitations carried from the data gate

- SEDRA's vocalized vocabulary is **NT-Peshitta-scoped**; the training corpus is
  classical authored prose/verse. Cross-register transfer of vocalization is a
  **research question**, reported by splitting accuracy into in-SEDRA-vocab vs.
  OOV-of-SEDRA forms — and on-thesis with the paper's OOV root-generalization
  result. See [`DATA.md`](DATA.md).
- SEDRA is license-restricted (academic-only, no redistribution of altered
  versions): we release regeneration **code**, not SEDRA-derived data.
