# DeFoG — honest assessment and findings

**Status (2026-06-24): this sub-project is *not* a strong core-ML contribution.**
The original framing did not survive its own controls. What remains is a modest
low-resource NLP / analysis result and a reusable benchmark. This document
records the idea, every experiment we ran, the numbers, and why the conclusion is
what it is. It is deliberately self-critical so the result is not over-sold later.

---

## 1. Bottom line up front

- The headline idea — that a **bipartite root×template generative model** is the
  right inductive bias for Semitic non-concatenative morphology — is *plausible
  and linguistically motivated*, but in our experiments the bespoke structure
  buys little over a generic, off-the-shelf baseline, and nothing over a
  training-free lookup at this data scale.
- The first task we evaluated (predict the **template** from root + features) is
  near-**trivial**: a training-free lookup table beats the neural model and never
  looks at the root. Templates are a function of the features, not the root, so
  there is almost no root-generalisation signal there.
- The genuinely hard, root-dependent task is **morphological reinflection**
  (root consonants + features → inflected consonantal form). Standard char-level
  seq2seq **collapses** on roots unseen in training (0.21 → 0.07 exact). That
  collapse is the one interesting phenomenon here.
- Fixing the collapse does **not** require the bipartite architecture: a generic
  pointer-generator **copy** mechanism recovers most of the gap, an explicit
  structured slot model is marginally better, and a **training-free structured
  lookup beats all of the neural models**.
- **Verdict:** not a new method. The defensible contributions are (1) a
  naturally-grounded compositional-generalisation **benchmark** where seq2seq
  fails, and (2) the mechanism-level **finding** that the failure is
  radical-identity overfitting, removable by decoupling identity from pattern.

---

## 2. The idea (what we set out to test)

Semitic morphology is the textbook case of *non-concatenative* word formation: a
consonantal root (e.g. k–t–b) is interdigitated into a vocalic/affixal template
determined by morphosyntactic features (number, gender, state, person, tense).
The project thesis was that this is structurally **bipartite graph co-generation**
— an unordered root *set* bound into an ordered template *sequence* — and that a
model with that factorisation (`model.py`, `BipartiteDeFoG`: Deep-Sets root
encoder × Transformer template encoder × cross-attention coupling × discrete flow
matching) should generalise to **unseen roots** better than a sequence model.

Data: real SEDRA IV (Syriac), read locally from `corpora/sedra_cache`
(`data.py`). Splits are **root-disjoint** (held-out roots never seen in training),
which is the correct test of systematic generalisation.

---

## 3. What we actually found (the journey)

### 3.1 Re-posing #1 — root → template was ill-posed
The first zero-shot task conditioned on the root alone and asked for a specific
inflected template. But each root realises **~9 distinct templates** (number ×
gender × state × tense …), so root→template is underdetermined. A 120-epoch /
300-root run plateaued at the marginal-pattern floor:

| zero-shot (root only) | result | majority baseline |
|---|---|---|
| template per-slot | ~0.50 | 0.54 |
| template exact | ~0.00 | 0.01 |

The model sat *below* the trivial baseline. Conditioning was missing.

### 3.2 Re-posing #2 — condition on the features (and the ablation that looked good)
We added morphological-feature conditioning (`graph.MORPH_SCHEMA`,
`model.MorphConditioning`, `--no-morph` ablation). On held-out roots (30 epochs,
150 roots, seed 42):

| zero-shot, novel roots | morph **ON** | morph **OFF** | majority baseline |
|---|---|---|---|
| template per-slot | 0.61 | 0.50 | 0.54 |
| template exact | 0.15 | 0.06 | 0.01 |

This *looked* like compositional generalisation (exact match ~18× the majority
baseline). **It was misleading**, as the next control showed.

### 3.3 The control that broke it — a training-free lookup beats the model
`lookup_baseline.py`: a table mapping a feature key → its majority template from
the training set, applied to held-out roots, no learning, no root:

| zero-shot, held-out roots | neural (morph ON) | lookup **feat+len** | lookup feat-only |
|---|---|---|---|
| template per-slot | 0.61 | **0.64** | 0.40 |
| template exact | 0.15 | **0.27** | 0.14 |

A table that **never sees the root** beats the bipartite model. Conclusion: the
template task carries almost no root-generalisation signal — templates are
feature-determined. The re-posing #2 "win" was the model partially recovering a
lookup.

### 3.4 Re-posing #3 — the real task is reinflection (generate the form)
The root-dependent quantity is the **surface form**, not the template. Evidence:

- a `(features, length) → surface-string` lookup scores **0.000** exact on
  held-out roots (so the form genuinely depends on the root), and
- **84%** of forms carry pattern affixes (m-, n-, t- …; mean ≈ 2 extra consonant
  slots beyond the radicals), so binding + affixation is non-trivial.

So we evaluate **morphological reinflection** (root consonants + features →
inflected consonantal form) — the standard SIGMORPHON setup — on the same
root-disjoint split. Three models:

- `reinflect_baseline.py` — GRU encoder–decoder + attention (unstructured ref).
- `reinflect_baseline.py --copy` — pointer-generator copy (radicals copied from
  input; the *honest strong* baseline).
- `structured_reinflect.py` — emits an abstract slot sequence (`ROOT_i` radical
  index / `AFFIX_c`), conditioned on features + a Deep-Sets root summary, then
  renders with the actual radicals. Radical **identity** is abstracted into
  indices, so a correct slot sequence transfers to a novel root by construction.
  (The slot representation reconstructs the consonantal surface at fidelity
  **1.000**, so it imposes no representational ceiling.)

**Held-out-root exact match (3 seeds: 42, 7, 13; 40 epochs, 150 roots):**

| model | held-out exact (mean) | per-seed | seen-root exact |
|---|---|---|---|
| plain char-seq2seq | **0.067** | .069 / .078 / .054 | 0.21 |
| copy-seq2seq (pointer-generator) | **0.153** | .179 / .146 / .135 | 0.17 |
| structured reinflector | **0.176** | .185 / .155 / .187 | 0.23 |
| structured-**lookup** (training-free) | **0.211** | — | — |

Per-character accuracy on held-out roots is ~0.32–0.39 for all three neural
models (so they get most characters but rarely the whole word).

---

## 4. Why this is not a strong contribution

1. **The bespoke architecture is not what wins.** Structured (0.176) beats a
   *generic 2017 copy mechanism* (0.153) by a hair, and both are beaten by a
   **training-free** structured lookup (0.211). The gain comes from *decoupling
   radical identity from the pattern* — which copy-attention, explicit slots, and
   a lookup table all achieve. None of that is novel ML.
2. **The first task was near-trivial** and the apparent "compositional
   generalisation" win was a lookup in disguise (§3.3).
3. **Scale is tiny and single-config.** 150 roots, 40 epochs, one architecture
   size, three seeds. These are diagnostic numbers, not a benchmarked system.
4. The original DeFoG/discrete-flow machinery (`model.py`) is, in the end, not
   needed for the part that matters (form generation); the reinflection models
   are plain autoregressive decoders.

---

## 5. What *is* defensible (if written up honestly)

- **A naturally-grounded compositional-generalisation benchmark.** Unlike
  synthetic SCAN/COGS, this is a real language with linguistically meaningful
  axes (root × pattern) and a clean root-disjoint split, where standard seq2seq
  measurably fails to generalise (seen 0.21 → held-out 0.07).
- **A mechanism-level finding.** The failure mode is *radical-identity
  overfitting*; it is removed three independent ways (generic copy, explicit
  slot/index abstraction, or a lookup table). The training-free lookup beating all
  neural models is itself a useful caution against "throw a transformer at it" on
  small morphological data.
- **Connection to paper 3 (disentanglement).** The same "abstract the radical
  identity, keep the pattern" decomposition is what the disentanglement probe
  measures; defog is the *generative* mirror of that analysis.

This is a **low-resource NLP / analysis** result, not a core-ML architecture
claim. It should not be framed as "a new model for Semitic morphology."

---

## 6. Honest limitations of these experiments

- Small scale (150 roots) and a single model size; no hyperparameter search; no
  confidence intervals beyond 3 seeds.
- Evaluated on the **consonantal** skeleton; full vocalised generation (the
  vowels/pointing) is not attempted here (that is `neural/`'s vocaliser).
- Greedy decoding only; no beam search (would lift all neural models similarly).
- The copy baseline is strong but not exhaustively tuned; a modern
  Transformer-with-copy or the UniMorph/SIGMORPHON neural baselines might change
  the *absolute* numbers (unlikely to change the ordering or the verdict).
- The `model.py` discrete-flow model is evaluated only on the (now-known-trivial)
  template task; we did not invest in scaling it to full-form generation because
  the reinflection results already answer the "does structure earn its keep"
  question.

---

## 7. File guide

| file | role |
|---|---|
| `data.py` | local SEDRA IV reader (`corpora/sedra_cache`) + morphological parser |
| `graph.py` | bipartite graph construction + `MORPH_SCHEMA` feature schema |
| `model.py` | `BipartiteDeFoG` discrete-flow model + feature conditioning (template task) |
| `train.py` | training loop + zero-shot (root, features) → template eval |
| `run.py` | entry point for the discrete-flow model (`--no-morph` ablation) |
| `lookup_baseline.py` | training-free feature→template table (shows template task is trivial) |
| `reinflect_baseline.py` | char seq2seq + `--copy` pointer-generator (reinflection baselines) |
| `structured_reinflect.py` | slot-sequence structured reinflector (the structural model) |

### Reproduce the decisive numbers
```bash
# template task is trivial (lookup beats the neural model):
.venv/bin/python -m defog.lookup_baseline

# reinflection showdown, held-out roots (run per seed 42/7/13):
.venv/bin/python -m defog.reinflect_baseline        --epochs 40 --max-roots 150 --seed 42
.venv/bin/python -m defog.reinflect_baseline --copy --epochs 40 --max-roots 150 --seed 42
.venv/bin/python -m defog.structured_reinflect      --epochs 40 --max-roots 150 --seed 42
```

---

## 8. If someone wants to revisit this

The only path to a *core-ML* claim is to make the structural prior win
**decisively** over a strong copy baseline, most plausibly where copy can't help:

- much lower data (few-shot per pattern), where index-abstraction should pull
  further ahead of copy;
- full vocalised forms (harder binding), not just consonants;
- broader cross-Semitic transfer (train Syriac → test Arabic/Hebrew patterns),
  using the UniMorph data already wired up in `disentangle/`.

Absent a decisive win there, treat defog as the benchmark+finding described in §5
and do not over-claim.
