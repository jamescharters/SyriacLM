# Syriac Bipartite DeFoG

> **Status (2026-06-24): exploratory; NOT a strong core-ML result.** The bipartite
> structure does not beat a generic copy baseline by much, and a training-free
> lookup beats every neural model at this scale. The original "template
> generation" framing turned out to be near-trivial. What stands up is a
> low-resource compositional-generalisation **benchmark** plus a mechanism
> **finding** (seq2seq fails on unseen roots by overfitting radical identity).
> Read [docs/FINDINGS.md](docs/FINDINGS.md) for the full, self-critical write-up
> with all numbers before building on this.

## What this is

A **generative** model of Semitic non-concatenative morphology: coupled discrete
flow matching over bipartite root–template graphs.

The original claim (which we then tested and largely walked back — see
[docs/FINDINGS.md](docs/FINDINGS.md)): Syriac/Semitic word generation is
structurally a *bipartite graph co-generation* problem, not sequence
transduction. A surface form like ܟܬܒ (ktb, "he wrote") is a consonantal root set
R={k,t,b} interdigitated into a positional template T=[C1aC2aC3], where the
template is determined by the morphosyntactic features (number, gender, state,
person, tense, category).

The CS/AI question we wanted to ask was *systematic generalisation*: does a model
that has seen a root inflected one way, and a pattern realised on other roots,
compose the two for a **root it has never seen**? The bipartite set×sequence
factorisation is the Semitic-specific inductive bias we tested — and the controls
below show it does not, on its own, earn its keep.

This implementation:
1. Reads real SEDRA IV data from the shared local `corpora/sedra_cache` (no network)
2. Parses Syriac words into (root_consonants, template_slots, surface) triples + morphology
3. Builds bipartite graphs: root-consonant nodes ↔ template-slot nodes
4. Trains a feature-conditioned bipartite discrete flow model (DeFoG-style)
5. Evaluates **zero-shot root transfer**: held-out roots, (root, features) → template

## Architecture

### Asymmetric Equivariance
- Root encoder: Deep Sets (permutation-invariant over consonant set)
- Template encoder: positional transformer (order-sensitive over slots)
- Coupling: cross-attention between R and T representations
- Morphological conditioning: per-feature embeddings summed into a pattern vector
  (intended to make template generation well-posed; see the caveat below)
- Flow: discrete CTMC (continuous-time Markov chain) over joint (R,T) state

### What the controls showed (read this before trusting the model)
Predicting the **template** from (root, features) turned out to be near-trivial:
a training-free lookup table beats the neural model **without ever seeing the
root** (held-out roots: lookup exact 0.27 vs model 0.15), because templates are
feature-determined. The genuinely root-dependent task is **reinflection**
(root + features → inflected *form*). On that task, held-out-root exact match
(3-seed means, 40 epochs / 150 roots):

| model | held-out exact | seen-root exact |
|---|---|---|
| plain char-seq2seq | 0.067 | 0.21 |
| copy-seq2seq (pointer-generator) | 0.153 | 0.17 |
| structured reinflector | 0.176 | 0.23 |
| structured-lookup (training-free) | **0.211** | — |

The structural model wins on every seed but only marginally over a generic copy
baseline, and the lookup tops them all. See [docs/FINDINGS.md](docs/FINDINGS.md).

### Data pipeline
```
corpora/sedra_cache (SEDRA IV) → root/lexeme/word JSON → Syriac consonant extraction →
(root_consonants, template_pattern, surface_form, morphology) tuples →
BipartiteMorphGraph objects → DataLoader
```

## Files
- `data.py`      — local SEDRA IV reader (`corpora/sedra_cache`) + morphological parser
- `graph.py`     — BipartiteMorphGraph construction + morphological feature schema
- `model.py`     — Bipartite DeFoG architecture + feature conditioning (template task)
- `train.py`     — Training loop + zero-shot (root, features) → template eval
- `run.py`       — Entry point for the discrete-flow model
- `lookup_baseline.py`     — training-free feature→template table (shows the template task is trivial)
- `reinflect_baseline.py`  — char seq2seq + `--copy` pointer-generator (reinflection baselines)
- `structured_reinflect.py`— slot-sequence structured reinflector (the structural model)
- `docs/FINDINGS.md`       — **honest assessment, full results, and the verdict**

## Usage

Run as a package from the repository root:

```bash
pip install torch
python -m defog.run                # train on real SEDRA IV data (local corpora cache)
python -m defog.run --no-morph     # ablation: drop feature conditioning
python -m defog.run --synthetic    # offline synthetic toy data (no SEDRA)
python -m defog.run --eval         # zero-shot root transfer eval (loads best.pt)
```

The SEDRA IV cache is license-restricted and git-ignored; (re)build it with
`.venv/bin/python -m corpora.sedra_scrape` (see `neural/docs/DATA.md`).
