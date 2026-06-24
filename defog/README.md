# Syriac Bipartite DeFoG

## What this is

A **generative** model of Semitic non-concatenative morphology: coupled discrete
flow matching over bipartite root–template graphs.

The core claim: Syriac/Semitic word generation is structurally a *bipartite graph
co-generation* problem, not sequence transduction. A surface form like ܟܬܒ (ktb,
"he wrote") is a consonantal root set R={k,t,b} interdigitated into a positional
template T=[C1aC2aC3]. Crucially the **template is determined by the
morphosyntactic features** (number, gender, state, person, tense, category), not
by the root — so the well-posed task is compositional:

> given a (possibly novel) root **and** a feature spec, generate the interdigitated form.

The CS/AI question is *systematic generalisation*: does a model that has seen a
root inflected one way, and a pattern realised on other roots, compose the two
for a **root it has never seen**? The bipartite set×sequence factorisation is the
Semitic-specific inductive bias we test this with.

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
  (the signal that makes template generation well-posed)
- Flow: discrete CTMC (continuous-time Markov chain) over joint (R,T) state

### Why feature conditioning matters (ablation)
Each root realises ~9 distinct templates, so **root alone underdetermines the
template**. On held-out roots (30 epochs, 150 roots, seed 42):

| zero-shot, novel roots | morph **ON** | morph **OFF** (`--no-morph`) | majority baseline |
|---|---|---|---|
| template per-slot acc  | **0.61** | 0.50 | 0.54 |
| template **exact match** | **0.15** | 0.06 | 0.01 |

With the feature spec the model composes pattern×root for unseen roots (exact
match ~18× baseline); without it, accuracy collapses to the marginal-pattern
floor. Longer training widens the gap.

### Data pipeline
```
corpora/sedra_cache (SEDRA IV) → root/lexeme/word JSON → Syriac consonant extraction →
(root_consonants, template_pattern, surface_form, morphology) tuples →
BipartiteMorphGraph objects → DataLoader
```

## Files
- `data.py`      — local SEDRA IV reader (`corpora/sedra_cache`) + morphological parser
- `graph.py`     — BipartiteMorphGraph construction + morphological feature schema
- `model.py`     — Bipartite DeFoG architecture + feature conditioning
- `train.py`     — Training loop + zero-shot (root, features) → template eval
- `run.py`       — Entry point

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
