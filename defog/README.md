# Syriac Bipartite DeFoG — Toy Implementation

## What this is

A proof-of-concept for **coupled discrete flow matching over bipartite root-template graphs**,
motivated by Semitic non-concatenative morphology.

The core claim: Syriac/Semitic word generation is structurally a *bipartite graph co-generation*
problem, not a sequence transduction problem. A surface form like ܟܬܒ (ktb, "he wrote") is the
product of a consonantal root set R={k,t,b} and a positional template T=[C1aC2aC3], interdigitated.

This toy:
1. Reads real SEDRA IV data from the shared local `corpora/sedra_cache` (no network)
2. Parses Syriac words into (root_consonants, template_slots, surface) triples
3. Builds bipartite graphs: root-consonant nodes ↔ template-slot nodes
4. Trains a bipartite discrete flow model (DeFoG-style) over these graphs
5. Evaluates zero-shot generalisation to unseen roots

## Architecture

### Asymmetric Equivariance
- Root encoder: Deep Sets (permutation-invariant over consonant set)
- Template encoder: positional transformer (order-sensitive over slots)
- Coupling: cross-attention between R and T representations
- Flow: discrete CTMC (continuous-time Markov chain) over joint (R,T) state

### Data pipeline
```
corpora/sedra_cache (SEDRA IV) → root/lexeme/word JSON → Syriac consonant extraction →
(root_consonants, template_pattern, surface_form) triples →
BipartiteMorphGraph objects → DataLoader
```

## Files
- `data.py`      — local SEDRA IV reader (`corpora/sedra_cache`) + morphological parser
- `graph.py`     — BipartiteMorphGraph construction
- `model.py`     — Bipartite DeFoG architecture
- `train.py`     — Training loop + zero-shot eval
- `run.py`       — Entry point

## Usage

Run as a package from the repository root:

```bash
pip install torch
python -m defog.run                # train on real SEDRA IV data (local corpora cache)
python -m defog.run --synthetic    # offline synthetic toy data (no SEDRA)
python -m defog.run --eval         # zero-shot root transfer eval (loads best.pt)
```

The SEDRA IV cache is license-restricted and git-ignored; (re)build it with
`.venv/bin/python -m corpora.sedra_scrape` (see `neural/docs/DATA.md`).
