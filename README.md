# Syriac Bipartite DeFoG — Toy Implementation

## What this is

A proof-of-concept for **coupled discrete flow matching over bipartite root-template graphs**,
motivated by Semitic non-concatenative morphology.

The core claim: Syriac/Semitic word generation is structurally a *bipartite graph co-generation*
problem, not a sequence transduction problem. A surface form like ܟܬܒ (ktb, "he wrote") is the
product of a consonantal root set R={k,t,b} and a positional template T=[C1aC2aC3], interdigitated.

This toy:
1. Fetches real data from the SEDRA IV API (3284 roots, 61445 words)
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
SEDRA API → root/word JSON → Syriac consonant extraction →
(root_consonants, template_pattern, surface_form) triples →
BipartiteMorphGraph objects → DataLoader
```

## Files
- `data.py`      — SEDRA API fetcher + morphological parser
- `graph.py`     — BipartiteMorphGraph construction
- `model.py`     — Bipartite DeFoG architecture
- `train.py`     — Training loop + zero-shot eval
- `run.py`       — Entry point

## Usage
```bash
pip install torch torch-geometric requests tqdm
python run.py --fetch      # fetch SEDRA data (needs internet)
python run.py --train      # train on cached data
python run.py --eval       # zero-shot root transfer eval
```
