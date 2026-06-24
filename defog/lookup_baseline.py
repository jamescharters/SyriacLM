"""Training-free lookup baselines for the (root, features) -> template task.

The decisive question for whether the neural bipartite model earns its keep: can
a trivial table that memorises "majority template for this feature combination"
already match it on *held-out roots*? If yes, the task is a feature->template
lookup and the root/structure machinery is decoration. Two variants:

* feat-only      : key = morph features (the marginal template per feature combo)
* feat+len(T)    : key = (features, template length) -- matches the neural model's
                   information, since the model is given the template length T.

We report per-slot accuracy and whole-template exact match on the same root
holdout split the neural model uses (seed 42).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .data import build_dataset
from .train import split_by_root
from .graph import MORPH_FIELDS


def _fkey(item: dict) -> tuple:
    m = item.get("morphology", {})
    return tuple(m.get(f) for f in MORPH_FIELDS)


def _majority(counter: Counter):
    return counter.most_common(1)[0][0] if counter else None


def evaluate(max_roots: int = 150, holdout_frac: float = 0.15, seed: int = 42,
             verbose: bool = True) -> dict:
    items = build_dataset(max_roots=max_roots, verbose=False)
    train_items, _val, zs_items = split_by_root(items, holdout_frac=holdout_frac, seed=seed)

    by_feat: dict = defaultdict(Counter)            # features -> template counter
    by_feat_len: dict = defaultdict(Counter)        # (features, T) -> template counter
    by_len: dict = defaultdict(Counter)             # T -> template counter (global fallback)
    for it in train_items:
        t = tuple(it["template"])
        if not t:
            continue
        by_feat[_fkey(it)][t] += 1
        by_feat_len[(_fkey(it), len(t))][t] += 1
        by_len[len(t)][t] += 1

    def run(use_len: bool) -> dict:
        slot_sum = 0.0
        exact = 0
        covered = 0
        n = 0
        for it in zs_items:
            true = tuple(it["template"])
            if not true:
                continue
            n += 1
            T = len(true)
            if use_len:
                cand = by_feat_len.get((_fkey(it), T))
            else:
                cand = by_feat.get(_fkey(it))
            if cand:
                covered += 1
                pred = _majority(cand)
            else:
                pred = _majority(by_len.get(T))   # back off to global majority of length T
            if pred is None:
                continue
            slot_sum += sum(1 for i in range(T) if i < len(pred) and pred[i] == true[i]) / T
            exact += int(pred == true)
        return {"per_slot": slot_sum / n, "exact": exact / n,
                "coverage": covered / n, "n": n}

    out = {"feat_only": run(False), "feat_len": run(True)}
    if verbose:
        for tag, r in out.items():
            print(f"  lookup [{tag:9}] per-slot {r['per_slot']:.3f}  "
                  f"exact {r['exact']:.3f}  coverage {r['coverage']:.3f}  (n={r['n']})")
    return out


if __name__ == "__main__":
    evaluate()
