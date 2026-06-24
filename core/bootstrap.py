#!/usr/bin/env python3
"""Author-cluster bootstrap for same/cross-author AUC.

A 95% percentile confidence interval for the same-vs-cross-author AUC, resampling
**authors** (clusters) rather than documents, because the document pairs are not
independent: every text shares its author label with its siblings. Pairs between
two resampled copies of the same author are skipped, not miscounted as cross.

Extracted from the paper-1 driver so that every paper (notably the ``neural``
authorship bake-off) can import the identical CI machinery from one place.
"""

from __future__ import annotations

import numpy as np

from core.stylometry import (
    auc_same_higher,
    l2_normalize,
    remove_common_component,
)


def bootstrap_auc_ci(matrix: np.ndarray, keys: np.ndarray, *, B: int, seed: int):
    """95% percentile CI for same/cross-author AUC, resampling authors (clusters)."""
    unit = l2_normalize(remove_common_component(matrix))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = unit @ unit.T
    authors = list(dict.fromkeys(keys.tolist()))
    idx_by_author = {a: np.where(keys == a)[0] for a in authors}
    rng = np.random.default_rng(seed)

    point = _auc_from_sims(sims, [(a, idx_by_author[a]) for a in authors])
    boots = []
    for _ in range(B):
        sampled = rng.choice(len(authors), size=len(authors), replace=True)
        groups = [(authors[i], idx_by_author[authors[i]]) for i in sampled]
        boots.append(_auc_from_sims(sims, groups))
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _auc_from_sims(sims, groups):
    """AUC over same- vs cross-author pairs.

    ``groups`` is a list of ``(author_id, index_array)``. Same-author pairs are
    the within-group pairs (a duplicated author, from cluster resampling,
    contributes its within-pairs again). Cross-author pairs are between groups
    with *different* author ids only -- pairs between two resampled copies of the
    same author are skipped, not miscounted as cross.
    """
    same, cross = [], []
    for _, g in groups:
        if len(g) >= 2:
            sub = sims[np.ix_(g, g)]
            iu = np.triu_indices(len(g), k=1)
            same.append(sub[iu])
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i][0] == groups[j][0]:
                continue  # duplicate of the same author -> not a cross pair
            cross.append(sims[np.ix_(groups[i][1], groups[j][1])].ravel())
    if not same or not cross:
        return float("nan")
    return auc_same_higher(np.concatenate(same), np.concatenate(cross))
