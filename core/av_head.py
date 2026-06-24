#!/usr/bin/env python3
"""A small supervised authorship-verification (AV) head over the FastText vectors.

This is the modern, *learned* representation baseline for the stylometry
experiments. Unlike the unsupervised rows (FastText/word2vec mean, Burrows's
Delta), it trains a tiny projection network with a supervised-contrastive
objective so that same-author documents are pulled together and different-author
documents pushed apart.

Crucially, it is evaluated **leave-one-author-out (LOAO)**: to embed an author's
texts, the head is trained only on the *other* authors. Every document therefore
receives an *out-of-sample* embedding from a head that never saw its author, so
the resulting same/cross-author AUC and attribution are not inflated by the
supervision. This makes it directly comparable to the unsupervised methods (which
use no author labels at all): the underlying FastText encoder is shared and
unsupervised; only the verification metric is learned, and that metric is tested
on held-out authors.

The model is deliberately tiny (a one-hidden-layer MLP on 100-d vectors) and
trains in a couple of seconds per fold on CPU -- no GPU required.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # pragma: no cover - handled by callers
    _TORCH = False


# Defaults chosen to resist overfitting on a small author pool (~11 authors):
# low capacity, dropout, and weight decay.
DEFAULTS = dict(hidden=64, out_dim=32, dropout=0.3, epochs=200,
                lr=1e-3, weight_decay=1e-2, temperature=0.1)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if _TORCH:

    class ProjectionHead(nn.Module):
        """One-hidden-layer MLP producing L2-normalized embeddings."""

        def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            z = self.net(x)
            return z / (z.norm(dim=1, keepdim=True) + 1e-9)

    def supcon_loss(z, labels, temperature: float):
        """Supervised contrastive loss (Khosla et al. 2020).

        For each anchor, positives are same-author samples in the batch; the loss
        maximizes their relative similarity against all other samples.
        """
        n = z.size(0)
        sim = (z @ z.T) / temperature
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability
        exp = torch.exp(sim).masked_fill(eye, 0.0)
        log_denom = torch.log(exp.sum(dim=1) + 1e-9)
        log_prob = sim - log_denom[:, None]
        pos = (labels[:, None] == labels[None, :]) & ~eye
        pos_counts = pos.sum(dim=1)
        has_pos = pos_counts > 0
        if not bool(has_pos.any()):
            return z.sum() * 0.0
        per_anchor = (log_prob * pos).sum(dim=1)[has_pos] / pos_counts[has_pos]
        return -per_anchor.mean()

    def _train_head(X: np.ndarray, y: np.ndarray, *, hidden, out_dim, dropout,
                    epochs, lr, weight_decay, temperature, seed):
        _set_seed(seed)
        device = torch.device("cpu")  # tiny model; CPU is fast and deterministic
        model = ProjectionHead(X.shape[1], hidden, out_dim, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.long, device=device)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = supcon_loss(model(Xt), yt, temperature)
            loss.backward()
            opt.step()
        model.eval()
        return model


def leave_author_out_projection(matrix: np.ndarray, keys: np.ndarray, *,
                                seed: int = 42, **over) -> np.ndarray:
    """Return out-of-sample learned embeddings, one row per input document.

    For each author, a fresh head is trained on all *other* authors' documents and
    used to embed this author's documents. The returned matrix is therefore free
    of train/test author overlap and can be scored with the same machinery as the
    unsupervised representations.
    """
    if not _TORCH:
        raise RuntimeError("PyTorch is required for the AV head.")
    cfg = {**DEFAULTS, **over}
    authors = list(dict.fromkeys(keys.tolist()))
    id_of = {a: i for i, a in enumerate(authors)}
    y = np.asarray([id_of[k] for k in keys])

    out = np.zeros((len(matrix), cfg["out_dim"]), dtype=np.float64)
    for a in authors:
        test = keys == a
        train = ~test
        if len(set(y[train].tolist())) < 2:
            continue  # need >=2 training authors to form contrastive pairs
        model = _train_head(
            matrix[train], y[train], hidden=cfg["hidden"], out_dim=cfg["out_dim"],
            dropout=cfg["dropout"], epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], temperature=cfg["temperature"], seed=seed)
        with torch.no_grad():
            xte = torch.tensor(matrix[test], dtype=torch.float32)
            out[test] = model(xte).cpu().numpy()
    return out
