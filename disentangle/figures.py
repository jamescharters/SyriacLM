#!/usr/bin/env python3
"""Figures for the disentanglement paper (supervised-axis views).

t-SNE/PCA organise an embedding by its dominant variance, which here is lexical
identity, so the small within-lemma factor offset is invisible in an unsupervised
2-D map. The claim is about *linear* structure, so we plot along the supervised
axes the analysis already uses: the factor direction (mean difference) and an
identity axis orthogonal to it.

* FD1 -- the factor "arrow": each lemma's two forms joined sg->pl, on (factor
         axis x identity axis). One consistent direction = a composable factor.
* FD2 -- factor-axis histograms, vocalised vs consonantal: shows whether the
         factor is separable and whether it is vowel- or consonant-borne.
* FD3 -- before vs after linearly erasing the factor: the separable factor
         collapses while lexical structure is untouched.

Figures are written to ``disentangle/paper/figures/*.pdf``. Needs matplotlib
(see ``disentangle/requirements-disentangle.txt``); everything else is the frozen
encoder + numpy.

    .venv/bin/python -m disentangle.figures --encoder canine --factor number
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:  # pragma: no cover
    _MPL = False

FIG_DIR = Path(__file__).resolve().parent / "paper" / "figures"


def _axes(Xtr, ytr, Xte):
    """Factor axis (mean diff, train) and an identity axis orthogonal to it."""
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    w = w / (np.linalg.norm(w) + 1e-12)
    perp = Xte - np.outer(Xte @ w, w)
    perp = perp - perp.mean(0)
    _, _, vt = np.linalg.svd(perp, full_matrices=False)
    return w, vt[0]


def make_figures(encoder: str = "canine", factor: str = "number",
                 seed: int = 42) -> list[Path]:
    from disentangle import disentangle as D
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = D._device()
    grid = D.load_grid(factor)
    wv, label = D.build_encoder(encoder, device)
    lex = np.array([g["lexeme"] for g in grid])
    y = np.array([g["label"] for g in grid])

    rng = np.random.default_rng(seed)
    uniq = np.unique(lex)
    rng.shuffle(uniq)
    te_lex = set(uniq[:max(2, int(0.3 * len(uniq)))].tolist())
    te = np.array([l in te_lex for l in lex])
    tr = ~te

    with np.errstate(all="ignore"):
        Xv = D.encode([g["voc"] for g in grid], wv)
        Xc = D.encode([g["cons"] for g in grid], wv)
        Xvtr, Xvte = D._standardise(Xv[tr], Xv[te])
        Xctr, Xcte = D._standardise(Xc[tr], Xc[te])
        w, u = _axes(Xvtr, y[tr], Xvte)
        out = []

        # FD1 -- factor arrows
        px, py = Xvte @ w, Xvte @ u
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        yte = y[te]
        ax.scatter(px[yte == 0], py[yte == 0], s=9, alpha=.45, label="value 0")
        ax.scatter(px[yte == 1], py[yte == 1], s=9, alpha=.45, label="value 1")
        lt = lex[te]
        drawn = 0
        for L in np.unique(lt):
            i0 = np.where((lt == L) & (yte == 0))[0]
            i1 = np.where((lt == L) & (yte == 1))[0]
            if len(i0) and len(i1) and drawn < 60:
                ax.plot([px[i0[0]], px[i1[0]]], [py[i0[0]], py[i1[0]]],
                        "k-", lw=.3, alpha=.35)
                drawn += 1
        ax.set_xlabel(f"{factor} axis (value1 $-$ value0)")
        ax.set_ylabel(f"identity axis $\\perp$ {factor}")
        ax.set_title(f"{label}: {factor} offset")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        p = FIG_DIR / f"FD1_arrow_{encoder}_{factor}.pdf"
        fig.savefig(p); plt.close(fig); out.append(p)

        # FD2 -- factor-axis histograms, vocalised vs consonantal
        wc, _ = _axes(Xctr, y[tr], Xcte)
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
        for ax, X, ww, title in ((axes[0], Xvte, w, "vocalised"),
                                 (axes[1], Xcte, wc, "consonantal")):
            pr = X @ ww
            ax.hist(pr[yte == 0], 30, alpha=.6, label="value 0")
            ax.hist(pr[yte == 1], 30, alpha=.6, label="value 1")
            ax.set_title(f"{factor} -- {title}")
            ax.set_xlabel(f"{factor} axis")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        p = FIG_DIR / f"FD2_hist_{encoder}_{factor}.pdf"
        fig.savefig(p); plt.close(fig); out.append(p)

        # FD3 -- before vs after erasing the factor
        P = D.erase_binary(Xvtr, y[tr])
        Xe = Xvte @ P
        we, ue = _axes(Xvtr @ P, y[tr], Xe)  # axes in erased space (factor ~gone)
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharex=True, sharey=True)
        for ax, XX, ww, uu, title in (
                (axes[0], Xvte, w, u, "original"),
                (axes[1], Xe, we, ue, "factor erased")):
            qx, qy = XX @ ww, XX @ uu
            ax.scatter(qx[yte == 0], qy[yte == 0], s=8, alpha=.45, label="value 0")
            ax.scatter(qx[yte == 1], qy[yte == 1], s=8, alpha=.45, label="value 1")
            ax.set_title(title)
            ax.set_xlabel(f"{factor} axis")
        axes[0].set_ylabel("identity axis")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        p = FIG_DIR / f"FD3_erase_{encoder}_{factor}.pdf"
        fig.savefig(p); plt.close(fig); out.append(p)

    D._free(device)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="canine")
    ap.add_argument("--factor", default="number")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not _MPL:
        print("matplotlib required:\n"
              "    .venv/bin/python -m pip install -r "
              "disentangle/requirements-disentangle.txt", file=sys.stderr)
        return 2
    paths = make_figures(args.encoder, args.factor, args.seed)
    print("wrote:\n  " + "\n  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
