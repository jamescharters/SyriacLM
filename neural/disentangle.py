#!/usr/bin/env python3
"""Does a frozen byte encoder factorise templatic morphology? (MVP probe).

Root-and-pattern morphology gives a rare gift: every Syriac word is a
ground-truth product ``word = lexical-identity x morphosyntactic-pattern``, and
SEDRA labels both factors. This lets us ask a representation-geometry question
that is usually unanswerable for lack of clean factor labels:

    In a *frozen* encoder that never trained on Syriac (off-the-shelf CANINE),
    are the two factors stored in linearly separable subspaces -- i.e. can we
    erase one with a linear projection while the other survives?

We build a **balanced grid** for each morphosyntactic factor P that varies within
a lexeme (number: sg/pl; gender: m/f; state: emphatic/absolute): every lexeme
attested with both values of P contributes exactly one form per value, so lexical
identity (factor R) and the factor P are statistically independent by
construction. We then encode each form with a frozen encoder and measure a 2x2
**cross-erasure selectivity matrix**:

                       factor acc.     lexeme NN
    original            > 0.5           high
    erase factor        ~ 0.5           (R should survive)
    erase lexeme        (P should survive)   ~ chance

A clean off-diagonal -- erasing P leaves R, erasing R leaves P -- is evidence the
encoder discovered the algebraic factorisation without supervision. The
**consonantal control** (same pipeline on the unvocalised skeleton) checks that
the vocalised result is not trivial input bookkeeping: a factor marked partly in
the vowels should be markedly less decodable when the vowels are removed.

We sweep this over factors {number, gender, state} and encoders {off-the-shelf
CANINE, Syriac-tuned CANINE (LoRA), Hebrew-transfer AlephBERT} to test whether the
factorisation generalises across morphosyntactic features and across models.

Erasure is iterative mean-difference (LDA-direction) nullspace projection in the
spirit of INLP (Ravfogel et al., 2020); LEACE (Belrose et al., 2023) is the
principled closed-form upgrade for a full study. Probes are plain logistic
regression. Everything is numpy + the frozen encoder, so there are no new
dependencies.

Data: the SEDRA IV API word records (``neural/sedra_cache/api/word``), which carry
structured ``number``/``gender``/``state`` plus the consonantal (``syriac``) and
vocalised (``western``) surface forms. SEDRA is licence-restricted (cite Kiraz);
no data is committed. See ``neural/docs/DATA.md``.

    .venv/bin/python -m neural.disentangle --demo
    .venv/bin/python -m neural.disentangle --demo --factors number,gender,state \
        --encoders canine,canine-syriac,hebrew
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import torch  # noqa: F401  (only needed via the encoder)
    from neural.canine_encoder import load_canine, CanineWordVectors, _device
    _HF = True
except Exception:  # pragma: no cover - heavy deps optional
    _HF = False

API_WORD_DIR = os.path.join(os.path.dirname(__file__), "sedra_cache", "api", "word")

# Morphosyntactic factors P that VARY WITHIN a lexeme, so a balanced grid can hold
# lexical identity R fixed while P flips -> R independent of P by construction.
# (value 0, value 1); a noun's POS does not vary within a lexeme, so it cannot be
# a factor here. Counts of qualifying lexemes in SEDRA IV: number 1512, gender 511,
# state 295.
FACTORS = {
    "number": ("number", ("singular", "plural")),
    "gender": ("gender", ("masculine", "feminine")),
    "state": ("state", ("emphatic", "absolute")),
}

# Syriac-tuned CANINE (LoRA continued-pretraining) checkpoint, if present.
SYRIAC_CANINE = os.path.expanduser(
    "~/.cache/syriac-neural/checkpoints/canine-lora")


# --------------------------------------------------------------------------- #
# Build the balanced lexeme x factor grid (factor R independent of factor P)
# --------------------------------------------------------------------------- #
def load_grid(factor: str = "number", word_dir: str = API_WORD_DIR,
              max_files: int | None = None, seed: int = 0) -> list[dict]:
    """Forms grouped so every kept lexeme has one form for each value of ``factor``.

    Each item: ``{lexeme, label(0/1), voc, cons}`` where ``voc`` is the vocalised
    (``western``) surface and ``cons`` the consonantal (``syriac``) skeleton. The
    within-lexeme balance makes lexical identity R independent of the factor P.
    """
    if factor not in FACTORS:
        raise ValueError(f"unknown factor {factor!r}; choose from {list(FACTORS)}")
    field, values = FACTORS[factor]
    if not os.path.isdir(word_dir):
        raise RuntimeError(
            f"No SEDRA API word records at {word_dir}. They are licence-restricted "
            "and not shipped; download them yourself (see neural/docs/DATA.md).")
    files = [f for f in os.listdir(word_dir)
             if f.endswith(".json") and not f.startswith("_")]
    files.sort()
    if max_files:
        files = files[:max_files]

    # lexeme -> factor value -> list of (voc, cons)
    by_lex: dict[int, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: {values[0]: [], values[1]: []})
    for fn in files:
        try:
            rec = json.load(open(os.path.join(word_dir, fn)))[0]
        except Exception:
            continue
        v = rec.get(field)
        if v not in values:
            continue
        lex = (rec.get("lexeme") or {}).get("id")
        voc = (rec.get("western") or "").strip()
        cons = (rec.get("syriac") or "").strip()
        if lex is None or not voc or not cons:
            continue
        by_lex[lex][v].append((voc, cons))

    rng = np.random.default_rng(seed)
    grid: list[dict] = []
    for lex, cells in by_lex.items():
        if not cells[values[0]] or not cells[values[1]]:
            continue
        a = cells[values[0]][int(rng.integers(len(cells[values[0]])))]
        b = cells[values[1]][int(rng.integers(len(cells[values[1]])))]
        grid.append({"lexeme": lex, "label": 0, "voc": a[0], "cons": a[1]})
        grid.append({"lexeme": lex, "label": 1, "voc": b[0], "cons": b[1]})
    return grid


# --------------------------------------------------------------------------- #
# Encoding + numpy probes / erasure
# --------------------------------------------------------------------------- #
def encode(forms: list[str], wv) -> np.ndarray:
    wv.precompute(forms)
    return np.stack([wv[f] for f in forms]).astype(np.float64)


def _standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(0)
    sd = train.std(0) + 1e-8
    return tuple((m - mu) / sd for m in (train, *others))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logreg(X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.5,
            l2: float = 1e-3) -> np.ndarray:
    """Binary logistic regression (numpy GD); returns weight vector incl. bias."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(steps):
        p = _sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / n + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return w


def factor_probe(Xtr, ytr, Xte, yte) -> float:
    """Binary linear-probe accuracy for a morphosyntactic factor."""
    w = _logreg(Xtr, ytr)
    p = _sigmoid(np.hstack([Xte, np.ones((len(Xte), 1))]) @ w)
    return float(((p > 0.5).astype(int) == yte).mean())


def lexeme_nn(emb: np.ndarray, lex: np.ndarray) -> float:
    """Self-excluded nearest-neighbour retrieval: share lexeme with NN? (cosine)."""
    n = len(emb)
    if n < 2:
        return float("nan")
    z = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    assert np.isfinite(z).all()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = z @ z.T                    # spurious FP flags on Apple Silicon
    np.fill_diagonal(sims, -1e9)
    nn = sims.argmax(1)
    return float(np.mean(lex == lex[nn]))


def erase_binary(Xfit: np.ndarray, y: np.ndarray, max_iter: int = 16,
                 tol: float = 0.55) -> np.ndarray:
    """Iterative mean-difference nullspace projection erasing a binary factor.

    Returns a (d x d) projection ``P`` such that ``X @ P`` is linearly
    non-predictive of ``y`` (INLP-style; LDA direction per iteration).
    """
    d = Xfit.shape[1]
    P = np.eye(d)
    Xr = Xfit.copy()
    for _ in range(max_iter):
        w = Xr[y == 1].mean(0) - Xr[y == 0].mean(0)
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            break
        w = w / nrm
        proj = Xr @ w
        thr = 0.5 * (proj[y == 1].mean() + proj[y == 0].mean())
        acc = max(((proj > thr).astype(int) == y).mean(),
                  ((proj <= thr).astype(int) == y).mean())
        Pi = np.eye(d) - np.outer(w, w)
        P = P @ Pi
        Xr = Xr @ Pi
        if acc <= tol:
            break
    return P


def erase_classmeans(Xfit: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Project out the top-k principal directions of the class-mean subspace.

    Removes linear information that separates the classes (here, lexemes).
    """
    classes = np.unique(labels)
    M = np.stack([Xfit[labels == c].mean(0) for c in classes])
    M = M - M.mean(0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, Vt.shape[0])
    B = Vt[:k]
    return np.eye(Xfit.shape[1]) - B.T @ B


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(grid: list[dict], wv, *, seed: int = 42,
        lex_k_grid: tuple[int, ...] = (16, 32, 64, 128, 256, 512)) -> dict:
    lex = np.array([g["lexeme"] for g in grid])
    y = np.array([g["label"] for g in grid])

    # split by lexeme so the factor-probe must learn the factor, not lexeme identity
    rng = np.random.default_rng(seed)
    uniq = np.unique(lex)
    rng.shuffle(uniq)
    n_te = max(2, int(0.3 * len(uniq)))
    te_lex = set(uniq[:n_te].tolist())
    te = np.array([l in te_lex for l in lex])
    tr = ~te
    chance_r = 1.0 / max(len(te_lex), 1)

    out = {"n_pairs": len(grid) // 2, "n_lexemes": len(uniq),
           "n_test_lexemes": len(te_lex), "chance_lexeme_nn": round(chance_r, 4),
           "results": {}}

    # encoding hits torch; the probes/erasure below raise only the spurious
    # Apple-Silicon Accelerate BLAS FP flags, which we silence.
    with np.errstate(all="ignore"):
        for cond in ("voc", "cons"):
            X = encode([g[cond] for g in grid], wv)
            Xtr, Xte = _standardise(X[tr], X[te])

            # original
            p_orig = factor_probe(Xtr, y[tr], Xte, y[te])
            r_orig = lexeme_nn(Xte, lex[te])

            # erase the FACTOR (low-rank binary): fit on train, re-measure on test
            Pf = erase_binary(Xtr, y[tr])
            p_after_f = factor_probe(Xtr @ Pf, y[tr], Xte @ Pf, y[te])
            r_keep = lexeme_nn(Xte @ Pf, lex[te])

            # erase LEXEME identity transductively: fit the erased subspace on the
            # very forms whose retrieval we measure (identity is per-form, not a
            # generalisable concept), and sweep its rank k. The factor reader is
            # trained on clean train forms, so a surviving factor-acc means the
            # factor does not live in the removed lexical subspace.
            sweep = []
            for k in lex_k_grid:
                Pl = erase_classmeans(Xte, lex[te], k)
                sweep.append({
                    "k": k,
                    "lexeme_nn": round(lexeme_nn(Xte @ Pl, lex[te]), 3),
                    "factor_acc": round(
                        factor_probe(Xtr, y[tr], Xte @ Pl, y[te]), 3),
                })
            # report the 2x2 at the smallest k that pushes lexeme-NN <= 2x chance
            picked = next((s for s in sweep if s["lexeme_nn"] <= 2 * chance_r),
                          sweep[-1])

            out["results"][cond] = {
                "factor_acc": round(p_orig, 3),
                "lexeme_nn": round(r_orig, 3),
                "erase_factor__factor_acc": round(p_after_f, 3),
                "erase_factor__lexeme_nn": round(r_keep, 3),
                "erase_lexeme_k": picked["k"],
                "erase_lexeme__lexeme_nn": picked["lexeme_nn"],
                "erase_lexeme__factor_acc": picked["factor_acc"],
                "lexeme_erase_sweep": sweep,
            }
    return out


def _print(out: dict, factor: str, encoder_label: str) -> None:
    print(f"\n######## {encoder_label}  |  factor = {factor}  ########")
    print(f"balanced grid: {out['n_pairs']:,} lexeme pairs, {out['n_lexemes']:,} "
          f"lexemes ({out['n_test_lexemes']:,} held out for test)")
    for cond, label in (("voc", "VOCALISED (main)"),
                        ("cons", "CONSONANTAL (control)")):
        r = out["results"][cond]
        print(f"\n=== {label} ===")
        print(f"  chance: {factor} 0.500, lexeme-NN {out['chance_lexeme_nn']:.4f}")
        print(f"  {'condition':<22}{factor + ' acc':>14}{'lexeme NN':>12}")
        print(f"  {'original':<22}{r['factor_acc']:>14.3f}{r['lexeme_nn']:>12.3f}")
        print(f"  {'erase ' + factor:<22}{r['erase_factor__factor_acc']:>14.3f}"
              f"{r['erase_factor__lexeme_nn']:>12.3f}   <- R should survive")
        klabel = f"erase lexeme (k={r['erase_lexeme_k']})"
        print(f"  {klabel:<22}{r['erase_lexeme__factor_acc']:>14.3f}"
              f"{r['erase_lexeme__lexeme_nn']:>12.3f}   <- P should survive")
        print("  lexeme-erase sweep (k: lexeme-NN / factor-acc): " + "  ".join(
            f"{s['k']}:{s['lexeme_nn']:.3f}/{s['factor_acc']:.3f}"
            for s in r["lexeme_erase_sweep"]))


def _print_summary(rows: list[tuple]) -> None:
    if not rows:
        return
    print("\n" + "=" * 96)
    print("SUMMARY  (vocalised; P = factor acc [chance .5], R = lexeme-NN; "
          "P(cons) = factor acc without vowels)")
    print("=" * 96)
    head = ("encoder", "factor", "P", "P|delP", "P|delR",
            "R", "R|delP", "R|delR", "P(cons)")
    print(f"{head[0]:<24}{head[1]:<8}" + "".join(f"{c:>9}" for c in head[2:]))
    for (label, f, p, pep, per, r, rep, rer, pc) in rows:
        short = (label.replace("off-the-shelf ", "os-")
                 .replace("Syriac-tuned ", "syr-").replace(" (LoRA)", "")
                 .replace("Hebrew-transfer ", "heb-").replace(" (AlephBERT)", ""))
        print(f"{short:<24}{f:<8}{p:>9.3f}{pep:>9.3f}{per:>9.3f}"
              f"{r:>9.3f}{rep:>9.3f}{rer:>9.3f}{pc:>9.3f}")
    print("\nDisentangled cell: erase-factor sends P->~.5 while R survives (R|delP ~ R),")
    print("AND erase-lexeme sends R->~0 while the factor survives (P|delR still > .5).")
    print("P(cons) < P means the factor signal is vowel-dependent. Cite SEDRA (Kiraz).")


# --------------------------------------------------------------------------- #
# Encoders (all expose the precompute / wv[token] interface)
# --------------------------------------------------------------------------- #
ENCODERS = ("canine", "canine-syriac", "hebrew")


def build_encoder(name: str, device):
    """Return ``(wv, label)`` for an encoder name; raises if unavailable."""
    if name == "canine":
        return CanineWordVectors(load_canine(None).to(device), device), \
            "off-the-shelf CANINE"
    if name == "canine-syriac":
        if not os.path.isdir(SYRIAC_CANINE):
            raise RuntimeError(
                f"no Syriac-tuned checkpoint at {SYRIAC_CANINE} "
                "(run neural.canine_pretrain first)")
        return CanineWordVectors(load_canine(SYRIAC_CANINE).to(device), device), \
            "Syriac-tuned CANINE (LoRA)"
    if name == "hebrew":
        from transformers import AutoModel, AutoTokenizer
        from neural.hf_encoder import HFWordVectors
        from neural.transliterate import syriac_to_hebrew
        mid = "onlplab/alephbert-base"
        model = AutoModel.from_pretrained(mid).to(device)
        tok = AutoTokenizer.from_pretrained(mid)
        return HFWordVectors(model, tok, device, preprocess=syriac_to_hebrew), \
            "Hebrew-transfer (AlephBERT)"
    raise ValueError(f"unknown encoder {name!r}; choose from {list(ENCODERS)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="build grids, encode, and print the selectivity matrices")
    ap.add_argument("--factors", default="number,gender,state",
                    help=f"comma list from {list(FACTORS)}")
    ap.add_argument("--encoders", default="canine,canine-syriac,hebrew",
                    help=f"comma list from {list(ENCODERS)}")
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap API word files scanned (speed; default all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not _HF:
        print("transformers + torch required. Install:\n"
              "    .venv/bin/python -m pip install -r neural/requirements-neural.txt",
              file=sys.stderr)
        return 2
    if not args.demo:
        ap.print_help()
        return 1

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    # grids are encoder-independent -> build each once
    grids: dict[str, list[dict]] = {}
    for f in factors:
        try:
            g = load_grid(f, max_files=args.max_files)
        except (RuntimeError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        if len(g) < 40:
            print(f"grid for {f!r} too small ({len(g)} forms); skipping.",
                  file=sys.stderr)
            continue
        grids[f] = g
    if not grids:
        print("no usable factor grids.", file=sys.stderr)
        return 2

    device = _device()
    summary: list[tuple] = []
    for enc in encoders:
        print(f"\nloading encoder {enc!r} on {device} ...", file=sys.stderr)
        try:
            wv, label = build_encoder(enc, device)
        except (RuntimeError, ValueError) as exc:
            print(f"skip encoder {enc!r}: {exc}", file=sys.stderr)
            continue
        for f, g in grids.items():
            out = run(g, wv, seed=args.seed)
            _print(out, f, label)
            rv, rc = out["results"]["voc"], out["results"]["cons"]
            summary.append((
                label, f, rv["factor_acc"], rv["erase_factor__factor_acc"],
                rv["erase_lexeme__factor_acc"], rv["lexeme_nn"],
                rv["erase_factor__lexeme_nn"], rv["erase_lexeme__lexeme_nn"],
                rc["factor_acc"]))
        del wv
        try:
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
