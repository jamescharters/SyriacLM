#!/usr/bin/env python3
"""The first neural Syriac vocaliser: pointing restoration as morphology (Twist 1).

Syriac is written as a *defective* abjad -- consonants are written, the vowels and
pointing are optional and usually omitted. In a root-and-pattern morphology a word
is ``root`` (consonants) + ``pattern`` (vocalisation), so **restoring the pointing
is recovering the pattern morpheme**. The published pipeline *discards* the
pointing; here we invert that and learn to *predict* it -- cheap, morphology-aligned
self-supervision, supervised by the SEDRA vocalised lexicon (``neural.sedra`` /
``neural.sedra_build``).

We frame it as **sequence labelling**, the canonical diacritisation setup: the
input is the consonant skeleton, and for each consonant slot the model predicts
the vowel/diacritic string that follows it (one of ~30 classes including "bare").
Every position is supervised (dense), so -- unlike a masked LM on tiny data -- the
model cannot collapse to the prior. A small bidirectional LSTM (the standard
diacritiser architecture) is enough.

Honest framing: neural diacritisation is established for Arabic/Hebrew; the novelty
here is Syriac-first and the *reframing* as morphological pretraining. The SEDRA
lexicon is New-Testament-scoped, so accuracy is reported on held-out SEDRA words
and the cross-register transfer to classical text is left as the open question it
is (no openly vocalised classical gold exists).

    .venv/bin/python -m neural.vocalizer --demo        # train + evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

from neural import sedra

PAD = 0


def load_pairs() -> list[tuple[str, list[str]]]:
    """Return (skeleton, pointing-labels) pairs from the SEDRA table."""
    src = sedra.find_sedra_source()
    if src is None:
        raise RuntimeError(
            "No SEDRA table found. Build it first:\n"
            "    git clone https://github.com/peshitta/sedrajs ~/.cache/sedrajs\n"
            "    .venv/bin/python -m neural.sedra_build --sedra-dir ~/.cache/sedrajs/sedra")
    pairs: list[tuple[str, list[str]]] = []
    for f in sedra.load_words(src):
        skel, pts = sedra.split_skeleton_pointing(f.vocalised)
        if skel:
            pairs.append((skel, [p if p is not None else "" for p in pts]))
    return pairs


def build_vocabs(pairs):
    chars = sorted({c for skel, _ in pairs for c in skel})
    labels = sorted({p for _, pts in pairs for p in pts})
    ctoi = {c: i + 1 for i, c in enumerate(chars)}     # 0 = PAD
    ltoi = {l: i for i, l in enumerate(labels)}        # labels dense from 0
    return ctoi, labels, ltoi


def split_pairs(pairs, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(len(pairs) * val_frac)
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


def majority_baseline(train, val, ltoi, labels):
    """Per-consonant most-frequent pointing -> position acc + word exact-match."""
    by_char: dict[str, Counter] = {}
    glob = Counter()
    for skel, pts in train:
        for c, p in zip(skel, pts):
            by_char.setdefault(c, Counter())[p] += 1
            glob[p] += 1
    default = glob.most_common(1)[0][0]
    pred_for = {c: cnt.most_common(1)[0][0] for c, cnt in by_char.items()}
    pos_ok = pos_tot = word_ok = 0
    for skel, pts in val:
        ok = True
        for c, p in zip(skel, pts):
            pred = pred_for.get(c, default)
            pos_tot += 1
            if pred == p:
                pos_ok += 1
            else:
                ok = False
        word_ok += int(ok)
    return pos_ok / max(pos_tot, 1), word_ok / max(len(val), 1)


if _TORCH:

    class BiLSTMVocalizer(nn.Module):
        def __init__(self, n_chars, n_labels, emb=64, hid=128, layers=2, dropout=0.2):
            super().__init__()
            self.embed = nn.Embedding(n_chars + 1, emb, padding_idx=PAD)
            self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout)
            self.head = nn.Linear(2 * hid, n_labels)

        def forward(self, x):
            h, _ = self.lstm(self.embed(x))
            return self.head(h)

    def _device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _encode(pairs, ctoi, ltoi, maxlen):
        X = np.zeros((len(pairs), maxlen), dtype=np.int64)
        Y = np.full((len(pairs), maxlen), -100, dtype=np.int64)
        for i, (skel, pts) in enumerate(pairs):
            for j, (c, p) in enumerate(zip(skel[:maxlen], pts[:maxlen])):
                X[i, j] = ctoi.get(c, 0)
                Y[i, j] = ltoi[p]
        return X, Y

    def train_vocalizer(*, seed=42, epochs=8, batch_size=128, lr=2e-3,
                        val_frac=0.1) -> dict:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        device = _device()

        pairs = load_pairs()
        ctoi, labels, ltoi = build_vocabs(pairs)
        train, val = split_pairs(pairs, val_frac, seed)
        maxlen = max(len(s) for s, _ in pairs)

        base_pos, base_word = majority_baseline(train, val, ltoi, labels)

        Xtr, Ytr = _encode(train, ctoi, ltoi, maxlen)
        Xva, Yva = _encode(val, ctoi, ltoi, maxlen)
        model = BiLSTMVocalizer(len(ctoi), len(labels)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(seed)

        t0 = time.time()
        model.train()
        steps_per = max(1, len(Xtr) // batch_size)
        for ep in range(1, epochs + 1):
            order = rng.permutation(len(Xtr))
            tot = 0.0
            for s in range(steps_per):
                b = order[s * batch_size:(s + 1) * batch_size]
                xb = torch.from_numpy(Xtr[b]).to(device)
                yb = torch.from_numpy(Ytr[b]).to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       yb.reshape(-1), ignore_index=-100)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            print(f"  epoch {ep}/{epochs}  loss {tot/steps_per:.3f}",
                  file=sys.stderr, flush=True)
        train_s = time.time() - t0

        pos_acc, word_acc = evaluate(model, Xva, Yva, val, ctoi, ltoi, labels, device)
        return {
            "params": n_params, "train_seconds": round(train_s, 1), "device": str(device),
            "train_words": len(train), "val_words": len(val),
            "n_labels": len(labels), "maxlen": maxlen,
            "pos_acc": pos_acc, "word_acc": word_acc,
            "baseline_pos_acc": base_pos, "baseline_word_acc": base_word,
            "model": model, "ctoi": ctoi, "ltoi": ltoi, "labels": labels,
            "maxlen_": maxlen, "device_": device,
        }

    @torch.no_grad()
    def evaluate(model, Xva, Yva, val, ctoi, ltoi, labels, device):
        model.eval()
        pos_ok = pos_tot = word_ok = 0
        for s in range(0, len(Xva), 256):
            xb = torch.from_numpy(Xva[s:s + 256]).to(device)
            pred = model(xb).argmax(-1).cpu().numpy()
            yb = Yva[s:s + 256]
            for r in range(yb.shape[0]):
                mask = yb[r] != -100
                p_ok = (pred[r][mask] == yb[r][mask])
                pos_ok += int(p_ok.sum()); pos_tot += int(mask.sum())
                word_ok += int(bool(p_ok.all()))
        model.train()
        return pos_ok / max(pos_tot, 1), word_ok / max(len(val), 1)

    @torch.no_grad()
    def vocalize_examples(res, k=6):
        """Show skeleton -> predicted vocalisation vs truth on held-out words."""
        model, ctoi, ltoi = res["model"], res["ctoi"], res["ltoi"]
        labels, maxlen, device = res["labels"], res["maxlen_"], res["device_"]
        pairs = load_pairs()
        _, val = split_pairs(pairs, 0.1, 42)
        rng = np.random.default_rng(1)
        out = []
        for i in rng.choice(len(val), size=k, replace=False):
            skel, pts = val[i]
            x = np.zeros((1, maxlen), dtype=np.int64)
            for j, c in enumerate(skel[:maxlen]):
                x[0, j] = ctoi.get(c, 0)
            pred = model(torch.from_numpy(x).to(device)).argmax(-1).cpu().numpy()[0]
            pred_voc = "".join(c + labels[pred[j]] for j, c in enumerate(skel))
            true_voc = "".join(c + (p or "") for c, p in zip(skel, pts))
            out.append((skel, pred_voc, true_voc))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="train + evaluate")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not _TORCH:
        print("torch is required for the vocaliser.", file=sys.stderr)
        return 2
    if not args.demo:
        ap.print_help()
        return 1

    try:
        res = train_vocalizer(seed=args.seed, epochs=args.epochs)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print("\n=== Syriac vocaliser (pointing restoration on held-out SEDRA words) ===")
    print(f"  params / device     : {res['params']:,} / {res['device']}")
    print(f"  train / val words    : {res['train_words']:,} / {res['val_words']:,}"
          f"  ({res['n_labels']} pointing classes)")
    print(f"  train time           : {res['train_seconds']}s")
    print(f"  per-position pointing accuracy : {res['pos_acc']:.3f}  "
          f"(majority baseline {res['baseline_pos_acc']:.3f})")
    print(f"  full-word exact match          : {res['word_acc']:.3f}  "
          f"(majority baseline {res['baseline_word_acc']:.3f})")
    print("\n  --- examples (CAL-ASCII: lowercase = restored vowels/diacritics) ---")
    for skel, pred, true in vocalize_examples(res):
        flag = "OK" if pred == true else "x"
        print(f"   [{flag}] {skel:<12} -> {pred:<18} (truth {true})")
    print("\nNote: SEDRA is New-Testament-scoped; these are held-out NT-vocabulary")
    print("words. Cross-register transfer to classical text is the open question.")
    print("Cite SEDRA (Kiraz); see neural/docs/DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
