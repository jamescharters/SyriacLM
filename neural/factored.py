#!/usr/bin/env python3
"""Twist 2: a factored root/pattern encoder, ablated against a flat one.

Semitic morphology is root-and-pattern, so a representation that *factors* a word
into its consonantal **root** tier and its vocalic **pattern** tier should be
morphologically better-behaved than a flat character model. SEDRA's vocalised
forms hand us the split for free (consonants vs. vowels/diacritics), so we can
test the inductive bias directly and honestly.

Setup (fair: both models see the same vocalised word):
  * **flat** -- a BiLSTM over the raw vocalised character sequence;
  * **factored** -- two parallel BiLSTMs over *aligned* streams of equal length
    (one consonant per slot, one pointing label per slot), fused position-wise.
Both are trained with the same supervised-contrastive objective on the SEDRA
root, and both produce a unit document vector for retrieval.

Evaluation -- **root coherence by nearest-neighbour retrieval**:
  * *seen roots*  -- held-out words whose root also has training words;
  * *unseen roots* -- words of roots entirely absent from training (the decisive
    generalisation test of whether the factorisation transfers).
We report the fraction of queries whose nearest neighbour shares the root, for
flat vs. factored. This is the neural analogue of the paper's morphology probe
(T3a) and OOV root-NN (T3b).

Honest scope: the factored model is *given* the consonant/vowel split (trivial
from the script -- vowels are combining marks). The question is whether making
that Semitic structure explicit helps a model learn from limited data, not
whether the split itself is hard. SEDRA is New-Testament-scoped (see DATA.md).

    .venv/bin/python -m neural.factored --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

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


def load_root_words():
    """Return list of (consonant_ids-as-chars, pointing-labels, root)."""
    src = sedra.find_sedra_source()
    if src is None:
        raise RuntimeError(
            "No SEDRA table found. Build it first:\n"
            "    git clone https://github.com/peshitta/sedrajs ~/.cache/sedrajs\n"
            "    .venv/bin/python -m neural.sedra_build --sedra-dir ~/.cache/sedrajs/sedra")
    out = []
    for f in sedra.load_words(src):
        if not f.root:
            continue
        skel, pts = sedra.split_skeleton_pointing(f.vocalised)
        if skel:
            out.append((skel, [p if p is not None else "" for p in pts], f.root))
    return out


def split_by_root(words, unseen_root_frac=0.1, heldout_word_frac=0.1, seed=42):
    """Partition into train / seen-query / unseen-query.

    * Roots are split into train-roots and unseen-roots.
    * Among train-roots, a fraction of words become seen-queries (their roots keep
      other words in the gallery, so retrieval is possible).
    """
    rng = np.random.default_rng(seed)
    by_root = defaultdict(list)
    for w in words:
        by_root[w[2]].append(w)
    roots = [r for r, ws in by_root.items() if len(ws) >= 2]  # need a peer
    rng.shuffle(roots)
    n_unseen = int(len(roots) * unseen_root_frac)
    unseen_roots = set(roots[:n_unseen])
    train_roots = roots[n_unseen:]

    train, seen_q = [], []
    for r in train_roots:
        ws = by_root[r][:]
        rng.shuffle(ws)
        n_hold = max(1, int(len(ws) * heldout_word_frac)) if len(ws) >= 3 else 0
        seen_q.extend(ws[:n_hold])
        train.extend(ws[n_hold:])
    unseen_q = [w for r in unseen_roots for w in by_root[r]]
    return train, seen_q, unseen_q


def build_vocabs(words):
    cons = sorted({c for skel, _, _ in words for c in skel})
    labels = sorted({p for _, pts, _ in words for p in pts})
    allchars = sorted({c for skel, pts, _ in words
                       for c in (skel + "".join(pts))})
    ctoi = {c: i + 1 for i, c in enumerate(cons)}
    ltoi = {l: i + 1 for i, l in enumerate(labels)}
    atoi = {c: i + 1 for i, c in enumerate(allchars)}
    return ctoi, ltoi, atoi


def _flat_seq(skel, pts):
    """Interleave consonant + its pointing back into the raw vocalised string."""
    s = []
    for c, p in zip(skel, pts):
        s.append(c)
        s.extend(p)
    return s


if _TORCH:

    def supcon(z, labels, temp=0.1):
        """Supervised-contrastive loss (Khosla 2020), in-batch positives."""
        n = z.size(0)
        sim = (z @ z.T) / temp
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim).masked_fill(eye, 0.0)
        logden = torch.log(exp.sum(1) + 1e-9)
        logp = sim - logden[:, None]
        pos = (labels[:, None] == labels[None, :]) & ~eye
        cnt = pos.sum(1)
        has = cnt > 0
        if not bool(has.any()):
            return z.sum() * 0.0
        per = (logp * pos).sum(1)[has] / cnt[has]
        return -per.mean()

    class FlatEncoder(nn.Module):
        def __init__(self, n_all, emb=64, hid=128, out=64):
            super().__init__()
            self.embed = nn.Embedding(n_all + 1, emb, padding_idx=PAD)
            self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(2 * hid, out)

        def forward(self, flat_ids, flat_mask, **_):
            h, _ = self.lstm(self.embed(flat_ids))
            m = flat_mask.unsqueeze(-1).type_as(h)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            z = self.proj(pooled)
            return z / (z.norm(dim=1, keepdim=True) + 1e-9)

    class FactoredEncoder(nn.Module):
        """Parallel consonant (root) and pointing (pattern) streams, aligned."""

        def __init__(self, n_cons, n_lab, emb=48, hid=96, out=64):
            super().__init__()
            self.cemb = nn.Embedding(n_cons + 1, emb, padding_idx=PAD)
            self.pemb = nn.Embedding(n_lab + 1, emb, padding_idx=PAD)
            self.clstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.plstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(4 * hid, out)

        def forward(self, cons_ids, pat_ids, slot_mask, **_):
            hc, _ = self.clstm(self.cemb(cons_ids))
            hp, _ = self.plstm(self.pemb(pat_ids))
            m = slot_mask.unsqueeze(-1).type_as(hc)
            pc = (hc * m).sum(1) / m.sum(1).clamp(min=1e-9)
            pp = (hp * m).sum(1) / m.sum(1).clamp(min=1e-9)
            z = self.proj(torch.cat([pc, pp], dim=1))
            return z / (z.norm(dim=1, keepdim=True) + 1e-9)

    def _device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _encode_slots(words, ctoi, ltoi, maxc):
        cons = np.zeros((len(words), maxc), dtype=np.int64)
        pat = np.zeros((len(words), maxc), dtype=np.int64)
        smask = np.zeros((len(words), maxc), dtype=np.float32)
        for i, (skel, pts, _) in enumerate(words):
            for j in range(min(len(skel), maxc)):
                cons[i, j] = ctoi.get(skel[j], 0)
                pat[i, j] = ltoi.get(pts[j], 0)
                smask[i, j] = 1.0
        return cons, pat, smask

    def _encode_flat(words, atoi, maxf):
        flat = np.zeros((len(words), maxf), dtype=np.int64)
        fmask = np.zeros((len(words), maxf), dtype=np.float32)
        for i, (skel, pts, _) in enumerate(words):
            seq = _flat_seq(skel, pts)[:maxf]
            for j, ch in enumerate(seq):
                flat[i, j] = atoi.get(ch, 0)
                fmask[i, j] = 1.0
        return flat, fmask

    def _embed_all(model, kind, words, packs, device, bs=512):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(words), bs):
                sl = slice(s, s + bs)
                if kind == "flat":
                    z = model(flat_ids=torch.from_numpy(packs["flat"][sl]).to(device),
                              flat_mask=torch.from_numpy(packs["fmask"][sl]).to(device))
                else:
                    z = model(cons_ids=torch.from_numpy(packs["cons"][sl]).to(device),
                              pat_ids=torch.from_numpy(packs["pat"][sl]).to(device),
                              slot_mask=torch.from_numpy(packs["smask"][sl]).to(device))
                outs.append(z.cpu().numpy())
        model.train()
        return np.vstack(outs)

    def _root_nn(query_emb, query_roots, gallery_emb, gallery_roots):
        """Fraction of queries whose nearest gallery neighbour shares the root."""
        if len(query_emb) == 0 or len(gallery_emb) == 0:
            return float("nan")
        assert np.isfinite(query_emb).all() and np.isfinite(gallery_emb).all()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = query_emb @ gallery_emb.T   # spurious FP flags on Apple Silicon
        nn = sims.argmax(1)
        return float(np.mean([query_roots[i] == gallery_roots[nn[i]]
                              for i in range(len(query_roots))]))

    def train_one(kind, train, seen_q, unseen_q, vocabs, *, epochs, bs, lr, seed):
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        device = _device()
        ctoi, ltoi, atoi = vocabs
        maxc = max(len(s) for s, _, _ in train + seen_q + unseen_q)
        maxf = max(len(_flat_seq(s, p)) for s, p, _ in train + seen_q + unseen_q)

        if kind == "flat":
            model = FlatEncoder(len(atoi)).to(device)
        else:
            model = FactoredEncoder(len(ctoi), len(ltoi)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # pack all splits once
        def pack(ws):
            cons, pat, smask = _encode_slots(ws, ctoi, ltoi, maxc)
            flat, fmask = _encode_flat(ws, atoi, maxf)
            return {"cons": cons, "pat": pat, "smask": smask, "flat": flat, "fmask": fmask}
        ptr, pseen, puns = pack(train), pack(seen_q), pack(unseen_q)
        tr_roots = np.array([w[2] for w in train], dtype=object)

        # root -> indices for positive sampling
        by_root = defaultdict(list)
        for i, w in enumerate(train):
            by_root[w[2]].append(i)
        roots_multi = [r for r, ix in by_root.items() if len(ix) >= 2]
        rng = np.random.default_rng(seed)

        t0 = time.time()
        model.train()
        steps = max(1, len(train) // bs)
        for ep in range(1, epochs + 1):
            tot = 0.0
            for _ in range(steps):
                # build a batch with same-root positives: pick roots, 2 words each
                chosen = rng.choice(len(roots_multi), size=bs // 2, replace=True)
                idx = []
                for ci in chosen:
                    pool = by_root[roots_multi[ci]]
                    idx.extend(rng.choice(pool, size=2, replace=len(pool) < 2))
                idx = np.array(idx)
                lab = torch.tensor([hash(train[i][2]) % (10 ** 8) for i in idx],
                                   device=device)
                if kind == "flat":
                    z = model(flat_ids=torch.from_numpy(ptr["flat"][idx]).to(device),
                              flat_mask=torch.from_numpy(ptr["fmask"][idx]).to(device))
                else:
                    z = model(cons_ids=torch.from_numpy(ptr["cons"][idx]).to(device),
                              pat_ids=torch.from_numpy(ptr["pat"][idx]).to(device),
                              slot_mask=torch.from_numpy(ptr["smask"][idx]).to(device))
                loss = supcon(z, lab)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item())
            print(f"  [{kind}] epoch {ep}/{epochs}  loss {tot/steps:.3f}",
                  file=sys.stderr, flush=True)
        train_s = time.time() - t0

        gal = _embed_all(model, kind, train, ptr, device)
        gal_roots = [w[2] for w in train]
        seen_emb = _embed_all(model, kind, seen_q, pseen, device)
        uns_emb = _embed_all(model, kind, unseen_q, puns, device)
        seen_acc = _root_nn(seen_emb, [w[2] for w in seen_q], gal, gal_roots)
        # unseen roots: retrieve among the unseen set itself (peers), exclude self
        uns_acc = _self_excluded_root_nn(uns_emb, [w[2] for w in unseen_q])
        return {"params": n_params, "train_seconds": round(train_s, 1),
                "seen_root_nn": seen_acc, "unseen_root_nn": uns_acc}

    def _self_excluded_root_nn(emb, roots):
        if len(emb) < 2:
            return float("nan")
        assert np.isfinite(emb).all()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = emb @ emb.T                 # spurious FP flags on Apple Silicon
        np.fill_diagonal(sims, -1e9)
        nn = sims.argmax(1)
        return float(np.mean([roots[i] == roots[nn[i]] for i in range(len(roots))]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="train flat + factored and compare")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not _TORCH:
        print("torch is required for the factored encoder.", file=sys.stderr)
        return 2
    if not args.demo:
        ap.print_help()
        return 1

    try:
        words = load_root_words()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    train, seen_q, unseen_q = split_by_root(words, seed=args.seed)
    vocabs = build_vocabs(words)
    print(f"SEDRA words with roots: {len(words):,}  "
          f"(train {len(train):,}, seen-query {len(seen_q):,}, "
          f"unseen-root query {len(unseen_q):,})", file=sys.stderr)

    results = {}
    for kind in ("flat", "factored"):
        results[kind] = train_one(kind, train, seen_q, unseen_q, vocabs,
                                  epochs=args.epochs, bs=args.batch_size,
                                  lr=args.lr, seed=args.seed)

    print("\n=== Twist 2: factored root/pattern vs flat (root-NN retrieval) ===")
    print(f"  {'model':<10}{'params':>10}{'seen-root NN':>15}{'unseen-root NN':>16}")
    for kind in ("flat", "factored"):
        r = results[kind]
        print(f"  {kind:<10}{r['params']:>10,}{r['seen_root_nn']:>15.3f}"
              f"{r['unseen_root_nn']:>16.3f}")
    df_seen = results["factored"]["seen_root_nn"] - results["flat"]["seen_root_nn"]
    df_uns = results["factored"]["unseen_root_nn"] - results["flat"]["unseen_root_nn"]
    print(f"\n  factored - flat:  seen {df_seen:+.3f}   unseen {df_uns:+.3f}")
    print("  (unseen-root NN is the decisive generalisation test.)")
    print("\nSEDRA is NT-scoped; cite Kiraz. See neural/docs/DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
