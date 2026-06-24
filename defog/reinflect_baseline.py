"""Character-level seq2seq baseline for Syriac morphological reinflection.

The decisive task (see ``lookup_baseline`` for why template prediction is too
easy): given a root's consonants and a morphosyntactic feature spec, generate the
inflected **consonantal surface string** -- the radicals bound into the pattern
*plus* the feature-determined affixes (m-, n-, t- ..., which 84% of forms carry).
A ``(features, length) -> surface`` lookup scores 0.000 exact on held-out roots,
so this task is genuinely root-dependent and is the standard morphological
(re)inflection setup (SIGMORPHON): lemma/root + tags -> form.

This module is the *unstructured* reference point: a vanilla GRU encoder-decoder
with attention that treats the problem as conditioned string transduction, with
**no** bipartite/root-template structure. It is evaluated on the same held-out
root split the structured model uses (seed 42), by whole-string exact match and
per-character accuracy. If a structure-aware model cannot beat this, the
structural prior is not earning its keep.

    .venv/bin/python -m defog.reinflect_baseline --epochs 40
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data import build_synthetic_dataset, build_dataset
from .graph import MORPH_FIELDS, MORPH_SCHEMA
from .train import split_by_root

PAD, BOS, EOS = 0, 1, 2
_SPECIAL = ["<pad>", "<bos>", "<eos>"]


# ── Vocabulary ──────────────────────────────────────────────────────────────

class Vocab:
    """Shared symbol table: special tokens, Syriac characters seen in surfaces and
    roots, and one token per (field=value) morphological feature."""

    def __init__(self, items: list[dict]):
        chars = set()
        for it in items:
            chars.update(it["surface"])
            chars.update(it["root_consonants"])
        self.chars = sorted(chars)
        # Output (decoder) vocabulary: specials + surface characters.
        self.itos = list(_SPECIAL) + self.chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        # Input (encoder) vocabulary: a contiguous block after the char tokens for
        # root characters (reuse stoi) and feature tokens.
        self.feat_tokens = [f"{f}={v}" for f in MORPH_FIELDS for v in MORPH_SCHEMA[f]]
        base = len(self.itos)
        self.feat_stoi = {t: base + i for i, t in enumerate(self.feat_tokens)}
        self.n_in = base + len(self.feat_tokens)   # encoder embedding size
        self.n_out = len(self.itos)                # decoder vocabulary size

    def encode_input(self, item: dict) -> list[int]:
        toks = [self.stoi[c] for c in item["root_consonants"] if c in self.stoi]
        for f in MORPH_FIELDS:
            v = (item.get("morphology", {}) or {}).get(f)
            if v is not None:
                tok = f"{f}={v}"
                if tok in self.feat_stoi:
                    toks.append(self.feat_stoi[tok])
        return toks or [self.stoi.get(self.chars[0], PAD)]

    def encode_output(self, item: dict) -> list[int]:
        return [BOS] + [self.stoi[c] for c in item["surface"] if c in self.stoi] + [EOS]


class ReinflectDataset(Dataset):
    def __init__(self, items: list[dict], vocab: Vocab):
        self.rows = [(vocab.encode_input(it), vocab.encode_output(it), it["surface"])
                     for it in items]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    src, tgt, surf = zip(*batch)
    sl = max(len(s) for s in src)
    tl = max(len(t) for t in tgt)
    src_p = torch.full((len(batch), sl), PAD, dtype=torch.long)
    tgt_p = torch.full((len(batch), tl), PAD, dtype=torch.long)
    src_len = torch.tensor([len(s) for s in src], dtype=torch.long)
    for i, (s, t) in enumerate(zip(src, tgt)):
        src_p[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_p[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return src_p, src_len, tgt_p, list(surf)


# ── Model: GRU encoder-decoder with Bahdanau attention ─────────────────────

class Encoder(nn.Module):
    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.embed = nn.Embedding(n_in, d, padding_idx=PAD)
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * d, d)

    def forward(self, src, src_len):
        x = self.embed(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # [B,S,2d]
        h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], dim=-1)))      # [B,d]
        return out, h


class Attention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(3 * d, d)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        S = enc_out.size(1)
        e = self.v(torch.tanh(self.w(
            torch.cat([dec_h.unsqueeze(1).expand(-1, S, -1), enc_out], dim=-1)))).squeeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        a = F.softmax(e, dim=-1)
        ctx = (a.unsqueeze(-1) * enc_out).sum(1)   # [B,2d]
        return ctx, a


class Decoder(nn.Module):
    def __init__(self, n_out: int, d: int):
        super().__init__()
        self.embed = nn.Embedding(n_out, d, padding_idx=PAD)
        self.attn = Attention(d)
        self.gru = nn.GRUCell(d + 2 * d, d)
        self.out = nn.Linear(d + 2 * d, n_out)
        self.p_gen = nn.Linear(d + 2 * d + d, 1)   # [h ; ctx ; emb] -> copy gate

    def step(self, tok, h, enc_out, mask):
        emb = self.embed(tok)                       # [B,d]
        ctx, a = self.attn(h, enc_out, mask)        # [B,2d], [B,S]
        h = self.gru(torch.cat([emb, ctx], dim=-1), h)
        feats = torch.cat([h, ctx], dim=-1)
        logits = self.out(feats)
        p_gen = torch.sigmoid(self.p_gen(torch.cat([feats, emb], dim=-1)))  # [B,1]
        return logits, a, p_gen, h


class Seq2Seq(nn.Module):
    def __init__(self, vocab: Vocab, d: int = 128, copy: bool = False):
        super().__init__()
        self.enc = Encoder(vocab.n_in, d)
        self.dec = Decoder(vocab.n_out, d)
        self.d = d
        self.copy = copy
        self.n_out = vocab.n_out

    def _logp(self, gen_logits, a, p_gen, src):
        """Final per-step log-probabilities. Plain softmax, or a pointer-generator
        mixture that can COPY a source character (token ids in [n_special, n_out))."""
        if not self.copy:
            return F.log_softmax(gen_logits, dim=-1)
        B, V = gen_logits.shape
        gen = F.softmax(gen_logits, dim=-1)
        copyable = (src >= len(_SPECIAL)) & (src < self.n_out)   # only chars
        a_copy = a * copyable.float()
        idx = src.clamp(min=0, max=V - 1)
        copy_dist = torch.zeros(B, V, device=gen.device).scatter_add(1, idx, a_copy)
        final = p_gen * gen + (1.0 - p_gen) * copy_dist
        return (final + 1e-9).log()

    def forward(self, src, src_len, tgt, teacher_forcing: float = 1.0):
        enc_out, h = self.enc(src, src_len)
        mask = (src != PAD)
        B, T = tgt.shape
        logps = []
        tok = tgt[:, 0]
        for t in range(1, T):
            lg, a, pg, h = self.dec.step(tok, h, enc_out, mask)
            logp = self._logp(lg, a, pg, src)
            logps.append(logp)
            tok = tgt[:, t] if random.random() < teacher_forcing else logp.argmax(-1)
        return torch.stack(logps, dim=1)            # [B,T-1,V] log-probs

    @torch.no_grad()
    def greedy(self, src, src_len, max_len: int = 24):
        enc_out, h = self.enc(src, src_len)
        mask = (src != PAD)
        B = src.size(0)
        tok = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        seqs = [[] for _ in range(B)]
        for _ in range(max_len):
            lg, a, pg, h = self.dec.step(tok, h, enc_out, mask)
            tok = self._logp(lg, a, pg, src).argmax(-1)
            for i in range(B):
                if not done[i]:
                    if tok[i].item() == EOS:
                        done[i] = True
                    else:
                        seqs[i].append(tok[i].item())
            if done.all():
                break
        return seqs


# ── Train / eval ────────────────────────────────────────────────────────────

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: Seq2Seq, loader: DataLoader, vocab: Vocab, device) -> dict:
    model.eval()
    exact = 0
    char_num = 0.0
    char_den = 0
    n = 0
    for src, src_len, tgt, surf in loader:
        src = src.to(device)
        seqs = model.greedy(src, src_len)
        for pred_ids, gold in zip(seqs, surf):
            pred = "".join(vocab.itos[i] for i in pred_ids)
            exact += int(pred == gold)
            # per-character accuracy over the gold length (alignment by position)
            L = len(gold)
            char_num += sum(1 for k in range(L) if k < len(pred) and pred[k] == gold[k])
            char_den += L
            n += 1
    return {"exact": exact / n, "char_acc": char_num / max(char_den, 1), "n": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-roots", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--copy", action="store_true",
                    help="Pointer-generator copy mechanism (radicals copied from input)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = _device()
    print(f"Device: {device}")

    if args.synthetic:
        raw = build_synthetic_dataset(n_samples=2000)
    else:
        raw = build_dataset(max_roots=args.max_roots, verbose=False)
    train_items, val_items, zs_items = split_by_root(raw, holdout_frac=0.15, seed=args.seed)

    vocab = Vocab(raw)
    print(f"Vocab: {vocab.n_out} output symbols, {vocab.n_in} input symbols "
          f"({len(vocab.feat_tokens)} feature tokens)")

    train_loader = DataLoader(ReinflectDataset(train_items, vocab),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    zs_loader = DataLoader(ReinflectDataset(zs_items, vocab),
                           batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    seen_loader = DataLoader(ReinflectDataset(val_items, vocab),
                             batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = Seq2Seq(vocab, d=args.d_model, copy=args.copy).to(device)
    kind = "copy-seq2seq (pointer-generator)" if args.copy else "seq2seq"
    print(f"{kind}: {sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tf = max(0.5, 1.0 - epoch / args.epochs)   # anneal teacher forcing
        total = 0.0
        nb = 0
        for src, src_len, tgt, _surf in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            logp = model(src, src_len, tgt, teacher_forcing=tf)
            loss = F.nll_loss(
                logp.reshape(-1, logp.size(-1)), tgt[:, 1:].reshape(-1),
                ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        if epoch % 5 == 0 or epoch == args.epochs:
            zs = evaluate(model, zs_loader, vocab, device)
            print(f"Epoch {epoch:3d} | loss {total/nb:.3f} | "
                  f"ZS-exact {zs['exact']:.3f} ZS-char {zs['char_acc']:.3f}")

    seen = evaluate(model, seen_loader, vocab, device)
    zs = evaluate(model, zs_loader, vocab, device)
    print(f"\n── {kind} reinflection (root + features -> consonantal form) ──")
    print(f"  Seen roots (val):   exact {seen['exact']:.3f}  char {seen['char_acc']:.3f}  (n={seen['n']})")
    print(f"  HELD-OUT roots:     exact {zs['exact']:.3f}  char {zs['char_acc']:.3f}  (n={zs['n']})")
    print("  (feature+length lookup baseline on held-out roots: exact 0.000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
