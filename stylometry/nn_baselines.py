#!/usr/bin/env python3
"""Tiny from-scratch neural language models for Classical Syriac.

These are deliberately small causal language models trained from scratch on the
srophe/syriac-corpus (no large pretrained Syriac LM exists). They serve as
*representation baselines* for the stylometry experiments, alongside the
character n-gram FastText model and Burrows's Delta:

  * ``ByteLM``         -- a small recurrent (LSTM) causal LM over UTF-8 *bytes*
                          (each Syriac codepoint is two bytes).
  * ``CharTransformer`` -- a tiny Transformer causal LM over Syriac *characters*
                          (Unicode codepoints), i.e. an attention model.

Each model provides three things used by ``paper_experiments.py``:

  1. intrinsic LM quality (held-out bits-per-byte / perplexity);
  2. a document vector  (mean-pooled final-layer hidden states over a text), and
  3. a word vector      (the same pooling over an isolated word form),

so the neural models join the FastText/word2vec/Delta comparison on identical
footing (same morphology probe, same same/cross-author AUC, same attribution).

Requires PyTorch. On Apple Silicon the MPS backend is used when available.
The models are tiny on ~2.18M tokens and are therefore *data-limited*: they are
a low-resource baseline, not an architecture ceiling.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:  # pragma: no cover - handled by callers
    _TORCH = False


# --------------------------------------------------------------------------- #
# Device & seeding
# --------------------------------------------------------------------------- #
def get_device() -> "torch.device":
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
class ByteTokenizer:
    """UTF-8 byte tokenizer (fixed vocabulary of 256)."""
    vocab_size = 256
    unit = "byte"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def bytes_per_unit(self) -> float:
        return 1.0  # one model step == one byte


class CharTokenizer:
    """Codepoint tokenizer over the characters seen in the training corpus."""
    unit = "char"

    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        # average UTF-8 bytes per character, to convert char-NLL into bits/byte
        total_bytes = len(corpus.encode("utf-8"))
        self._bpu = total_bytes / max(len(corpus), 1)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(c, unk) for c in text]

    def bytes_per_unit(self) -> float:
        return self._bpu


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
if _TORCH:

    class ByteLM(nn.Module):
        """Small LSTM causal language model."""

        def __init__(self, vocab_size: int, d_model: int = 128,
                     hidden: int = 256, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.lstm = nn.LSTM(d_model, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab_size)
            self.hidden_size = hidden

        def forward(self, x):
            h, _ = self.lstm(self.embed(x))
            h = self.norm(h)
            return self.head(h)

        def hidden_states(self, x):
            h, _ = self.lstm(self.embed(x))
            return self.norm(h)

    class CharTransformer(nn.Module):
        """Tiny Transformer causal language model (decoder-style, self-attn)."""

        def __init__(self, vocab_size: int, d_model: int = 128, layers: int = 2,
                     heads: int = 4, ff: int = 256, block_size: int = 128,
                     dropout: float = 0.1):
            super().__init__()
            self.block_size = block_size
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(block_size, d_model)
            enc = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=ff, dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True)
            self.blocks = nn.TransformerEncoder(enc, num_layers=layers,
                                                enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
            self.hidden_size = d_model

        def _mask(self, t: int, device):
            return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool),
                              diagonal=1)

        def hidden_states(self, x):
            t = x.size(1)
            pos = torch.arange(t, device=x.device)
            h = self.tok(x) + self.pos(pos)[None, :, :]
            h = self.blocks(h, mask=self._mask(t, x.device))
            return self.norm(h)

        def forward(self, x):
            return self.head(self.hidden_states(x))


# --------------------------------------------------------------------------- #
# Config + training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 64
    max_steps: int = 3000          # hard cap to bound wall-clock
    lr: float = 3e-3
    weight_decay: float = 0.01
    val_fraction: float = 0.05
    seed: int = 42
    log_every: int = 500


@dataclass
class TrainResult:
    model: object
    tokenizer: object
    name: str
    params: int
    val_bits_per_byte: float
    val_perplexity_unit: float
    train_seconds: float
    device: str
    steps: int
    history: list = field(default_factory=list)


def _make_chunks(ids: list[int], block_size: int):
    """Chop a token-id stream into (input, target) blocks for next-token LM."""
    n = (len(ids) - 1) // block_size
    if n <= 0:
        raise ValueError("corpus too small for the chosen block_size")
    x = torch.tensor(ids[: n * block_size], dtype=torch.long).view(n, block_size)
    y = torch.tensor(ids[1: n * block_size + 1], dtype=torch.long).view(n, block_size)
    return x, y


def train_lm(model, tokenizer, corpus_text: str, name: str,
             cfg: TrainConfig) -> TrainResult:
    """Train a causal LM on a single concatenated corpus string."""
    device = get_device()
    set_seed(cfg.seed)
    model = model.to(device)

    ids = tokenizer.encode(corpus_text)
    x, y = _make_chunks(ids, cfg.block_size)
    n_val = max(1, int(len(x) * cfg.val_fraction))
    perm = torch.randperm(len(x))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())

    start = time.time()
    step = 0
    history: list[tuple[int, float]] = []
    model.train()
    done = False
    while not done:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                history.append((step, float(loss.item())))
            if step >= cfg.max_steps:
                done = True
                break

    # Validation NLL (nats / unit) -> bits per unit -> bits per byte.
    model.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            l = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            nll += float(l.item()) * yb.numel()
            ntok += yb.numel()
    nll_per_unit = nll / max(ntok, 1)
    bits_per_unit = nll_per_unit / math.log(2)
    bits_per_byte = bits_per_unit / tokenizer.bytes_per_unit()
    perplexity_unit = math.exp(nll_per_unit)

    return TrainResult(
        model=model, tokenizer=tokenizer, name=name, params=params,
        val_bits_per_byte=bits_per_byte, val_perplexity_unit=perplexity_unit,
        train_seconds=time.time() - start, device=str(device),
        steps=step, history=history)


# --------------------------------------------------------------------------- #
# Embedding extraction (doc + word vectors from pooled hidden states)
# --------------------------------------------------------------------------- #
class NeuralEncoder:
    """Wraps a trained LM to produce mean-pooled hidden-state vectors."""

    def __init__(self, result: TrainResult):
        self.model = result.model
        self.tok = result.tokenizer
        self.name = result.name
        self.block_size = getattr(result.model, "block_size", 128)
        self.device = next(result.model.parameters()).device
        self.dim = result.model.hidden_size

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray | None:
        ids = self.tok.encode(text)
        if not ids:
            return None
        self.model.eval()
        bs = self.block_size
        total = np.zeros(self.dim, dtype=np.float64)
        count = 0
        for i in range(0, len(ids), bs):
            chunk = ids[i:i + bs]
            if not chunk:
                continue
            xb = torch.tensor(chunk, dtype=torch.long, device=self.device)[None, :]
            h = self.model.hidden_states(xb)[0]   # (t, dim)
            # MPS has no float64; sum in float32 on-device, accumulate in numpy.
            total += h.sum(dim=0).float().cpu().numpy().astype(np.float64)
            count += h.size(0)
        if count == 0:
            return None
        return total / count


def build_corpus_text(data_dir: Path, normalize: bool, exclude_ids=None) -> str:
    """One whitespace-joined string of the whole corpus, for LM training."""
    from core.stylometry import load_texts
    texts = load_texts(data_dir, normalize, exclude_ids=exclude_ids or set())
    # Reconstruct a token stream per document; join documents with newlines.
    docs = []
    for t in texts:
        words = []
        for form, c in t.counts.items():
            words.extend([form] * c)
        docs.append(" ".join(words))
    return "\n".join(docs)


def default_models(vocab_byte: int, char_tok: "CharTokenizer"):
    """Construct the two baseline models with paper-default sizes."""
    byte_lm = ByteLM(vocab_byte, d_model=128, hidden=256, layers=2)
    char_tf = CharTransformer(char_tok.vocab_size, d_model=128, layers=2,
                              heads=4, ff=256, block_size=128)
    return byte_lm, char_tf


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _smoke(argv: list[str] | None = None) -> int:
    import argparse
    from core.script import DEFAULT_CACHE, ensure_corpus

    if not _TORCH:
        print("error: PyTorch is not installed; cannot run neural baselines.")
        return 1

    ap = argparse.ArgumentParser(description="Train tiny Syriac LMs (smoke test).")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data_dir = ensure_corpus(args.cache_dir)
    corpus = build_corpus_text(data_dir, normalize=True)
    print(f"corpus chars: {len(corpus):,} | bytes: {len(corpus.encode('utf-8')):,}")

    cfg = TrainConfig(max_steps=args.max_steps, seed=args.seed)
    byte_tok = ByteTokenizer()
    char_tok = CharTokenizer(corpus)
    byte_lm, char_tf = default_models(byte_tok.vocab_size, char_tok)

    for model, tok, name in ((byte_lm, byte_tok, "byte-LM"),
                             (char_tf, char_tok, "char-Transformer")):
        res = train_lm(model, tok, corpus, name, cfg)
        print(f"{name:<18} params={res.params:,} "
              f"bpb={res.val_bits_per_byte:.3f} "
              f"ppl_{tok.unit}={res.val_perplexity_unit:.1f} "
              f"{res.train_seconds:.1f}s on {res.device}")
        enc = NeuralEncoder(res)
        # quick morphology sanity: king vs kingdom should beat king vs father
        def cos(a, b):
            va, vb = enc.encode(a), enc.encode(b)
            return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
        king, kingdom, father = "\u0721\u0720\u071f\u0710", "\u0721\u0720\u071f\u0718\u072c\u0710", "\u0710\u0712\u0710"
        print(f"    cos(king,kingdom)={cos(king, kingdom):.3f}  "
              f"cos(king,father)={cos(king, father):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
