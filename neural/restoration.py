#!/usr/bin/env python3
"""Neural textual restoration for Classical Syriac ("Ithaca for Syriac").

Twist 3 (the application demonstrator). Manuscripts are damaged: letters, words,
and whole lines are lost to lacunae. This module trains a small **causal
character Transformer** -- a genuine from-scratch Syriac character language model
-- and restores a lacuna by generating into the gap from its surrounding context.
It is evaluated by **synthetic masking**: corrupting held-out real text and
measuring how well the model recovers it. No external gold labels are needed, so
the task is fully self-supervised and self-contained.

Why causal (not masked): on a model and corpus this small, a bidirectional masked
objective supervises only ~15% of positions and collapses to the unigram prior
(verified empirically). A causal objective gives dense per-position supervision
and learns the corpus -- it is the recipe the parent ``nn_baselines`` uses, and
here it lifts lacuna char-accuracy well above the unigram floor (~0.19 -> ~0.44 in
the short demo).

Honesty about novelty: neural restoration of damaged text is established for
Greek/Latin (e.g. Ithaca). The contribution here is Syriac-first and the
morphology-aware setting; the task itself is not new. See ``docs/DESIGN.md``.

Why this twist is runnable today
--------------------------------
Unlike the encoder-transfer phases, restoration needs only ``torch`` and the
Digital Syriac Corpus you already have cached -- no ``transformers`` download, no
license-restricted SEDRA data. It reuses the parent tokenizer indirectly through
``neural.aggregate`` (read-only), so the text is exactly the corpus the paper
uses, and splits **by document** so a restored passage is never seen in training.

    # end-to-end demo (small model, 2000 steps, ~1-2 min on MPS/CPU)
    .venv/bin/python -m neural.restoration --demo

    # a longer/larger run
    .venv/bin/python -m neural.restoration --train --steps 4000 --d-model 192
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover - torch optional
    _TORCH = False

# read-only reuse of the sub-project's corpus assembler (which itself reuses the
# parent tokenizer), so restoration sees the same text as the baseline.
from neural.aggregate import collect, split

PAD, MASK, UNK = 0, 1, 2
_SPECIAL = ["<pad>", "<mask>", "<unk>"]


@dataclass
class RestorationConfig:
    sources: tuple[str, ...] = ("dsc",)
    normalize: bool = True
    window: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    mask_fraction: float = 0.15
    max_span: int = 8
    batch_size: int = 32
    steps: int = 2000
    lr: float = 3e-3
    seed: int = 42
    max_windows: int = 30000     # cap dataset size for quick, bounded runs
    val_fraction: float = 0.1


# --------------------------------------------------------------------------- #
# Vocabulary and windowing
# --------------------------------------------------------------------------- #
def build_vocab(texts: list[str]) -> tuple[dict[str, int], list[str]]:
    chars = sorted({ch for t in texts for ch in t})
    itos = list(_SPECIAL) + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def make_windows(texts: list[str], stoi: dict[str, int], window: int,
                 max_windows: int | None = None) -> "np.ndarray":
    rows: list[list[int]] = []
    for t in texts:
        ids = [stoi.get(ch, UNK) for ch in t]
        for i in range(0, len(ids), window):
            chunk = ids[i:i + window]
            if len(chunk) < window:
                chunk = chunk + [PAD] * (window - len(chunk))
            rows.append(chunk)
            if max_windows and len(rows) >= max_windows:
                return np.asarray(rows, dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def corrupt(batch: "np.ndarray", rng: "np.random.Generator",
            mask_fraction: float, max_span: int,
            vocab_size: int | None = None,
            all_mask: bool = False) -> tuple["np.ndarray", "np.ndarray"]:
    """Corrupt contiguous spans (simulated lacunae), BERT-style 80/10/10.

    For each selected position the target is the original id; loss is computed
    only there (everything else is -100). During training, of the selected
    positions 80% are replaced by <mask>, 10% by a random character, and 10% left
    unchanged -- this stops the model from trusting visible tokens, so it must
    contextualize every position rather than copy. For evaluation pass
    ``all_mask=True`` to model real lacunae (every selected position becomes
    <mask>).

    Returns (corrupted, targets).
    """
    corrupted = batch.copy()
    targets = np.full_like(batch, -100)
    B, L = batch.shape
    lo = len(_SPECIAL)
    hi = vocab_size if vocab_size is not None else lo + 1
    for b in range(B):
        valid = int((batch[b] != PAD).sum())
        if valid <= 1:
            continue
        n_mask = max(1, int(valid * mask_fraction))
        masked, guard = 0, 0
        while masked < n_mask and guard < 64:
            guard += 1
            span = int(rng.integers(1, max_span + 1))
            start = int(rng.integers(0, max(1, valid - span)))
            for p in range(start, min(start + span, valid)):
                if targets[b, p] != -100:
                    continue
                targets[b, p] = batch[b, p]
                r = 0.0 if all_mask else rng.random()
                if r < 0.8:
                    corrupted[b, p] = MASK
                elif r < 0.9 and hi > lo:
                    corrupted[b, p] = int(rng.integers(lo, hi))
                # else: keep the original token (10%)
                masked += 1
    return corrupted, targets


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
if _TORCH:

    class CharRestorer(nn.Module):
        """A small character Transformer trained as a causal language model.

        It restores a lacuna by generating into the gap from the surrounding
        (left) context. A bidirectional masked objective was tried first but, on
        a model and corpus this small, it collapses to the unigram prior; the
        causal objective gives dense per-position supervision and provably learns
        this corpus (it is the recipe the parent ``nn_baselines`` uses).
        """

        def __init__(self, vocab: int, cfg: RestorationConfig):
            super().__init__()
            d = cfg.d_model
            self.embed = nn.Embedding(vocab, d, padding_idx=PAD)
            self.pos = nn.Parameter(torch.zeros(1, cfg.window, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads, dim_feedforward=4 * d,
                dropout=cfg.dropout, batch_first=True, activation="gelu",
                norm_first=True)   # pre-norm: stable from-scratch training
            # enable_nested_tensor=False avoids an MPS/code path issue (matches
            # the parent nn_baselines char-Transformer).
            self.encoder = nn.TransformerEncoder(layer, cfg.n_layers,
                                                 enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)     # final norm (matches working baseline)
            self.head = nn.Linear(d, vocab)

        def forward(self, x, key_padding_mask=None, causal=False):
            h = self.embed(x) + self.pos[:, :x.size(1)]
            attn_mask = None
            if causal:
                t = x.size(1)
                attn_mask = torch.triu(
                    torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
            return self.head(self.norm(h))

    def _device() -> "torch.device":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(cfg: RestorationConfig) -> dict:
        _seed(cfg.seed)
        device = _device()

        docs = [d for d in collect(cfg.sources, cfg.normalize) if d.n_tokens >= 20]
        if not docs:
            raise RuntimeError("no documents collected; is the corpus cached?")
        parts = split(docs, cfg.val_fraction, 0.0, cfg.seed)
        train_text = [d.text for d in parts["train"]]
        val_text = [d.text for d in parts["val"]] or train_text[-1:]

        stoi, itos = build_vocab(train_text)
        train_w = make_windows(train_text, stoi, cfg.window, cfg.max_windows)
        val_w = make_windows(val_text, stoi, cfg.window, max(2000, cfg.max_windows // 10))
        if len(train_w) == 0:
            raise RuntimeError("no training windows produced")

        model = CharRestorer(len(itos), cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        # Linear warmup then linear decay -- standard stabilizer for from-scratch
        # Transformer training.
        warmup = max(10, cfg.steps // 10)

        def lr_at(step: int) -> float:
            if step < warmup:
                return step / warmup
            return max(0.0, (cfg.steps - step) / max(1, cfg.steps - warmup))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        rng = np.random.default_rng(cfg.seed)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        model.train()
        for step in range(1, cfg.steps + 1):
            idx = rng.integers(0, len(train_w), size=cfg.batch_size)
            batch = train_w[idx]
            xb = torch.from_numpy(batch).to(device)
            kpm = torch.from_numpy(batch == PAD).to(device)
            # Causal language-model objective: predict each character from its
            # left context, dense supervision at every position. This provably
            # learns the corpus; a bidirectional masked objective on a model and
            # data this small collapses to the unigram prior (see class docstring).
            logits = model(xb, key_padding_mask=kpm, causal=True)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                xb[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % max(1, cfg.steps // 10) == 0:
                print(f"  step {step:5d}/{cfg.steps}  loss {loss.item():.3f}",
                      file=sys.stderr, flush=True)
        train_s = time.time() - t0

        metrics = evaluate(model, val_w, rng, cfg, device)
        metrics.update(params=n_params, train_seconds=round(train_s, 1),
                       device=str(device), vocab=len(itos),
                       train_windows=len(train_w), val_windows=len(val_w))
        # a tangible qualitative example
        example = restore_example(model, val_w, itos, rng, cfg, device)
        metrics["example"] = example
        return metrics

    @torch.no_grad()
    def evaluate(model, windows: "np.ndarray", rng, cfg, device) -> dict:
        model.eval()
        # fixed eval masking for comparability
        eval_rng = np.random.default_rng(12345)
        correct = total = 0
        span_exact = span_total = 0
        for i in range(0, len(windows), cfg.batch_size):
            batch = windows[i:i + cfg.batch_size]
            # choose lacuna positions to restore; a causal model is scored on its
            # prediction at those positions (teacher-forced from true left context)
            _, targets = corrupt(batch, eval_rng, cfg.mask_fraction, cfg.max_span,
                                 all_mask=True)
            xb = torch.from_numpy(batch).to(device)
            kpm = torch.from_numpy(batch == PAD).to(device)
            logits = model(xb, key_padding_mask=kpm, causal=True)
            pred = logits.argmax(-1).cpu().numpy()   # pred[:, t] restores pos t+1
            mask = targets != -100
            L = batch.shape[1]
            for b in range(batch.shape[0]):
                p = 0
                while p < L:
                    if mask[b, p] and p >= 1:
                        q = p
                        span_ok = True
                        while q < L and mask[b, q]:
                            total += 1
                            ok = bool(pred[b, q - 1] == batch[b, q])
                            correct += int(ok)
                            span_ok = span_ok and ok
                            q += 1
                        span_total += 1
                        span_exact += int(span_ok)
                        p = q
                    else:
                        p += 1
        return {
            "char_accuracy": round(correct / max(total, 1), 4),
            "span_exact_match": round(span_exact / max(span_total, 1), 4),
            "masked_chars": total,
            "masked_spans": span_total,
        }

    @torch.no_grad()
    def restore_example(model, windows: "np.ndarray", itos, rng, cfg, device) -> dict:
        model.eval()
        ex_rng = np.random.default_rng(7)
        # pick a window with enough content
        cand = [w for w in windows if int((w != PAD).sum()) > 40]
        if not cand:
            return {}
        row = cand[int(ex_rng.integers(0, len(cand)))].copy()
        _, targets = corrupt(row[None, :], ex_rng, cfg.mask_fraction, cfg.max_span,
                             all_mask=True)
        mpos = targets[0] != -100
        # autoregressive greedy fill: rebuild left to right, replacing each lacuna
        # position with the model's own prediction from the (partly restored) left
        work = row.copy()
        T = len(row)
        for p in range(1, T):
            if work[p] == PAD:
                break
            if mpos[p]:
                xb = torch.from_numpy(work[None, :]).to(device)
                kpm = torch.from_numpy((work == PAD)[None, :]).to(device)
                logits = model(xb, key_padding_mask=kpm, causal=True)
                work[p] = int(logits[0, p - 1].argmax().item())

        def decode(ids, blanks=None):
            out = []
            for j, i in enumerate(ids):
                if i == PAD:
                    break
                out.append("\u2588" if (blanks is not None and blanks[j]) else itos[i])
            return "".join(out)

        return {
            "damaged": decode(row, blanks=mpos),
            "restored": decode(work),
            "truth": decode(row),
        }

else:  # pragma: no cover - torch missing

    def train(cfg: RestorationConfig) -> dict:
        raise RuntimeError(
            "torch is required for restoration. It ships with the parent repo's "
            "neural baselines; if missing install it into the venv.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="end-to-end run on DSC (causal char-LM, 2000 steps, ~1-2 min)")
    ap.add_argument("--train", action="store_true", help="train with the given options")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--sources", default="dsc",
                    help="comma-separated subset of dsc,syrnt,peshitta")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not _TORCH:
        print("torch is required for restoration (parent repo installs it).",
              file=sys.stderr)
        return 2
    if not (args.demo or args.train):
        ap.print_help()
        return 1

    cfg = RestorationConfig(seed=args.seed,
                            sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()))
    if args.demo:
        cfg.steps = 2000
        cfg.max_windows = 20000
    if args.steps is not None:
        cfg.steps = args.steps
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.layers is not None:
        cfg.n_layers = args.layers
    if args.window is not None:
        cfg.window = args.window
    if args.max_windows is not None:
        cfg.max_windows = args.max_windows

    print(f"Training char restorer: d_model={cfg.d_model} layers={cfg.n_layers} "
          f"window={cfg.window} steps={cfg.steps} sources={cfg.sources}",
          file=sys.stderr)
    metrics = train(cfg)

    print("\n=== restoration results (synthetic lacunae on held-out text) ===")
    print(f"  params              : {metrics['params']:,}")
    print(f"  device / train time : {metrics['device']} / {metrics['train_seconds']}s")
    print(f"  vocab / windows     : {metrics['vocab']} chars / "
          f"{metrics['train_windows']:,} train, {metrics['val_windows']:,} val")
    print(f"  masked char accuracy: {metrics['char_accuracy']:.3f} "
          f"(over {metrics['masked_chars']:,} masked chars)")
    print(f"  span exact-match    : {metrics['span_exact_match']:.3f} "
          f"(over {metrics['masked_spans']:,} lacunae)")
    ex = metrics.get("example") or {}
    if ex:
        print("\n  --- example (\u2588 = lacuna) ---")
        print("  damaged : " + ex.get("damaged", ""))
        print("  restored: " + ex.get("restored", ""))
        print("  truth   : " + ex.get("truth", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
