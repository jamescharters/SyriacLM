#!/usr/bin/env python3
"""LoRA continued-pretraining of CANINE-c on Syriac (Phase 1).

Adapts the multilingual, tokenizer-free CANINE-c encoder to Classical Syriac by
masked-codepoint denoising on the aggregated corpus, training only small LoRA
adapters (the methodologically correct choice at ~2.6M tokens -- full fine-tuning
of 132M parameters would overfit). The adapted encoder is then scored by
``neural.canine_encoder`` on the paper's exact authorship cohort, so the gain
from continued pretraining is measured against the off-the-shelf baseline and the
FastText/word2vec/Delta references.

There is no ``CanineForMaskedLM`` in transformers, so we attach a lightweight
linear codepoint-prediction head over CANINE's character-resolution
``last_hidden_state`` and train it together with the LoRA adapters. Masking is
BERT-style 80/10/10 over spans, loss only on masked positions.

Requires the optional extras (``transformers``, ``peft``). TLS to HuggingFace is
verified through the OS trust store via ``truststore``.

    .venv/bin/python -m neural.canine_pretrain --steps 1500 --out ~/.cache/syriac-neural/checkpoints/canine-lora
    .venv/bin/python -m neural.canine_encoder --checkpoint ~/.cache/syriac-neural/checkpoints/canine-lora
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:  # pragma: no cover - environment dependent
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from neural.config import DEFAULT_NEURAL_CACHE

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import CanineModel
    from peft import LoraConfig, get_peft_model
    _HF = True
except Exception:  # pragma: no cover
    _HF = False

DEFAULT_BASE = "google/canine-c"
CLS, SEP, PAD, MASK = 0xE000, 0xE001, 0, 0xE003
# CANINE hashes codepoints into a fixed embedding space; the output vocabulary we
# predict over is the set of codepoints actually present in the corpus.


def load_shard_texts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv/bin/python -m neural.aggregate --out "
            f"{path.parent}")
    return [json.loads(line)["text"] for line in
            path.read_text(encoding="utf-8").splitlines()]


def build_codepoint_vocab(texts: list[str]) -> tuple[dict[int, int], list[int]]:
    """Map the corpus codepoints to a compact output index space (+specials)."""
    cps = sorted({ord(c) for t in texts for c in t})
    itos = [PAD, MASK] + cps
    stoi = {cp: i for i, cp in enumerate(itos)}
    return stoi, itos


def make_windows(texts: list[str], window: int, max_windows: int | None) -> np.ndarray:
    rows: list[list[int]] = []
    for t in texts:
        cps = [ord(c) for c in t]
        for i in range(0, len(cps), window - 2):
            chunk = cps[i:i + window - 2]
            seq = [CLS] + chunk + [SEP]
            if len(seq) < window:
                seq = seq + [PAD] * (window - len(seq))
            rows.append(seq)
            if max_windows and len(rows) >= max_windows:
                return np.asarray(rows, dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


if _HF:

    class CanineForCodepointMLM(nn.Module):
        """CANINE encoder (LoRA-adapted) + a linear codepoint-prediction head."""

        def __init__(self, base: str, out_vocab: int, lora_r: int, lora_alpha: int,
                     lora_dropout: float, freeze_encoder: bool = False):
            super().__init__()
            enc = CanineModel.from_pretrained(base)
            self.frozen = freeze_encoder
            if freeze_encoder:
                # Linear-probe baseline: the encoder is frozen and only the
                # codepoint head trains. The gap between this and the LoRA model
                # isolates exactly what adapting the encoder bought.
                for p in enc.parameters():
                    p.requires_grad = False
                self.encoder = enc
            else:
                lora = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    bias="none",
                    target_modules=["query", "key", "value"])
                self.encoder = get_peft_model(enc, lora)
            self.head = nn.Linear(enc.config.hidden_size, out_vocab)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.head(out.last_hidden_state)

        def save_adapter(self, out_dir: Path) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)
            if not self.frozen:
                self.encoder.save_pretrained(str(out_dir))
            torch.save(self.head.state_dict(), out_dir / "codepoint_head.pt")

    def _device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def mask_batch(batch: np.ndarray, stoi: dict[int, int], rng, vocab_size: int,
                   mask_fraction: float, max_span: int):
        """BERT 80/10/10 span masking. Returns (input_cp, target_idx) where target
        is the output-vocab index at masked positions and -100 elsewhere."""
        inp = batch.copy()
        tgt = np.full(batch.shape, -100, dtype=np.int64)
        B, L = batch.shape
        cp_choices = [cp for cp in stoi if cp not in (PAD, MASK)]
        for b in range(B):
            valid = int((batch[b] != PAD).sum())
            content = [p for p in range(valid) if batch[b, p] not in (CLS, SEP)]
            if not content:
                continue
            n_mask = max(1, int(len(content) * mask_fraction))
            masked, guard = 0, 0
            while masked < n_mask and guard < 64:
                guard += 1
                span = int(rng.integers(1, max_span + 1))
                start = int(rng.integers(0, max(1, valid - span)))
                for p in range(start, min(start + span, valid)):
                    if batch[b, p] in (CLS, SEP, PAD) or tgt[b, p] != -100:
                        continue
                    tgt[b, p] = stoi[int(batch[b, p])]
                    r = rng.random()
                    if r < 0.8:
                        inp[b, p] = MASK
                    elif r < 0.9:
                        inp[b, p] = int(rng.choice(cp_choices))
                    masked += 1
        return inp, tgt

    def train(*, base: str, data_dir: Path, out_dir: Path, steps: int, window: int,
              batch_size: int, lr: float, mask_fraction: float, max_span: int,
              lora_r: int, lora_alpha: int, lora_dropout: float, max_windows: int,
              seed: int, freeze_encoder: bool = False) -> dict:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        device = _device()

        train_texts = load_shard_texts(data_dir / "train.jsonl")
        val_texts = load_shard_texts(data_dir / "val.jsonl")
        stoi, itos = build_codepoint_vocab(train_texts + val_texts)
        train_w = make_windows(train_texts, window, max_windows)
        val_w = make_windows(val_texts, window, max(1000, max_windows // 10))

        model = CanineForCodepointMLM(base, len(itos), lora_r, lora_alpha,
                                      lora_dropout, freeze_encoder=freeze_encoder).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=lr, weight_decay=0.01)
        warmup = max(10, steps // 10)

        def lr_at(s):
            return s / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        rng = np.random.default_rng(seed)

        t0 = time.time()
        model.train()
        for step in range(1, steps + 1):
            idx = rng.integers(0, len(train_w), size=batch_size)
            batch = train_w[idx]
            inp, tgt = mask_batch(batch, stoi, rng, len(itos), mask_fraction, max_span)
            ids = torch.from_numpy(inp).to(device)
            attn = torch.from_numpy((batch != PAD).astype(np.int64)).to(device)
            yb = torch.from_numpy(tgt).to(device)
            logits = model(ids, attn)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   yb.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            sched.step()
            if step % max(1, steps // 10) == 0:
                print(f"  step {step:5d}/{steps}  loss {loss.item():.3f}",
                      file=sys.stderr, flush=True)
        train_s = time.time() - t0

        # held-out masked-codepoint accuracy + masked pseudo bits-per-byte
        val_acc = evaluate(model, val_w, stoi, len(itos), mask_fraction, max_span, device)
        val_bpb = masked_bits_per_byte(model, val_w, val_texts, stoi, len(itos),
                                       mask_fraction, max_span, device)

        model.save_adapter(out_dir)
        (out_dir / "pretrain_meta.json").write_text(json.dumps({
            "base": base, "steps": steps, "trainable_params": trainable,
            "total_params": total, "train_seconds": round(train_s, 1),
            "val_masked_accuracy": val_acc, "val_masked_bits_per_byte": val_bpb,
            "frozen_encoder": freeze_encoder, "device": str(device),
            "out_vocab": len(itos), "window": window,
        }, indent=2), encoding="utf-8")
        return {"trainable": trainable, "total": total, "train_seconds": round(train_s, 1),
                "val_masked_accuracy": val_acc, "val_masked_bits_per_byte": val_bpb,
                "frozen": freeze_encoder, "out_dir": str(out_dir),
                "device": str(device)}

    @torch.no_grad()
    def evaluate(model, windows, stoi, vocab_size, mask_fraction, max_span, device) -> float:
        model.eval()
        rng = np.random.default_rng(12345)
        correct = total = 0
        for i in range(0, len(windows), 32):
            batch = windows[i:i + 32]
            inp, tgt = mask_batch(batch, stoi, rng, vocab_size, mask_fraction, max_span)
            ids = torch.from_numpy(inp).to(device)
            attn = torch.from_numpy((batch != PAD).astype(np.int64)).to(device)
            pred = model(ids, attn).argmax(-1).cpu().numpy()
            m = tgt != -100
            correct += int((pred[m] == tgt[m]).sum())
            total += int(m.sum())
        model.train()
        return round(correct / max(total, 1), 4)

    @torch.no_grad()
    def masked_bits_per_byte(model, windows, texts, stoi, vocab_size,
                             mask_fraction, max_span, device) -> float:
        """Masked **pseudo** bits-per-byte on held-out text.

        CANINE is bidirectional, so this is a *pseudo*-likelihood (each masked
        codepoint scored from both-side context) and is therefore NOT directly
        comparable to the autoregressive byte-LM's bits-per-byte -- it only
        supports comparing CANINE variants to each other (frozen vs LoRA). The
        per-codepoint NLL is converted to bits/byte using the corpus's mean UTF-8
        bytes per codepoint.
        """
        rng = np.random.default_rng(2024)
        nll, n = 0.0, 0
        for i in range(0, len(windows), 32):
            batch = windows[i:i + 32]
            inp, tgt = mask_batch(batch, stoi, rng, vocab_size, mask_fraction, max_span)
            ids = torch.from_numpy(inp).to(device)
            attn = torch.from_numpy((batch != PAD).astype(np.int64)).to(device)
            yb = torch.from_numpy(tgt).to(device)
            logits = model(ids, attn)
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                yb.reshape(-1), ignore_index=-100, reduction="sum")
            nll += float(l.item())
            n += int((tgt != -100).sum())
        bits_per_cp = (nll / max(n, 1)) / np.log(2)
        total_chars = sum(len(t) for t in texts)
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        bytes_per_cp = total_bytes / max(total_chars, 1)
        return round(bits_per_cp / bytes_per_cp, 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_NEURAL_CACHE)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_NEURAL_CACHE / "checkpoints" / "canine-lora")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mask-fraction", type=float, default=0.15)
    ap.add_argument("--max-span", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-windows", type=int, default=12000)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="linear-probe baseline: freeze the encoder, train only the head")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not _HF:
        print("transformers + peft + torch required. Install:\n"
              "    .venv/bin/python -m pip install -r neural/requirements-neural.txt",
              file=sys.stderr)
        return 2

    mode = "linear-probe (frozen encoder)" if args.freeze_encoder else "LoRA continued-pretraining"
    print(f"{mode}: {args.base} on {args.data} ...", file=sys.stderr)
    m = train(base=args.base, data_dir=args.data, out_dir=args.out, steps=args.steps,
              window=args.window, batch_size=args.batch_size, lr=args.lr,
              mask_fraction=args.mask_fraction, max_span=args.max_span,
              lora_r=args.lora_r, lora_alpha=args.lora_alpha,
              lora_dropout=args.lora_dropout, max_windows=args.max_windows,
              seed=args.seed, freeze_encoder=args.freeze_encoder)
    print(f"\n=== {mode} done ===")
    print(f"  trainable params : {m['trainable']:,} / {m['total']:,} "
          f"({100*m['trainable']/m['total']:.2f}% trained)")
    print(f"  train time       : {m['train_seconds']}s on {m['device']}")
    print(f"  val masked acc   : {m['val_masked_accuracy']:.3f}")
    print(f"  val masked p-bpb : {m['val_masked_bits_per_byte']:.3f}  "
          f"(pseudo bits/byte; CANINE variants only, NOT vs the autoregressive byte-LM)")
    print(f"  saved to         : {m['out_dir']}")
    if not args.freeze_encoder:
        print("\nevaluate authorship AUC with:")
        print(f"  .venv/bin/python -m neural.canine_encoder --checkpoint {m['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
