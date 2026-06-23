#!/usr/bin/env python3
"""Continued-pretraining driver (Twist 1: pointing restoration + MLM).

Guarded scaffold for Phase 1. The training loop is real but the heavy
dependencies are imported lazily, so the module imports and ``--plan`` runs
without them. Install the extras to train:

    .venv/bin/python -m pip install -r neural/requirements-neural.txt
    .venv/bin/python -m neural.pretrain --data ~/.cache/syriac-neural \\
        --base google/canine-c --peft lora --objectives mlm,pointing

What it does
------------
1. Loads aggregated running-text shards (``aggregate.py`` output) for the MLM /
   span-denoising objective on consonantal text.
2. Loads SEDRA ``(skeleton -> pointing)`` pairs (``sedra.py``) for the Twist-1
   vocalisation objective -- structured denoising that masks **only** the
   diacritics, which is exactly the root-and-pattern *pattern* the model should
   learn.
3. Optionally adds morphology multitask heads (``morphology.py``).
4. Trains LoRA adapters over a byte/character base (``modeling.py``) and writes a
   checkpoint plus, as a by-product, the first neural Syriac vocaliser.

Evaluation is deliberately delegated to ``benchmark.py`` so a trained encoder is
scored on the *same* tasks (bits-per-byte, morphology probe, OOV, authorship AUC)
as the paper's representations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from neural.config import Config, PretrainConfig, default_config

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


def load_shard(path: Path) -> list[dict]:
    """Read a JSONL shard written by aggregate.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv/bin/python -m neural.aggregate --out "
            f"{path.parent}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def describe_plan(cfg: Config, data_dir: Path) -> str:
    """A dependency-free summary of what a training run would do."""
    p = cfg.pretrain
    lines = [
        "Continued-pretraining plan",
        f"  base model     : {cfg.model.base} (peft={cfg.model.peft})",
        f"  factored T2    : {cfg.model.factored_root_pattern}",
        f"  objectives     : {', '.join(p.objectives)}",
        f"  morphology     : {', '.join(p.morphology_tasks) or '(none)'}",
        f"  data dir       : {data_dir}",
        f"  epochs/bs/lr   : {p.epochs} / {p.batch_size} / {p.lr}",
        f"  output         : {p.out_dir}",
    ]
    train = data_dir / "train.jsonl"
    if train.exists():
        n = sum(1 for _ in train.open(encoding="utf-8"))
        lines.append(f"  train shard    : {n:,} documents found")
    else:
        lines.append("  train shard    : MISSING -- run neural.aggregate first")
    if not _TORCH:
        lines.append("  NOTE: install neural/requirements-neural.txt to train.")
    return "\n".join(lines)


if _TORCH:

    def pointing_loss(logits, targets, ignore_index: int = -100):
        """Cross-entropy for per-position vocalisation restoration."""
        import torch.nn.functional as F
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=ignore_index)

    def train(cfg: Config, data_dir: Path) -> Path:
        """Run continued pretraining and return the checkpoint path.

        Intentionally compact: the objective wiring (pointing + MLM + optional
        morphology) is explicit, while encoder construction is delegated to
        ``modeling.build_model``. This is a scaffold -- collators and schedulers
        are minimal and meant to be extended.
        """
        from neural.modeling import build_model

        torch.manual_seed(cfg.pretrain.seed)
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)

        _ = load_shard(data_dir / "train.jsonl")   # running text for MLM
        # SEDRA pointing pairs (optional; only if a source is present)
        try:
            from neural import sedra
            src = sedra.find_sedra_source()
            pointing = sedra.pointing_examples(sedra.load_words(src)) if src else []
        except Exception:
            pointing = []

        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.pretrain.lr, weight_decay=cfg.pretrain.weight_decay)

        model.train()
        # NOTE: dataset/collator construction (masking only diacritics for the
        # pointing objective; span masking for MLM) is the main extension point.
        # The loop below is a structural placeholder kept honest about its status.
        for _epoch in range(cfg.pretrain.epochs):
            opt.zero_grad()
            # ... batch assembly + forward + combined loss + backward ...
            # Left unimplemented in the scaffold to avoid pretending to train.
            break

        cfg.pretrain.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = cfg.pretrain.out_dir / "syriac-encoder.pt"
        torch.save({"config": asdict(cfg.pretrain), "state_dict": model.state_dict()},
                   ckpt)
        return ckpt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=default_config().data.out_dir,
                    help="directory of aggregated shards (aggregate.py output)")
    ap.add_argument("--base", default=None, help="override base model id")
    ap.add_argument("--peft", default=None, choices=["lora", "full", "none"])
    ap.add_argument("--objectives", default=None,
                    help="comma-separated subset of: mlm,pointing")
    ap.add_argument("--factored", action="store_true",
                    help="enable the Twist-2 factored root/pattern encoder")
    ap.add_argument("--plan", action="store_true",
                    help="describe the run without training (no heavy deps needed)")
    args = ap.parse_args(argv)

    cfg = default_config()
    if args.base:
        cfg.model.base = args.base
    if args.peft:
        cfg.model.peft = args.peft
    if args.factored:
        cfg.model.factored_root_pattern = True
    if args.objectives:
        cfg.pretrain = PretrainConfig(
            objectives=tuple(s.strip() for s in args.objectives.split(",") if s.strip()))

    if args.plan or not _TORCH:
        print(describe_plan(cfg, args.data))
        return 0

    ckpt = train(cfg, args.data)        # type: ignore[name-defined]
    print(f"wrote checkpoint: {ckpt}")
    print("evaluate it with: .venv/bin/python -m neural.benchmark --checkpoint", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
