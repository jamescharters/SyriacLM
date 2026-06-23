#!/usr/bin/env python3
"""Model definitions: a byte/character Syriac encoder with morphology-aware heads.

This is the modeling scaffold for Phases 1-2. It is **guarded**: the heavy
dependencies (``torch``, ``transformers``, optionally ``peft``) are imported
lazily, so the module always imports and ``--info`` always runs. Install the
extras to instantiate models:

    .venv/bin/python -m pip install -r neural/requirements-neural.txt

Design (see ``docs/DESIGN.md``)
------------------------------
* **Base**: a tokenizer-free encoder over Unicode codepoints / UTF-8 bytes
  (default CANINE-c), so coverage of the Syriac script is total -- the neural
  continuation of the paper's subword/OOV-free argument.
* **PEFT**: LoRA adapters by default; full fine-tuning is available but
  inappropriate at this data scale.
* **Twist 1 -- PointingHead**: restores the vocalisation (pattern morpheme) of a
  consonantal form. A per-position classifier over a small vocalisation vocabulary.
* **Twist 2 -- FactoredEncoder**: an explicit consonant-skeleton (root) stream and
  a vocalic-pattern stream, fused before the task heads. The two channels come
  for free from SEDRA's encoding (``sedra.consonant_vowel_channels``).
* **MorphologyHead**: multitask token classification over root/state/tense/...
* **ProjectionHead**: a SupCon projection for authorship verification, mirroring
  the parent ``av_head.py`` so the encoder plugs into the existing bake-off.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from neural.config import Config, ModelConfig, default_config

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel
    _HF = True
except Exception:  # pragma: no cover - heavy deps optional
    _HF = False

try:
    from peft import LoraConfig, get_peft_model
    _PEFT = True
except Exception:  # pragma: no cover
    _PEFT = False


# Vocalisation target vocabulary for the pointing head: the SEDRA vowels and
# diacritics plus a "bare" symbol. Kept tiny and fixed so the head is cheap.
from neural import sedra

POINTING_VOCAB: list[str] = ["<bare>"] + sorted(sedra.SEDRA_VOWELS | sedra.SEDRA_DIACRITICS)
POINTING_STOI: dict[str, int] = {s: i for i, s in enumerate(POINTING_VOCAB)}


@dataclass
class HeadSpec:
    """Describes a morphology classification head (name -> number of classes)."""

    name: str
    num_classes: int


if _HF:

    def _maybe_lora(model, cfg: ModelConfig):
        if cfg.peft != "lora":
            return model
        if not _PEFT:
            raise RuntimeError(
                "peft is required for LoRA. Install neural/requirements-neural.txt, "
                "or set ModelConfig.peft='full'/'none'.")
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, bias="none")
        return get_peft_model(model, lora)

    class PointingHead(nn.Module):
        """Per-position classifier restoring vocalisation (Twist 1)."""

        def __init__(self, hidden: int, vocab: int = len(POINTING_VOCAB)):
            super().__init__()
            self.proj = nn.Linear(hidden, vocab)

        def forward(self, hidden_states):           # (B, T, H) -> (B, T, V)
            return self.proj(hidden_states)

    class MorphologyHead(nn.Module):
        """Multitask token classification over morphological features."""

        def __init__(self, hidden: int, specs: list[HeadSpec]):
            super().__init__()
            self.heads = nn.ModuleDict(
                {s.name: nn.Linear(hidden, s.num_classes) for s in specs})

        def forward(self, hidden_states):
            return {name: head(hidden_states) for name, head in self.heads.items()}

    class ProjectionHead(nn.Module):
        """L2-normalized projection for supervised-contrastive authorship (cf. av_head.py)."""

        def __init__(self, hidden: int, out_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))

        def forward(self, pooled):
            z = self.net(pooled)
            return z / (z.norm(dim=-1, keepdim=True) + 1e-9)

    class FactoredEncoder(nn.Module):
        """Twist 2: parallel consonant (root) and vowel (pattern) streams.

        A lightweight character embedding for the vowel/pattern channel is fused
        (concatenate + project) with the base encoder's contextual states, so the
        model is biased toward the root/pattern factorization rather than having
        to discover it. When ``ModelConfig.factored_root_pattern`` is False, the
        base encoder is used directly (the ablation baseline).
        """

        def __init__(self, base, hidden: int, pattern_dim: int):
            super().__init__()
            self.base = base
            self.pattern_embed = nn.Embedding(256, pattern_dim)   # byte-level pattern channel
            self.fuse = nn.Linear(hidden + pattern_dim, hidden)

        def forward(self, input_ids, attention_mask=None, pattern_ids=None):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state
            if pattern_ids is not None:
                p = self.pattern_embed(pattern_ids)
                h = self.fuse(torch.cat([h, p], dim=-1))
            return h

    class SyriacEncoderModel(nn.Module):
        """Base encoder + optional factoring + the requested heads + mean pooling."""

        def __init__(self, cfg: ModelConfig, morph_specs: list[HeadSpec] | None = None):
            super().__init__()
            self.cfg = cfg
            base = AutoModel.from_pretrained(cfg.base)
            base = _maybe_lora(base, cfg)
            hidden = AutoConfig.from_pretrained(cfg.base).hidden_size
            self.hidden = hidden
            if cfg.factored_root_pattern:
                self.encoder = FactoredEncoder(base, hidden, cfg.pattern_channel_dim)
                self._factored = True
            else:
                self.encoder = base
                self._factored = False
            self.pointing = PointingHead(hidden)
            self.morphology = MorphologyHead(hidden, morph_specs or [])
            self.projection = ProjectionHead(hidden)

        def encode(self, input_ids, attention_mask=None, pattern_ids=None):
            if self._factored:
                return self.encoder(input_ids, attention_mask, pattern_ids)
            return self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state

        @staticmethod
        def mean_pool(hidden_states, attention_mask=None):
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            m = attention_mask.unsqueeze(-1).type_as(hidden_states)
            return (hidden_states * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

    def build_model(cfg: Config | None = None,
                    morph_specs: list[HeadSpec] | None = None) -> "SyriacEncoderModel":
        cfg = cfg or default_config()
        return SyriacEncoderModel(cfg.model, morph_specs)

else:  # pragma: no cover - exercised only without heavy deps

    def build_model(cfg: Config | None = None, morph_specs=None):
        raise RuntimeError(
            "torch + transformers are required to build models. Install:\n"
            "    .venv/bin/python -m pip install -r neural/requirements-neural.txt")


def info() -> str:
    lines = [
        "Syriac neural encoder (scaffold)",
        f"  heavy deps available : torch+transformers={_HF}, peft={_PEFT}",
        f"  pointing vocab size  : {len(POINTING_VOCAB)} -> {POINTING_VOCAB}",
        f"  default base model   : {default_config().model.base}",
        f"  factored (Twist 2)   : {default_config().model.factored_root_pattern}",
    ]
    if not _HF:
        lines.append("  NOTE: install neural/requirements-neural.txt to instantiate models.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect the modeling scaffold.")
    ap.add_argument("--info", action="store_true", help="print availability + defaults")
    args = ap.parse_args(argv)
    if args.info or True:
        print(info())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
