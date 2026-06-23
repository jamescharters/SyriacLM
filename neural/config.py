#!/usr/bin/env python3
"""Configuration dataclasses for the Syriac neural sub-project.

Pure standard library: importing or running this module needs no heavy
dependencies, so it doubles as a quick check that the package is importable and
as living documentation of the chosen defaults.

    .venv/bin/python -m neural.config        # prints the default configs

The defaults encode the design decisions argued in ``docs/DESIGN.md``:

* a *byte/character* base model (OOV-free, the neural continuation of the paper's
  subword thesis);
* *parameter-efficient* continued pretraining (LoRA), the methodologically
  correct choice at tens-of-millions of tokens;
* *consonantal* normalization matching the parent pipeline, except where the
  pointing-restoration objective deliberately needs the diacritics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# Default cache for aggregated corpora and checkpoints. Kept under ~/.cache (not
# in the repo) like the parent project's corpus/model caches.
DEFAULT_NEURAL_CACHE = Path.home() / ".cache" / "syriac-neural"


@dataclass
class DataConfig:
    """Where the text comes from and how it is split.

    ``sources`` names the corpora aggregate.py will assemble. ``dsc`` is the
    Digital Syriac Corpus (parent cache); ``syrnt``/``peshitta`` are the ETCBC
    corpora the parent already knows how to clone. SEDRA is handled separately
    (license-restricted; see sedra.py) and is never written into shared shards.
    """

    out_dir: Path = DEFAULT_NEURAL_CACHE
    sources: tuple[str, ...] = ("dsc", "syrnt", "peshitta")
    normalize: bool = True            # strip combining diacritics (parent default)
    # Leakage-safe split is by *document* (never by line), so no document
    # contributes to more than one split.
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    split_seed: int = 42
    # Near-duplicate filtering (biblical/liturgical text repeats heavily).
    dedup_shingle: int = 8            # word-shingle size for Jaccard dedup
    dedup_threshold: float = 0.8      # drop a doc if Jaccard >= this vs a kept doc
    min_doc_tokens: int = 20


@dataclass
class ModelConfig:
    """The base checkpoint and how it is adapted.

    ``base`` defaults to CANINE-c: a tokenizer-free encoder over Unicode
    codepoints, so every Syriac character is covered with zero OOV. Alternatives
    discussed in docs: ``google/byt5-small`` (byte seq2seq) and a Semitic warm
    start via transliteration (see transliterate.py).
    """

    base: str = "google/canine-c"
    # Parameter-efficient finetuning keeps us from estimating hundreds of millions
    # of parameters from tens of millions of tokens.
    peft: str = "lora"               # one of: "lora", "full", "none"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 512
    # Twist 2: a factored representation with an explicit consonantal-skeleton
    # (root) channel and a vocalic-pattern channel. When False, a plain flat
    # encoder is used (the ablation baseline).
    factored_root_pattern: bool = False
    pattern_channel_dim: int = 64


@dataclass
class PretrainConfig:
    """The continued-pretraining objective(s) and optimization budget.

    Two objectives can be combined (multitask):
      * ``mlm`` -- masked-language / span-corruption denoising on running text;
      * ``pointing`` -- Twist 1: restore the vocalisation diacritics of a
        vocalised word given its consonantal skeleton (supervised by SEDRA).
    """

    objectives: tuple[str, ...] = ("mlm", "pointing")
    mlm_probability: float = 0.15
    # For the pointing objective we mask ONLY the diacritics, keeping the
    # consonantal skeleton visible -- structured denoising, not random masking.
    pointing_mask_all: bool = True
    # Optional morphology multitask heads supervised by SEDRA/ETCBC labels.
    morphology_tasks: tuple[str, ...] = ()   # e.g. ("root", "state", "tense")
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    out_dir: Path = DEFAULT_NEURAL_CACHE / "checkpoints"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def default_config() -> Config:
    return Config()


def _format(cfg: Config) -> str:
    import json

    def _coerce(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj))

    return json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=_coerce)


if __name__ == "__main__":
    print(_format(default_config()))
