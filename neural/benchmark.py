#!/usr/bin/env python3
"""Evaluate a trained Syriac encoder on the paper's existing tasks.

The point of this adapter is comparability: a neural encoder should be scored on
exactly the metrics the published bake-off uses, so its numbers sit in the same
table as char-n-gram FastText, word2vec, Burrows's Delta, and the byte-LM /
char-Transformer baselines. To that end it imports the parent project's metric
helpers **read-only**:

* ``stylometry.remove_common_component`` / ``l2_normalize`` / ``separation`` --
  the anisotropy correction and same/cross-author AUC; and
* ``stylometry.load_texts`` -- the authorship cohort, tokenized identically.

A trained encoder is wrapped as an ``EncoderAdapter`` exposing ``doc_vector`` and
``word_vector`` -- the same surface the parent ``nn_baselines.NeuralEncoder``
presents -- so downstream code is representation-agnostic.

    # dependency-free description of the evaluation protocol
    .venv/bin/python -m neural.benchmark --plan
    # with deps + a checkpoint
    .venv/bin/python -m neural.benchmark --checkpoint ~/.cache/syriac-neural/checkpoints/syriac-encoder.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# read-only reuse of the parent metric stack (never modified)
from core.stylometry import l2_normalize, remove_common_component, separation

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


# Tasks this adapter can score, mirroring the paper's tables.
EVAL_TASKS = ("authorship_auc", "morphology_probe", "oov_generalization", "bits_per_byte")


class EncoderAdapter:
    """Wrap a trained encoder to expose doc/word vectors (parent-compatible).

    Mirrors ``nn_baselines.NeuralEncoder``: ``doc_vector(text)`` mean-pools the
    encoder's final hidden states over a document; ``word_vector(form)`` does the
    same over an isolated form (for the morphology probe).
    """

    def __init__(self, checkpoint: Path):
        if not _TORCH:
            raise RuntimeError(
                "torch is required to load a checkpoint. Install:\n"
                "    .venv/bin/python -m pip install -r neural/requirements-neural.txt")
        from neural.config import default_config
        from neural.modeling import build_model

        blob = torch.load(checkpoint, map_location="cpu")
        cfg = default_config()
        self.model = build_model(cfg)
        self.model.load_state_dict(blob["state_dict"], strict=False)
        self.model.eval()
        self._dim = getattr(self.model, "hidden", 768)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, text: str) -> np.ndarray:
        # Placeholder tokenization hook: a real run uses the base model's
        # tokenizer (CANINE = codepoints / ByT5 = bytes). Kept abstract so the
        # adapter shape is stable while Phase-1 wiring lands.
        raise NotImplementedError(
            "Tokenizer/encode wiring is completed in Phase 1 (see docs/ROADMAP.md).")

    def doc_vector(self, text: str) -> np.ndarray:
        return self._encode(text)

    def word_vector(self, form: str) -> np.ndarray:
        return self._encode(form)


def authorship_auc_from_matrix(matrix: np.ndarray, labels: np.ndarray) -> dict:
    """Apply the paper's exact pipeline: mean-center -> L2 -> same/cross AUC."""
    centered = l2_normalize(remove_common_component(matrix))
    return separation(centered, labels)


def describe_protocol() -> str:
    return "\n".join([
        "Evaluation protocol (matches the paper's harness)",
        "  authorship_auc     : doc vectors -> remove_common_component -> l2 ->",
        "                       same/cross-author AUC (author-cluster bootstrap)",
        "  morphology_probe   : word vectors for root-sharing vs control pairs (T3a)",
        "  oov_generalization : vectorize held-out OOV forms; root-NN rate (T3b)",
        "  bits_per_byte      : held-out bpb of the LM head (TLM)",
        "",
        "  Discipline: real-only val/test; vocalised/transliterated text never in",
        "  eval; provenance tags from aggregate.py block leakage; report deltas vs",
        "  a plain continued-pretrain baseline with the parent's bootstrap CIs.",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="trained encoder checkpoint (pretrain.py output)")
    ap.add_argument("--plan", action="store_true",
                    help="print the evaluation protocol and exit (no deps needed)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the AUC pipeline on synthetic vectors (no deps)")
    args = ap.parse_args(argv)

    if args.plan or (args.checkpoint is None and not args.selftest):
        print(describe_protocol())
        return 0

    if args.selftest:
        rng = np.random.default_rng(0)
        # Two well-separated author clusters -> AUC should be high.
        a = rng.normal(0.0, 0.1, size=(10, 16)) + np.r_[np.ones(8), np.zeros(8)]
        b = rng.normal(0.0, 0.1, size=(10, 16)) + np.r_[np.zeros(8), np.ones(8)]
        matrix = np.vstack([a, b])
        labels = np.array([0] * 10 + [1] * 10)
        res = authorship_auc_from_matrix(matrix, labels)
        auc = res["auc"]
        print(f"synthetic same/cross AUC = {auc:.3f} (expect > 0.9)")
        print("selftest:", "PASS" if auc > 0.9 else "FAIL")
        return 0 if auc > 0.9 else 1

    adapter = EncoderAdapter(args.checkpoint)
    print(f"loaded encoder (dim={adapter.dim}); per-task wiring completes in Phase 1.")
    print(describe_protocol())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
