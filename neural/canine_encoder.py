#!/usr/bin/env python3
"""CANINE-c as a tokenizer-free Syriac word/document encoder (Phase 1).

The neural continuation of the paper's subword thesis: CANINE operates directly
on Unicode codepoints, so every Syriac character is covered with **zero OOV** and
no tokenizer to train. This module wraps a (optionally LoRA-adapted) ``CanineModel``
as a drop-in replacement for the parent project's ``wv`` word-vector object, so a
neural encoder is scored on *exactly* the same authorship pipeline as char-n-gram
FastText, word2vec, and Burrows's Delta -- byte-for-byte the same cohort,
tokenizer, mean-centering, and same/cross-author AUC.

Requires the optional extras (``transformers``); see
``neural/requirements-neural.txt``. On a corporate-TLS machine the HuggingFace
download is verified through the OS trust store via ``truststore`` (injected at
import); no certificate verification is disabled.

    # off-the-shelf CANINE-c authorship AUC, comparable to Table 6
    .venv/bin/python -m neural.canine_encoder --floors 1000,2000

    # evaluate a LoRA-continued-pretrained checkpoint
    .venv/bin/python -m neural.canine_encoder --checkpoint ~/.cache/syriac-neural/checkpoints/canine-lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Verify TLS through the OS trust store before any HuggingFace network access
# (corporate MITM proxies present a CA that OpenSSL's bundle rejects; truststore
# delegates verification to the system store, exactly like curl). Best-effort.
try:  # pragma: no cover - environment dependent
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# read-only reuse of the parent cohort + metric stack (never modified)
from script import DEFAULT_CACHE, ensure_corpus
from stylometry import (
    load_texts, filter_min_texts, filter_min_tokens, remove_common_component,
    separation,
)
from authorship import doc_matrix, key_labels, parse_ids

try:
    import torch
    from transformers import CanineModel
    _HF = True
except Exception:  # pragma: no cover - heavy deps optional
    _HF = False

DEFAULT_BASE = "google/canine-c"
DISPUTED_DEFAULT = "690,219-227,519"   # same held-out set as the paper's bake-off
CLS, SEP, PAD = 0xE000, 0xE001, 0

# Optional codepoint normalization map (kept empty: the parent tokenizer already
# applied consonantal normalization upstream).


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if _HF:

    class CanineWordVectors:
        """A ``wv``-compatible, zero-OOV CANINE word encoder.

        Exposes ``vector_size``, ``__contains__`` (always True -- no form is OOV),
        and ``wv[token] -> np.ndarray`` (mean-pooled contextual encoding of the
        token's codepoints), with caching. This is the exact interface the
        parent ``doc_vector``/``doc_matrix`` expect, so the whole authorship
        pipeline is reused unchanged.
        """

        def __init__(self, model, device, batch_size: int = 128):
            self.model = model
            self.device = device
            self.batch_size = batch_size
            self.vector_size = int(model.config.hidden_size)
            self._cache: dict[str, np.ndarray] = {}

        def __contains__(self, token: str) -> bool:  # zero OOV by construction
            return True

        def _encode_batch(self, forms: list[str]) -> None:
            maxlen = max(len(f) for f in forms) + 2  # CLS + SEP
            ids = np.full((len(forms), maxlen), PAD, dtype=np.int64)
            mask = np.zeros((len(forms), maxlen), dtype=np.int64)
            for i, f in enumerate(forms):
                seq = [CLS] + [ord(c) for c in f] + [SEP]
                ids[i, :len(seq)] = seq
                mask[i, :len(seq)] = 1
            t_ids = torch.from_numpy(ids).to(self.device)
            t_mask = torch.from_numpy(mask).to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=t_ids, attention_mask=t_mask)
            h = out.last_hidden_state          # (B, T, H)
            m = t_mask.unsqueeze(-1).type_as(h)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            pooled = pooled.float().cpu().numpy().astype(np.float64)
            for f, v in zip(forms, pooled):
                self._cache[f] = v

        def precompute(self, forms) -> None:
            todo = sorted({f for f in forms if f and f not in self._cache}, key=len)
            for i in range(0, len(todo), self.batch_size):
                self._encode_batch(todo[i:i + self.batch_size])

        def __getitem__(self, token: str) -> np.ndarray:
            v = self._cache.get(token)
            if v is None:
                self._encode_batch([token])
                v = self._cache[token]
            return v

    def load_canine(checkpoint: str | None = None, base: str = DEFAULT_BASE):
        """Load CANINE-c, optionally applying a saved LoRA/continued-pretrain checkpoint."""
        model = CanineModel.from_pretrained(base)
        if checkpoint:
            ckpt = Path(checkpoint)
            if (ckpt / "adapter_config.json").exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(ckpt))
                model = model.merge_and_unload()
            elif ckpt.with_suffix(".pt").exists() or ckpt.suffix == ".pt":
                blob = torch.load(ckpt if ckpt.suffix == ".pt" else ckpt.with_suffix(".pt"),
                                  map_location="cpu")
                state = blob.get("encoder_state", blob.get("state_dict", blob))
                model.load_state_dict(state, strict=False)
            else:
                print(f"warning: no recognizable checkpoint at {ckpt}; using base.",
                      file=sys.stderr)
        model.eval()
        return model


def authorship_auc(wv, floors: list[int], normalize: bool = True) -> list[dict]:
    """Compute centered same/cross-author AUC for CANINE on each token floor,
    using the paper's exact cohort and metric."""
    data_dir = ensure_corpus(DEFAULT_CACHE)
    genuine = load_texts(data_dir, normalize,
                         exclude_ids=set(parse_ids(DISPUTED_DEFAULT)),
                         drop_anonymous=True)
    # warm the cache with every form that can appear in any cohort
    allforms = {tok for t in genuine for tok in t.counts}
    if hasattr(wv, "precompute"):
        print(f"encoding {len(allforms):,} unique forms with CANINE ...",
              file=sys.stderr)
        wv.precompute(allforms)

    results = []
    for floor in floors:
        cohort = filter_min_texts(filter_min_tokens(genuine, floor), 3)
        M, kept = doc_matrix(cohort, wv, None)
        if M is None or len(kept) < 4:
            results.append({"floor": floor, "auc": float("nan"),
                            "texts": 0, "authors": 0})
            continue
        labels = key_labels(kept)
        auc = separation(remove_common_component(M), labels)["auc"]
        results.append({"floor": floor, "auc": auc, "texts": len(kept),
                        "authors": len(set(labels.tolist()))})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floors", default="1000,2000",
                    help="comma-separated token floors (default 1000,2000)")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA/continued-pretrain checkpoint to evaluate")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args(argv)

    if not _HF:
        print("transformers + torch required. Install:\n"
              "    .venv/bin/python -m pip install -r neural/requirements-neural.txt",
              file=sys.stderr)
        return 2

    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    device = _device()
    print(f"loading {args.base}"
          + (f" (+ checkpoint {args.checkpoint})" if args.checkpoint else "")
          + f" on {device} ...", file=sys.stderr)
    model = load_canine(args.checkpoint, args.base).to(device)
    wv = CanineWordVectors(model, device)

    tag = "CANINE-c (continued-pretrained)" if args.checkpoint else "CANINE-c (off-the-shelf)"
    rows = authorship_auc(wv, floors, normalize=not args.no_normalize)
    print(f"\n=== authorship separation AUC: {tag} ===")
    print("(centered cosine; same cohort + metric as paper Table 6)")
    print(f"  {'floor':>6}  {'AUC':>6}  texts  authors")
    for r in rows:
        print(f"  {r['floor']:>6}  {r['auc']:.3f}   {r['texts']:>3}    {r['authors']:>2}")
    print("\nReference (paper Table 6): FastText 0.915/0.900, word2vec 0.946/0.938,")
    print("Delta 0.907/0.940, AV head 0.966/0.954 at floors 1000/2000.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
