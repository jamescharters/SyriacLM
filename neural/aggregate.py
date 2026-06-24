#!/usr/bin/env python3
"""Aggregate real Syriac corpora into leakage-safe, deduplicated shards.

Phase-0 of the neural plan and the one step that genuinely increases the token
count: assemble every openly-licensed Syriac corpus we already have access to
into normalized, provenance-tagged documents, drop near-duplicates (biblical and
liturgical text repeats heavily), and split **by document** so no text leaks
across train/val/test.

Reuse, not reimplementation
---------------------------
To guarantee that a neural model is compared against the released FastText
baseline on byte-for-byte the same tokenization, this module imports the parent
project's tokenizer and corpus loaders **read-only** and never modifies them:

* ``script.ensure_corpus`` / ``find_body`` / ``iter_words`` -- the Digital Syriac
  Corpus (TEI), tokenized exactly as in the paper;
* ``script.iter_words_text`` + ``etcbc_corpus.ensure_etcbc`` -- the ETCBC SyrNT
  and Peshitta plain-text corpora, in original order (the parent's
  ``load_etcbc_texts`` returns unordered counts, so we read the plain files here
  to preserve running text for language modeling).

Vocalisation note: running text in all three corpora is consonantal. The
vocalisation supervision for the pointing objective comes from SEDRA
(``sedra.py``), not from here -- so this aggregator deliberately produces the
same consonantal text the baseline sees.

    .venv/bin/python -m neural.aggregate --out ~/.cache/syriac-neural
    .venv/bin/python -m neural.aggregate --sources dsc --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

# --- read-only reuse of the parent project (never modified) ---------------- #
from core.script import (
    DEFAULT_CACHE,
    ensure_corpus,
    find_body,
    iter_words,
    iter_words_text,
    strip_marks,
)

try:
    from core.etcbc_corpus import ETCBC_SOURCES, ensure_etcbc
    _ETCBC = True
except Exception:  # pragma: no cover - etcbc optional
    _ETCBC = False

from neural.config import DEFAULT_NEURAL_CACHE


@dataclass
class Document:
    doc_id: str
    provenance: str
    tokens: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


# --------------------------------------------------------------------------- #
# Source loaders (each best-effort: a failing source warns and is skipped)
# --------------------------------------------------------------------------- #
def load_dsc(normalize: bool) -> list[Document]:
    data_dir = ensure_corpus(DEFAULT_CACHE)
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        body = find_body(root)
        if body is None:
            continue
        toks = [strip_marks(t) if normalize else t for t in iter_words(body)]
        if toks:
            docs.append(Document(doc_id=f"dsc:{path.stem}", provenance="dsc", tokens=toks))
    return docs


def load_etcbc(name: str, normalize: bool) -> list[Document]:
    if not _ETCBC or name not in ETCBC_SOURCES:
        return []
    plain = ensure_etcbc(name)
    docs: list[Document] = []
    for path in sorted(plain.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        toks = [strip_marks(t) if normalize else t for t in iter_words_text(raw)]
        if toks:
            docs.append(Document(doc_id=f"{name.lower()}:{path.stem}",
                                 provenance=name.lower(), tokens=toks))
    return docs


_SOURCE_FNS = {
    "dsc": lambda norm: load_dsc(norm),
    "syrnt": lambda norm: load_etcbc("SyrNT", norm),
    "peshitta": lambda norm: load_etcbc("Peshitta", norm),
}


def collect(sources: tuple[str, ...], normalize: bool) -> list[Document]:
    docs: list[Document] = []
    for name in sources:
        fn = _SOURCE_FNS.get(name)
        if fn is None:
            print(f"warning: unknown source {name!r}; skipping", file=sys.stderr)
            continue
        try:
            got = fn(normalize)
        except SystemExit as exc:           # ensure_* calls sys.exit on hard errors
            print(f"warning: source {name!r} unavailable ({exc}); skipping",
                  file=sys.stderr)
            got = []
        except Exception as exc:            # pragma: no cover - defensive
            print(f"warning: source {name!r} failed ({exc!r}); skipping",
                  file=sys.stderr)
            got = []
        print(f"  {name:10} {len(got):4d} docs, "
              f"{sum(d.n_tokens for d in got):>9,} tokens", file=sys.stderr)
        docs.extend(got)
    return docs


# --------------------------------------------------------------------------- #
# Near-duplicate filtering (word-shingle Jaccard)
# --------------------------------------------------------------------------- #
def _shingles(tokens: list[str], k: int) -> frozenset[int]:
    if len(tokens) < k:
        return frozenset({hash(tuple(tokens))}) if tokens else frozenset()
    return frozenset(
        hash(tuple(tokens[i:i + k])) for i in range(len(tokens) - k + 1)
    )


def dedup(docs: list[Document], k: int, threshold: float) -> tuple[list[Document], int]:
    """Greedy near-duplicate removal: drop a doc if its shingle-Jaccard against
    any already-kept doc is >= ``threshold``. Returns (kept, n_dropped)."""
    kept: list[Document] = []
    kept_sh: list[frozenset[int]] = []
    dropped = 0
    # Process longer documents first so we keep the fuller witness of a text.
    for doc in sorted(docs, key=lambda d: d.n_tokens, reverse=True):
        sh = _shingles(doc.tokens, k)
        is_dup = False
        for other in kept_sh:
            if not sh or not other:
                continue
            inter = len(sh & other)
            if inter == 0:
                continue
            union = len(sh | other)
            if union and inter / union >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
        else:
            kept.append(doc)
            kept_sh.append(sh)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Leakage-safe split (by document)
# --------------------------------------------------------------------------- #
def _bucket(doc_id: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split(docs: list[Document], val_frac: float, test_frac: float,
          seed: int) -> dict[str, list[Document]]:
    splits: dict[str, list[Document]] = {"train": [], "val": [], "test": []}
    for doc in docs:
        r = _bucket(doc.doc_id, seed)
        if r < test_frac:
            splits["test"].append(doc)
        elif r < test_frac + val_frac:
            splits["val"].append(doc)
        else:
            splits["train"].append(doc)
    return splits


def _assert_no_leakage(splits: dict[str, list[Document]]) -> None:
    seen: dict[str, str] = {}
    for name, docs in splits.items():
        for d in docs:
            if d.doc_id in seen:
                raise AssertionError(
                    f"leakage: {d.doc_id} in both {seen[d.doc_id]} and {name}")
            seen[d.doc_id] = name


# --------------------------------------------------------------------------- #
# Writing + reporting
# --------------------------------------------------------------------------- #
def write_shards(splits: dict[str, list[Document]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"splits": {}, "by_provenance": {}}
    for name, docs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(
                    {"doc_id": d.doc_id, "provenance": d.provenance,
                     "n_tokens": d.n_tokens, "text": d.text},
                    ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "docs": len(docs),
            "tokens": sum(d.n_tokens for d in docs),
        }
    # provenance tallies over everything
    alldocs = [d for ds in splits.values() for d in ds]
    provs = sorted({d.provenance for d in alldocs})
    for p in provs:
        sel = [d for d in alldocs if d.provenance == p]
        manifest["by_provenance"][p] = {
            "docs": len(sel), "tokens": sum(d.n_tokens for d in sel)}
    types = len({t for d in alldocs for t in d.tokens})
    manifest["total"] = {
        "docs": len(alldocs),
        "tokens": sum(d.n_tokens for d in alldocs),
        "types": types,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_NEURAL_CACHE,
                    help="output directory for shards (default: ~/.cache/syriac-neural)")
    ap.add_argument("--sources", default="dsc,syrnt,peshitta",
                    help="comma-separated subset of: dsc,syrnt,peshitta")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep combining diacritics (default: strip, matching the paper)")
    ap.add_argument("--shingle", type=int, default=8, help="word-shingle size for dedup")
    ap.add_argument("--dedup-threshold", type=float, default=0.8,
                    help="Jaccard threshold to treat docs as duplicates")
    ap.add_argument("--min-tokens", type=int, default=20, help="drop docs shorter than this")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="report statistics but do not write shards")
    args = ap.parse_args(argv)

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    normalize = not args.no_normalize

    print("Collecting sources ...", file=sys.stderr)
    docs = collect(sources, normalize)
    docs = [d for d in docs if d.n_tokens >= args.min_tokens]
    if not docs:
        print("error: no documents collected (are the corpora cached / is git "
              "available?).", file=sys.stderr)
        return 2

    kept, dropped = dedup(docs, args.shingle, args.dedup_threshold)
    splits = split(kept, args.val_fraction, args.test_fraction, args.seed)
    _assert_no_leakage(splits)

    total_tok = sum(d.n_tokens for d in kept)
    types = len({t for d in kept for t in d.tokens})
    print("\n=== aggregation summary ===")
    print(f"documents:     {len(docs):,} collected -> {len(kept):,} kept "
          f"({dropped:,} near-duplicates dropped)")
    print(f"tokens:        {total_tok:,}")
    print(f"types:         {types:,}")
    for name in ("train", "val", "test"):
        ds = splits[name]
        print(f"  {name:5}: {len(ds):4d} docs, {sum(d.n_tokens for d in ds):>9,} tokens")
    print("leakage check: PASS (no document in more than one split)")

    if args.dry_run:
        print("\n(dry run -- no shards written)")
        return 0

    manifest = write_shards(splits, args.out)
    print(f"\nwrote shards + manifest.json to {args.out}")
    print(json.dumps(manifest["total"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
