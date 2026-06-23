#!/usr/bin/env python3
"""Frequency statistics for the srophe/syriac-corpus vocabulary.

Reuses the corpus cache and Syriac tokenizer from ``script.py`` and reports:

* total tokens,
* unique forms (vocabulary size),
* hapax legomena (forms occurring exactly once),
* rare forms (forms occurring 5 times or fewer), and
* mean frequency (total tokens / unique forms).

By default these are computed over surface word forms; pass ``--normalize`` to
fold away diacritics (seyame, vowel points, etc.) first.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from script import DEFAULT_CACHE, ensure_corpus, find_body, iter_words, strip_marks


def build_frequencies(data_dir: Path, normalize: bool) -> tuple[Counter[str], int]:
    """Return (form -> count, number of files parsed)."""
    freq: Counter[str] = Counter()
    parsed = 0
    for path in sorted(data_dir.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        parsed += 1
        body = find_body(root)
        if body is None:
            continue
        for token in iter_words(body):
            freq[strip_marks(token) if normalize else token] += 1
    return freq, parsed


def compute_stats(freq: Counter[str]) -> dict[str, float | int]:
    total_tokens = sum(freq.values())
    unique_forms = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    rare = sum(1 for c in freq.values() if c <= 5)
    mean_frequency = total_tokens / unique_forms if unique_forms else 0.0
    return {
        "total_tokens": total_tokens,
        "unique_forms": unique_forms,
        "hapax": hapax,
        "rare": rare,
        "mean_frequency": mean_frequency,
    }


def report(stats: dict, data_dir: Path, parsed: int, normalize: bool) -> None:
    unique = stats["unique_forms"] or 1
    print()
    print("=" * 60)
    print("Syriac corpus frequency statistics")
    print("=" * 60)
    print(f"Corpus location : {data_dir}")
    print(f"Files parsed    : {parsed}")
    print(f"Form counting   : {'diacritic-stripped' if normalize else 'surface forms'}")
    print("-" * 60)
    print(f"Total tokens            : {stats['total_tokens']:>12,}")
    print(f"Unique forms            : {stats['unique_forms']:>12,}")
    print(f"Hapax legomena (n=1)    : {stats['hapax']:>12,}"
          f"  ({stats['hapax'] / unique:6.2%} of vocab)")
    print(f"Rare forms (n<=5)       : {stats['rare']:>12,}"
          f"  ({stats['rare'] / unique:6.2%} of vocab)")
    print(f"Mean frequency          : {stats['mean_frequency']:>12.2f}  tokens/form")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"where the corpus is cached (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="delete the cache and re-clone before computing stats")
    parser.add_argument("--update", action="store_true",
                        help="git pull the cached corpus before computing stats")
    parser.add_argument("--normalize", action="store_true",
                        help="strip diacritics before counting word forms")
    args = parser.parse_args(argv)

    data_dir = ensure_corpus(args.cache_dir, refresh=args.refresh, update=args.update)
    freq, parsed = build_frequencies(data_dir, args.normalize)
    stats = compute_stats(freq)
    report(stats, data_dir, parsed, args.normalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
