#!/usr/bin/env python3
"""Train a character n-gram FastText model on the srophe/syriac-corpus and probe
its morphological coherence.

FastText represents every word as a bag of character n-grams, so two Syriac
forms that share a triconsonantal root -- for example ܡܠܟܐ "king" and ܡܠܟܘܬܐ
"kingdom" (both built on the root m-l-k) -- end up with similar vectors even
when one of the forms is rare. This script:

  1. builds a training corpus from the cached TEI files (reusing the Syriac
     tokenizer and corpus helpers from script.py),
  2. trains a gensim FastText model with character n-grams and min_count=1, and
  3. runs a morphological-coherence test that checks whether root-sharing word
     pairs (king/kingdom, write/book) sit closer in vector space than
     semantically-adjacent but morphologically-unrelated control pairs
     (king/father, write/read).

Requires gensim (``pip install gensim``). Only the corpus tokenizer is shared
with the other scripts in this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from script import DEFAULT_CACHE, ensure_corpus, find_body, iter_words, strip_marks

# The user's requirement: every form counts, no minimum-frequency cutoff.
MIN_COUNT = 1

# Canonical Syriac surface forms (consonantal skeletons) for the probe words.
# Each value is a list of candidate spellings; the first one present in the
# trained vocabulary is used. Approximate frequencies in the cached corpus:
#   ܡܠܟܐ king 3243 | ܡܠܟܘܬܐ kingdom 916 | ܐܒܐ father 1069
#   ܟܬܒܐ book 909  | ܟܬܒ write 371      | ܩܪܐ read 1779
CONCEPTS: dict[str, list[str]] = {
    "king":    ["ܡܠܟܐ"],
    "kingdom": ["ܡܠܟܘܬܐ"],
    "father":  ["ܐܒܐ"],
    "book":    ["ܟܬܒܐ"],
    "write":   ["ܟܬܒ"],
    "read":    ["ܩܪܐ"],
}

GLOSS = {
    "king": "king (malka)",
    "kingdom": "kingdom (malkuta)",
    "father": "father (aba)",
    "book": "book (ktaba)",
    "write": "write / he wrote (ktab)",
    "read": "read / he read (qra)",
}

# Each case pits a morphologically related pair (which shares a Semitic root)
# against a semantically adjacent control pair built on a *different* root.
# A coherent subword model should score the related pair higher than the control.
COHERENCE_CASES = [
    {"root": "m-l-k (ܡܠܟ)", "related": ("king", "kingdom"), "control": ("king", "father")},
    {"root": "k-t-b (ܟܬܒ)", "related": ("write", "book"), "control": ("write", "read")},
]


def build_sentences(data_dir: Path, normalize: bool, limit: int = 0):
    """Return (sentences, files_considered, unparseable). One sentence per file."""
    sentences: list[list[str]] = []
    paths = sorted(data_dir.glob("*.xml"))
    if limit:
        paths = paths[:limit]
    skipped = 0
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            skipped += 1
            continue
        body = find_body(root)
        if body is None:
            continue
        tokens = [strip_marks(t) if normalize else t for t in iter_words(body)]
        if tokens:
            sentences.append(tokens)
    return sentences, len(paths), skipped


def train_model(sentences, *, dim, window, epochs, min_n, max_n, buckets, sg, workers, seed):
    """Train and return a gensim FastText model (character n-grams, min_count=1)."""
    try:
        from gensim.models import FastText
    except ImportError:
        sys.exit("error: gensim is required to train the model (pip install gensim).")

    model = FastText(
        vector_size=dim,
        window=window,
        min_count=MIN_COUNT,
        min_n=min_n,            # smallest character n-gram
        max_n=max_n,            # largest character n-gram
        bucket=buckets,         # hashing space for the n-gram vectors
        sg=sg,                  # 1 = skip-gram (better for morphology/small data)
        workers=workers,
        seed=seed,
        epochs=epochs,
    )
    model.build_vocab(corpus_iterable=sentences)

    # gensim 4.x compiled against numpy 2.x emits a benign, unraisable
    # "Exception ignored in: 'gensim...our_dot_float'" line from its Cython BLAS
    # wrapper once per worker batch. It does not affect the trained vectors, so
    # we capture stderr during training and forward only the non-noise lines.
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=model.epochs)
    for line in buf.getvalue().splitlines():
        if "our_dot" in line or "word2vec_inner" in line:
            continue
        print(line, file=sys.stderr)
    return model


def resolve_form(model, candidates: list[str], normalize: bool) -> tuple[str, bool]:
    """Pick the first candidate present in the vocabulary (normalizing to match)."""
    forms = [strip_marks(c) if normalize else c for c in candidates]
    for form in forms:
        if form in model.wv.key_to_index:
            return form, True
    return forms[0], False


def vocab_count(model, form: str) -> int:
    try:
        return int(model.wv.get_vecattr(form, "count"))
    except KeyError:
        return 0


def morphological_coherence_test(model, normalize: bool, topn: int = 8) -> bool:
    """Print the coherence report; return True if every case passes."""
    print()
    print("=" * 70)
    print("Morphological coherence test")
    print("=" * 70)

    resolved: dict[str, str] = {}
    print("Probe words (resolved against the trained vocabulary):")
    for concept, candidates in CONCEPTS.items():
        form, in_vocab = resolve_form(model, candidates, normalize)
        resolved[concept] = form
        status = f"count={vocab_count(model, form):,}" if in_vocab else "OOV (vector from n-grams only)"
        print(f"  {GLOSS[concept]:<26} -> {form}   [{status}]")
    print()

    def sim(a: str, b: str) -> float:
        return float(model.wv.similarity(resolved[a], resolved[b]))

    print("-" * 70)
    print("Root-sharing pair vs. semantically-adjacent control pair")
    print("-" * 70)
    passed = 0
    for case in COHERENCE_CASES:
        ra, rb = case["related"]
        ca, cb = case["control"]
        s_rel, s_ctl = sim(ra, rb), sim(ca, cb)
        ok = s_rel > s_ctl
        passed += ok
        print(f"root {case['root']}")
        print(f"    related  cos({ra:<7}, {rb:<7}) = {s_rel:+.3f}   (shared root)")
        print(f"    control  cos({ca:<7}, {cb:<7}) = {s_ctl:+.3f}   (different root)")
        print(f"    => {'PASS' if ok else 'FAIL'}: related pair is "
              f"{'closer' if ok else 'NOT closer'} (margin {s_rel - s_ctl:+.3f})")
        print()

    print("-" * 70)
    print(f"Nearest neighbours (top {topn}) -- a qualitative view of each root family")
    print("-" * 70)
    for concept in CONCEPTS:
        form = resolved[concept]
        try:
            neighbours = model.wv.most_similar(form, topn=topn)
        except KeyError:
            print(f"  {GLOSS[concept]:<26} {form}: (OOV, no neighbours)")
            continue
        joined = "  ".join(f"{w}:{s:.2f}" for w, s in neighbours)
        print(f"  {GLOSS[concept]:<26} {form}")
        print(f"      {joined}")
    print()

    total = len(COHERENCE_CASES)
    verdict = ("all root-sharing pairs beat their controls" if passed == total
               else "some pairs did not separate")
    print("=" * 70)
    print(f"Coherence result: {passed}/{total} cases passed ({verdict})")
    print("=" * 70)
    return passed == total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"where the corpus is cached (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="delete the cache and re-clone before training")
    parser.add_argument("--update", action="store_true",
                        help="git pull the cached corpus before training")

    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true",
                      help="strip diacritics before training (default; aligns forms by root)")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false",
                      help="train on surface forms, keeping all diacritics")
    parser.set_defaults(normalize=True)

    parser.add_argument("--dim", type=int, default=100, help="embedding dimensionality")
    parser.add_argument("--window", type=int, default=5, help="context window size")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--min-n", type=int, default=2, help="smallest character n-gram")
    parser.add_argument("--max-n", type=int, default=5, help="largest character n-gram")
    parser.add_argument("--buckets", type=int, default=200_000, help="n-gram hashing buckets")
    parser.add_argument("--cbow", action="store_true", help="use CBOW instead of skip-gram")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="training worker threads")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--limit", type=int, default=0,
                        help="train on only the first N TEI files (0 = all; for quick smoke tests)")
    parser.add_argument("--save", type=Path, default=None,
                        help="save the trained model to this path")
    parser.add_argument("--topn", type=int, default=8, help="neighbours to show per probe word")
    args = parser.parse_args(argv)

    data_dir = ensure_corpus(args.cache_dir, refresh=args.refresh, update=args.update)

    print(f"Building corpus from {data_dir} ...", file=sys.stderr)
    sentences, n_files, skipped = build_sentences(data_dir, args.normalize, args.limit)
    n_tokens = sum(len(s) for s in sentences)
    detail = f" ({skipped} unparseable)" if skipped else ""
    print(f"  documents: {len(sentences):,} / {n_files:,} files{detail} | "
          f"tokens: {n_tokens:,} | mode: "
          f"{'diacritic-stripped' if args.normalize else 'surface forms'}", file=sys.stderr)
    if not sentences:
        sys.exit("error: no training data found.")

    print(f"Training FastText (char n-grams {args.min_n}-{args.max_n}, dim={args.dim}, "
          f"{'CBOW' if args.cbow else 'skip-gram'}, min_count={MIN_COUNT}, "
          f"epochs={args.epochs}) ...", file=sys.stderr)
    model = train_model(
        sentences,
        dim=args.dim, window=args.window, epochs=args.epochs,
        min_n=args.min_n, max_n=args.max_n, buckets=args.buckets,
        sg=0 if args.cbow else 1, workers=args.workers, seed=args.seed,
    )
    print(f"  vocabulary: {len(model.wv):,} forms | n-gram buckets: {args.buckets:,}",
          file=sys.stderr)

    if args.save:
        model.save(str(args.save))
        print(f"  saved model to {args.save}", file=sys.stderr)

    ok = morphological_coherence_test(model, args.normalize, topn=args.topn)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
