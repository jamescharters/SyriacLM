#!/usr/bin/env python3
"""Clone (and cache) the srophe/syriac-corpus and report basic statistics.

The corpus is a collection of TEI XML files (one per text) under ``data/tei``.
This script clones the repository into a local cache the first time it runs and
reuses that cache on later runs (so it does not download the corpus every time).
It then reports:

* how many unique authors there are,
* how many texts each author has, and
* the overall Syriac vocabulary size (number of distinct word forms).

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_URL = "https://github.com/srophe/syriac-corpus.git"
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

DEFAULT_CACHE = Path.home() / ".cache" / "syriac-corpus"
UNATTRIBUTED = "(unattributed)"

# Runs of Syriac letters/marks plus the combining-diacritics block (U+0300-U+036F),
# which is where seyame (combining diaeresis, U+0308) lives in this corpus.
WORD_RE = re.compile(r"[\u0710-\u074F\u0300-\u036F]+")
# A "real" word must contain at least one Syriac *letter* (not just marks).
LETTER_RE = re.compile(r"[\u0710\u0712-\u072F\u074D-\u074F]")
# Invisible joiners / bidi controls that should not split or distinguish words.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF,
     0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
     0x2066, 0x2067, 0x2068, 0x2069],
    None,
)


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def run_git(args: list[str]) -> None:
    subprocess.run(["git", *args], check=True)


def ensure_corpus(cache_dir: Path, refresh: bool = False, update: bool = False) -> Path:
    """Make sure the corpus is available locally and return the data/tei dir."""
    if shutil.which("git") is None:
        sys.exit("error: 'git' is required but was not found on PATH.")

    if refresh and cache_dir.exists():
        print(f"Removing existing cache at {cache_dir} ...", file=sys.stderr)
        shutil.rmtree(cache_dir)

    if not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {REPO_URL} into {cache_dir} ...", file=sys.stderr)
        run_git(["clone", "--depth", "1", REPO_URL, str(cache_dir)])
    elif update:
        print(f"Updating cached corpus in {cache_dir} ...", file=sys.stderr)
        run_git(["-C", str(cache_dir), "pull", "--ff-only"])
    else:
        print(f"Using cached corpus at {cache_dir}", file=sys.stderr)

    data_dir = cache_dir / "data" / "tei"
    if not data_dir.is_dir():
        sys.exit(f"error: expected TEI files in {data_dir}, but it does not exist.")
    return data_dir


def extract_authors(root: ET.Element) -> list[tuple[str, str]]:
    """Return a list of (key, display_name) for the work's author(s)."""
    authors: list[tuple[str, str]] = []
    for el in root.findall(".//tei:fileDesc/tei:titleStmt/tei:author", NS):
        name = normalize_space("".join(el.itertext()))
        ref = (el.get("ref") or "").strip()
        if not name and not ref:
            continue
        key = ref or name.casefold()
        authors.append((key, name or ref))
    return authors


def extract_title(root: ET.Element) -> str:
    titles = root.findall(".//tei:fileDesc/tei:titleStmt/tei:title", NS)
    for el in titles:
        if el.get("level") == "a":
            return normalize_space("".join(el.itertext()))
    return normalize_space("".join(titles[0].itertext())) if titles else ""


def extract_series(root: ET.Element) -> str:
    """Return the work's series title (TEI title level="s"), if any.

    The corpus boilerplate series "Digital Syriac Corpus" is ignored, so this
    returns the meaningful collection a text belongs to (e.g. "Hymns on
    Nativity", "Prose Refutations of Mani, Marcion, and Bardaisan"), which is the
    only place genre is recoverable in these headers.
    """
    for el in root.findall(".//tei:fileDesc/tei:titleStmt/tei:title", NS):
        if el.get("level") == "s":
            text = normalize_space("".join(el.itertext()))
            if text and text != "Digital Syriac Corpus":
                return text
    return ""


def find_body(root: ET.Element) -> ET.Element | None:
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        body = root.find(".//tei:body", NS)
    return body


def iter_words(body: ET.Element):
    for chunk in body.itertext():
        if not chunk:
            continue
        chunk = chunk.translate(_INVISIBLE)
        for token in WORD_RE.findall(chunk):
            if LETTER_RE.search(token):
                yield token


def iter_words_text(text: str):
    """Tokenize a plain Unicode-Syriac string with the same rules as iter_words.

    Used for corpora distributed as plain ``.txt`` (e.g. the ETCBC datasets)
    rather than TEI. Latin digits, verse numbers and references are ignored
    because only runs containing a Syriac letter are yielded.
    """
    if not text:
        return
    text = text.translate(_INVISIBLE)
    for token in WORD_RE.findall(text):
        if LETTER_RE.search(token):
            yield token


def strip_marks(token: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", token) if not unicodedata.combining(ch)
    )


def collect_stats(data_dir: Path):
    files = sorted(data_dir.glob("*.xml"))
    author_text_counts: Counter[str] = Counter()
    author_names: dict[str, str] = {}
    vocab: set[str] = set()
    total_tokens = 0
    multi_author_texts = 0
    parse_errors: list[tuple[str, str]] = []
    parsed = 0

    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            parse_errors.append((path.name, str(exc)))
            continue
        parsed += 1

        authors = extract_authors(root) or [(UNATTRIBUTED, UNATTRIBUTED)]
        if len(authors) > 1:
            multi_author_texts += 1
        for key in {k for k, _ in authors}:
            author_text_counts[key] += 1
        for key, name in authors:
            author_names.setdefault(key, name)

        body = find_body(root)
        if body is not None:
            for token in iter_words(body):
                total_tokens += 1
                vocab.add(token)

    # Some files name an author (e.g. "Ephrem the Syrian") without the
    # syriaca.org @ref that the same author carries elsewhere. Merge those
    # name-only entries into their URI-identified counterpart so each real
    # author is counted once.
    uri_by_name = {}
    for key in author_text_counts:
        if key.startswith("http"):
            uri_by_name.setdefault(author_names.get(key, "").casefold(), key)
    for key in list(author_text_counts):
        if key.startswith("http") or key == UNATTRIBUTED:
            continue
        target = uri_by_name.get(author_names.get(key, "").casefold())
        if target and target != key:
            author_text_counts[target] += author_text_counts.pop(key)
            author_names.pop(key, None)

    vocab_normalized = {strip_marks(tok) for tok in vocab}

    return {
        "files": len(files),
        "parsed": parsed,
        "parse_errors": parse_errors,
        "author_text_counts": author_text_counts,
        "author_names": author_names,
        "multi_author_texts": multi_author_texts,
        "vocab": vocab,
        "vocab_normalized": vocab_normalized,
        "total_tokens": total_tokens,
    }


def report(stats: dict, data_dir: Path, top: int) -> None:
    counts: Counter[str] = stats["author_text_counts"]
    names: dict[str, str] = stats["author_names"]

    attributed_keys = [k for k in counts if k != UNATTRIBUTED]
    unattributed = counts.get(UNATTRIBUTED, 0)

    print()
    print("=" * 70)
    print("Syriac corpus statistics (srophe/syriac-corpus)")
    print("=" * 70)
    print(f"Corpus location : {data_dir}")
    print(f"TEI files found : {stats['files']}")
    print(f"Parsed OK       : {stats['parsed']}")
    if stats["parse_errors"]:
        print(f"Parse errors    : {len(stats['parse_errors'])}")
        for name, msg in stats["parse_errors"]:
            print(f"    - {name}: {msg}")

    print()
    print(f"Unique authors  : {len(attributed_keys)}")
    if unattributed:
        print(f"Unattributed texts (no <author>): {unattributed}")
    if stats["multi_author_texts"]:
        print(f"Texts crediting more than one author: {stats['multi_author_texts']}")
        print("(such texts are counted once for each of their authors below)")

    print()
    print("-" * 70)
    print("Texts per author (descending)")
    print("-" * 70)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], names.get(kv[0], "").casefold()))
    if top > 0:
        ordered = ordered[:top]
    width = max((len(str(c)) for _, c in ordered), default=1)
    for key, count in ordered:
        name = names.get(key, key)
        ref = key if key.startswith("http") else ""
        suffix = f"  <{ref}>" if ref else ""
        print(f"  {count:>{width}}  {name}{suffix}")

    print()
    print("-" * 70)
    print("Vocabulary")
    print("-" * 70)
    print(f"Total Syriac word tokens          : {stats['total_tokens']:,}")
    print(f"Distinct word forms (surface)     : {len(stats['vocab']):,}")
    print(f"Distinct word forms (no diacritics): {len(stats['vocab_normalized']):,}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"where to clone/cache the corpus (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="delete the cache and re-clone before computing stats")
    parser.add_argument("--update", action="store_true",
                        help="git pull the cached corpus before computing stats")
    parser.add_argument("--top", type=int, default=0,
                        help="only show the N authors with the most texts (default: show all)")
    args = parser.parse_args(argv)

    data_dir = ensure_corpus(args.cache_dir, refresh=args.refresh, update=args.update)
    stats = collect_stats(data_dir)
    report(stats, data_dir, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
