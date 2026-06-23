#!/usr/bin/env python3
"""Load independent ETCBC Syriac corpora as cross-corpus validation sets.

The Digital Syriac Corpus (used for training) is a heterogeneous collection of
original compositions. To test that the released FastText model and the
stylometric findings generalize, we use two *independent*, openly licensed
corpora distributed by the Eep Talstra Centre for Bible and Computer (ETCBC):

  * the Syriac New Testament (SyrNT) -- a translation from Greek; and
  * the Peshitta Old Testament      -- a translation from Hebrew.

Both ship as plain Unicode-Syriac ``.txt`` files (one per biblical book), in the
same script and (consonantal, unvocalized) orthography as the training corpus, so
they are processed with the same tokenizer. They are biblical *translations*,
hence add register and known-translation reference points rather than new
authors; they are used for out-of-vocabulary coverage and as external anchors for
the translationese analysis, not to enlarge the author pool.

Data: https://github.com/ETCBC/syrnt (MIT), https://github.com/ETCBC/peshitta (MIT).
"""

from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

from script import iter_words_text, run_git, strip_marks
from stylometry import Text

DEFAULT_ETCBC_CACHE = Path.home() / ".cache"

# name -> (git repo, plain-text subdirectory, display label)
ETCBC_SOURCES: dict[str, tuple[str, str, str]] = {
    "SyrNT": ("https://github.com/ETCBC/syrnt.git", "plain/0.1",
              "Syriac New Testament (translation from Greek)"),
    "Peshitta": ("https://github.com/ETCBC/peshitta.git", "plain/0.2",
                 "Peshitta Old Testament (translation from Hebrew)"),
}


def ensure_etcbc(name: str, cache_dir: Path = DEFAULT_ETCBC_CACHE,
                 refresh: bool = False) -> Path:
    """Clone the ETCBC corpus if needed and return its plain-text directory."""
    if name not in ETCBC_SOURCES:
        raise ValueError(f"unknown ETCBC corpus {name!r}; "
                         f"choose from {sorted(ETCBC_SOURCES)}")
    if shutil.which("git") is None:
        sys.exit("error: 'git' is required but was not found on PATH.")
    repo, subdir, _ = ETCBC_SOURCES[name]
    dest = cache_dir / f"etcbc-{name.lower()}"
    if refresh and dest.exists():
        shutil.rmtree(dest)
    if not dest.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {repo} into {dest} ...", file=sys.stderr)
        run_git(["clone", "--depth", "1", repo, str(dest)])
    else:
        print(f"Using cached ETCBC corpus at {dest}", file=sys.stderr)
    plain = dest / subdir
    if not plain.is_dir():
        sys.exit(f"error: expected plain text in {plain}, but it does not exist.")
    return plain


def load_etcbc_texts(name: str, normalize: bool = True,
                     cache_dir: Path = DEFAULT_ETCBC_CACHE,
                     refresh: bool = False) -> list[Text]:
    """Return one ``Text`` per biblical book (author_key = corpus name)."""
    plain = ensure_etcbc(name, cache_dir, refresh)
    _, _, label = ETCBC_SOURCES[name]
    texts: list[Text] = []
    for path in sorted(plain.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        counts: Counter[str] = Counter()
        for token in iter_words_text(raw):
            counts[strip_marks(token) if normalize else token] += 1
        if not counts:
            continue
        texts.append(Text(path.name, name, label, counts,
                          text_id=f"{name}:{path.stem}", series=name, title=path.stem))
    return texts


def corpus_frequencies(texts: list[Text]) -> Counter:
    freq: Counter[str] = Counter()
    for t in texts:
        freq.update(t.counts)
    return freq


if __name__ == "__main__":
    for nm in ETCBC_SOURCES:
        ts = load_etcbc_texts(nm)
        freq = corpus_frequencies(ts)
        print(f"{nm:10} books={len(ts):3d}  tokens={sum(freq.values()):>8,}  "
              f"types={len(freq):>7,}")
