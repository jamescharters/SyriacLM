#!/usr/bin/env python3
"""Probe the FastText embeddings for *stylometric* signal in the srophe/syriac-corpus.

Idea: if the character n-gram embeddings capture an author's style (not just
topic), then two texts by the *same* author should have more similar averaged
vectors than two texts by *different* authors.

Method (classic authorship-verification setup):

  1. Load the corpus split by author (one TEI file == one text, keeping only
     texts with a single attributed author; reuses script.py helpers).
  2. Represent each text as the (token-frequency-weighted) mean of its word
     vectors, then L2-normalise it.
  3. Score every pair of texts by cosine similarity and split the pairs into
     SAME-author and CROSS-author groups.
  4. Measure separation with AUC = P(a random same-author pair is more similar
     than a random cross-author pair).  AUC 0.5 = no signal, 1.0 = perfect.

Averaged embeddings are strongly anisotropic: every document vector shares a
large common component, so raw cosines are all close to 1 and the signal is
compressed.  We therefore also report a "mean-centered" view that subtracts the
corpus-wide mean document vector (an unsupervised, label-free step) before
scoring; this removes the shared component and exposes the real separation.

Two versions of the test are run on the *identical* set of texts:

  * "all words"            - the mean uses every token in the text;
  * "function words only"  - the mean uses only tokens among the N most frequent
                             forms in the corpus (default N=200).  Function words
                             (particles, pronouns, prepositions, conjunctions...)
                             are the traditional stylometric markers because they
                             are topic-independent.

The script then reports which version separates same- from cross-author pairs
better, i.e. whether function-word vectors carry a cleaner stylometric signal
than all-word vectors.

Two cohorts are reported side by side so the pooled AUC is not dominated by a
few very large authors:

  * "full"       - every author with >=2 texts, any length; and
  * "restricted" - only prolific authors (>=3 texts) writing long texts
                   (>=2000 tokens each), which evens out corpus-size imbalance.

Finally, a false-positive (negative control) splits ONE author's genuine texts
into two random "pseudo-authors" and re-runs the test. With no real authorship
boundary the AUC should collapse to ~0.5; if it stays high, the pipeline is
picking up topic/length rather than authorship. The control AUC is averaged over
several seeded random half-splits.

Requires gensim + numpy.  By default it loads the model saved by
fasttext_model.py; if that file is missing it trains one with the same settings.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from script import (
    DEFAULT_CACHE,
    ensure_corpus,
    extract_authors,
    extract_series,
    extract_title,
    find_body,
    iter_words,
    strip_marks,
)

DEFAULT_MODEL = Path("syriac_fasttext.model")
DEFAULT_NUM_FUNCTION_WORDS = 200

# Restricted-cohort defaults: prolific authors writing long texts. This removes
# the size imbalance where a few huge authors dominate the pooled statistics.
DEFAULT_MIN_TEXTS = 3
DEFAULT_MIN_TOKENS = 2000

# False-positive (negative control) defaults.
DEFAULT_CONTROL_AUTHOR = "Ephrem"
DEFAULT_CONTROL_SPLITS = 20

# Training settings used only when no saved model is found. These mirror the
# defaults in fasttext_model.py so the two scripts stay consistent.
_TRAIN_DEFAULTS = dict(
    dim=100, window=5, epochs=10, min_n=2, max_n=5,
    buckets=200_000, sg=1, seed=42,
)


@dataclass
class Text:
    """One TEI text: its file name, canonical author, token counts, metadata."""
    name: str
    author_key: str
    author_name: str
    counts: Counter = field(default_factory=Counter)
    text_id: str = ""       # file stem, e.g. "690"
    series: str = ""        # TEI series title (genre proxy), e.g. "Hymns on Nativity"
    title: str = ""         # work title (TEI title level="a")


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def _read_text(path: Path, normalize: bool):
    """Parse one TEI file -> (authors, series, title, counts) or None if unusable."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    body = find_body(root)
    if body is None:
        return None
    counts: Counter[str] = Counter()
    for token in iter_words(body):
        counts[strip_marks(token) if normalize else token] += 1
    if not counts:
        return None
    return extract_authors(root), extract_series(root), extract_title(root), counts


def load_texts(data_dir: Path, normalize: bool, *,
               exclude_ids: set[str] | None = None,
               drop_anonymous: bool = False) -> list[Text]:
    """Load single-author texts, canonicalising author identity.

    Only texts crediting exactly one author are kept (multi-author and
    unattributed texts are ambiguous for a per-author style test). Authors named
    without a syriaca.org @ref are merged into their URI-identified counterpart,
    matching the logic in script.py's collect_stats.

    ``exclude_ids`` drops files by stem (e.g. disputed texts held out for
    attribution). ``drop_anonymous`` removes the collapsed "Anonymous"
    pseudo-author (role="anonymous" with no @ref), which is not a real author.
    """
    exclude_ids = exclude_ids or set()
    parsed: list[tuple[str, str, str, str, str, str, Counter]] = []
    name_to_uri: dict[str, str] = {}

    for path in sorted(data_dir.glob("*.xml")):
        if path.stem in exclude_ids:
            continue
        rec = _read_text(path, normalize)
        if rec is None:
            continue
        authors, series, title, counts = rec
        keys = {k for k, _ in authors}
        if len(keys) != 1:  # skip unattributed and multi-author texts
            continue
        (key,) = keys
        name = next(n for k, n in authors if k == key)
        if drop_anonymous and (key == "anonymous" or name.casefold() == "anonymous"):
            continue

        parsed.append((path.stem, path.name, key, name, series, title, counts))
        if key.startswith("http"):
            name_to_uri.setdefault(name.casefold(), key)

    texts: list[Text] = []
    for text_id, fname, key, name, series, title, counts in parsed:
        if not key.startswith("http"):
            key = name_to_uri.get(name.casefold(), key)
        texts.append(Text(fname, key, name, counts,
                          text_id=text_id, series=series, title=title))
    return texts


def load_one_text(data_dir: Path, text_id: str, normalize: bool) -> Text | None:
    """Load any TEI file by id, regardless of attribution (for disputed texts)."""
    path = data_dir / f"{text_id}.xml"
    if not path.is_file():
        return None
    rec = _read_text(path, normalize)
    if rec is None:
        return None
    authors, series, title, counts = rec
    key, name = authors[0] if authors else ("(unattributed)", "(unattributed)")
    return Text(path.name, key, name, counts,
                text_id=str(text_id), series=series, title=title)




def filter_min_texts(texts: list[Text], min_texts: int) -> list[Text]:
    """Keep only texts whose author has at least ``min_texts`` texts."""
    per_author = Counter(t.author_key for t in texts)
    keep = {a for a, c in per_author.items() if c >= min_texts}
    return [t for t in texts if t.author_key in keep]


def text_length(text: Text) -> int:
    """Total token count of a text."""
    return sum(text.counts.values())


def filter_min_tokens(texts: list[Text], min_tokens: int) -> list[Text]:
    """Keep only texts with at least ``min_tokens`` tokens (longer = more stable mean)."""
    if min_tokens <= 0:
        return list(texts)
    return [t for t in texts if text_length(t) >= min_tokens]


def function_word_set(texts: list[Text], n: int) -> tuple[set[str], Counter]:
    """Return (top-N most frequent forms, full frequency counter)."""
    freq: Counter[str] = Counter()
    for t in texts:
        freq.update(t.counts)
    top = {form for form, _ in freq.most_common(n)}
    return top, freq


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
def doc_vector(wv, counts: Counter, allowed: set[str] | None) -> np.ndarray | None:
    """Token-frequency-weighted mean of word vectors (None if no tokens match)."""
    acc: np.ndarray | None = None
    total = 0
    for token, c in counts.items():
        if allowed is not None and token not in allowed:
            continue
        try:
            vec = wv[token]  # FastText synthesises OOV vectors from char n-grams
        except KeyError:
            continue
        if acc is None:
            acc = np.zeros(wv.vector_size, dtype=np.float64)
        acc += c * vec
        total += c
    if acc is None or total == 0:
        return None
    return acc / total


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def remove_common_component(matrix: np.ndarray) -> np.ndarray:
    """Subtract the mean document vector (label-free) to undo embedding anisotropy."""
    return matrix - matrix.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# Separation statistics
# --------------------------------------------------------------------------- #
def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share their mean rank), like scipy rankdata."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.flatnonzero(obs), a.size]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def auc_same_higher(same: np.ndarray, cross: np.ndarray) -> float:
    """P(same-author similarity > cross-author similarity) via Mann-Whitney U."""
    n1, n2 = same.size, cross.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    ranks = _avg_ranks(np.concatenate([same, cross]))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def cohens_d(same: np.ndarray, cross: np.ndarray) -> float:
    """Standardised mean difference (pooled SD) between the two pair groups."""
    n1, n2 = same.size, cross.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = same.var(ddof=1), cross.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return float((same.mean() - cross.mean()) / pooled)


def separation(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """Compare same- vs cross-author cosine similarities over all text pairs."""
    unit = l2_normalize(vectors)
    # NumPy's matmul SIMD kernel can trip spurious divide/overflow/invalid FP
    # flags on Apple Silicon even when inputs and outputs are finite; the cosine
    # values below are valid, so silence those flags for this product only.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = unit @ unit.T
    iu = np.triu_indices(len(labels), k=1)
    pair_sims = sims[iu]
    same_pair = labels[iu[0]] == labels[iu[1]]
    same = pair_sims[same_pair]
    cross = pair_sims[~same_pair]
    return {
        "n_same": int(same.size),
        "n_cross": int(cross.size),
        "mean_same": float(same.mean()) if same.size else float("nan"),
        "mean_cross": float(cross.mean()) if cross.size else float("nan"),
        "diff": float(same.mean() - cross.mean()) if same.size and cross.size else float("nan"),
        "auc": auc_same_higher(same, cross),
        "cohen_d": cohens_d(same, cross),
    }


# --------------------------------------------------------------------------- #
# Cohort vectors
# --------------------------------------------------------------------------- #
def build_matrices(texts: list[Text], wv, func_set: set[str]):
    """Return (all_matrix, func_matrix, kept_texts) or None.

    Each text becomes one row in both the all-word and function-word matrix;
    texts that yield no vector for *either* representation are dropped so the two
    versions are always compared over the same documents.
    """
    all_rows: list[np.ndarray] = []
    func_rows: list[np.ndarray] = []
    kept: list[Text] = []
    for t in texts:
        v_all = doc_vector(wv, t.counts, None)
        v_func = doc_vector(wv, t.counts, func_set)
        if v_all is None or v_func is None:
            continue
        all_rows.append(v_all)
        func_rows.append(v_func)
        kept.append(t)
    if not all_rows:
        return None
    return np.vstack(all_rows), np.vstack(func_rows), kept


def author_labels(texts: list[Text]) -> tuple[np.ndarray, dict[str, int]]:
    """Map each text to an integer author id (by canonical author_key)."""
    ids: dict[str, int] = {}
    labels = [ids.setdefault(t.author_key, len(ids)) for t in texts]
    return np.asarray(labels), ids


def four_separations(all_matrix: np.ndarray, func_matrix: np.ndarray,
                     labels: np.ndarray) -> dict:
    """Raw + mean-centered separation for both all-word and function-word vectors."""
    return {
        ("raw", "all"): separation(all_matrix, labels),
        ("raw", "func"): separation(func_matrix, labels),
        ("centered", "all"): separation(remove_common_component(all_matrix), labels),
        ("centered", "func"): separation(remove_common_component(func_matrix), labels),
    }


def select_author(texts: list[Text], substring: str):
    """Find the author whose name contains ``substring`` and has the most texts.

    Returns (author_key, display_name, [texts]) or None. Different spellings that
    share a canonical author_key (e.g. name-only vs syriaca.org @ref) are merged.
    """
    sub = substring.casefold()
    by_key: dict[str, list[Text]] = {}
    names: dict[str, Counter] = {}
    for t in texts:
        by_key.setdefault(t.author_key, []).append(t)
        names.setdefault(t.author_key, Counter())[t.author_name] += 1
    matches = [k for k in by_key if any(sub in nm.casefold() for nm in names[k])]
    if not matches:
        return None
    best = max(matches, key=lambda k: len(by_key[k]))
    display = names[best].most_common(1)[0][0]
    return best, display, by_key[best]


def false_positive_test(texts: list[Text], wv, func_set: set[str], *,
                        n_splits: int, seed: int):
    """Negative control: split ONE author's texts into two random pseudo-authors.

    With no genuine authorship difference the within-half ("same") and
    across-half ("cross") cosine distributions should coincide, so AUC ~ 0.5.
    The AUC is averaged over ``n_splits`` seeded random half-splits; the first
    split is also returned as a worked example.

    Returns ("ok", n_kept, summary, example, aucs) where summary maps each
    (mode, vocab) key to (mean, std, min, max); or ("too_few", n_kept); or None.
    """
    built = build_matrices(texts, wv, func_set)
    if built is None:
        return None
    all_m, func_m, _ = built
    n = all_m.shape[0]
    if n < 4:  # need >=2 texts per half so each half has within-pairs
        return ("too_few", n)

    keys = [("centered", "all"), ("centered", "func"), ("raw", "all"), ("raw", "func")]
    aucs: dict[tuple[str, str], list[float]] = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    example: dict | None = None
    for s in range(n_splits):
        perm = rng.permutation(n)
        labels = np.zeros(n, dtype=int)
        labels[perm[: n // 2]] = 1  # two halves (larger half keeps label 0 if n is odd)
        res = four_separations(all_m, func_m, labels)
        for k in keys:
            aucs[k].append(res[k]["auc"])
        if s == 0:
            example = res
    summary = {
        k: (float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)))
        for k, v in aucs.items()
    }
    return ("ok", n, summary, example, {k: np.asarray(v) for k, v in aucs.items()})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def get_model(model_path: Path, data_dir: Path, normalize: bool):
    """Load the saved FastText model, or train (and save) one if absent."""
    try:
        from gensim.models import FastText
    except ImportError:
        sys.exit("error: gensim is required (pip install gensim).")

    if model_path.exists():
        print(f"Loading FastText model from {model_path} ...", file=sys.stderr)
        return FastText.load(str(model_path))

    print(f"No model at {model_path}; training a new one ...", file=sys.stderr)
    try:
        from fasttext_model import build_sentences, train_model
    except ImportError:
        sys.exit("error: cannot import fasttext_model.py to train a model.")
    import os

    sentences, _, _ = build_sentences(data_dir, normalize, 0)
    model = train_model(sentences, workers=os.cpu_count() or 1, **_TRAIN_DEFAULTS)
    model.save(str(model_path))
    print(f"Saved model to {model_path}", file=sys.stderr)
    return model


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _sep_row(label: str, s: dict) -> None:
    print(f"{label:<32}{s['mean_same']:>9.3f}{s['mean_cross']:>9.3f}"
          f"{s['diff']:>+8.3f}{s['auc']:>8.3f}{s['cohen_d']:>9.2f}")


def report(results: dict, *, cohort_name: str, n_texts: int, n_authors: int,
           min_texts: int, min_tokens: int, num_function_words: int,
           function_preview: list[tuple[str, int]], dim: int, dropped: int,
           normalize: bool) -> None:
    sample = results[("centered", "all")]
    cohort_detail = f">={min_texts} texts/author"
    if min_tokens:
        cohort_detail += f", >={min_tokens:,} tokens/text"
    print()
    print("=" * 74)
    print(f"Stylometric separation -- {cohort_name} cohort  ({cohort_detail})")
    print("=" * 74)
    print(f"Texts analysed        : {n_texts:,}  (authors: {n_authors})")
    print(f"Vector space          : {dim}-dim FastText, "
          f"{'diacritic-stripped' if normalize else 'surface'} tokens, freq-weighted mean")
    print(f"Same-author pairs     : {sample['n_same']:,}")
    print(f"Cross-author pairs    : {sample['n_cross']:,}")
    if dropped:
        print(f"Texts dropped (no usable tokens): {dropped}")

    preview = "  ".join(f"{w}:{c:,}" for w, c in function_preview)
    print(f"Function words        : top {num_function_words} forms by corpus frequency")
    print(f"    most frequent ->  {preview}")

    print()
    print("-" * 74)
    print(f"{'version':<32}{'same':>9}{'cross':>9}{'diff':>8}{'AUC':>8}{'Cohen d':>9}")
    print("-" * 74)
    print("common component removed (mean-centered):")
    _sep_row("  all words", results[("centered", "all")])
    _sep_row("  function words only", results[("centered", "func")])
    print("raw cosine (uncentered):")
    _sep_row("  all words", results[("raw", "all")])
    _sep_row("  function words only", results[("raw", "func")])
    print("-" * 74)
    print("AUC = P(same-author pair more similar than cross-author pair); 0.5 = no signal.")

    ca, cf = results[("centered", "all")]["auc"], results[("centered", "func")]["auc"]
    ra, rf = results[("raw", "all")]["auc"], results[("raw", "func")]["auc"]
    print()
    if abs(ca - cf) < 1e-6:
        print(f"Function vs all : equal separation (AUC {ca:.3f}).")
    elif cf > ca:
        print(f"Function vs all : function words separate BETTER "
              f"(AUC {cf:.3f} vs {ca:.3f}, +{cf - ca:.3f}).")
    else:
        print(f"Function vs all : all words separate better "
              f"(AUC {ca:.3f} vs {cf:.3f}); function words still carry signal "
              f"({cf:.3f}).")
    best = max(ca, cf)
    tone = ("clear" if best >= 0.75 else "moderate" if best >= 0.6 else "weak")
    print(f"Stylometric signal: {tone} -- best AUC {best:.3f} (mean-centered).")


def report_cohort_comparison(full: dict, restricted: dict) -> None:
    """One-line side-by-side of the two cohorts' mean-centered AUCs."""
    fa, ff = full[("centered", "all")]["auc"], full[("centered", "func")]["auc"]
    ra, rf = restricted[("centered", "all")]["auc"], restricted[("centered", "func")]["auc"]
    print()
    print("=" * 74)
    print("Full vs restricted cohort  (mean-centered AUC)")
    print("=" * 74)
    print(f"  all words           : full {fa:.3f}   restricted {ra:.3f}   "
          f"(delta {ra - fa:+.3f})")
    print(f"  function words only : full {ff:.3f}   restricted {rf:.3f}   "
          f"(delta {rf - ff:+.3f})")
    note = ("higher" if ra > fa else "lower" if ra < fa else "unchanged")
    print(f"  => restricting to long texts by prolific authors leaves all-word "
          f"separation {note}")
    print(f"     (the restricted set removes size imbalance from one-off / tiny texts).")


def report_control(display_name: str, n_used: int, n_total: int, summary: dict,
                   example: dict, *, n_splits: int, seed: int, min_tokens: int) -> None:
    print()
    print("=" * 74)
    print("False-positive control -- one author split into two pseudo-authors")
    print("=" * 74)
    floor = f">={min_tokens:,} tokens/text" if min_tokens else "no token floor"
    print(f"Author                : {display_name}")
    print(f"Texts used            : {n_used} of {n_total} ({floor})")
    print(f"Random half-splits    : {n_splits} (seed {seed})")
    print("Expectation           : AUC ~ 0.50  (no real author boundary to detect)")

    print()
    print("-" * 74)
    print(f"{'version':<32}{'mean AUC':>10}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 74)
    labels = {
        ("centered", "all"): "centered  all words",
        ("centered", "func"): "centered  function words",
        ("raw", "all"): "raw       all words",
        ("raw", "func"): "raw       function words",
    }
    for key in (("centered", "all"), ("centered", "func"), ("raw", "all"), ("raw", "func")):
        mean, std, lo, hi = summary[key]
        print(f"{labels[key]:<32}{mean:>10.3f}{std:>8.3f}{lo:>8.3f}{hi:>8.3f}")
    print("-" * 74)

    ex = example[("centered", "all")]
    print(f"Example split (#1): {ex['n_same']:,} within-half pairs, "
          f"{ex['n_cross']:,} across-half pairs")
    print(f"{'version':<32}{'same':>9}{'cross':>9}{'diff':>8}{'AUC':>8}{'Cohen d':>9}")
    _sep_row("  centered all words", example[("centered", "all")])
    _sep_row("  centered function words", example[("centered", "func")])

    mean_ca = summary[("centered", "all")][0]
    print()
    print("=" * 74)
    if abs(mean_ca - 0.5) < 0.10:
        print(f"Result: PASS -- mean AUC {mean_ca:.3f} ~ 0.5; no spurious separation "
              f"within a single author.")
        print("        The same/cross-author signal seen above is genuine, not an artefact.")
    else:
        print(f"Result: WARN -- mean AUC {mean_ca:.3f} departs from 0.5; the pipeline "
              f"separates")
        print("        even one author's own texts (topic/length leakage). Interpret "
              "real AUCs with caution.")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"where the corpus is cached (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="delete the cache and re-clone before analysing")
    parser.add_argument("--update", action="store_true",
                        help="git pull the cached corpus before analysing")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"FastText model to load/train (default: {DEFAULT_MODEL})")

    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true",
                      help="strip diacritics from tokens (default; must match the model)")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false",
                      help="keep diacritics (use only if the model was trained that way)")
    parser.set_defaults(normalize=True)

    parser.add_argument("--num-function-words", type=int, default=DEFAULT_NUM_FUNCTION_WORDS,
                        help="how many top-frequency forms count as function words")
    parser.add_argument("--min-texts", type=int, default=DEFAULT_MIN_TEXTS,
                        help=f"restricted cohort: authors with >= this many texts "
                             f"(default: {DEFAULT_MIN_TEXTS})")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                        help=f"restricted cohort: keep texts with >= this many tokens "
                             f"(default: {DEFAULT_MIN_TOKENS})")
    parser.add_argument("--control-author", default=DEFAULT_CONTROL_AUTHOR,
                        help=f"author for the false-positive control "
                             f"(default: {DEFAULT_CONTROL_AUTHOR!r})")
    parser.add_argument("--control-splits", type=int, default=DEFAULT_CONTROL_SPLITS,
                        help=f"random half-splits to average for the control "
                             f"(default: {DEFAULT_CONTROL_SPLITS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the control splits (default: 42)")
    parser.add_argument("--no-control", dest="control", action="store_false",
                        help="skip the false-positive control")
    parser.set_defaults(control=True)
    args = parser.parse_args(argv)

    data_dir = ensure_corpus(args.cache_dir, refresh=args.refresh, update=args.update)

    print("Loading corpus split by author ...", file=sys.stderr)
    all_texts = load_texts(data_dir, args.normalize)
    if len(all_texts) < 3:
        sys.exit("error: too few single-author texts to analyse.")

    # One fixed function-word inventory (top-N by corpus frequency), computed
    # over the whole single-author corpus and reused for every cohort + control.
    func_set, full_freq = function_word_set(all_texts, args.num_function_words)
    function_preview = full_freq.most_common(12)

    model = get_model(args.model, data_dir, args.normalize)
    wv = model.wv

    def run_cohort(name: str, texts: list[Text], *, min_texts: int, min_tokens: int):
        built = build_matrices(texts, wv, func_set)
        if built is None:
            print(f"warning: {name} cohort has no usable texts; skipping.", file=sys.stderr)
            return None
        all_m, func_m, kept = built
        labels, ids = author_labels(kept)
        if len(ids) < 2:
            print(f"warning: {name} cohort has <2 authors; skipping.", file=sys.stderr)
            return None
        res = four_separations(all_m, func_m, labels)
        report(res, cohort_name=name, n_texts=len(kept), n_authors=len(ids),
               min_texts=min_texts, min_tokens=min_tokens,
               num_function_words=args.num_function_words,
               function_preview=function_preview, dim=wv.vector_size,
               dropped=len(texts) - len(kept), normalize=args.normalize)
        return res

    # Cohort 1: full set -- every author with >=2 texts, any length.
    full_texts = filter_min_texts(all_texts, 2)
    full_res = run_cohort("full", full_texts, min_texts=2, min_tokens=0)

    # Cohort 2: restricted set -- prolific authors writing long texts only.
    restricted_texts = filter_min_texts(
        filter_min_tokens(all_texts, args.min_tokens), args.min_texts)
    restricted_res = run_cohort("restricted", restricted_texts,
                                min_texts=args.min_texts, min_tokens=args.min_tokens)

    if full_res and restricted_res:
        report_cohort_comparison(full_res, restricted_res)

    # Negative control: split one prolific author into two pseudo-authors.
    if args.control:
        picked = select_author(all_texts, args.control_author)
        if picked is None:
            names = sorted({t.author_name for t in all_texts})
            print(f"\nwarning: control author matching {args.control_author!r} not found; "
                  f"skipping control.", file=sys.stderr)
            print("available authors include: "
                  + ", ".join(names[:10]) + (" ..." if len(names) > 10 else ""),
                  file=sys.stderr)
        else:
            _, display, author_texts = picked
            kept_texts = filter_min_tokens(author_texts, args.min_tokens)
            outcome = false_positive_test(kept_texts, wv, func_set,
                                          n_splits=args.control_splits, seed=args.seed)
            if outcome is None or outcome[0] == "too_few":
                got = 0 if outcome is None else outcome[1]
                print(f"\nwarning: {display} has only {got} text(s) >= "
                      f"{args.min_tokens:,} tokens; need >=4 for the control. Skipping.",
                      file=sys.stderr)
            else:
                _, n_used, summary, example, _ = outcome
                if n_used < 6:
                    print(f"\nnote: only {n_used} control texts; each half is small, so "
                          f"the null AUC is noisier than usual.", file=sys.stderr)
                report_control(display, n_used, len(author_texts), summary, example,
                               n_splits=args.control_splits, seed=args.seed,
                               min_tokens=args.min_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
