#!/usr/bin/env python3
"""Authorship experiments on the srophe/syriac-corpus FastText embeddings.

Three analyses, all reusing the corpus loader, tokenizer and document-vector
machinery from stylometry.py (which in turn reuses script.py):

  1. fastText vs Burrows's Delta -- the classic stylometric baseline. Both
     methods are scored on the SAME author cohort by two yardsticks:
       * same/cross-author AUC (do same-author text pairs sit closer?), and
       * leave-one-out top-1 / top-3 attribution accuracy (nearest author
         centroid).
     Delta is swept over several most-frequent-word (MFW) vocabularies, and the
     whole comparison is repeated at two minimum-token floors (1000 and 2000) as
     a sensitivity analysis.

  2. Attribution of disputed / pseudonymous texts. After validating the nearest-
     centroid attributor with leave-one-out accuracy on known authors, three
     held-out cases are ranked against the known author centroids:
       * 690  - a letter transmitted under Ephrem's name in the (spurious)
                "Letters of Papa bar Aggai" dossier;
       * 219-227 - the Pseudo-Clementine Recognitions/Homilies (a translation
                attributed to Ps.-Clement of Rome); and
       * 519  - the Chronicle of Zuqnin (encoded as Anonymous; traditionally
                mis-attributed to Dionysius of Tel-Mahre, hence "Pseudo-
                Dionysius").

  3. Genre control inside Ephrem's genuine corpus: do his madrase (verse hymns)
     separate from his prose? Strong within-author genre separation would mean
     the headline authorship signal is partly a genre/register effect.

Requires gensim + numpy. Loads the model saved by fasttext_model.py.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from core.script import DEFAULT_CACHE, ensure_corpus
from core.stylometry import (
    DEFAULT_MODEL,
    Text,
    auc_same_higher,
    doc_vector,
    filter_min_texts,
    filter_min_tokens,
    function_word_set,
    get_model,
    l2_normalize,
    load_one_text,
    load_texts,
    remove_common_component,
    select_author,
    separation,
    text_length,
)

DEFAULT_MIN_TEXTS = 3
DEFAULT_TOKEN_FLOORS = "1000,2000"
DEFAULT_DELTA_MFW = "100,200,500"
DEFAULT_DISPUTED = "690,219-227,519"
DEFAULT_CONTROL_AUTHOR = "Ephrem"
DEFAULT_GENRE_FLOORS = "0,500,1000"


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_ids(spec: str) -> list[str]:
    """Expand "690,219-227,519" -> ['690','219',...,'227','519']."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
# Document matrices
# --------------------------------------------------------------------------- #
def doc_matrix(texts: list[Text], wv, allowed: set[str] | None):
    """Stack one freq-weighted mean vector per text; drop texts with no vector."""
    rows: list[np.ndarray] = []
    kept: list[Text] = []
    for t in texts:
        v = doc_vector(wv, t.counts, allowed)
        if v is not None:
            rows.append(v)
            kept.append(t)
    if not rows:
        return None, []
    return np.vstack(rows), kept


def key_labels(texts: list[Text]) -> np.ndarray:
    return np.asarray([t.author_key for t in texts], dtype=object)


# --------------------------------------------------------------------------- #
# Burrows's Delta
# --------------------------------------------------------------------------- #
def delta_profiles(texts: list[Text], k_mfw: int) -> np.ndarray:
    """Burrows's Delta features: per-text z-scored relative frequencies of the
    top-``k_mfw`` corpus word forms."""
    freq: Counter[str] = Counter()
    for t in texts:
        freq.update(t.counts)
    vocab = [w for w, _ in freq.most_common(k_mfw)]
    index = {w: j for j, w in enumerate(vocab)}

    x = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, t in enumerate(texts):
        total = text_length(t)
        if total == 0:
            continue
        for w, c in t.counts.items():
            j = index.get(w)
            if j is not None:
                x[i, j] = c / total
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def delta_distance_matrix(z: np.ndarray) -> np.ndarray:
    """Mean absolute difference of z-profiles (Burrows's Delta distance)."""
    n, k = z.shape
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist[i] = np.abs(z - z[i]).sum(axis=1) / k
    return dist


def auc_from_distance(dist: np.ndarray, labels: np.ndarray) -> float:
    """Same/cross-author AUC where smaller distance = more similar."""
    iu = np.triu_indices(len(labels), k=1)
    pair = dist[iu]
    same = labels[iu[0]] == labels[iu[1]]
    # auc_same_higher wants same-group scores to be larger when "closer"; negate.
    return auc_same_higher(-pair[same], -pair[~same])


# --------------------------------------------------------------------------- #
# Nearest-centroid leave-one-out attribution
# --------------------------------------------------------------------------- #
def centroid_loo(x: np.ndarray, labels: np.ndarray, metric: str) -> dict:
    """Leave-one-out nearest author-centroid attribution.

    metric="cosine": vectors are mean-centered, centroid cosine (for fastText).
    metric="l1":     centroid mean-absolute distance on z-profiles (for Delta).
    The held-out text is removed from its own author's centroid before scoring.
    """
    labels = np.asarray(labels, dtype=object)
    uniq = list(dict.fromkeys(labels.tolist()))

    if metric == "cosine":
        acc = x - x.mean(axis=0, keepdims=True)   # accumulation/centroid space
        query = l2_normalize(acc)
        higher_better = True

        def score(cent: np.ndarray, i: int) -> float:
            c = cent / (np.linalg.norm(cent) or 1.0)
            return float(query[i] @ c)
    else:  # "l1" on standardized profiles
        acc = x
        higher_better = False

        def score(cent: np.ndarray, i: int) -> float:
            return float(np.abs(cent - x[i]).mean())

    sums = {a: acc[labels == a].sum(axis=0) for a in uniq}
    counts = {a: int((labels == a).sum()) for a in uniq}

    top1 = top3 = n = 0
    confusions: list[tuple[str, str]] = []
    for i in range(len(labels)):
        a = labels[i]
        cand: list[tuple[float, str]] = []
        for b in uniq:
            if b == a:
                if counts[b] < 2:
                    continue  # no centroid without the held-out text
                cent = (sums[b] - acc[i]) / (counts[b] - 1)
            else:
                cent = sums[b] / counts[b]
            cand.append((score(cent, i), b))
        if not any(b == a for _, b in cand):
            continue
        cand.sort(reverse=higher_better)
        ranked = [b for _, b in cand]
        n += 1
        top1 += ranked[0] == a
        top3 += a in ranked[:3]
        if ranked[0] != a:
            confusions.append((a, ranked[0]))
    return {
        "n": n,
        "top1": top1 / n if n else float("nan"),
        "top3": top3 / n if n else float("nan"),
        "confusions": confusions,
    }


# --------------------------------------------------------------------------- #
# Author centroids (for ranking disputed texts)
# --------------------------------------------------------------------------- #
def fit_centroids(texts: list[Text], wv, allowed: set[str] | None):
    """Return (centroids{key->unit vec}, names{key->name}, mean) in centered space."""
    matrix, kept = doc_matrix(texts, wv, allowed)
    mean = matrix.mean(axis=0)
    unit = l2_normalize(matrix - mean)
    centroids: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}
    for key in dict.fromkeys(t.author_key for t in kept):
        idx = [i for i, t in enumerate(kept) if t.author_key == key]
        c = unit[idx].mean(axis=0)
        centroids[key] = c / (np.linalg.norm(c) or 1.0)
        names[key] = next(t.author_name for t in kept if t.author_key == key)
    return centroids, names, mean


def project_unit(vec: np.ndarray, mean: np.ndarray) -> np.ndarray:
    v = vec - mean
    return v / (np.linalg.norm(v) or 1.0)


def rank_against(vec: np.ndarray, centroids: dict[str, np.ndarray],
                 mean: np.ndarray) -> list[tuple[float, str]]:
    u = project_unit(vec, mean)
    return sorted(((float(u @ c), key) for key, c in centroids.items()), reverse=True)


# --------------------------------------------------------------------------- #
# Genre classification (from series/title; no formal genre tags in the corpus)
# --------------------------------------------------------------------------- #
def classify_genre(text: Text) -> str:
    s = f"{text.series} {text.title}".casefold()
    if "prose" in s:
        return "prose"
    if "hymn" in s or "madras" in s or "madr\u0101\u0161" in s or "\u0721\u0715\u072a" in s:
        return "madrase (hymns)"
    if any(w in s for w in ("homily", "homilies", "memra", "m\u0113mr", "mimro",
                            "discourse", "sermon", "verse homily")):
        return "memre (verse homilies)"
    if "letter" in s or "epistle" in s:
        return "letter"
    return "other"


# --------------------------------------------------------------------------- #
# Analysis 1: fastText vs Burrows's Delta
# --------------------------------------------------------------------------- #
def analysis_compare(genuine: list[Text], wv, func_set: set[str],
                     floors: list[int], min_texts: int, delta_mfws: list[int]) -> None:
    print()
    print("#" * 74)
    print("# Analysis 1 -- fastText vs Burrows's Delta")
    print("#" * 74)
    print("AUC = same/cross-author separation; top1/top3 = leave-one-out attribution.")

    for floor in floors:
        cohort = filter_min_texts(filter_min_tokens(genuine, floor), min_texts)
        labels = key_labels(cohort)
        n_authors = len(set(labels.tolist()))
        print()
        print("=" * 74)
        print(f"Cohort: >={min_texts} texts/author, >={floor:,} tokens/text  "
              f"-> {len(cohort)} texts, {n_authors} authors")
        print("=" * 74)
        if n_authors < 2:
            print("  (too few authors at this floor; skipping)")
            continue

        print(f"{'method / feature':<26}{'AUC':>9}{'top-1':>9}{'top-3':>9}")
        print("-" * 74)

        # fastText: centered document vectors, two vocabularies.
        for vocab_name, allowed in (("all words", None), ("function words", func_set)):
            matrix, kept = doc_matrix(cohort, wv, allowed)
            klab = key_labels(kept)
            auc = separation(remove_common_component(matrix), klab)["auc"]
            loo = centroid_loo(matrix, klab, metric="cosine")
            print(f"{'fastText ' + vocab_name:<26}{auc:>9.3f}"
                  f"{loo['top1']:>9.3f}{loo['top3']:>9.3f}")

        # Burrows's Delta over several MFW vocabularies.
        for k in delta_mfws:
            z = delta_profiles(cohort, k)
            auc = auc_from_distance(delta_distance_matrix(z), labels)
            loo = centroid_loo(z, labels, metric="l1")
            print(f"{'Delta MFW=' + str(k):<26}{auc:>9.3f}"
                  f"{loo['top1']:>9.3f}{loo['top3']:>9.3f}")


# --------------------------------------------------------------------------- #
# Analysis 2: attribution of disputed texts
# --------------------------------------------------------------------------- #
def analysis_attribution(genuine: list[Text], disputed: list[Text], wv,
                         floor: int, min_texts: int, topn: int = 5) -> None:
    print()
    print("#" * 74)
    print("# Analysis 2 -- attribution of disputed / pseudonymous texts")
    print("#" * 74)

    reference = filter_min_texts(filter_min_tokens(genuine, floor), min_texts)
    labels = key_labels(reference)
    n_authors = len(set(labels.tolist()))
    print(f"Reference cohort: >={min_texts} texts/author, >={floor:,} tokens/text "
          f"-> {len(reference)} texts, {n_authors} known authors")

    # Validate the attributor on known authors (fastText, all words).
    matrix, kept = doc_matrix(reference, wv, None)
    loo = centroid_loo(matrix, key_labels(kept), metric="cosine")
    print(f"Leave-one-out nearest-centroid accuracy on knowns: "
          f"top-1 {loo['top1']:.3f}, top-3 {loo['top3']:.3f}  (n={loo['n']})")

    centroids, names, mean = fit_centroids(reference, wv, None)

    # Per-author distribution of cosine-to-own-centroid, for percentile context.
    own_cos: dict[str, list[float]] = {}
    for i, t in enumerate(kept):
        u = (matrix[i] - mean)
        u = u / (np.linalg.norm(u) or 1.0)
        own_cos.setdefault(t.author_key, []).append(float(u @ centroids[t.author_key]))

    # Group disputed texts by series for reporting.
    groups: dict[str, list[Text]] = {}
    for t in disputed:
        groups.setdefault(t.series or t.title or t.text_id, []).append(t)

    for series, members in groups.items():
        members = sorted(members, key=lambda t: t.text_id)
        print()
        print("=" * 74)
        tagged = Counter(t.author_name for t in members).most_common(1)[0][0]
        ids = ", ".join(t.text_id for t in members)
        print(f"{series}")
        print(f"  files {ids}  |  encoded author: {tagged}  |  {len(members)} text(s)")
        print("-" * 74)

        # Nearest known author for each disputed text.
        vecs = []
        for t in members:
            v = doc_vector(wv, t.counts, None)
            vecs.append(v)
            ranking = rank_against(v, centroids, mean)
            best = ranking[:topn]
            shown = "  ".join(f"{names[k].split(',')[0][:18]}:{s:.2f}" for s, k in best)
            flag = "  [!short]" if text_length(t) < 500 else ""
            print(f"  {t.text_id} ({text_length(t):,} tok){flag} -> {shown}")

        # Group cohesion vs nearest external author (outlier signal).
        if len(members) > 1:
            mat = l2_normalize(np.vstack(vecs) - mean)
            sims = mat @ mat.T
            iu = np.triu_indices(len(members), k=1)
            internal = float(sims[iu].mean())
            ext_best = max(
                max(s for s, _ in rank_against(v, centroids, mean)) for v in vecs)
            print(f"  group cohesion (mean pairwise cos): {internal:.3f}  |  "
                  f"best match to any known author: {ext_best:.3f}")
            if internal > ext_best:
                print("  => the texts resemble each other more than any known author")
                print("     -> a distinct hand (consistent with a single translator).")
            else:
                print("  => their closest affinities are to other works rendered from")
                print("     Greek (e.g. the Syriac NT / Eusebius) rather than to each")
                print("     other -> style dominated by translationese, not the author.")

        # Special read-out for the Ephrem-attributed letter: percentile vs his
        # genuine texts, and genre-matched comparison (prose vs hymns).
        eph_key = next((k for k, nm in names.items() if "ephrem" in nm.casefold()), None)
        if eph_key and any("papa" in (t.series or "").casefold() for t in members):
            t = members[0]
            v = doc_vector(wv, t.counts, None)
            u = project_unit(v, mean)
            cos_eph = float(u @ centroids[eph_key])
            dist = np.asarray(own_cos[eph_key])
            pct = float((dist < cos_eph).mean() * 100)
            rank_pos = [k for _, k in rank_against(v, centroids, mean)].index(eph_key) + 1
            print(f"  Ephrem check: cos-to-Ephrem {cos_eph:.3f} "
                  f"(percentile {pct:.0f}% of his {len(dist)} genuine texts; "
                  f"Ephrem ranks #{rank_pos} of {n_authors}).")
            print("  (his reference profile here is built from his long, mostly prose")
            print("   works, so this is a genre-appropriate comparison for a prose letter.)")
            if pct < 25:
                print("  => sits at/below the bottom of Ephrem's own range "
                      "-> weak support for the traditional ascription.")
            else:
                print("  => within Ephrem's own range -> consistent with his style.")


# --------------------------------------------------------------------------- #
# Analysis 3: genre control within Ephrem
# --------------------------------------------------------------------------- #
def _genre_separation(texts: list[Text], wv) -> dict | None:
    by_genre: dict[str, list[Text]] = {}
    for t in texts:
        by_genre.setdefault(classify_genre(t), []).append(t)
    usable = {g: m for g, m in by_genre.items() if len(m) >= 2}
    if len(usable) < 2:
        return None
    members = [t for m in usable.values() for t in m]
    matrix, kept = doc_matrix(members, wv, None)
    klab = np.asarray([classify_genre(t) for t in kept], dtype=object)
    sep = separation(remove_common_component(matrix), klab)
    sep["sizes"] = {g: len(m) for g, m in sorted(usable.items(), key=lambda kv: -len(kv[1]))}
    return sep


def analysis_genre(genuine: list[Text], wv, control_author: str,
                   genre_floors: list[int]) -> None:
    print()
    print("#" * 74)
    print("# Analysis 3 -- genre control within one author")
    print("#" * 74)

    picked = select_author(genuine, control_author)
    if picked is None:
        print(f"  author matching {control_author!r} not found; skipping.")
        return
    _, display, author_texts = picked

    by_genre: dict[str, list[Text]] = {}
    for t in author_texts:
        by_genre.setdefault(classify_genre(t), []).append(t)

    print(f"Author: {display}  ({len(author_texts)} texts)")
    print("-" * 74)
    print(f"{'genre':<26}{'texts':>7}{'mean tokens':>14}")
    for g, members in sorted(by_genre.items(), key=lambda kv: -len(kv[1])):
        mean_tok = sum(text_length(t) for t in members) / len(members)
        print(f"{g:<26}{len(members):>7}{mean_tok:>14,.0f}")
    print("(genre and length are confounded here: hymns are short, prose is long,")
    print(" so we sweep a token floor to see genre separation at comparable lengths.)")

    # Genre AUC at several token floors: low floor keeps short hymns (noisy),
    # higher floors compare length-matched texts (fewer, but cleaner).
    print()
    print(f"{'token floor':<14}{'genres (sizes)':<28}{'AUC':>8}{'Cohen d':>9}")
    print("-" * 74)
    rows: list[tuple[int, float]] = []
    for floor in sorted(genre_floors):
        sep = _genre_separation(filter_min_tokens(author_texts, floor), wv)
        if sep is None:
            print(f">={floor:<12,}{'(too few per genre)':<28}")
            continue
        sizes = ", ".join(f"{g.split()[0]}:{n}" for g, n in sep["sizes"].items())
        print(f">={floor:<12,}{sizes:<28}{sep['auc']:>8.3f}{sep['cohen_d']:>+9.2f}")
        rows.append((floor, sep["auc"]))

    print()
    print("=" * 74)
    if len(rows) >= 2 and rows[-1][1] - rows[0][1] >= 0.12:
        print(f"Genre signal within {display} is LENGTH-DEPENDENT:")
        print(f"  weak at the low floor (AUC {rows[0][1]:.3f}, dominated by short, noisy")
        print(f"  hymns) but strong once lengths are matched (AUC {rows[-1][1]:.3f}).")
        print("  => genre/register is a real confound. The headline cross-author")
        print("     numbers partly reflect genre; compare like with like (e.g. the")
        print("     disputed prose letter against Ephrem's prose, not his hymns).")
    elif rows and max(a for _, a in rows) >= 0.6:
        print(f"Genre signal within {display}: present "
              f"(best AUC {max(a for _, a in rows):.3f}).")
        print("  Part of the cross-author signal reflects genre/register; compare")
        print("  like with like where possible.")
    else:
        print(f"Genre signal within {display}: weak at all tested lengths.")
        print("  The cross-author signal is mostly authorship, not a genre artefact.")
    print("=" * 74)



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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

    parser.add_argument("--analyses", default="delta,attribution,genre",
                        help="comma list of analyses to run (default: all)")
    parser.add_argument("--num-function-words", type=int, default=200,
                        help="how many top-frequency forms count as function words")
    parser.add_argument("--min-texts", type=int, default=DEFAULT_MIN_TEXTS,
                        help=f"authors with >= this many texts (default: {DEFAULT_MIN_TEXTS})")
    parser.add_argument("--min-tokens", default=DEFAULT_TOKEN_FLOORS,
                        help=f"token floor(s), comma-separated (default: {DEFAULT_TOKEN_FLOORS})")
    parser.add_argument("--delta-mfw", default=DEFAULT_DELTA_MFW,
                        help=f"Delta most-frequent-word sizes (default: {DEFAULT_DELTA_MFW})")
    parser.add_argument("--disputed-ids", default=DEFAULT_DISPUTED,
                        help=f"disputed text ids/ranges (default: {DEFAULT_DISPUTED})")
    parser.add_argument("--control-author", default=DEFAULT_CONTROL_AUTHOR,
                        help=f"author for the genre control (default: {DEFAULT_CONTROL_AUTHOR!r})")
    parser.add_argument("--genre-min-tokens", default=DEFAULT_GENRE_FLOORS,
                        help=f"token floor(s) for the genre control, comma-separated "
                             f"(default: {DEFAULT_GENRE_FLOORS}; sweeps to expose the "
                             f"length/genre confound)")
    args = parser.parse_args(argv)

    analyses = {a.strip() for a in args.analyses.split(",") if a.strip()}
    floors = parse_int_list(args.min_tokens)
    delta_mfws = parse_int_list(args.delta_mfw)
    disputed_ids = parse_ids(args.disputed_ids)

    data_dir = ensure_corpus(args.cache_dir, refresh=args.refresh, update=args.update)

    print("Loading corpus ...", file=sys.stderr)
    # Genuine cohort: drop the collapsed "Anonymous" pseudo-author and hold out
    # every disputed text so it cannot pollute the known-author centroids.
    genuine = load_texts(data_dir, args.normalize,
                         exclude_ids=set(disputed_ids), drop_anonymous=True)
    disputed = [t for t in (load_one_text(data_dir, i, args.normalize)
                            for i in disputed_ids) if t is not None]
    print(f"  genuine single-author texts: {len(genuine)} | "
          f"disputed loaded: {len(disputed)}", file=sys.stderr)

    func_set, _ = function_word_set(genuine, args.num_function_words)

    model = get_model(args.model, data_dir, args.normalize)
    wv = model.wv

    if "delta" in analyses:
        analysis_compare(genuine, wv, func_set, floors, args.min_texts, delta_mfws)
    if "attribution" in analyses:
        analysis_attribution(genuine, disputed, wv, min(floors), args.min_texts)
    if "genre" in analyses:
        analysis_genre(genuine, wv, args.control_author,
                       parse_int_list(args.genre_min_tokens))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
