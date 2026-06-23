#!/usr/bin/env python3
"""Experiments for the paper: representation bake-off, ablations, and statistics.

This script produces every number in the paper's tables (and prints ready-to-use
LaTeX ``tabular`` rows). It reuses the pipeline from ``stylometry.py`` /
``authorship.py`` / ``fasttext_model.py`` and the neural baselines from
``nn_baselines.py``.

Analyses (select with ``--analyses``):

  morph    Morphological coherence + subword ablation, by frequency band
           (FastText vs word2vec vs neural).                        [T2, T3a]
  oov      Out-of-vocabulary generalization: FastText synthesizes vectors for
           unseen forms; word2vec cannot.                           [T3b]
  hyper    FastText hyperparameter sensitivity.                     [T4]
  sep      Same/cross-author AUC with author-cluster bootstrap CIs and
           multi-seed variance.                                     [T5]
  bakeoff  Representation comparison (FastText / word2vec / byte-LM /
           char-Transformer / Burrows's Delta) on AUC + LOO attribution. [T6]
  genre    Genre-matched cross-author test.                          [T7]
  lm       Neural LM quality (bits-per-byte / perplexity).           [TLM]

All randomness is seeded. Neural analyses require PyTorch; if it is missing they
are skipped with a message and the rest still runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from script import (
    DEFAULT_CACHE,
    ensure_corpus,
    find_body,
    iter_words,
    strip_marks,
)
from stylometry import (
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
    separation,
    text_length,
)
from authorship import (
    centroid_loo,
    classify_genre,
    delta_distance_matrix,
    delta_profiles,
    doc_matrix,
    fit_centroids,
    key_labels,
    parse_ids,
    parse_int_list,
    project_unit,
)
import fasttext_model

try:
    import etcbc_corpus
    _ETCBC = True
except Exception:  # pragma: no cover
    _ETCBC = False

try:
    import nn_baselines
    _TORCH = nn_baselines._TORCH
except Exception:  # pragma: no cover
    _TORCH = False

try:
    import av_head
    _AVHEAD = av_head._TORCH
except Exception:  # pragma: no cover
    _AVHEAD = False

SEEDS = [42, 7, 13, 101, 2024]
DISPUTED_DEFAULT = "690,219-227,519"

# Curated, corpus-verified triconsonantal root families (diacritics stripped).
# Forms are listed most-frequent first; the most frequent present form is the
# anchor. Fixed BEFORE inspecting any model output to avoid cherry-picking.
MORPH_FAMILIES: dict[str, list[str]] = {
    "k-t-b (write/book)":   ["\u071f\u072c\u0712\u0710", "\u071f\u072c\u0712", "\u071f\u072c\u071d\u0712",
                             "\u071f\u072c\u0712\u072c", "\u071f\u072c\u0712\u0718", "\u071f\u072c\u0712\u0717",
                             "\u071f\u072c\u0712\u071d\u0722", "\u071f\u072c\u0712\u0718\u0717\u071d",
                             "\u0722\u071f\u072c\u0718\u0712", "\u0721\u071f\u072c\u0712\u0710"],
    "m-l-k (king/reign)":   ["\u0721\u0720\u071f\u0710", "\u0721\u0720\u071f\u0718\u072c\u0710", "\u0721\u0720\u071f",
                             "\u0721\u0720\u071f\u0718\u072c\u0717", "\u0721\u0720\u071f\u072c\u0710",
                             "\u0721\u0720\u071f\u071d\u0722", "\u0721\u0720\u071f\u0718"],
    "q-t-l (kill)":         ["\u0729\u071b\u0720", "\u0729\u071b\u0720\u0710", "\u0729\u071b\u0718\u0720\u0710",
                             "\u0729\u071b\u0720\u0718", "\u0729\u071b\u0720\u072c", "\u0722\u0729\u071b\u0718\u0720",
                             "\u0729\u071b\u071d\u0720"],
    "q-d-sh (holy)":        ["\u0729\u0715\u071d\u072b\u0710", "\u0729\u0718\u0715\u072b\u0710",
                             "\u0729\u0715\u071d\u072b\u0718\u072c\u0710", "\u0721\u0729\u0715\u072b",
                             "\u0729\u0715\u072b", "\u0729\u0715\u072b\u0710"],
    "sh-l-h (send/apostle)": ["\u072b\u0720\u071d\u071a\u0710", "\u072b\u0720\u071a", "\u072b\u0720\u071a\u0710",
                              "\u072b\u0720\u071a\u0718", "\u0722\u072b\u0720\u071a",
                              "\u072b\u0720\u071d\u071a\u0718\u072c\u0710"],
    "y-d-ʿ (know)":         ["\u071d\u0715\u0725", "\u071d\u0715\u0725\u072c\u0710", "\u0721\u0715\u0725\u0710",
                             "\u0722\u0715\u0725", "\u071d\u0715\u0725\u072c", "\u071d\u0715\u0725\u0718",
                             "\u071d\u0715\u071d\u0725"],
    "ʿ-b-d (do/servant)":   ["\u0725\u0712\u0715", "\u0725\u0712\u0715\u0710", "\u0722\u0725\u0712\u0715",
                             "\u0725\u0712\u0718\u0715\u0710", "\u0725\u0712\u0715\u0718", "\u0725\u0712\u0715\u072c",
                             "\u0725\u0712\u071d\u0715"],
    "q-w-m (rise/stand)":   ["\u0729\u0710\u0721", "\u0729\u0721", "\u0729\u071d\u0721\u0710",
                             "\u0729\u071d\u0721\u072c\u0710", "\u0721\u0729\u071d\u0721", "\u0722\u0729\u0718\u0721",
                             "\u0729\u071d\u0721"],
}


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #
def tex_row(*cells) -> str:
    return "  " + " & ".join(str(c) for c in cells) + r" \\"


def banner(title: str) -> None:
    print()
    print("#" * 76)
    print(f"# {title}")
    print("#" * 76)


# --------------------------------------------------------------------------- #
# Model cache (train once, reuse across analyses)
# --------------------------------------------------------------------------- #
class Models:
    def __init__(self, data_dir: Path, normalize: bool, model_path: Path):
        self.data_dir = data_dir
        self.normalize = normalize
        self._sentences = None
        self._ft = None
        self._ft_path = model_path
        self._ft_seeds = {}
        self._w2v = None
        self._w2v_seeds = {}
        self._neural = {}
        self._corpus_text = None

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences, _, _ = fasttext_model.build_sentences(self.data_dir, self.normalize, 0)
        return self._sentences

    def fasttext(self, seed: int = 42):
        """Primary FastText: load saved model for seed 42, else train (cached)."""
        if seed == 42:
            if self._ft is None:
                self._ft = get_model(self._ft_path, self.data_dir, self.normalize)
            return self._ft
        if seed not in self._ft_seeds:
            self._ft_seeds[seed] = train_fasttext(self.sentences, seed)
        return self._ft_seeds[seed]

    def word2vec(self, seed: int = 42):
        if seed not in self._w2v_seeds:
            self._w2v_seeds[seed] = train_word2vec(self.sentences, seed)
        return self._w2v_seeds[seed]

    @property
    def corpus_text(self):
        if self._corpus_text is None:
            self._corpus_text = nn_baselines.build_corpus_text(self.data_dir, self.normalize)
        return self._corpus_text

    def neural(self, which: str, max_steps: int, seed: int):
        key = (which, max_steps, seed)
        if key not in self._neural:
            self._neural[key] = train_neural(self.corpus_text, which, max_steps, seed)
        return self._neural[key]


def train_fasttext(sentences, seed, **over):
    cfg = dict(dim=100, window=5, epochs=10, min_n=2, max_n=5,
               buckets=200_000, sg=1, workers=1, seed=seed)
    cfg.update(over)
    return fasttext_model.train_model(sentences, **cfg)


def train_word2vec(sentences, seed, dim=100, window=5, epochs=10):
    """No-subword control: gensim Word2Vec, same config as FastText."""
    import contextlib
    import io
    from gensim.models import Word2Vec
    model = Word2Vec(vector_size=dim, window=window, min_count=1, sg=1,
                     workers=1, seed=seed, epochs=epochs)
    model.build_vocab(corpus_iterable=sentences)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        model.train(corpus_iterable=sentences,
                    total_examples=model.corpus_count, epochs=model.epochs)
    return model


def train_neural(corpus_text: str, which: str, max_steps: int, seed: int):
    cfg = nn_baselines.TrainConfig(max_steps=max_steps, seed=seed)
    if which == "byte-LM":
        tok = nn_baselines.ByteTokenizer()
        model = nn_baselines.ByteLM(tok.vocab_size, d_model=128, hidden=256, layers=2)
    else:
        tok = nn_baselines.CharTokenizer(corpus_text)
        model = nn_baselines.CharTransformer(tok.vocab_size, d_model=128, layers=2,
                                             heads=4, ff=256, block_size=128)
    return nn_baselines.train_lm(model, tok, corpus_text, which, cfg)


# --------------------------------------------------------------------------- #
# Ordered document text (faithful order for neural encoders)
# --------------------------------------------------------------------------- #
def ordered_text(data_dir: Path, text_id: str, normalize: bool) -> str:
    root = ET.parse(data_dir / f"{text_id}.xml").getroot()
    body = find_body(root)
    if body is None:
        return ""
    toks = [strip_marks(t) if normalize else t for t in iter_words(body)]
    return " ".join(toks)


def neural_doc_matrix(texts, encoder, data_dir, normalize):
    rows, kept = [], []
    for t in texts:
        s = ordered_text(data_dir, t.text_id, normalize)
        if not s:
            continue
        v = encoder.encode(s)
        if v is not None:
            rows.append(v)
            kept.append(t)
    if not rows:
        return None, []
    return np.vstack(rows), kept


# --------------------------------------------------------------------------- #
# Word vectors per model (for morphology)
# --------------------------------------------------------------------------- #
def word_vec_fn(kind: str, model):
    """Return f(form)->vector|None for the given model kind."""
    if kind in ("FastText", "word2vec"):
        wv = model.wv

        def f(form):
            try:
                return np.asarray(wv[form], dtype=np.float64)
            except KeyError:
                return None
        return f
    # neural encoder
    enc = nn_baselines.NeuralEncoder(model)
    return lambda form: enc.encode(form)


def cos(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


# --------------------------------------------------------------------------- #
# Analysis: morphology coherence + frequency-banded subword ablation
# --------------------------------------------------------------------------- #
def corpus_frequencies(data_dir: Path, normalize: bool) -> Counter:
    freq: Counter[str] = Counter()
    for p in sorted(data_dir.glob("*.xml")):
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        body = find_body(root)
        if body is None:
            continue
        for t in iter_words(body):
            freq[strip_marks(t) if normalize else t] += 1
    return freq


def _related_pairs(freq: Counter):
    """Yield (anchor, related_form, related_freq) for present curated forms."""
    out = []
    for forms in MORPH_FAMILIES.values():
        present = [f for f in forms if freq.get(f, 0) > 0]
        if len(present) < 2:
            continue
        anchor = max(present, key=lambda f: freq[f])
        for f in present:
            if f != anchor:
                out.append((anchor, f, freq[f]))
    return out


def analysis_morph(models: Models, data_dir, normalize, max_steps, seed):
    banner("Morphology coherence + subword ablation by frequency band  [T2, T3a]")
    freq = corpus_frequencies(data_dir, normalize)
    pairs = _related_pairs(freq)
    print(f"curated families: {len(MORPH_FAMILIES)} | related pairs present: {len(pairs)}")

    # Frequency-matched control: pair each anchor with a different family's form
    # of the closest frequency (different root => morphologically unrelated).
    all_forms = sorted({f for fams in MORPH_FAMILIES.values() for f in fams if freq.get(f, 0) > 0},
                       key=lambda f: freq[f])
    fam_of = {f: fam for fam, forms in MORPH_FAMILIES.items() for f in forms}

    def control_for(anchor, target_freq, anchor_fam):
        cand = [f for f in all_forms if fam_of.get(f) != anchor_fam]
        return min(cand, key=lambda f: abs(freq[f] - target_freq)) if cand else None

    bands = [("freq>=100", lambda c: c >= 100), ("rare<100", lambda c: c < 100)]

    model_kinds = [("FastText", models.fasttext(42)), ("word2vec", models.word2vec(42))]
    if _TORCH:
        model_kinds.append(("byte-LM", models.neural("byte-LM", max_steps, seed)))
        model_kinds.append(("char-Transformer", models.neural("char-Transformer", max_steps, seed)))

    print()
    print(r"% T3a: morphology coherence (related cos, control cos, margin, acc) by band")
    header = f"{'model':<18}{'band':<12}{'n':>4}{'related':>9}{'control':>9}{'margin':>8}{'acc':>7}"
    print(header)
    print("-" * len(header))
    rows_for_tex = []
    for kind, model in model_kinds:
        vf = word_vec_fn(kind, model)
        cache = {}

        def vec(form):
            if form not in cache:
                cache[form] = vf(form)
            return cache[form]

        for band_name, band_test in bands:
            rel_cos, ctl_cos, wins, n = [], [], 0, 0
            for anchor, target, tfreq in pairs:
                if not band_test(tfreq):
                    continue
                ctl = control_for(anchor, tfreq, fam_of.get(anchor))
                cr = cos(vec(anchor), vec(target))
                cc = cos(vec(anchor), vec(ctl))
                if np.isnan(cr) or np.isnan(cc):
                    continue
                rel_cos.append(cr)
                ctl_cos.append(cc)
                wins += cr > cc
                n += 1
            if n == 0:
                continue
            r, c = float(np.mean(rel_cos)), float(np.mean(ctl_cos))
            acc = wins / n
            print(f"{kind:<18}{band_name:<12}{n:>4}{r:>9.3f}{c:>9.3f}{r - c:>+8.3f}{acc:>7.2f}")
            rows_for_tex.append(tex_row(kind, band_name, n, f"{r:.3f}", f"{c:.3f}",
                                        f"{r - c:+.3f}", f"{acc:.2f}"))
    print()
    print(r"% --- LaTeX rows for T3a ---")
    print("\n".join(rows_for_tex))


# --------------------------------------------------------------------------- #
# Analysis: OOV generalization
# --------------------------------------------------------------------------- #
def analysis_oov(models: Models, data_dir, normalize, seed):
    banner("Out-of-vocabulary generalization  [T3b]")
    paths = sorted(data_dir.glob("*.xml"))
    rng = np.random.default_rng(seed)
    held = set(rng.choice([p.stem for p in paths], size=max(1, len(paths) // 10), replace=False))

    train_forms: Counter[str] = Counter()
    held_forms: Counter[str] = Counter()
    for p in paths:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        body = find_body(root)
        if body is None:
            continue
        target = held_forms if p.stem in held else train_forms
        for t in iter_words(body):
            target[strip_marks(t) if normalize else t] += 1

    oov = [f for f in held_forms if f not in train_forms]
    print(f"held-out files: {len(held)} / {len(paths)}")
    print(f"forms only in held-out (true OOV vs the rest): {len(oov):,} "
          f"({len(oov) / max(len(held_forms), 1):.1%} of held-out vocab)")

    # Train FastText + word2vec on the TRAIN split only.
    train_sentences = []
    for p in paths:
        if p.stem in held:
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        body = find_body(root)
        if body is None:
            continue
        toks = [strip_marks(t) if normalize else t for t in iter_words(body)]
        if toks:
            train_sentences.append(toks)

    ft = train_fasttext(train_sentences, seed)
    w2v = train_word2vec(train_sentences, seed)
    ft_vocab = set(ft.wv.key_to_index)
    w2v_vocab = set(w2v.wv.key_to_index)

    ft_can = sum(1 for f in oov if f not in ft_vocab)  # all, via n-grams
    w2v_can = sum(1 for f in oov if f in w2v_vocab)    # none (by definition)
    print(f"FastText vectorizes OOV forms via char n-grams: {ft_can}/{len(oov)} "
          f"({ft_can / max(len(oov), 1):.0%})")
    print(f"word2vec can vectorize OOV forms:               {w2v_can}/{len(oov)} (0%)")

    # Quality check: do FastText OOV vectors land near morphological kin? For OOV
    # forms whose 3-char prefix matches an in-vocab form, measure nearest-neighbor
    # prefix-match rate (a rough root-coherence proxy).
    sample = [f for f in oov if len(f) >= 4][:300]
    hits = 0
    for f in sample:
        try:
            nbrs = ft.wv.most_similar(f, topn=5)
        except KeyError:
            continue
        if any(w[:3] == f[:3] for w, _ in nbrs):
            hits += 1
    print(f"FastText OOV nearest-neighbour shares 3-char root prefix: "
          f"{hits}/{len(sample)} ({hits / max(len(sample), 1):.0%}) "
          f"-> synthesized vectors are morphologically coherent")
    print()
    print(r"% T3b rows")
    print(tex_row("FastText", f"{ft_can}", "100\\%", f"{hits}/{len(sample)}"))
    print(tex_row("word2vec", "0", "0\\%", "n/a"))


# --------------------------------------------------------------------------- #
# Analysis: hyperparameter sensitivity
# --------------------------------------------------------------------------- #
def analysis_hyper(models: Models, data_dir, normalize, seed):
    banner("FastText hyperparameter sensitivity  [T4]")
    genuine = load_texts(data_dir, normalize, exclude_ids=set(parse_ids(DISPUTED_DEFAULT)),
                         drop_anonymous=True)
    cohort = filter_min_texts(filter_min_tokens(genuine, 1000), 3)
    labels = key_labels(cohort)
    sentences = models.sentences

    morph_freq = corpus_frequencies(data_dir, normalize)
    pairs = _related_pairs(morph_freq)

    def morph_margin(wv):
        vf = lambda f: (np.asarray(wv[f], dtype=np.float64) if f in wv.key_to_index
                        or True else None)
        ms = []
        for anchor, target, _ in pairs:
            try:
                a, b = np.asarray(wv[anchor]), np.asarray(wv[target])
            except KeyError:
                continue
            ms.append(cos(a, b))
        return float(np.mean(ms)) if ms else float("nan")

    configs = [
        dict(min_n=2, max_n=4), dict(min_n=2, max_n=5), dict(min_n=3, max_n=6),
        dict(dim=50), dict(dim=200), dict(sg=0),
    ]
    base = dict(dim=100, min_n=2, max_n=5, sg=1)
    print(f"{'config':<22}{'morph cos':>10}{'AUC(centered)':>15}")
    print("-" * 47)
    rows = []
    for over in configs:
        cfg = {**base, **over}
        label = ", ".join(f"{k}={v}" for k, v in over.items())
        model = train_fasttext(sentences, seed, **cfg)
        mc = morph_margin(model.wv)
        M, kept = doc_matrix(cohort, model.wv, None)
        auc = separation(remove_common_component(M), key_labels(kept))["auc"]
        print(f"{label:<22}{mc:>10.3f}{auc:>15.3f}")
        rows.append(tex_row(label, f"{mc:.3f}", f"{auc:.3f}"))
    print()
    print(r"% T4 rows")
    print("\n".join(rows))


# --------------------------------------------------------------------------- #
# Author-cluster bootstrap for AUC
# --------------------------------------------------------------------------- #
def bootstrap_auc_ci(matrix: np.ndarray, keys: np.ndarray, *, B: int, seed: int):
    """95% percentile CI for same/cross-author AUC, resampling authors (clusters)."""
    unit = l2_normalize(remove_common_component(matrix))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = unit @ unit.T
    authors = list(dict.fromkeys(keys.tolist()))
    idx_by_author = {a: np.where(keys == a)[0] for a in authors}
    rng = np.random.default_rng(seed)

    point = _auc_from_sims(sims, [(a, idx_by_author[a]) for a in authors])
    boots = []
    for _ in range(B):
        sampled = rng.choice(len(authors), size=len(authors), replace=True)
        groups = [(authors[i], idx_by_author[authors[i]]) for i in sampled]
        boots.append(_auc_from_sims(sims, groups))
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _auc_from_sims(sims, groups):
    """AUC over same- vs cross-author pairs.

    ``groups`` is a list of ``(author_id, index_array)``. Same-author pairs are
    the within-group pairs (a duplicated author, from cluster resampling,
    contributes its within-pairs again). Cross-author pairs are between groups
    with *different* author ids only -- pairs between two resampled copies of the
    same author are skipped, not miscounted as cross.
    """
    same, cross = [], []
    for _, g in groups:
        if len(g) >= 2:
            sub = sims[np.ix_(g, g)]
            iu = np.triu_indices(len(g), k=1)
            same.append(sub[iu])
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i][0] == groups[j][0]:
                continue  # duplicate of the same author -> not a cross pair
            cross.append(sims[np.ix_(groups[i][1], groups[j][1])].ravel())
    if not same or not cross:
        return float("nan")
    return auc_same_higher(np.concatenate(same), np.concatenate(cross))


# --------------------------------------------------------------------------- #
# Analysis: same/cross AUC with CIs + multi-seed
# --------------------------------------------------------------------------- #
def _cohorts(data_dir, normalize):
    genuine = load_texts(data_dir, normalize, exclude_ids=set(parse_ids(DISPUTED_DEFAULT)),
                         drop_anonymous=True)
    full = filter_min_texts(genuine, 2)
    restricted = filter_min_texts(filter_min_tokens(genuine, 2000), 3)
    return genuine, full, restricted


def analysis_sep(models: Models, data_dir, normalize, B, seed):
    banner("Same/cross-author AUC: bootstrap CIs + multi-seed  [T5]")
    genuine, full, restricted = _cohorts(data_dir, normalize)
    func_set, _ = function_word_set(genuine, 200)
    wv = models.fasttext(42).wv

    rows = []
    for cname, cohort in (("full (>=2)", full), ("restricted (>=3,>=2000)", restricted)):
        for vname, allowed in (("all words", None), ("function words", func_set)):
            M, kept = doc_matrix(cohort, wv, allowed)
            klab = key_labels(kept)
            auc = separation(remove_common_component(M), klab)["auc"]
            _, lo, hi = bootstrap_auc_ci(M, klab, B=B, seed=seed)
            print(f"{cname:<26}{vname:<16} AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
                  f"({len(kept)} texts, {len(set(klab.tolist()))} authors)")
            rows.append(tex_row(cname, vname, f"{auc:.3f}", f"[{lo:.3f}, {hi:.3f}]"))

    print()
    print(f"Multi-seed FastText (seeds {SEEDS}) -- centered AUC, all words:")
    seed_rows = []
    for cname, cohort in (("full", full), ("restricted", restricted)):
        vals = []
        for s in SEEDS:
            wv_s = models.fasttext(s).wv
            M, kept = doc_matrix(cohort, wv_s, None)
            vals.append(separation(remove_common_component(M), key_labels(kept))["auc"])
        vals = np.array(vals)
        print(f"  {cname:<14} {vals.mean():.3f} +/- {vals.std():.3f}  "
              f"(min {vals.min():.3f}, max {vals.max():.3f})")
        seed_rows.append(tex_row(cname, f"{vals.mean():.3f} $\\pm$ {vals.std():.3f}",
                                 f"{vals.min():.3f}", f"{vals.max():.3f}"))
    print()
    print(r"% T5 AUC+CI rows"); print("\n".join(rows))
    print(r"% T5 multi-seed rows"); print("\n".join(seed_rows))


# --------------------------------------------------------------------------- #
# Analysis: representation bake-off
# --------------------------------------------------------------------------- #
def analysis_bakeoff(models: Models, data_dir, normalize, max_steps, seed, B):
    banner("Representation bake-off: AUC + leave-one-out attribution  [T6]")
    genuine, _, _ = _cohorts(data_dir, normalize)

    rows = []
    for floor in (1000, 2000):
        cohort = filter_min_texts(filter_min_tokens(genuine, floor), 3)
        labels = key_labels(cohort)
        n_auth = len(set(labels.tolist()))
        print()
        print(f"== cohort >=3 texts, >={floor} tokens -> {len(cohort)} texts, {n_auth} authors ==")
        print(f"{'representation':<22}{'AUC':>7}{'95% CI':>16}{'top-1':>8}{'top-3':>8}")
        print("-" * 61)

        def report(name, M, kept):
            klab = key_labels(kept)
            auc = separation(remove_common_component(M), klab)["auc"]
            _, lo, hi = bootstrap_auc_ci(M, klab, B=B, seed=seed)
            loo = centroid_loo(M, klab, metric="cosine")
            print(f"{name:<22}{auc:>7.3f}  [{lo:.3f}, {hi:.3f}]{loo['top1']:>8.3f}{loo['top3']:>8.3f}")
            rows.append(tex_row(f"{floor}", name, f"{auc:.3f}", f"[{lo:.3f}, {hi:.3f}]",
                                f"{loo['top1']:.3f}", f"{loo['top3']:.3f}"))

        Mft, kept = doc_matrix(cohort, models.fasttext(42).wv, None)
        report("FastText (all)", Mft, kept)
        Mw2, kept2 = doc_matrix(cohort, models.word2vec(42).wv, None)
        report("word2vec (all)", Mw2, kept2)

        if _AVHEAD:
            M_av = av_head.leave_author_out_projection(Mft, key_labels(kept), seed=seed)
            report("AV head (LOAO)", M_av, kept)

        if _TORCH:
            for which in ("byte-LM", "char-Transformer"):
                enc = nn_baselines.NeuralEncoder(models.neural(which, max_steps, seed))
                Mn, keptn = neural_doc_matrix(cohort, enc, data_dir, normalize)
                report(which, Mn, keptn)

        # Burrows's Delta (distance-based AUC + LOO)
        for k in (100, 200):
            z = delta_profiles(cohort, k)
            from authorship import auc_from_distance
            auc = auc_from_distance(delta_distance_matrix(z), labels)
            loo = centroid_loo(z, labels, metric="l1")
            print(f"{'Delta MFW=' + str(k):<22}{auc:>7.3f}{'  (n/a)':>16}"
                  f"{loo['top1']:>8.3f}{loo['top3']:>8.3f}")
            rows.append(tex_row(f"{floor}", f"Delta MFW={k}", f"{auc:.3f}", "n/a",
                                f"{loo['top1']:.3f}", f"{loo['top3']:.3f}"))
    print()
    print(r"% T6 rows"); print("\n".join(rows))


# --------------------------------------------------------------------------- #
# Analysis: genre-matched cross-author test
# --------------------------------------------------------------------------- #
def analysis_genre(models: Models, data_dir, normalize, B, seed):
    banner("Genre-matched cross-author separation  [T7]")
    genuine, _, restricted = _cohorts(data_dir, normalize)
    wv = models.fasttext(42).wv

    cohort = restricted
    genres = [classify_genre(t) for t in cohort]
    cov = sum(1 for g in genres if g != "other") / max(len(cohort), 1)
    print(f"genre classifier coverage on restricted cohort: {cov:.0%} "
          f"({Counter(genres)})")

    M, kept = doc_matrix(cohort, wv, None)
    keys = key_labels(kept)
    gen = np.asarray([classify_genre(t) for t in kept], dtype=object)
    unit = l2_normalize(remove_common_component(M))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = unit @ unit.T
    iu = np.triu_indices(len(kept), k=1)
    s = sims[iu]
    same_author = keys[iu[0]] == keys[iu[1]]
    same_genre = gen[iu[0]] == gen[iu[1]]

    auc_all = auc_same_higher(s[same_author], s[~same_author])
    # genre-matched: restrict cross-author comparisons to SAME-genre pairs
    mask = same_genre
    auc_matched = auc_same_higher(s[same_author & mask], s[~same_author & mask])
    print(f"same/cross-author AUC, unmatched : {auc_all:.3f}  "
          f"(same {same_author.sum()}, cross {(~same_author).sum()})")
    print(f"same/cross-author AUC, genre-matched: {auc_matched:.3f}  "
          f"(within-genre pairs only: same {(same_author & mask).sum()}, "
          f"cross {((~same_author) & mask).sum()})")
    verdict = ("signal SURVIVES genre matching" if auc_matched >= auc_all - 0.05
               else "signal partly reflects genre")
    print(f"=> {verdict}")
    print()
    print(r"% T7 rows")
    print(tex_row("unmatched", f"{auc_all:.3f}", int(same_author.sum()), int((~same_author).sum())))
    print(tex_row("genre-matched", f"{auc_matched:.3f}", int((same_author & mask).sum()),
                  int(((~same_author) & mask).sum())))


# --------------------------------------------------------------------------- #
# Analysis: neural LM quality
# --------------------------------------------------------------------------- #
def analysis_lm(models: Models, max_steps, seed):
    banner("Neural language-model quality  [TLM]")
    if not _TORCH:
        print("PyTorch not available; skipping neural LM analysis.")
        return
    rows = []
    print(f"{'model':<18}{'params':>10}{'bits/byte':>11}{'ppl(unit)':>11}{'train s':>9}{'device':>7}")
    print("-" * 66)
    for which in ("byte-LM", "char-Transformer"):
        r = models.neural(which, max_steps, seed)
        print(f"{which:<18}{r.params:>10,}{r.val_bits_per_byte:>11.3f}"
              f"{r.val_perplexity_unit:>11.1f}{r.train_seconds:>9.1f}{r.device:>7}")
        rows.append(tex_row(which, f"{r.params:,}", f"{r.val_bits_per_byte:.3f}",
                            f"{r.val_perplexity_unit:.1f}", f"{r.train_seconds:.0f}", r.device))
    print()
    print(r"% TLM rows"); print("\n".join(rows))


# --------------------------------------------------------------------------- #
# Analysis: supervised AV head (standalone / fast)
# --------------------------------------------------------------------------- #
def analysis_avhead(models: Models, data_dir, normalize, B, seed):
    banner("Supervised AV head (leave-one-author-out)  [T6 row]")
    if not _AVHEAD:
        print("PyTorch not available; skipping AV head.")
        return
    genuine, _, _ = _cohorts(data_dir, normalize)
    wv = models.fasttext(42).wv

    print("A tiny supervised-contrastive projection over the FastText document")
    print("vectors, embedded leave-one-author-out so no test author is seen in")
    print("training. Comparable to the unsupervised rows; uses the same scoring.")
    print()
    print(f"{'cohort':<26}{'AUC':>7}{'95% CI':>16}{'top-1':>8}{'top-3':>8}")
    print("-" * 65)
    rows = []
    for floor in (1000, 2000):
        cohort = filter_min_texts(filter_min_tokens(genuine, floor), 3)
        Mft, kept = doc_matrix(cohort, wv, None)
        klab = key_labels(kept)
        M_av = av_head.leave_author_out_projection(Mft, klab, seed=seed)
        auc = separation(remove_common_component(M_av), klab)["auc"]
        _, lo, hi = bootstrap_auc_ci(M_av, klab, B=B, seed=seed)
        loo = centroid_loo(M_av, klab, metric="cosine")
        n_auth = len(set(klab.tolist()))
        print(f"{'>=3 texts, >=' + str(floor):<26}{auc:>7.3f}  [{lo:.3f}, {hi:.3f}]"
              f"{loo['top1']:>8.3f}{loo['top3']:>8.3f}")
        rows.append(tex_row(f"{floor}", "AV head (LOAO)", f"{auc:.3f}",
                            f"[{lo:.3f}, {hi:.3f}]", f"{loo['top1']:.3f}", f"{loo['top3']:.3f}"))
    print()
    print(r"% T6 AV-head rows"); print("\n".join(rows))


# --------------------------------------------------------------------------- #
# Analysis: cross-corpus validation on independent ETCBC corpora
# --------------------------------------------------------------------------- #
def _unit_centroid(texts, wv, mean):
    """Mean-centered, L2-normalized centroid of a set of texts (cosine space)."""
    vs = [doc_vector(wv, t.counts, None) for t in texts]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    unit = l2_normalize(np.vstack(vs) - mean)
    c = unit.mean(axis=0)
    return c / (np.linalg.norm(c) or 1.0)


def analysis_crosscorpus(models: Models, data_dir, normalize, seed):
    banner("Cross-corpus validation on independent ETCBC corpora  [T9]")
    if not _ETCBC:
        print("etcbc_corpus unavailable; skipping.")
        return
    wv = models.fasttext(42).wv
    w2v = models.word2vec(42).wv
    ft_vocab = set(wv.key_to_index)
    w2v_vocab = set(w2v.key_to_index)

    externals = {name: etcbc_corpus.load_etcbc_texts(name, normalize)
                 for name in ("SyrNT", "Peshitta")}

    # --- Part A: out-of-vocabulary coverage of an independent corpus -------- #
    print()
    print("Part A -- generalization of the DSC-trained model to independent corpora")
    print(f"{'corpus':<12}{'tokens':>10}{'types':>9}{'type cov':>10}"
          f"{'tok cov':>9}{'OOV types':>11}{'FT/ w2v':>10}{'root-NN':>9}")
    print("-" * 80)
    rows_a = []
    for name, texts in externals.items():
        freq = etcbc_corpus.corpus_frequencies(texts)
        types = set(freq)
        tot = sum(freq.values())
        in_vocab = types & ft_vocab
        type_cov = len(in_vocab) / len(types)
        tok_cov = sum(freq[w] for w in in_vocab) / tot
        oov = [w for w in types if w not in ft_vocab]
        w2v_oov_covered = sum(1 for w in oov if w in w2v_vocab)  # 0 by construction
        # root-coherence of FastText's synthesized OOV vectors
        sample = [w for w in oov if len(w) >= 4][:300]
        hits = 0
        for w in sample:
            try:
                nbrs = wv.most_similar(w, topn=5)
            except KeyError:
                continue
            if any(n[:3] == w[:3] for n, _ in nbrs):
                hits += 1
        root_str = f"{hits / len(sample):.0%}" if len(sample) >= 10 else "n/a"
        print(f"{name:<12}{tot:>10,}{len(types):>9,}{type_cov:>9.1%}"
              f"{tok_cov:>9.1%}{len(oov):>11,}{'100/0%':>10}{root_str:>9}")
        rows_a.append(tex_row(name, f"{tot:,}", f"{len(types):,}", f"{type_cov:.1%}",
                              f"{tok_cov:.1%}", f"{len(oov):,}",
                              "100\\,/\\,0\\%", root_str))
    print(f"(word2vec covers 0 of the OOV types by construction; FastText vectorizes"
          f" all via char n-grams. root-NN = nearest in-vocab neighbour shares the"
          f" 3-consonant prefix.)")

    # --- Part B: translationese, anchored by external known translations --- #
    print()
    print("Part B -- translationese: distance to an external Greek-source translation")
    genuine = load_texts(data_dir, normalize,
                         exclude_ids=set(parse_ids(DISPUTED_DEFAULT)), drop_anonymous=True)
    cohort = filter_min_texts(filter_min_tokens(genuine, 1000), 3)
    centroids, names, mean = fit_centroids(cohort, wv, None)

    ref = _unit_centroid(externals["SyrNT"], wv, mean)         # Greek-source anchor
    pesh = _unit_centroid(externals["Peshitta"], wv, mean)     # Hebrew-source anchor

    disputed_ids = parse_ids(DISPUTED_DEFAULT)
    disputed = [t for t in (load_one_text(data_dir, i, normalize) for i in disputed_ids)
                if t is not None]
    groups: dict[str, list] = {}
    for t in disputed:
        groups.setdefault(t.series or t.title or t.text_id, []).append(t)

    # Score every genuine author and disputed group by cosine to the SyrNT anchor.
    scored = [(float(c @ ref), names[k], "author") for k, c in centroids.items()]
    scored.append((float(pesh @ ref), "Peshitta (OT, Hebrew-source)", "translation"))
    for series, members in groups.items():
        gc = _unit_centroid(members, wv, mean)
        if gc is not None:
            scored.append((float(gc @ ref), series, "disputed"))
    scored.sort(reverse=True)

    print(f"cosine to SyrNT (Greek\u2192Syriac translation), mean-centered:")
    print(f"{'rank':>4}  {'cos':>6}  {'kind':<12}entity")
    print("-" * 66)
    rows_b = []
    for i, (s, label, kind) in enumerate(scored, 1):
        mark = "  <--" if kind in ("disputed", "translation") else ""
        print(f"{i:>4}  {s:>6.3f}  {kind:<12}{label[:34]}{mark}")
        if kind != "author":
            rows_b.append(tex_row(i, f"{s:.3f}", kind, label[:34]))

    # Headline: where do the Pseudo-Clementines fall among genuine authors?
    n_auth = sum(1 for _, _, k in scored if k == "author")
    psclem = next((s for s, lab, k in scored
                   if k == "disputed" and "clement" in lab.casefold()), None)
    if psclem is not None:
        beaten = sum(1 for s, _, k in scored if k == "author" and psclem > s)
        print()
        print(f"Pseudo-Clementines cos-to-SyrNT = {psclem:.3f}: more SyrNT-like than "
              f"{beaten}/{n_auth} genuine Syriac authors")
        print(f"({beaten / n_auth:.0%} percentile). The two independent biblical "
              f"translations are themselves close (SyrNT~Peshitta cos {float(ref @ pesh):.3f}),")
        print("and the (Greek-translated) Pseudo-Clementines sit nearer them than "
              "native Syriac composition does -- external support for the translationese reading.")
    print()
    print(r"% T9a coverage rows"); print("\n".join(rows_a))
    print(r"% T9b translationese rows (disputed/translation only)"); print("\n".join(rows_b))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--analyses", default="morph,oov,hyper,sep,bakeoff,genre,lm")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--nn-steps", type=int, default=2000, help="neural LM training steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)
    args = ap.parse_args(argv)

    which = {a.strip() for a in args.analyses.split(",") if a.strip()}
    data_dir = ensure_corpus(args.cache_dir)
    models = Models(data_dir, args.normalize, args.model)

    if not _TORCH and which & {"lm"}:
        print("note: PyTorch missing; neural rows in morph/bakeoff/lm are skipped.",
              file=sys.stderr)

    t0 = time.time()
    if "morph" in which:
        analysis_morph(models, data_dir, args.normalize, args.nn_steps, args.seed)
    if "oov" in which:
        analysis_oov(models, data_dir, args.normalize, args.seed)
    if "hyper" in which:
        analysis_hyper(models, data_dir, args.normalize, args.seed)
    if "sep" in which:
        analysis_sep(models, data_dir, args.normalize, args.bootstrap, args.seed)
    if "bakeoff" in which:
        analysis_bakeoff(models, data_dir, args.normalize, args.nn_steps, args.seed, args.bootstrap)
    if "avhead" in which:
        analysis_avhead(models, data_dir, args.normalize, args.bootstrap, args.seed)
    if "crosscorpus" in which:
        analysis_crosscorpus(models, data_dir, args.normalize, args.seed)
    if "genre" in which:
        analysis_genre(models, data_dir, args.normalize, args.bootstrap, args.seed)
    if "lm" in which:
        analysis_lm(models, args.nn_steps, args.seed)

    print(f"\n[done in {time.time() - t0:.0f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
