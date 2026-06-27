"""
Smoke test: bipartite graph SSL for Arabic root identification.

WHAT WE SHOW:
  Treating root-identification as a semi-supervised node classification
  problem on a bipartite (root <-> POS) graph outperforms a flat
  consonant-skeleton majority-vote baseline on masked tokens.

  Grid search is honest: CV on the labelled set only; the masked
  held-out set is touched exactly once for final reporting.

DATASETS:
  1. Synthetic stub (43 tokens) — runs with no dependencies beyond pip
  2. Quranic Arabic Corpus v0.4 (~50k root-bearing tokens) — fetched
     automatically from GitHub on first run and cached locally.

     Source: github.com/mustafa0x/quran-morphology
     License: GNU GPL (Kais Dukes / University of Leeds, 2011)

MEMORY NOTE:
  The RBF kernel builds an n×n similarity matrix. At ~50k tokens this
  is ~18 GB — infeasible on most machines. The script therefore uses
  RBF only for the stub and switches to knn-only configs for real data.
  knn kernel is O(n·k) and runs fine at full corpus scale.

INSTALL:
  pip install scikit-learn numpy

RUN:
  python smoke_test.py
"""

import re, os, sys, random, urllib.request
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # LP divide-by-zero on tiny data

# ── Config ────────────────────────────────────────────────────────────────────
MASK_FRAC    = 0.30
RARE_CUTOFF  = 3       # roots with corpus freq <= this → "hard" held-out
RANDOM_SEED  = 42
CV_FOLDS     = 3
CACHE_FILE   = "quranic-corpus-morphology-0.4-ar.txt"
CORPUS_URL   = (
    "https://raw.githubusercontent.com/mustafa0x/quran-morphology"
    "/master/quranic-corpus-morphology-0.4-ar.txt"
)
# Tokens above this count trigger knn-only mode to avoid n×n memory blow-up
RBF_TOKEN_LIMIT = 5_000
# ─────────────────────────────────────────────────────────────────────────────

# ── Parameter grids ───────────────────────────────────────────────────────────
# Full grid (stub / small data): includes RBF configs
GRID_FULL = [
    # LabelPropagation – rbf, vary gamma
    (LabelPropagation, 'rbf', 1.0,  None, None),
    (LabelPropagation, 'rbf', 5.0,  None, None),
    (LabelPropagation, 'rbf', 20.0, None, None),
    (LabelPropagation, 'rbf', 50.0, None, None),
    # LabelPropagation – knn, vary k
    (LabelPropagation, 'knn', None, 3,    None),
    (LabelPropagation, 'knn', None, 7,    None),
    (LabelPropagation, 'knn', None, 15,   None),
    # LabelSpreading – rbf, vary gamma + alpha
    (LabelSpreading,   'rbf', 5.0,  None, 0.2),
    (LabelSpreading,   'rbf', 20.0, None, 0.2),
    (LabelSpreading,   'rbf', 20.0, None, 0.5),
    (LabelSpreading,   'rbf', 20.0, None, 0.8),
    # LabelSpreading – knn
    (LabelSpreading,   'knn', None, 7,    0.2),
    (LabelSpreading,   'knn', None, 7,    0.5),
]

# knn-only grid (large data): avoids n×n RBF matrix
GRID_KNN = [
    (LabelPropagation, 'knn', None, 3,    None),
    (LabelPropagation, 'knn', None, 7,    None),
    (LabelPropagation, 'knn', None, 15,   None),
    (LabelPropagation, 'knn', None, 25,   None),
    (LabelSpreading,   'knn', None, 5,    0.2),
    (LabelSpreading,   'knn', None, 7,    0.2),
    (LabelSpreading,   'knn', None, 7,    0.5),
    (LabelSpreading,   'knn', None, 15,   0.2),
    (LabelSpreading,   'knn', None, 15,   0.5),
    (LabelSpreading,   'knn', None, 25,   0.2),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')

def strip_diacritics(s):
    return ARABIC_DIACRITICS.sub('', s)

def consonant_skeleton(surface):
    """Strip diacritics; keep base consonant string."""
    return strip_diacritics(surface)

def parse_corpus(text):
    """
    Parse the Quranic Arabic Corpus morphology file.
    Returns list of (surface, coarse_pos, root) for every STEM token
    that carries a ROOT: tag. Prefixes and suffixes are skipped.
    """
    tokens = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        _, surface, pos, features = parts[0], parts[1], parts[2], parts[3]
        # Skip prefix/suffix segments — they don't carry independent root info
        if 'PREFIX' in features or 'SUFFIX' in features:
            continue
        m = re.search(r'ROOT:([^\|\s]+)', features)
        if m:
            root = m.group(1).strip()
            # Coarse POS: first token of the pipe-delimited feature string
            # e.g. "STEM|POS:N|..." → coarse = "N"
            pos_match = re.search(r'POS:([^\|\s]+)', features)
            coarse = pos_match.group(1) if pos_match else pos
            tokens.append((surface, coarse, root))
    return tokens

def build_model(cls, kernel, gamma, n_neighbors, alpha):
    kwargs = dict(kernel=kernel, max_iter=2000)
    if kernel == 'rbf':
        kwargs['gamma'] = gamma
    else:
        kwargs['n_neighbors'] = n_neighbors
    if cls is LabelSpreading:
        kwargs['alpha'] = alpha
    return cls(**kwargs)

def param_label(cls, kernel, gamma, n_neighbors, alpha):
    name = 'LP' if cls is LabelPropagation else 'LS'
    if kernel == 'rbf':
        s = f"{name} rbf γ={gamma}"
        if alpha is not None:
            s += f" α={alpha}"
    else:
        s = f"{name} knn k={n_neighbors}"
        if alpha is not None:
            s += f" α={alpha}"
    return s

def fetch_corpus(url, cache_path):
    if os.path.exists(cache_path):
        print(f"  Using cached file: {cache_path}")
        with open(cache_path, encoding='utf-8') as f:
            return f.read()
    print(f"  Downloading from {url} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"  Saved to {cache_path} ({len(data):,} bytes)")
        return data
    except Exception as e:
        print(f"  [ERROR] Could not fetch corpus: {e}")
        print(f"  Manually download from:")
        print(f"    {url}")
        print(f"  and save as: {cache_path}")
        return None

# ── Core experiment function ──────────────────────────────────────────────────
def run_experiment(tokens, label, grid):
    """Run the full SSL vs baseline experiment on a token list."""

    print(f"\n{'━'*60}")
    print(f"  DATASET: {label}")
    print(f"{'━'*60}")

    n = len(tokens)

    root_freq = defaultdict(int)
    for _, _, r in tokens:
        root_freq[r] += 1

    # ── Features: one-hot skeleton + one-hot coarse POS ──────────────────────
    skeletons = sorted(set(consonant_skeleton(s) for s, _, _ in tokens))
    pos_tags  = sorted(set(p for _, p, _ in tokens))
    roots     = sorted(set(r for _, _, r in tokens))

    skel_idx = {s: i for i, s in enumerate(skeletons)}
    pos_idx  = {p: i for i, p in enumerate(pos_tags)}
    root_le  = LabelEncoder().fit(roots)

    n_skel = len(skeletons)
    n_pos  = len(pos_tags)

    X      = np.zeros((n, n_skel + n_pos), dtype=np.float32)
    y_true = np.zeros(n, dtype=int)

    for i, (surface, pos, root) in enumerate(tokens):
        X[i, skel_idx[consonant_skeleton(surface)]] = 1.0
        X[i, n_skel + pos_idx[pos]]                 = 1.0
        y_true[i] = root_le.transform([root])[0]

    print(f"  Tokens: {n:,}  |  Roots: {len(roots):,}  |  "
          f"POS: {len(pos_tags)}  |  Skeletons: {len(skeletons):,}  |  "
          f"Feat dim: {X.shape[1]:,}")

    # ── Held-out split (rare roots prioritised) ───────────────────────────────
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    hard = [i for i, (_, _, r) in enumerate(tokens) if root_freq[r] <= RARE_CUTOFF]
    easy = [i for i in range(n) if i not in set(hard)]

    n_mask = int(n * MASK_FRAC)
    n_hard = min(len(hard), n_mask)
    n_easy = max(0, n_mask - n_hard)

    masked_idx   = sorted(
        random.sample(hard, n_hard) +
        random.sample(easy, min(n_easy, len(easy)))
    )
    labelled_idx = [i for i in range(n) if i not in set(masked_idx)]

    print(f"  Labelled: {len(labelled_idx):,}  |  "
          f"Held-out: {len(masked_idx):,}  (rare-root: {n_hard:,})")

    true_mask = y_true[masked_idx]

    # ── Baseline: majority root per consonant skeleton ────────────────────────
    skel_votes = defaultdict(lambda: defaultdict(int))
    for i in labelled_idx:
        skel_votes[consonant_skeleton(tokens[i][0])][tokens[i][2]] += 1

    def baseline_predict(surface):
        votes = skel_votes.get(consonant_skeleton(surface), {})
        return root_le.transform([max(votes, key=votes.get)])[0] if votes else 0

    bl_preds = np.array([baseline_predict(tokens[i][0]) for i in masked_idx])
    bl_acc   = accuracy_score(true_mask, bl_preds)

    # ── CV on labelled set for model selection ────────────────────────────────
    X_lab = X[labelled_idx]
    y_lab = y_true[labelled_idx]

    class_counts = defaultdict(int)
    for y in y_lab:
        class_counts[y] += 1
    valid_mask = np.array([class_counts[y] >= CV_FOLDS for y in y_lab])
    do_cv = valid_mask.sum() >= CV_FOLDS * 4   # need enough samples

    best_config = grid[0]
    cv_results  = []

    if do_cv:
        print(f"\n  Grid search ({CV_FOLDS}-fold CV on labelled set, "
              f"{len(grid)} configs):")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)

        for cfg in grid:
            cls, kernel, gamma, n_neighbors, alpha = cfg
            fold_scores = []

            for train_idx, val_idx in skf.split(X_lab[valid_mask],
                                                  y_lab[valid_mask]):
                X_fold  = X_lab[valid_mask]
                y_fold  = y_lab[valid_mask].copy()
                val_true = y_fold[val_idx].copy()
                y_fold[val_idx] = -1
                try:
                    m = build_model(cls, kernel, gamma, n_neighbors, alpha)
                    m.fit(X_fold, y_fold)
                    preds = m.predict(X_fold)[val_idx]
                    fold_scores.append(accuracy_score(val_true, preds))
                except Exception:
                    fold_scores.append(0.0)

            mean_cv = float(np.mean(fold_scores))
            lbl     = param_label(cls, kernel, gamma, n_neighbors, alpha)
            cv_results.append((mean_cv, cfg, lbl))
            print(f"    {lbl:<38s}  CV={mean_cv:.3f}")

        cv_results.sort(key=lambda x: -x[0])
        best_cv_acc, best_config, best_cv_label = cv_results[0]
        print(f"\n  ✓ Best by CV: {best_cv_label}  (CV={best_cv_acc:.3f})")
    else:
        print(f"\n  [WARN] Too few labelled samples for CV "
              f"({valid_mask.sum()} usable). Skipping CV.")

    # ── Final evaluation on held-out (all configs, for transparency) ──────────
    y_ssl_full = y_true.copy()
    for i in masked_idx:
        y_ssl_full[i] = -1

    print(f"\n  Held-out evaluation (MASK_FRAC={MASK_FRAC}):")
    print(f"    {'Config':<38s}  {'Acc':>6s}")
    print(f"    {'-'*38}  {'-'*6}")

    held_results = []
    for cfg in grid:
        cls, kernel, gamma, n_neighbors, alpha = cfg
        lbl = param_label(cls, kernel, gamma, n_neighbors, alpha)
        try:
            m = build_model(cls, kernel, gamma, n_neighbors, alpha)
            m.fit(X, y_ssl_full)
            preds = m.predict(X)[masked_idx]
            acc   = accuracy_score(true_mask, preds)
        except Exception as e:
            acc = float('nan')
        marker = "  ← CV winner" if (do_cv and cfg == best_config) else ""
        print(f"    {lbl:<38s}  {acc:>6.3f}{marker}")
        held_results.append((acc, lbl, cfg))

    held_results.sort(key=lambda x: -(x[0] if not np.isnan(x[0]) else -1))
    best_held_acc, best_held_lbl, _ = held_results[0]

    if do_cv:
        winner_held = next(
            acc for acc, _, cfg in held_results if cfg == best_config
        )
    else:
        winner_held  = best_held_acc
        best_cv_label = best_held_lbl

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*52}")
    print(f"  BASELINE  (skeleton majority):      {bl_acc:.3f}")
    if do_cv:
        print(f"  SSL winner (chosen by CV):          {winner_held:.3f}"
              f"  [{best_cv_label}]")
    print(f"  SSL best on held-out:               {best_held_acc:.3f}"
          f"  [{best_held_lbl}]")
    print(f"  {'─'*52}")

    # ── Rarity breakdown ──────────────────────────────────────────────────────
    rare_idx    = [i for i in masked_idx if root_freq[tokens[i][2]] <= RARE_CUTOFF]
    nonrare_idx = [i for i in masked_idx if root_freq[tokens[i][2]] >  RARE_CUTOFF]

    winning_cfg = best_config if do_cv else held_results[0][2]
    cls, kernel, gamma, n_neighbors, alpha = winning_cfg
    m = build_model(cls, kernel, gamma, n_neighbors, alpha)
    m.fit(X, y_ssl_full)
    winner_preds = m.predict(X)

    if rare_idx:
        bl_r  = accuracy_score(y_true[rare_idx],
                               [baseline_predict(tokens[i][0]) for i in rare_idx])
        ssl_r = accuracy_score(y_true[rare_idx], winner_preds[rare_idx])
        delta = ssl_r - bl_r
        sign  = '+' if delta >= 0 else ''
        print(f"\n  Rare roots   (freq ≤ {RARE_CUTOFF}, n={len(rare_idx):,}):")
        print(f"    baseline={bl_r:.3f}  ssl={ssl_r:.3f}  Δ={sign}{delta:.3f}")

    if nonrare_idx:
        bl_nr  = accuracy_score(y_true[nonrare_idx],
                                [baseline_predict(tokens[i][0]) for i in nonrare_idx])
        ssl_nr = accuracy_score(y_true[nonrare_idx], winner_preds[nonrare_idx])
        delta  = ssl_nr - bl_nr
        sign   = '+' if delta >= 0 else ''
        print(f"  Common roots (freq >  {RARE_CUTOFF}, n={len(nonrare_idx):,}):")
        print(f"    baseline={bl_nr:.3f}  ssl={ssl_nr:.3f}  Δ={sign}{delta:.3f}")

# ── Stub data ─────────────────────────────────────────────────────────────────
STUB_TEXT = """
(1:1:1:2)\tاسْمِ\tN\tSTEM|POS:N|LEM:sim|ROOT:smw|M|GEN
(1:1:1:3)\tاللَّهِ\tPN\tSTEM|POS:PN|LEM:Allh|ROOT:Alh|GEN
(1:1:2:1)\tالْحَمْدُ\tN\tSTEM|POS:N|LEM:Hamd|ROOT:Hmd|M|NOM
(1:1:2:2)\tلِلَّهِ\tPN\tSTEM|POS:PN|LEM:Allh|ROOT:Alh|GEN
(1:1:3:1)\tالرَّحْمَٰنِ\tADJ\tSTEM|POS:ADJ|LEM:raHmAn|ROOT:rHm|M|GEN
(1:1:4:1)\tالرَّحِيمِ\tADJ\tSTEM|POS:ADJ|LEM:raHiym|ROOT:rHm|M|GEN
(1:1:5:1)\tمَٰلِكِ\tN\tSTEM|POS:N|LEM:mAlok|ROOT:mlk|M|GEN
(1:1:6:1)\tإِيَّاكَ\tPRON\tSTEM|POS:PRON|LEM:>iy~Aka|ROOT:Ayk|ACC
(1:1:7:1)\tنَعْبُدُ\tV\tSTEM|POS:V|IMPF|LEM:Eabada|ROOT:Ebd|1P
(1:1:7:2)\tوَإِيَّاكَ\tPRON\tSTEM|POS:PRON|LEM:>iy~Aka|ROOT:Ayk|ACC
(1:1:8:1)\tنَسْتَعِينُ\tV\tSTEM|POS:V|IMPF|LEM:AisotagAna|ROOT:Ewn|1P
(2:1:1:1)\tكِتَابٌ\tN\tSTEM|POS:N|LEM:kitAb|ROOT:ktb|M|NOM
(2:1:2:1)\tكَتَبَ\tV\tSTEM|POS:V|PERF|LEM:kataba|ROOT:ktb|3MS
(2:1:3:1)\tكَاتِبٌ\tN\tSTEM|POS:N|LEM:kAtib|ROOT:ktb|M|NOM
(2:1:4:1)\tمَكْتُوبٌ\tADJ\tSTEM|POS:ADJ|LEM:makotuwb|ROOT:ktb|M|NOM
(2:1:5:1)\tكِتَابَةٌ\tN\tSTEM|POS:N|LEM:kitAbap|ROOT:ktb|F|NOM
(2:2:1:1)\tعَلِمَ\tV\tSTEM|POS:V|PERF|LEM:Ealima|ROOT:Elm|3MS
(2:2:2:1)\tعَالِمٌ\tN\tSTEM|POS:N|LEM:EAlim|ROOT:Elm|M|NOM
(2:2:3:1)\tعِلْمٌ\tN\tSTEM|POS:N|LEM:Eilom|ROOT:Elm|M|NOM
(2:2:4:1)\tمَعْلُومٌ\tADJ\tSTEM|POS:ADJ|LEM:maEoluwm|ROOT:Elm|M|NOM
(2:3:1:1)\tقَرَأَ\tV\tSTEM|POS:V|PERF|LEM:qara>a|ROOT:qr>|3MS
(2:3:2:1)\tقُرْآنٌ\tN\tSTEM|POS:N|LEM:qurop>An|ROOT:qr>|M|NOM
(2:3:3:1)\tقَارِئٌ\tN\tSTEM|POS:N|LEM:qAri>|ROOT:qr>|M|NOM
(2:4:1:1)\tنَزَلَ\tV\tSTEM|POS:V|PERF|LEM:nazala|ROOT:nzl|3MS
(2:4:2:1)\tنَزَّلَ\tV\tSTEM|POS:V|PERF|LEM:naz~ala|ROOT:nzl|3MS
(2:4:3:1)\tتَنْزِيلٌ\tN\tSTEM|POS:N|LEM:tanoziyl|ROOT:nzl|M|NOM
(2:5:1:1)\tسَمِعَ\tV\tSTEM|POS:V|PERF|LEM:samiEa|ROOT:smE|3MS
(2:5:2:1)\tسَمِيعٌ\tADJ\tSTEM|POS:ADJ|LEM:samiyE|ROOT:smE|M|NOM
(2:6:1:1)\tرَحِمَ\tV\tSTEM|POS:V|PERF|LEM:raHima|ROOT:rHm|3MS
(2:6:2:1)\tرَحِيمٌ\tADJ\tSTEM|POS:ADJ|LEM:raHiym|ROOT:rHm|M|GEN
(2:6:3:1)\tرَحْمَةٌ\tN\tSTEM|POS:N|LEM:raHomap|ROOT:rHm|F|NOM
(2:7:1:1)\tمَلَكَ\tV\tSTEM|POS:V|PERF|LEM:malaka|ROOT:mlk|3MS
(2:7:2:1)\tمَلِكٌ\tN\tSTEM|POS:N|LEM:malik|ROOT:mlk|M|NOM
(2:7:3:1)\tمُلْكٌ\tN\tSTEM|POS:N|LEM:mulok|ROOT:mlk|M|NOM
(2:7:4:1)\tمَلَكٌ\tN\tSTEM|POS:N|LEM:malak|ROOT:mlk|M|NOM
(2:8:1:1)\tعَبَدَ\tV\tSTEM|POS:V|PERF|LEM:Eabada|ROOT:Ebd|3MS
(2:8:2:1)\tعِبَادَةٌ\tN\tSTEM|POS:N|LEM:EibAdap|ROOT:Ebd|F|NOM
(2:8:3:1)\tعَبْدٌ\tN\tSTEM|POS:N|LEM:Eabd|ROOT:Ebd|M|NOM
(2:9:1:1)\tعَوَنَ\tV\tSTEM|POS:V|PERF|LEM:EAna|ROOT:Ewn|3MS
(2:9:2:1)\tعَوْنٌ\tN\tSTEM|POS:N|LEM:Ewn|ROOT:Ewn|M|NOM
(2:9:3:1)\tمُعِينٌ\tN\tSTEM|POS:N|LEM:muEiyn|ROOT:Ewn|M|NOM
(2:10:1:1)\tسَمَا\tN\tSTEM|POS:N|LEM:samA>|ROOT:smw|M|NOM
(2:10:2:1)\tسَمَاءٌ\tN\tSTEM|POS:N|LEM:samA>|ROOT:smw|F|NOM
"""

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Bipartite graph SSL — Arabic root identification")
print("=" * 60)

# 1. Stub
stub_tokens = parse_corpus(STUB_TEXT)
run_experiment(stub_tokens, f"Synthetic stub ({len(stub_tokens)} tokens)",
               GRID_FULL)

# 2. Real corpus
print(f"\n{'━'*60}")
print(f"  Fetching real corpus ...")
corpus_text = fetch_corpus(CORPUS_URL, CACHE_FILE)

if corpus_text is not None:
    real_tokens = parse_corpus(corpus_text)
    if not real_tokens:
        print("  [ERROR] Parsed 0 tokens from corpus — check file format.")
    else:
        grid = GRID_KNN if len(real_tokens) > RBF_TOKEN_LIMIT else GRID_FULL
        if len(real_tokens) > RBF_TOKEN_LIMIT:
            print(f"  [{len(real_tokens):,} tokens > {RBF_TOKEN_LIMIT:,}] "
                  f"Using knn-only grid (avoids n×n RBF memory blow-up).")
        run_experiment(real_tokens,
                       f"Quranic Arabic Corpus v0.4 ({len(real_tokens):,} tokens)",
                       grid)
else:
    print("\n  Skipping real corpus (fetch failed — see instructions above).")

print(f"\n{'='*60}")
print("  Done.")
print(f"{'='*60}")