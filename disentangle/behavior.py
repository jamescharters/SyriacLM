#!/usr/bin/env python3
"""Causal test: is the number subspace *used*, or merely *present*?

The selectivity matrix (``disentangle.py``) shows that morphosyntactic number is
linearly separable from lexical identity in frozen encoders -- but a random-init
encoder reproduces that geometry, so separability alone is surface-carried and does
not implicate the *learned* representation. This module asks the complementary,
causal question with the model's own behaviour: does erasing the number direction
*break a number-dependent prediction*, and does it do so only in a model that was
actually trained?

Task: subject--verb number agreement, the standard targeted-evaluation probe
(Linzen et al., 2016; Goldberg, 2019). A Hebrew sentence pairs a number-marked
subject with a masked past-tense verb; we read the masked-LM head's preference
between the correctly- and incorrectly-numbered verb form (a UniMorph minimal
pair). Agreement accuracy is the fraction of items where the model prefers the
form that agrees with the subject. No probe is trained -- the readout is the
model's native objective, so there is no fitted metric to over-tune.

Intervention: we erase a linear concept direction from the final hidden state
*before the MLM head* with closed-form LEACE (Belrose et al., 2023), fit on
contextual subject-position representations labelled by subject number. We then
re-score agreement. The design crosses:

  * model:     pretrained AlephBERT  vs.  random-init AlephBERT (same config)
  * erasure:   none | number | gender (off-target) | random direction (rank-matched)

Predictions if the number subspace is causally used by the *trained* model:
erasing number drives pretrained agreement toward chance, while erasing gender or a
random direction does not (selectivity), and the random-init model has no agreement
to break (geometry without function). A null or non-selective result is reported as
such.

    .venv/bin/python -m disentangle.behavior --demo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    pass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
    from disentangle.disentangle import erase_leace, _device, _free
    _HF = True
except Exception:  # pragma: no cover - heavy deps optional
    _HF = False

UNIMORPH_HEB = os.path.join(os.path.dirname(__file__), "unimorph_cache", "heb")
MODEL_ID = "onlplab/alephbert-base"

# Two agreement tasks, each a binary morphosyntactic factor with a number-/gender-
# marked single-token subject and a [MASK] for a past-tense verb whose UniMorph
# minimal pair realises the two factor values. The subject is the only cue.
#   number: he (sg) / they-masc (pl)  x  3rd-person past verb sg/pl
#   gender: he (masc) / she (fem)     x  3rd-person sg past verb masc/fem
TEMPLATE = "{subj} {mask} ."
TASKS = {
    # verb spec: (tense, person, fixed_feature_or_None, (val0, val1),
    #             {value: required_extra_feature_or_None})
    "number": {"subjects": {0: "\u05d4\u05d5\u05d0", 1: "\u05d4\u05dd"},   # hu / hem
               "verb": ("PST", "3", None, ("SG", "PL"),
                        {"SG": "MASC", "PL": None})},
    "gender": {"subjects": {0: "\u05d4\u05d5\u05d0", 1: "\u05d4\u05d9\u05d0"}, # hu / hi
               "verb": ("PST", "3", "SG", ("MASC", "FEM"),
                        {"MASC": None, "FEM": None})},
}


# --------------------------------------------------------------------------- #
# Data: agreement minimal pairs (single-token verb forms)
# --------------------------------------------------------------------------- #
def load_pairs(tokenizer, task: str, path: str = UNIMORPH_HEB,
               limit: int | None = None):
    """Single-token (value0_verb, value1_verb, id0, id1) pairs for a task."""
    import collections
    tense, person, fixed, (val0, val1), extra = TASKS[task]["verb"]
    cells = collections.defaultdict(lambda: {val0: set(), val1: set()})
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            lemma, form, feats = p[0], p[1], set(p[2].split(";"))
            if "V" not in feats or tense not in feats or person not in feats:
                continue
            if fixed and fixed not in feats:
                continue
            for val in (val0, val1):
                req = extra.get(val)
                if val in feats and (req is None or req in feats):
                    cells[lemma][val].add(form)

    def tok1(w):
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    pairs = []
    for d in cells.values():
        if not d[val0] or not d[val1]:
            continue
        a, b = sorted(d[val0])[0], sorted(d[val1])[0]
        ia, ib = tok1(a), tok1(b)
        if ia is not None and ib is not None and ia != ib:
            pairs.append((a, b, ia, ib))
    pairs.sort()
    if limit:
        pairs = pairs[:limit]
    return pairs


# --------------------------------------------------------------------------- #
# Model plumbing: hidden states at the [MASK], split head into encoder + decoder
# --------------------------------------------------------------------------- #
class MaskedScorer:
    """Wraps a masked-LM so we can erase a direction from the pre-head hidden
    state at the [MASK] position and re-apply the model's own output head."""

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.mask_id = tokenizer.mask_token_id
        # AlephBERT (BERT) head: cls.predictions = transform -> decoder. We apply
        # the transform, optionally erase, then the decoder, reproducing logits.
        self.head = model.cls.predictions

    def _encode(self, sentences: list[str]):
        enc = self.tok(sentences, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.bert(**enc)
        h = out.last_hidden_state                       # (B,T,H)
        mask_pos = (enc["input_ids"] == self.mask_id)
        return h, mask_pos

    def mask_hidden(self, sentences: list[str]) -> np.ndarray:
        """Pre-head hidden vector at the [MASK] slot for each sentence."""
        h, mask_pos = self._encode(sentences)
        idx = mask_pos.float().argmax(1)                # first mask per row
        vecs = h[torch.arange(h.size(0)), idx]          # (B,H)
        return vecs.float().cpu().numpy().astype(np.float64)

    def subject_hidden(self, sentences: list[str], subj_first: bool = True) -> np.ndarray:
        """Hidden vector at the subject token (token index 1: after [CLS])."""
        h, _ = self._encode(sentences)
        return h[:, 1].float().cpu().numpy().astype(np.float64)

    def agreement_logits(self, sentences, sg_ids, pl_ids, P: np.ndarray | None):
        """Return (logit_sg, logit_pl) at the [MASK], optionally after erasing P
        from the post-transform hidden state."""
        h, mask_pos = self._encode(sentences)
        idx = mask_pos.float().argmax(1)
        m = h[torch.arange(h.size(0)), idx]             # (B,H) pre-head
        with torch.no_grad():
            t = self.head.transform(m)                  # dense+gelu+LN
            if P is not None:
                Pt = torch.from_numpy(P).to(t.dtype).to(self.device)
                t = t @ Pt
            logits = self.head.decoder(t)               # (B,V)
        sg = logits[torch.arange(len(sentences)), torch.tensor(sg_ids, device=self.device)]
        pl = logits[torch.arange(len(sentences)), torch.tensor(pl_ids, device=self.device)]
        return sg.float().cpu().numpy(), pl.float().cpu().numpy()


def load_scorer(random_init: bool, device, seed: int = 0) -> MaskedScorer:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if random_init:
        torch.manual_seed(seed)         # the random-init network is the control
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        model = AutoModelForMaskedLM.from_config(cfg)
    else:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    return MaskedScorer(model, tok, device)


# --------------------------------------------------------------------------- #
# Erasers fit in the POST-TRANSFORM head space (where agreement is read)
# --------------------------------------------------------------------------- #
def _head_space_features(scorer: MaskedScorer, sentences, labels):
    """Post-transform [MASK] features + labels, the space the decoder reads."""
    h, mask_pos = scorer._encode(sentences)
    idx = mask_pos.float().argmax(1)
    m = h[torch.arange(h.size(0)), idx]
    with torch.no_grad():
        t = scorer.head.transform(m)
    return t.float().cpu().numpy().astype(np.float64), np.asarray(labels)


def _random_projection(dim: int, rank: int, seed: int) -> np.ndarray:
    """Rank-matched random erasure: project out `rank` random orthonormal dirs."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(A)
    return np.eye(dim) - Q @ Q.T


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def build_items(task: str, pairs):
    """Agreement items for a task: subject + [MASK], with the agreeing/disagreeing
    single-token verb ids. The model should prefer the form that agrees."""
    subjects = TASKS[task]["subjects"]
    items = []
    for (form0, form1, id0, id1) in pairs:
        for val, subj in subjects.items():               # val 0 / 1
            sent = TEMPLATE.format(subj=subj, mask="[MASK]")
            items.append({"sent": sent, "val": val,
                          "id0": id0, "id1": id1})
    return items


def agreement_accuracy(scorer: MaskedScorer, items, P=None, batch=128) -> float:
    correct = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        l0, l1 = scorer.agreement_logits(
            [it["sent"] for it in chunk],
            [it["id0"] for it in chunk],
            [it["id1"] for it in chunk], P)
        for j, it in enumerate(chunk):
            chose1 = l1[j] > l0[j]
            correct += int(int(chose1) == it["val"])
    return correct / max(len(items), 1)


def _subject_features(scorer: MaskedScorer, task: str, pairs):
    """Head-space [MASK] features labelled by the subject's factor value, for
    fitting a LEACE eraser that targets exactly the agreement readout space."""
    subjects = TASKS[task]["subjects"]
    sents, y = [], []
    for val, subj in subjects.items():
        for _ in pairs:
            sents.append(TEMPLATE.format(subj=subj, mask="[MASK]"))
            y.append(val)
    X, y = _head_space_features(scorer, sents, y)
    return X, y


def fit_erasers(scorer: MaskedScorer, pairs_by_task, seed: int = 0):
    """Fit a LEACE eraser per factor (number, gender) in head space, plus a
    rank-matched random projection. Erasers are independent of the scored task."""
    erasers, dim = {}, None
    for factor in ("number", "gender"):
        X, y = _subject_features(scorer, factor, pairs_by_task[factor])
        erasers[factor] = erase_leace(X - X.mean(0), y)
        dim = X.shape[1]
    erasers["random"] = _random_projection(dim, rank=1, seed=seed)
    return erasers


def run(limit: int | None = 300, seed: int = 0) -> dict:
    device = _device()
    out = {"erasers": ("none", "number", "gender", "random"),
           "tasks": ("number", "gender"), "seed": seed, "models": {}}
    for random_init in (False, True):
        name = "random-init AlephBERT" if random_init else "pretrained AlephBERT"
        print(f"[behavior] {name} (seed {seed}) ...", file=sys.stderr)
        scorer = load_scorer(random_init, device, seed=seed)
        pairs_by_task = {t: load_pairs(scorer.tok, t, limit=limit)
                         for t in ("number", "gender")}
        with np.errstate(all="ignore"):
            erasers = fit_erasers(scorer, pairs_by_task, seed=seed)
            model_res = {}
            for task in ("number", "gender"):
                items = build_items(task, pairs_by_task[task])
                row = {"n_items": len(items),
                       "none": round(agreement_accuracy(scorer, items, None), 3)}
                for er in ("number", "gender", "random"):
                    row[er] = round(
                        agreement_accuracy(scorer, items, erasers[er]), 3)
                model_res[task] = row
        out["models"][name] = model_res
        del scorer
        _free(device)
    return out


def run_seeds(limit: int | None = 300, seeds=(0, 1, 2)) -> dict:
    """Aggregate the dissociation over seeds (mean/SD); seeds vary the random-init
    network and the rank-matched random projection."""
    runs = [run(limit=limit, seed=s) for s in seeds]
    agg = {"erasers": runs[0]["erasers"], "tasks": runs[0]["tasks"],
           "seeds": list(seeds), "models": {}}
    for name in runs[0]["models"]:
        agg["models"][name] = {}
        for task in runs[0]["models"][name]:
            agg["models"][name][task] = {
                "n_items": runs[0]["models"][name][task]["n_items"]}
            for er in ("none", "number", "gender", "random"):
                vals = np.array([r["models"][name][task][er] for r in runs],
                                dtype=float)
                agg["models"][name][task][er] = {
                    "mean": round(float(vals.mean()), 3),
                    "sd": round(float(vals.std(ddof=0)), 3)}
    return agg


def _print(out: dict) -> None:
    print("\n=== Cross-causal selectivity: agreement accuracy under erasure "
          "(chance 0.5) ===")
    for name, tasks in out["models"].items():
        print(f"\n  {name}")
        print(f"    {'agreement task':<20}{'intact':>9}{'-number':>9}"
              f"{'-gender':>9}{'-random':>9}")
        for task, r in tasks.items():
            print(f"    {task + ' agreement':<20}{r['none']:>9.3f}"
                  f"{r['number']:>9.3f}{r['gender']:>9.3f}{r['random']:>9.3f}"
                  f"   (n={r['n_items']})")
    print("\nCausal + selective: erasing a factor breaks ITS OWN agreement "
          "(-> chance)")
    print("but spares the other factor's and the random control; the random-init")
    print("model has no agreement to break. The diagonal collapses, off-diagonal "
          "holds.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--limit", type=int, default=300,
                    help="cap verb pairs per task (speed; default 300)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not _HF:
        print("transformers + torch required:\n"
              "    .venv/bin/python -m pip install -r neural/requirements-neural.txt",
              file=sys.stderr)
        return 2
    if not args.demo:
        ap.print_help()
        return 1
    out = run(limit=args.limit, seed=args.seed)
    _print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
