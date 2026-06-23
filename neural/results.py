#!/usr/bin/env python3
"""Consolidate every ``neural/`` result into one manifest and emit the paper tables.

This is the single source of truth for the numbers in the companion paper. Each
experiment is a function returning a JSON-serialisable dict; results are merged
into a manifest (``neural/paper/results.json``) and rendered to LaTeX table files
(``neural/paper/tables/``). It mirrors the parent ``paper_experiments.py`` pattern
(compute the numbers, then print the ``\\begin{tabular}`` rows) so that no value in
the paper is transcribed by hand.

Experiments split into two cost tiers:

* **light** -- need only ``torch`` plus the cached SEDRA/DSC data, run in seconds
  to a couple of minutes: ``vocalizer`` (held-out SEDRA + multi-seed cross-register
  on classical DSC), ``factored`` (Twist 2), ``restoration`` (Twist 3).
* **transfer** -- need ``transformers`` and a model download, a few minutes each:
  ``canine`` / ``hebrew`` / ``glot500`` authorship transfer and tokenizer coverage,
  ``intrinsic_lm`` (frozen vs. LoRA pseudo-bits-per-byte).

    .venv/bin/python -m neural.results --run vocalizer --seeds 5
    .venv/bin/python -m neural.results --run factored,restoration
    .venv/bin/python -m neural.results --run canine,hebrew,glot500
    .venv/bin/python -m neural.results --emit-latex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # neural/
PAPER_DIR = HERE / "paper"
TABLES_DIR = PAPER_DIR / "tables"
MANIFEST = PAPER_DIR / "results.json"

LIGHT = ("vocalizer", "factored", "restoration")
TRANSFER = ("canine", "hebrew", "glot500", "intrinsic_lm")


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = dict(sorted(manifest.items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def _mean_sd(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": [round(float(x), 4) for x in a]}


# --------------------------------------------------------------------------- #
# Light experiments (torch + cached SEDRA/DSC only)
# --------------------------------------------------------------------------- #
def run_vocalizer(seeds: int = 5, **_) -> dict:
    """Held-out SEDRA pointing + multi-seed cross-register vowel accuracy on DSC."""
    from neural import vocalizer, dsc_gold

    seed_list = [42, 7, 13, 101, 2024][:max(1, seeds)]
    # held-out SEDRA numbers come from the primary seed (42)
    primary = vocalizer.train_vocalizer(seed=seed_list[0])
    sedra = {k: primary[k] for k in (
        "params", "train_words", "val_words", "n_labels",
        "pos_acc", "word_acc", "baseline_pos_acc", "baseline_word_acc")}

    # harvest the classical gold once, then reuse across seeds
    harvested = dsc_gold.harvest()
    gs = harvested["gold"]
    in_vocab, oov, in_base, oov_base = [], [], [], []
    derived = harvested["derived"]
    for s in seed_list:
        res = primary if s == seed_list[0] else vocalizer.train_vocalizer(seed=s)
        ev = vocalizer.cross_register_eval(res, seed=s, harvested=harvested)
        in_vocab.append(ev["in_vocab"]["vowel_acc"])
        oov.append(ev["oov"]["vowel_acc"])
        in_base.append(ev["in_vocab"]["baseline_acc"])
        oov_base.append(ev["oov"]["baseline_acc"])

    return {
        "sedra_heldout": sedra,
        "cross_register": {
            "seeds": seed_list,
            "in_vocab_vowel_acc": _mean_sd(in_vocab),
            "oov_vowel_acc": _mean_sd(oov),
            "in_vocab_baseline": _mean_sd(in_base),
            "oov_baseline": _mean_sd(oov_base),
            "n_in_vocab_slots": ev["in_vocab"]["marked_slots"],
            "n_oov_slots": ev["oov"]["marked_slots"],
        },
        "gold": {
            "n_tokens": gs["n_tokens"], "n_marked_slots": gs["n_marked_slots"],
            "n_types": gs["n_types"],
            "n_in_vocab": len(gs["in_vocab"]), "n_oov": len(gs["oov"]),
            "aligned_tokens": derived["aligned_tokens"],
            "vowel_map": {m: v for m, v in derived["vowel_map"].items()},
            "vowel_map_report": [
                {"mark": n, "vowel": v, "purity": round(p, 3), "count": c}
                for n, v, p, c in derived["report"][:18]],
        },
    }


def run_factored(seed: int = 42, **_) -> dict:
    """Twist 2: factored vs. flat root/pattern encoder, root-NN retrieval."""
    from neural import factored

    words = factored.load_root_words()
    train, seen_q, unseen_q = factored.split_by_root(words, seed=seed)
    vocabs = factored.build_vocabs(words)
    out = {"n_words": len(words), "n_train": len(train),
           "n_seen_q": len(seen_q), "n_unseen_q": len(unseen_q)}
    for kind in ("flat", "factored"):
        r = factored.train_one(kind, train, seen_q, unseen_q, vocabs,
                               epochs=10, bs=128, lr=2e-3, seed=seed)
        out[kind] = {"params": r["params"],
                     "seen_root_nn": round(r["seen_root_nn"], 4),
                     "unseen_root_nn": round(r["unseen_root_nn"], 4)}
    out["delta_seen"] = round(out["factored"]["seen_root_nn"] - out["flat"]["seen_root_nn"], 4)
    out["delta_unseen"] = round(out["factored"]["unseen_root_nn"] - out["flat"]["unseen_root_nn"], 4)
    return out


def run_restoration(steps: int = 2000, seed: int = 42, **_) -> dict:
    """Twist 3: causal character Transformer lacuna restoration on held-out DSC."""
    from neural import restoration

    cfg = restoration.RestorationConfig(steps=steps, seed=seed)
    res = restoration.train(cfg)
    keep = ("params", "char_accuracy", "span_exact_match", "masked_chars",
            "masked_spans", "train_windows", "val_windows", "vocab",
            "train_seconds", "device")
    return {k: res[k] for k in keep if k in res}


# --------------------------------------------------------------------------- #
# Transfer experiments (transformers + a model download)
# --------------------------------------------------------------------------- #
def _auc_rows_to_dict(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        entry = {"auc": round(r["auc"], 4), "texts": r["texts"], "authors": r["authors"]}
        if r.get("ci"):
            entry["ci"] = [round(r["ci"][0], 4), round(r["ci"][1], 4)]
        out[str(r["floor"])] = entry
    return out


# The off-the-shelf encoder vectors are deterministic, but the supervised AV-head
# projection is trained on a tiny cohort (the floor-2000 split is ~11 authors) and
# is sensitive to backend (MPS) nondeterminism, so a single run is unreliable.
# We sweep seeds and report mean +/- SD, mirroring the parent paper's multi-seed
# FastText protocol; one seed also carries the author-cluster bootstrap CI.
AV_HEAD_SEEDS = [42, 7, 13, 101, 2024]


def _multiseed_av_head(auc_fn, wv, floors, seeds=AV_HEAD_SEEDS) -> dict:
    per_floor: dict[str, list[float]] = {str(f): [] for f in floors}
    ci_seed0: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for i, s in enumerate(seeds):
        rows = auc_fn(wv, list(floors), use_av_head=True, seed=s,
                      bootstrap=(i == 0))
        for r in rows:
            per_floor[str(r["floor"])].append(round(r["auc"], 4))
            meta[str(r["floor"])] = {"texts": r["texts"], "authors": r["authors"]}
            if i == 0 and r.get("ci"):
                ci_seed0[str(r["floor"])] = [round(r["ci"][0], 4), round(r["ci"][1], 4)]
    out = {}
    for f, vals in per_floor.items():
        ms = _mean_sd(vals)
        out[f] = {"auc": ms["mean"], "sd": ms["sd"], "values": ms["values"],
                  "ci_seed0": ci_seed0.get(f), **meta.get(f, {})}
    return out


def run_canine(floors=(1000, 2000), seed: int = 42, **_) -> dict:
    """Off-the-shelf CANINE-c authorship transfer; multi-seed AV head (mean+/-SD)."""
    from neural.canine_encoder import load_canine, CanineWordVectors, authorship_auc, _device

    device = _device()
    model = load_canine(None).to(device)
    wv = CanineWordVectors(model, device)
    base = authorship_auc(wv, list(floors), use_av_head=False, seed=seed)
    avh = _multiseed_av_head(authorship_auc, wv, floors)
    return {"off_the_shelf": _auc_rows_to_dict(base),
            "av_head": avh, "av_head_seeds": AV_HEAD_SEEDS}


def _hf_transfer(transliterate: str, floors, seed: int, with_av: bool) -> dict:
    from neural import hf_encoder as H

    model_id = H.DEFAULT_MODEL_FOR[transliterate]
    preprocess = H.PREPROCESSORS[transliterate]
    tok = H._load_tokenizer(model_id)
    forms = H._all_forms(True)
    cov = H.tokenizer_coverage(tok, forms, preprocess=preprocess)
    device = H._device()
    model = H.AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    wv = H.HFWordVectors(model, tok, device, preprocess=preprocess)
    out = {"model_id": model_id,
           "coverage": {"mean_subwords_per_form": cov["mean_subwords_per_form"],
                        "frac_with_any_unk": cov["frac_with_any_unk"],
                        "frac_entirely_unk": cov["frac_entirely_unk"]},
           "off_the_shelf": _auc_rows_to_dict(
               H.authorship_auc(wv, list(floors), use_av_head=False, seed=seed))}
    if with_av:
        out["av_head"] = _multiseed_av_head(H.authorship_auc, wv, floors)
        out["av_head_seeds"] = AV_HEAD_SEEDS
    return out


def run_hebrew(floors=(1000, 2000), seed: int = 42, **_) -> dict:
    """Hebrew (AlephBERT) transfer on Syriac transliterated into Hebrew script."""
    return _hf_transfer("hebrew", floors, seed, with_av=True)


def run_glot500(floors=(1000, 2000), seed: int = 42, **_) -> dict:
    """Glot500-m transfer (no transliteration); tokenizer coverage explains it."""
    return _hf_transfer("none", floors, seed, with_av=False)


def run_intrinsic_lm(steps: int = 1500, seed: int = 42, **_) -> dict:
    """Frozen linear probe vs. LoRA continued-pretraining: masked pseudo-bpb."""
    from neural import canine_pretrain as C
    from neural.config import DEFAULT_NEURAL_CACHE
    from neural.canine_pretrain import DEFAULT_BASE

    data_dir = DEFAULT_NEURAL_CACHE
    common = dict(base=DEFAULT_BASE, data_dir=data_dir, steps=steps, window=256,
                  batch_size=8, lr=5e-4, mask_fraction=0.15, max_span=5,
                  lora_r=16, lora_alpha=32, lora_dropout=0.05, max_windows=12000,
                  seed=seed)
    frozen = C.train(out_dir=data_dir / "checkpoints" / "canine-frozen",
                     freeze_encoder=True, **common)
    lora = C.train(out_dir=data_dir / "checkpoints" / "canine-lora",
                   freeze_encoder=False, **common)
    pick = lambda d: {"masked_acc": round(d["val_masked_accuracy"], 4),
                      "pseudo_bpb": round(d["val_masked_bits_per_byte"], 4),
                      "trainable_params": d["trainable"], "total_params": d["total"]}
    return {"frozen_probe": pick(frozen), "lora": pick(lora)}


EXPERIMENTS = {
    "vocalizer": run_vocalizer,
    "factored": run_factored,
    "restoration": run_restoration,
    "canine": run_canine,
    "hebrew": run_hebrew,
    "glot500": run_glot500,
    "intrinsic_lm": run_intrinsic_lm,
}


# --------------------------------------------------------------------------- #
# LaTeX emission
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _auc_cell(entry: dict | None) -> str:
    if not entry:
        return "--"
    s = _fmt(entry["auc"])
    if entry.get("sd") is not None and entry.get("values") and len(entry["values"]) > 1:
        s += f"$\\pm${_fmt(entry['sd'], 3)}"      # multi-seed AV head
    elif entry.get("ci"):
        s += f"\\,[{_fmt(entry['ci'][0])}, {_fmt(entry['ci'][1])}]"
    return s


def emit_vocalizer_table(m: dict) -> str | None:
    v = m.get("vocalizer")
    if not v:
        return None
    cr = v["cross_register"]
    iv, ov = cr["in_vocab_vowel_acc"], cr["oov_vowel_acc"]
    ivb, ovb = cr["in_vocab_baseline"], cr["oov_baseline"]
    sd = v["sedra_heldout"]
    lines = [
        "% Generated by neural/results.py -- do not edit by hand.",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & vowel/pointing acc. & baseline \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Held-out SEDRA (NT lexicon)}} \\\\",
        f"\\quad per-position pointing & {_fmt(sd['pos_acc'])} & {_fmt(sd['baseline_pos_acc'])} \\\\",
        f"\\quad full-word exact match & {_fmt(sd['word_acc'])} & {_fmt(sd['baseline_word_acc'])} \\\\",
        "\\midrule",
        f"\\multicolumn{{3}}{{l}}{{\\emph{{Classical DSC, {cr['seeds'].__len__()} seeds}}}} \\\\",
        f"\\quad in SEDRA vocabulary & {_fmt(iv['mean'])}$\\pm${_fmt(iv['sd'],3)} & {_fmt(ivb['mean'])} \\\\",
        f"\\quad out of SEDRA vocab. & {_fmt(ov['mean'])}$\\pm${_fmt(ov['sd'],3)} & {_fmt(ovb['mean'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def emit_vowelmap_table(m: dict) -> str | None:
    v = m.get("vocalizer")
    if not v:
        return None
    rep = v["gold"]["vowel_map_report"]
    accepted = set(v["gold"]["vowel_map"])
    lines = ["% Generated by neural/results.py -- do not edit by hand.",
             "\\begin{tabular}{llrr}", "\\toprule",
             "Syriac mark (U+0730--074A) & $\\to$ CAL vowel & purity & count \\\\",
             "\\midrule"]
    for row in rep:
        if not row["vowel"]:
            continue
        star = "$^\\ast$" if row["mark"] in accepted else ""
        name = row["mark"].title().replace("_", " ")
        lines.append(f"{name}{star} & {row['vowel']} & {_fmt(row['purity'],2)} & {row['count']:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_bakeoff_table(m: dict) -> str | None:
    if not any(k in m for k in ("canine", "hebrew", "glot500")):
        return None
    # context numbers from the parent paper (count-based baselines), labelled as such
    rows = [
        ("FastText (parent)", {"1000": {"auc": 0.915}, "2000": {"auc": 0.900}}, None),
        ("word2vec (parent)", {"1000": {"auc": 0.946}, "2000": {"auc": 0.938}}, None),
        ("Burrows's Delta (parent)", {"1000": {"auc": 0.907}, "2000": {"auc": 0.940}}, None),
    ]
    g = m.get("glot500")
    if g:
        rows.append(("Glot500-m (char fallback)", g["off_the_shelf"], None))
    c = m.get("canine")
    if c:
        rows.append(("CANINE-c off-the-shelf", c["off_the_shelf"], None))
        rows.append(("\\quad + AV head (LOAO)", c["av_head"], None))
    h = m.get("hebrew")
    if h:
        rows.append(("Hebrew transliteration", h["off_the_shelf"], None))
        if "av_head" in h:
            rows.append(("\\quad + AV head (LOAO)", h["av_head"], None))
    lines = ["% Generated by neural/results.py -- do not edit by hand.",
             "\\begin{tabular}{lcc}", "\\toprule",
             "Representation & AUC (floor 1000) & AUC (floor 2000) \\\\", "\\midrule"]
    for name, data, _ in rows:
        lines.append(f"{name} & {_auc_cell(data.get('1000'))} & {_auc_cell(data.get('2000'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_intrinsic_table(m: dict) -> str | None:
    il = m.get("intrinsic_lm")
    if not il:
        return None
    fr, lo = il["frozen_probe"], il["lora"]
    lines = [
        "% Generated by neural/results.py -- do not edit by hand.",
        "\\begin{tabular}{lrcc}",
        "\\toprule",
        " & trainable params & masked acc. & pseudo-bpb \\\\",
        "\\midrule",
        f"Frozen linear probe & {fr['trainable_params']:,} & "
        f"{_fmt(fr['masked_acc'])} & {_fmt(fr['pseudo_bpb'])} \\\\",
        f"LoRA continued-pretrain & {lo['trainable_params']:,} & "
        f"{_fmt(lo['masked_acc'])} & {_fmt(lo['pseudo_bpb'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def emit_factored_table(m: dict) -> str | None:
    f = m.get("factored")
    if not f:
        return None
    lines = [
        "% Generated by neural/results.py -- do not edit by hand.",
        "\\begin{tabular}{lrcc}",
        "\\toprule",
        " & params & seen-root NN & unseen-root NN \\\\",
        "\\midrule",
        f"Flat (vocalised chars) & {f['flat']['params']:,} & "
        f"{_fmt(f['flat']['seen_root_nn'])} & {_fmt(f['flat']['unseen_root_nn'])} \\\\",
        f"Factored (root/pattern) & {f['factored']['params']:,} & "
        f"{_fmt(f['factored']['seen_root_nn'])} & {_fmt(f['factored']['unseen_root_nn'])} \\\\",
        "\\midrule",
        f"$\\Delta$ (factored $-$ flat) & & {_fmt(f['delta_seen'],3)} & {_fmt(f['delta_unseen'],3)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


EMITTERS = {
    "TN_vocalizer.tex": emit_vocalizer_table,
    "TN_vowelmap.tex": emit_vowelmap_table,
    "TN_bakeoff.tex": emit_bakeoff_table,
    "TN_intrinsic.tex": emit_intrinsic_table,
    "TN_factored.tex": emit_factored_table,
}


def emit_latex(manifest: dict) -> list[str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, fn in EMITTERS.items():
        tex = fn(manifest)
        if tex is None:
            continue
        (TABLES_DIR / fname).write_text(tex + "\n", encoding="utf-8")
        written.append(fname)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="",
                    help="comma-separated experiments, 'light', 'transfer', or 'all'")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds for the multi-seed vocaliser cross-register")
    ap.add_argument("--emit-latex", action="store_true",
                    help="render manifest -> neural/paper/tables/*.tex")
    ap.add_argument("--show", action="store_true", help="print the current manifest")
    args = ap.parse_args(argv)

    manifest = load_manifest()

    if args.run:
        if args.run == "all":
            names = list(EXPERIMENTS)
        elif args.run == "light":
            names = list(LIGHT)
        elif args.run == "transfer":
            names = list(TRANSFER)
        else:
            names = [n.strip() for n in args.run.split(",") if n.strip()]
        for name in names:
            if name not in EXPERIMENTS:
                print(f"unknown experiment: {name} (have: {', '.join(EXPERIMENTS)})",
                      file=sys.stderr)
                return 2
            print(f"\n=== running: {name} ===", file=sys.stderr)
            t0 = time.time()
            try:
                manifest[name] = EXPERIMENTS[name](seeds=args.seeds)
            except RuntimeError as exc:
                print(f"  skipped ({exc})", file=sys.stderr)
                continue
            manifest[name]["_seconds"] = round(time.time() - t0, 1)
            save_manifest(manifest)
            print(f"  done in {manifest[name]['_seconds']}s -> {MANIFEST}",
                  file=sys.stderr)

    if args.emit_latex:
        written = emit_latex(manifest)
        print(f"wrote {len(written)} table(s) to {TABLES_DIR}: {', '.join(written)}")

    if args.show or (not args.run and not args.emit_latex):
        print(json.dumps(manifest, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
