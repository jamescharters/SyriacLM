#!/usr/bin/env python3
"""Consolidate every disentanglement result into one manifest and emit the tables.

Mirrors ``neural/results.py``: each experiment block writes a JSON manifest
(``disentangle/paper/results.json``, git-ignored) and the manifest is rendered to
the LaTeX tables the paper ``\\input``s (``disentangle/paper/tables/``). Numbers in
the paper therefore trace to one regenerable source.

Blocks
------
* ``syriac``    -- the selectivity matrix over Syriac encoders x morphosyntactic
                   factors (multi-seed, INLP), with the consonantal control and
                   the compositionality measures.
* ``controls``  -- the random-init-CANINE floor and the LEACE-vs-INLP agreement.
* ``crosslang`` -- the same design replicated on Arabic and Hebrew (UniMorph +
                   native encoders).

    .venv/bin/python -m disentangle.results --compute syriac,controls,crosslang
    .venv/bin/python -m disentangle.results --emit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # disentangle/
PAPER_DIR = HERE / "paper"
TABLES_DIR = PAPER_DIR / "tables"
MANIFEST = PAPER_DIR / "results.json"

SEEDS = (42, 7, 13, 101, 2024)
SYR_ENCODERS = ("canine", "canine-syriac", "hebrew")
SYR_FACTORS = ("number", "gender", "state")
XLANG = (("ara", ("number", "gender")), ("heb", ("number", "gender")))


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Compute blocks
# --------------------------------------------------------------------------- #
def compute_syriac(seeds=SEEDS) -> dict:
    """Selectivity matrix: Syriac encoders x factors, multi-seed INLP."""
    from disentangle import disentangle as D
    device = D._device()
    grids = {f: D.load_grid(f) for f in SYR_FACTORS}
    block = {"seeds": list(seeds), "grids": {}, "cells": {}}
    for f, g in grids.items():
        block["grids"][f] = {"n_pairs": len(g) // 2,
                             "n_lexemes": len({x["lexeme"] for x in g})}
    for enc in SYR_ENCODERS:
        print(f"[syriac] encoder {enc} ...", file=sys.stderr)
        wv, label = D.build_encoder(enc, device)
        for f, g in grids.items():
            agg = D.run_seeds(g, wv, seeds=seeds, erase="inlp")
            block["cells"][f"{enc}|{f}"] = {"label": label, **agg}
        del wv
        D._free(device)
    return block


def compute_controls(seeds=SEEDS) -> dict:
    """Random-encoder floor + LEACE-vs-INLP agreement (canine)."""
    from disentangle import disentangle as D
    device = D._device()
    grids = {f: D.load_grid(f) for f in SYR_FACTORS}
    block = {"seeds": list(seeds), "random": {}, "leace": {}, "inlp": {}}

    print("[controls] random-init CANINE floor ...", file=sys.stderr)
    wv, label = D.build_encoder("canine-random", device)
    for f, g in grids.items():
        block["random"][f] = {"label": label,
                              **D.run_seeds(g, wv, seeds=seeds, erase="inlp")}
    del wv
    D._free(device)

    print("[controls] LEACE vs INLP on off-the-shelf CANINE ...", file=sys.stderr)
    wv, label = D.build_encoder("canine", device)
    for f, g in grids.items():
        block["inlp"][f] = D.run_seeds(g, wv, seeds=seeds, erase="inlp")
        block["leace"][f] = D.run_seeds(g, wv, seeds=seeds, erase="leace")
    block["label"] = label
    del wv
    D._free(device)
    return block


def compute_crosslang(seeds=SEEDS, langs=None, prior=None) -> dict:
    """Replicate the matrix on Arabic and Hebrew (UniMorph + native encoders).

    ``langs`` restricts which languages to (re)compute; ``prior`` is an existing
    crosslang block to merge into, so languages can be added incrementally (e.g.
    Hebrew now from cache, Arabic later when its encoder can be downloaded).
    """
    from disentangle import disentangle as D
    from disentangle import unimorph as U
    device = D._device()
    # merge into any prior crosslang block so languages can be added incrementally
    block = prior or {"seeds": list(seeds), "grids": {}, "cells": {}}
    block.setdefault("grids", {})
    block.setdefault("cells", {})
    want = {l: fs for l, fs in XLANG if langs is None or l in langs}
    for lang, factors in want.items():
        print(f"[crosslang] {lang} ...", file=sys.stderr)
        try:
            wv, label = U.build_encoder(lang, device)
        except Exception as exc:  # e.g. encoder download blocked -- skip, keep rest
            print(f"[crosslang] skip {lang}: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", file=sys.stderr)
            continue
        for f in factors:
            # CAMeLBERT-CA cannot tokenise diacritised Arabic (every fully-vocalised
            # UniMorph form -> <unk>); use the undiacritised surface it actually
            # reads, which makes Arabic parallel to (unvocalised) Hebrew.
            keep_diac = lang not in ("ara",)
            g = U.load_grid_unimorph(lang, f, keep_diacritics=keep_diac)
            block["grids"][f"{lang}|{f}"] = {
                "n_pairs": len(g) // 2,
                "n_lexemes": len({x["lexeme"] for x in g}),
                "vocalised": any(x["voc"] != x["cons"] for x in g)}
            block["cells"][f"{lang}|{f}"] = {
                "label": label, "lang": lang,
                **D.run_seeds(g, wv, seeds=seeds, erase="inlp")}
        del wv
        D._free(device)
    return block


def compute_behavior(seeds=SEEDS, limit=300) -> dict:
    """Causal cross-dissociation: agreement accuracy under concept erasure on
    AlephBERT's own masked-LM head, pretrained vs random-init."""
    from disentangle import behavior as B
    bseeds = tuple(range(min(3, max(1, len(seeds)))))   # 3 seeds is enough here
    return B.run_seeds(limit=limit, seeds=bseeds)


COMPUTE = {"syriac": compute_syriac, "controls": compute_controls,
           "crosslang": compute_crosslang, "behavior": compute_behavior}


# --------------------------------------------------------------------------- #
# Table emission
# --------------------------------------------------------------------------- #
def _ms(cell: dict, cond: str, key: str):
    """mean$\\pm$sd string for a metric, or '--'."""
    d = cell["results"][cond][key]
    return f"${_fmt(d['mean'])}\\pm{_fmt(d['sd'], 3)}$"


_SHORT = {"off-the-shelf CANINE": "CANINE (frozen)",
          "Syriac-tuned CANINE (LoRA)": "CANINE (Syriac LoRA)",
          "Hebrew-transfer (AlephBERT)": "Hebrew transfer",
          "random-init CANINE": "CANINE (random)"}


def emit_grid_table(m: dict) -> str | None:
    syr = m.get("syriac")
    xl = m.get("crosslang")
    if not syr and not xl:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llrr}", "\\toprule",
             "Language & Factor & Lemma pairs & Chance (\\textsc{nn}) \\\\",
             "\\midrule"]
    if syr:
        for f in SYR_FACTORS:
            g = syr["grids"][f]
            ch = 1.0 / max(int(0.3 * g["n_lexemes"]), 1)
            lines.append(f"Syriac & {f} & {g['n_pairs']:,} & {_fmt(ch, 4)} \\\\")
    if xl:
        for key, g in xl["grids"].items():
            lang, f = key.split("|")
            name = {"ara": "Arabic", "heb": "Hebrew"}[lang]
            ch = 1.0 / max(int(0.3 * g["n_lexemes"]), 1)
            lines.append(f"{name} & {f} & {g['n_pairs']:,} & {_fmt(ch, 4)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _matrix_rows(cells: dict, keys: list[str], cond: str = "voc") -> list[str]:
    rows = []
    for key in keys:
        cell = cells[key]
        enc, f = key.split("|")
        label = _SHORT.get(cell["label"], cell["label"])
        rows.append(
            f"{label} & {f} & {_ms(cell, cond, 'factor_acc')} & "
            f"{_ms(cell, cond, 'erase_factor__factor_acc')} & "
            f"{_ms(cell, cond, 'erase_lexeme__factor_acc')} & "
            f"{_ms(cell, cond, 'lexeme_nn')} & "
            f"{_ms(cell, cond, 'erase_factor__lexeme_nn')} & "
            f"{_ms(cell, cond, 'erase_lexeme__lexeme_nn')} \\\\")
    return rows


def emit_matrix_table(m: dict) -> str | None:
    syr = m.get("syriac")
    if not syr:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llcccccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Factor acc.\\ (chance .5)} "
             "& \\multicolumn{3}{c}{Lexeme \\textsc{nn}} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Encoder & Factor & orig.\\ & $-$factor & $-$lexeme "
             "& orig.\\ & $-$factor & $-$lexeme \\\\", "\\midrule"]
    keys = [f"{e}|{f}" for e in SYR_ENCODERS for f in SYR_FACTORS]
    lines += _matrix_rows(syr["cells"], keys, cond="voc")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_control_table(m: dict) -> str | None:
    syr = m.get("syriac")
    if not syr:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llcc}", "\\toprule",
             "Encoder & Factor & Factor acc.\\ (vocalised) "
             "& Factor acc.\\ (consonantal) \\\\", "\\midrule"]
    for e in SYR_ENCODERS:
        for f in SYR_FACTORS:
            cell = syr["cells"][f"{e}|{f}"]
            label = _SHORT.get(cell["label"], cell["label"])
            lines.append(f"{label} & {f} & {_ms(cell, 'voc', 'factor_acc')} "
                         f"& {_ms(cell, 'cons', 'factor_acc')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_floor_leace_table(m: dict) -> str | None:
    c = m.get("controls")
    if not c:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llccc}", "\\toprule",
             "Setting & Factor & Factor acc.\\ & $-$factor (erased) "
             "& Lexeme \\textsc{nn} \\\\", "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Random-init CANINE (architecture/input "
             "floor)}} \\\\"]
    for f in SYR_FACTORS:
        cell = c["random"][f]
        lines.append(f"\\quad random CANINE & {f} & {_ms(cell, 'voc', 'factor_acc')} "
                     f"& {_ms(cell, 'voc', 'erase_factor__factor_acc')} "
                     f"& {_ms(cell, 'voc', 'lexeme_nn')} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{Erasure method on frozen CANINE "
                 "(post-erasure factor acc.)}} \\\\")
    for f in SYR_FACTORS:
        inlp = c["inlp"][f]["results"]["voc"]["erase_factor__factor_acc"]
        leace = c["leace"][f]["results"]["voc"]["erase_factor__factor_acc"]
        lines.append(f"\\quad INLP vs.\\ LEACE & {f} & "
                     f"${_fmt(inlp['mean'])}\\pm{_fmt(inlp['sd'],3)}$ & "
                     f"${_fmt(leace['mean'])}\\pm{_fmt(leace['sd'],3)}$ & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_composition_table(m: dict) -> str | None:
    syr = m.get("syriac")
    if not syr:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llcc}", "\\toprule",
             "Encoder & Factor & Offset consistency & Analogy top-1 "
             "(chance) \\\\", "\\midrule"]
    for e in ("canine", "canine-syriac"):
        for f in SYR_FACTORS:
            cell = syr["cells"].get(f"{e}|{f}")
            if not cell:
                continue
            comp = cell["results"]["voc"]["composition"]
            label = _SHORT.get(cell["label"], cell["label"])
            cons = comp["offset_consistency"]
            ana = comp["analogy_top1"]
            ch = comp["analogy_chance"]
            lines.append(
                f"{label} & {f} & ${_fmt(cons['mean'])}\\pm{_fmt(cons['sd'],3)}$ "
                f"& ${_fmt(ana['mean'])}\\pm{_fmt(ana['sd'],3)}$ ({_fmt(ch,4)}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def emit_crosslang_table(m: dict) -> str | None:
    xl = m.get("crosslang")
    if not xl:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llcccccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Factor acc.\\ (chance .5)} "
             "& \\multicolumn{3}{c}{Lexeme \\textsc{nn}} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Language & Factor & orig.\\ & $-$factor & $-$lexeme "
             "& orig.\\ & $-$factor & $-$lexeme \\\\", "\\midrule"]
    name = {"ara": "Arabic", "heb": "Hebrew"}
    for key in xl["cells"]:
        cell = xl["cells"][key]
        lang, f = key.split("|")
        lines.append(
            f"{name[lang]} & {f} & {_ms(cell, 'voc', 'factor_acc')} & "
            f"{_ms(cell, 'voc', 'erase_factor__factor_acc')} & "
            f"{_ms(cell, 'voc', 'erase_lexeme__factor_acc')} & "
            f"{_ms(cell, 'voc', 'lexeme_nn')} & "
            f"{_ms(cell, 'voc', 'erase_factor__lexeme_nn')} & "
            f"{_ms(cell, 'voc', 'erase_lexeme__lexeme_nn')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _bm(d: dict):
    """mean$\\pm$sd from a behavior cell ({mean,sd})."""
    return f"${_fmt(d['mean'])}\\pm{_fmt(d['sd'], 3)}$"


def emit_behavior_table(m: dict) -> str | None:
    b = m.get("behavior")
    if not b:
        return None
    lines = ["% Generated by disentangle/results.py -- do not edit by hand.",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "& & \\multicolumn{4}{c}{Agreement accuracy (chance .5) after "
             "erasing} \\\\",
             "\\cmidrule(lr){3-6}",
             "Model & Agreement task & none & $-$number & $-$gender "
             "& $-$random \\\\", "\\midrule"]
    order = ["pretrained AlephBERT", "random-init AlephBERT"]
    for name in order:
        tasks = b["models"].get(name, {})
        for ti, task in enumerate(("number", "gender")):
            r = tasks.get(task)
            if not r:
                continue
            label = name if ti == 0 else ""
            lines.append(
                f"{label} & {task} agr.\\ & {_bm(r['none'])} & "
                f"{_bm(r['number'])} & {_bm(r['gender'])} & {_bm(r['random'])} \\\\")
        if name == order[0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


EMITTERS = {
    "TD1_grids.tex": emit_grid_table,
    "TD2_matrix.tex": emit_matrix_table,
    "TD3_control.tex": emit_control_table,
    "TD4_floor_leace.tex": emit_floor_leace_table,
    "TD5_composition.tex": emit_composition_table,
    "TD6_crosslang.tex": emit_crosslang_table,
    "TD7_behavior.tex": emit_behavior_table,
}


def emit_all(m: dict) -> list[str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, fn in EMITTERS.items():
        tex = fn(m)
        if tex is None:
            continue
        (TABLES_DIR / fname).write_text(tex + "\n", encoding="utf-8")
        written.append(fname)
    return written


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", default="",
                    help=f"comma list from {list(COMPUTE)} (or 'all')")
    ap.add_argument("--langs", default="",
                    help="crosslang only: comma subset of ara,heb (default both)")
    ap.add_argument("--emit", action="store_true",
                    help="render the manifest -> disentangle/paper/tables/*.tex")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args(argv)

    manifest = load_manifest()
    if args.compute:
        blocks = list(COMPUTE) if args.compute == "all" else \
            [b.strip() for b in args.compute.split(",") if b.strip()]
        seeds = tuple(SEEDS[:max(1, args.seeds)])
        langs = [l.strip() for l in args.langs.split(",") if l.strip()] or None
        for b in blocks:
            if b not in COMPUTE:
                print(f"unknown block {b!r}; choose from {list(COMPUTE)}",
                      file=sys.stderr)
                return 2
            if b == "crosslang":
                manifest[b] = COMPUTE[b](seeds=seeds, langs=langs,
                                         prior=manifest.get(b))
            else:
                manifest[b] = COMPUTE[b](seeds=seeds)
            save_manifest(manifest)
            print(f"computed + saved block {b!r}", file=sys.stderr)

    if args.emit:
        written = emit_all(manifest)
        print(f"wrote {len(written)} table(s) to {TABLES_DIR}: {', '.join(written)}")

    if not args.compute and not args.emit:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
