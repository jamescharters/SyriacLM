#!/usr/bin/env python3
"""Build a local (license-aware) SEDRA word table for the pointing objective.

Parses the raw SEDRA 3 text database (``WORDS.TXT`` + ``LEXEMES.TXT`` +
``ROOTS.TXT``) into the JSON that :mod:`neural.sedra` consumes, joining each word
to its lexeme and root so the pointing objective (Twist 1) and the factored
root/pattern encoder (Twist 2) both have their supervision.

LICENSE (important). SEDRA 3 is "MIT **with restrictions**": academic/personal
use only, no redistribution of altered versions, and any publication must cite
Kiraz (see ``neural.sedra.SEDRA_CITATION``). Accordingly this repo ships **no**
SEDRA data: you obtain the raw files yourself (e.g. ``git clone
https://github.com/peshitta/sedrajs`` -> ``sedra/``) and this script writes a
derived table to the git-ignored ``neural/sedra_cache/`` (or ``~/.cache``). The
derived JSON is never committed.

    # after cloning sedrajs somewhere, e.g. ~/.cache/sedrajs
    .venv/bin/python -m neural.sedra_build --sedra-dir ~/.cache/sedrajs/sedra

Record formats (from SEDRA3.DOC), comma-separated, fields 0-indexed:
  WORDS.TXT   : id, lexeme_addr, "consonantal", "vocalised", morph_type, attrs
  LEXEMES.TXT : id, root_addr,   "lexeme",       morph_type,  attrs
  ROOTS.TXT   : id, "root",       "sort",        attrs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from neural.sedra import (
    SEDRA_VOWELS, SEDRA_DIACRITICS, SEDRA_CITATION, split_skeleton_pointing,
)

DEFAULT_SOURCES = [
    Path.home() / ".cache" / "sedrajs" / "sedra",
    Path("neural/sedra_cache/sedra"),
]
DEFAULT_OUT = Path("neural/sedra_cache/words.json")


def _read_csv(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            rows.append(next(csv.reader([line])))
    return rows


def _strip_pointing(s: str) -> str:
    return "".join(c for c in s if c not in SEDRA_VOWELS and c not in SEDRA_DIACRITICS)


def find_sedra_dir(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if (explicit / "WORDS.TXT").exists() else None
    for cand in DEFAULT_SOURCES:
        if (cand / "WORDS.TXT").exists():
            return cand
    return None


def build(sedra_dir: Path) -> list[dict]:
    roots = {r[0]: r[1] for r in _read_csv(sedra_dir / "ROOTS.TXT") if len(r) >= 2}
    # lexeme address -> (lexeme string, root address)
    lexemes: dict[str, tuple[str, str]] = {}
    for r in _read_csv(sedra_dir / "LEXEMES.TXT"):
        if len(r) >= 3:
            lexemes[r[0]] = (r[2], r[1])

    records: list[dict] = []
    for r in _read_csv(sedra_dir / "WORDS.TXT"):
        if len(r) < 4:
            continue
        _wid, lex_addr, cons, voc = r[0], r[1], r[2], r[3]
        if not voc:
            continue
        skel, _pts = split_skeleton_pointing(voc)
        lex_str, root_addr = lexemes.get(lex_addr, ("", ""))
        root_str = roots.get(root_addr, "")
        records.append({
            "skeleton": skel or _strip_pointing(cons),
            "vocalised": voc,
            "root": root_str,
            "lexeme": lex_str,
        })
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sedra-dir", type=Path, default=None,
                    help="directory with WORDS.TXT/LEXEMES.TXT/ROOTS.TXT")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output JSON (git-ignored; never commit SEDRA-derived data)")
    args = ap.parse_args(argv)

    sedra_dir = find_sedra_dir(args.sedra_dir)
    if sedra_dir is None:
        print("No SEDRA source found. SEDRA is license-restricted and not shipped.\n"
              "Obtain it, e.g.:\n"
              "    git clone https://github.com/peshitta/sedrajs ~/.cache/sedrajs\n"
              "then re-run with --sedra-dir ~/.cache/sedrajs/sedra", file=sys.stderr)
        return 2

    records = build(sedra_dir)
    if not records:
        print(f"error: no records parsed from {sedra_dir}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    n_root = sum(1 for r in records if r["root"])
    print(f"parsed {len(records):,} SEDRA words ({n_root:,} with a root) from {sedra_dir}")
    print(f"wrote derived table to {args.out}  (git-ignored)")
    print("\nReminder -- cite SEDRA in any publication:")
    print("  " + SEDRA_CITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
