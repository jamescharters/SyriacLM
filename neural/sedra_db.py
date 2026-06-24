#!/usr/bin/env python3
"""Build a queryable SQLite database from the scraped SEDRA JSON files.

``neural/sedra_scrape.py`` mirrors the SEDRA API to one ``<id>.json`` file per
record under ``neural/sedra_cache/api/<endpoint>/`` (``word``, ``lexeme``,
``root``). That layout is faithful to the API but awkward to query. This script
loads every ``<id>.json`` into a single SQLite database with a normalised,
query-friendly schema:

* ``words``         -- one row per word form: Syriac spelling, western/eastern
                       vocalised pointing, stem, part of speech, morphology flags
                       and the owning ``lexeme_id``.
* ``glosses``       -- one row per ``(word_id, language, gloss)`` translation
                       (``eng``/``ara``/``fre`` ... three-letter codes).
* ``lexemes``       -- one row per dictionary headword: Syriac, part of speech
                       and the owning ``root_id``.
* ``lexeme_glosses``-- ``(lexeme_id, language, gloss)`` (full language names
                       ``English``/``Arabic``/``French`` ... as the API returns).
* ``etymologies``   -- ``(lexeme_id, source_language, term)`` loan-word origins
                       embedded in each lexeme (e.g. Greek ``\u1f00\u03ae\u03c1``).
* ``roots``         -- one row per consonantal root (Syriac).
* ``gloss_fts`` / ``lexeme_gloss_fts`` -- FTS5 full-text indexes over the word
                       and lexeme glosses (skipped if this SQLite lacks FTS5).
* ``meta``          -- provenance: source dir, counts, build time, citation.

For every row a scalar field without its own column is preserved in an
``attributes`` JSON blob and the untouched record is kept in ``raw_json`` -- so
nothing from the API is ever lost. The ``words``->``lexemes``->``roots`` chain is
navigable via the ``lexeme_id``/``root_id`` foreign keys.

LICENSE. SEDRA is academic/personal use with restrictions (no redistribution of
altered versions; cite Kiraz -- see ``SEDRA_CITATION``). The database is written
under the git-ignored ``sedra_cache/`` and is never committed; it is a local
convenience rebuilt from your own download.

Examples
--------
    # build neural/sedra_cache/sedra.db from neural/sedra_cache/api/
    .venv/bin/python neural/sedra_db.py

    # custom locations / no full-text index
    .venv/bin/python neural/sedra_db.py --src some/dir --out /tmp/sedra.db --no-fts

    # an etymology query afterwards
    sqlite3 neural/sedra_cache/sedra.db \
      "SELECT l.syriac, e.lang, e.term FROM lexemes l
       JOIN etymologies e ON e.lexeme_id=l.lexeme_id LIMIT 5;"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

SEDRA_CITATION = (
    "This work makes use of the Syriac Electronic Data Retrieval Archive (SEDRA) "
    "by George A. Kiraz, distributed by the Syriac Computing Institute."
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


# Base dir holding the per-endpoint subdirs word/ lexeme/ root/.
DEFAULT_SRC = _script_dir() / "sedra_cache" / "api"
DEFAULT_DB = _script_dir() / "sedra_cache" / "sedra.db"

# Nested keys handled specially (not stored as scalar columns / attributes).
NESTED_KEYS = frozenset({"lexeme", "word", "glosses"})
# String-boolean fields ("true"/"false") stored as 0/1 INTEGER columns.
BOOL_KEYS = frozenset({"isLexicalForm", "isEnclitic", "hasSeyame", "isTheoretical"})
# JSON key -> SQLite column. Everything scalar but absent here lands in
# ``attributes`` (so unseen morphology fields are preserved, not dropped).
SCALAR_COLS: dict[str, str] = {
    "syriac": "syriac",
    "western": "western",
    "eastern": "eastern",
    "stem": "stem",
    "category": "category",
    "number": "number",
    "person": "person",
    "gender": "gender",
    "state": "state",
    "tense": "tense",
    "isLexicalForm": "is_lexical_form",
    "isEnclitic": "is_enclitic",
    "hasSeyame": "has_seyame",
    "isTheoretical": "is_theoretical",
}

WORD_COLUMNS: list[str] = [
    "word_id", "lexeme_id", "syriac", "western", "eastern", "stem", "category",
    "number", "person", "gender", "state", "tense", "is_lexical_form",
    "is_enclitic", "has_seyame", "is_theoretical", "attributes", "raw_json",
]

# Lexeme records: id under rec["lexeme"]["id"], owning root under rec["root"]["id"].
LEXEME_COLUMNS: list[str] = [
    "lexeme_id", "root_id", "syriac", "category", "attributes", "raw_json",
]
LEXEME_NESTED = frozenset({"lexeme", "root", "glosses", "etymologies", "words"})
LEXEME_SCALAR: dict[str, str] = {"syriac": "syriac", "category": "category"}

# Root records: id is the top-level rec["id"]; member lexemes are URI strings
# (derivable via lexemes.root_id, so kept only in raw_json).
ROOT_COLUMNS: list[str] = ["root_id", "syriac", "attributes", "raw_json"]
ROOT_NESTED = frozenset({"id", "lexemes"})
ROOT_SCALAR: dict[str, str] = {"syriac": "syriac"}

SCHEMA = """
CREATE TABLE words (
    word_id          INTEGER PRIMARY KEY,
    lexeme_id        INTEGER,
    syriac           TEXT,
    western          TEXT,
    eastern          TEXT,
    stem             TEXT,
    category         TEXT,
    number           TEXT,
    person           TEXT,
    gender           TEXT,
    state            TEXT,
    tense            TEXT,
    is_lexical_form  INTEGER,
    is_enclitic      INTEGER,
    has_seyame       INTEGER,
    is_theoretical   INTEGER,
    attributes       TEXT,
    raw_json         TEXT
);
CREATE TABLE glosses (
    word_id  INTEGER NOT NULL,
    lang     TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    gloss    TEXT NOT NULL
);
CREATE TABLE lexemes (
    lexeme_id   INTEGER PRIMARY KEY,
    root_id     INTEGER,
    syriac      TEXT,
    category    TEXT,
    attributes  TEXT,
    raw_json    TEXT
);
CREATE TABLE lexeme_glosses (
    lexeme_id  INTEGER NOT NULL,
    lang       TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    gloss      TEXT NOT NULL
);
CREATE TABLE etymologies (
    lexeme_id  INTEGER NOT NULL,
    lang       TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    term       TEXT NOT NULL
);
CREATE TABLE roots (
    root_id     INTEGER PRIMARY KEY,
    syriac      TEXT,
    attributes  TEXT,
    raw_json    TEXT
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_words_lexeme   ON words(lexeme_id);
CREATE INDEX idx_words_category ON words(category);
CREATE INDEX idx_glosses_word   ON glosses(word_id);
CREATE INDEX idx_glosses_lang   ON glosses(lang);
CREATE INDEX idx_lexemes_root   ON lexemes(root_id);
CREATE INDEX idx_lexgloss_lex   ON lexeme_glosses(lexeme_id);
CREATE INDEX idx_lexgloss_lang  ON lexeme_glosses(lang);
CREATE INDEX idx_etym_lex       ON etymologies(lexeme_id);
CREATE INDEX idx_etym_lang      ON etymologies(lang);
"""


def _as_bool(value: object) -> int | None:
    """SEDRA stores booleans as the strings ``"true"``/``"false"``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "true":
            return 1
        if s == "false":
            return 0
    return None


def _nested_id(rec: dict, key: str) -> int | None:
    node = rec.get(key)
    if isinstance(node, dict) and node.get("id") is not None:
        try:
            return int(node["id"])
        except (TypeError, ValueError):
            return None
    return None


def parse_record(rec: dict, fallback_id: int) -> tuple[list, list[tuple]]:
    """Turn one API record into a ``words`` row (list aligned to ``WORD_COLUMNS``)
    plus its ``glosses`` rows. Scalar fields without a dedicated column are kept
    in ``attributes``; the whole record is kept in ``raw_json``."""
    word_id = _nested_id(rec, "word")
    if word_id is None:
        word_id = fallback_id

    row: dict[str, object] = {c: None for c in WORD_COLUMNS}
    row["word_id"] = word_id
    row["lexeme_id"] = _nested_id(rec, "lexeme")
    row["raw_json"] = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))

    extra: dict[str, object] = {}
    for key, value in rec.items():
        if key in NESTED_KEYS:
            continue
        col = SCALAR_COLS.get(key)
        if col is None:
            extra[key] = value
        elif key in BOOL_KEYS:
            row[col] = _as_bool(value)
        elif value is None or isinstance(value, (str, int, float)):
            row[col] = value
        else:
            row[col] = json.dumps(value, ensure_ascii=False)
    row["attributes"] = json.dumps(extra, ensure_ascii=False) if extra else None

    glosses: list[tuple] = []
    g = rec.get("glosses")
    if isinstance(g, dict):
        for lang, items in g.items():
            seq_items = items if isinstance(items, list) else [items]
            for seq, gloss in enumerate(seq_items):
                if gloss is None:
                    continue
                glosses.append((word_id, str(lang), seq, str(gloss)))

    return [row[c] for c in WORD_COLUMNS], glosses


def _flatten_lang_map(node: object, owner_id: int) -> list[tuple]:
    """Flatten a SEDRA ``{language: [items]}`` map (glosses or etymologies) into
    ``(owner_id, lang, seq, item)`` rows, preserving order via ``seq``."""
    out: list[tuple] = []
    if isinstance(node, dict):
        for lang, items in node.items():
            seq_items = items if isinstance(items, list) else [items]
            for seq, item in enumerate(seq_items):
                if item is None:
                    continue
                out.append((owner_id, str(lang), seq, str(item)))
    return out


def _scalar_row(
    rec: dict, columns: list[str], id_col: str, id_val: int,
    scalar_map: dict[str, str], nested: frozenset[str],
    extra_cols: dict[str, int] | None = None,
) -> list:
    """Generic record->row: map known scalars to columns, stash unknown scalars in
    ``attributes`` and the full record in ``raw_json`` (nothing is lost)."""
    row: dict[str, object] = {c: None for c in columns}
    row[id_col] = id_val
    if extra_cols:
        for col, val in extra_cols.items():
            row[col] = val
    row["raw_json"] = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    extra: dict[str, object] = {}
    for key, value in rec.items():
        if key in nested:
            continue
        col = scalar_map.get(key)
        if col is None:
            extra[key] = value
        elif value is None or isinstance(value, (str, int, float)):
            row[col] = value
        else:
            row[col] = json.dumps(value, ensure_ascii=False)
    row["attributes"] = json.dumps(extra, ensure_ascii=False) if extra else None
    return [row[c] for c in columns]


def parse_lexeme(rec: dict, fallback_id: int) -> tuple[list, list[tuple], list[tuple]]:
    """Lexeme record -> (``lexemes`` row, ``lexeme_glosses`` rows, ``etymologies``
    rows). The lexeme id is under ``rec['lexeme']['id']``; the owning root under
    ``rec['root']['id']``; glosses use full language names; etymologies are the
    embedded ``{source_language: [terms]}`` loan-word origins."""
    lexeme_id = _nested_id(rec, "lexeme")
    if lexeme_id is None:
        lexeme_id = fallback_id
    row = _scalar_row(
        rec, LEXEME_COLUMNS, "lexeme_id", lexeme_id, LEXEME_SCALAR, LEXEME_NESTED,
        extra_cols={"root_id": _nested_id(rec, "root")},
    )
    glosses = _flatten_lang_map(rec.get("glosses"), lexeme_id)
    etyms = _flatten_lang_map(rec.get("etymologies"), lexeme_id)
    return row, glosses, etyms


def parse_root(rec: dict, fallback_id: int) -> list:
    """Root record -> ``roots`` row. The id is the top-level ``rec['id']``; member
    lexemes are URI strings (recoverable via ``lexemes.root_id``) kept in raw_json."""
    try:
        root_id = int(rec.get("id"))
    except (TypeError, ValueError):
        root_id = fallback_id
    return _scalar_row(
        rec, ROOT_COLUMNS, "root_id", root_id, ROOT_SCALAR, ROOT_NESTED
    )


def _load_records(path: Path) -> list[dict] | None:
    """Load a cached ``<id>.json`` as a list of record dicts, or ``None`` if the
    file is unreadable/invalid. Each API document is a JSON array (occasionally a
    bare object)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = data if isinstance(data, list) else [data]
    return [r for r in records if isinstance(r, dict)]


def _id_files(src: Path) -> list[tuple[int, Path]]:
    """All ``<digits>.json`` files in ``src``, sorted by id (skips bookkeeping
    files like ``_missing.txt`` / ``_summary.json`` / ``*.tmp``). Returns [] if
    ``src`` is absent."""
    if not src.is_dir():
        return []
    files = [
        (int(p.stem), p)
        for p in src.glob("*.json")
        if p.stem.isdigit()
    ]
    files.sort()
    return files


def build(src: Path, db_path: Path, fts: bool) -> dict:
    # Per-endpoint subdirs (current layout); fall back to a flat word dir for an
    # older cache that stored word files directly under src/.
    word_dir = src / "word"
    if not word_dir.is_dir() and _id_files(src):
        word_dir = src
    word_files = _id_files(word_dir)
    lexeme_files = _id_files(src / "lexeme")
    root_files = _id_files(src / "root")

    if not (word_files or lexeme_files or root_files):
        raise SystemExit(
            f"no <id>.json files found under {src}\n"
            f"run the scraper first: .venv/bin/python neural/sedra_scrape.py"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # full rebuild; the DB is a derived, regenerable cache

    con = sqlite3.connect(db_path)
    try:
        # Derived cache: trade durability for build speed (rebuilt on demand).
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.executescript(SCHEMA)
        cur = con.cursor()
        counts = {
            k: 0 for k in (
                "words", "glosses", "lexemes", "lexeme_glosses", "etymologies",
                "roots", "bad",
            )
        }

        # ---- words (+ glosses) ----
        insert_word = (
            f"INSERT OR REPLACE INTO words ({','.join(WORD_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(WORD_COLUMNS))})"
        )
        insert_gloss = (
            "INSERT INTO glosses (word_id, lang, seq, gloss) VALUES (?,?,?,?)"
        )
        con.execute("BEGIN")
        for wid, path in word_files:
            recs = _load_records(path)
            if recs is None:
                counts["bad"] += 1
                continue
            for rec in recs:
                word_row, glosses = parse_record(rec, wid)
                cur.execute(insert_word, word_row)
                if glosses:
                    cur.executemany(insert_gloss, glosses)
                    counts["glosses"] += len(glosses)
                counts["words"] += 1
        con.commit()

        # ---- lexemes (+ glosses, etymologies) ----
        insert_lex = (
            f"INSERT OR REPLACE INTO lexemes ({','.join(LEXEME_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(LEXEME_COLUMNS))})"
        )
        insert_lexgloss = (
            "INSERT INTO lexeme_glosses (lexeme_id, lang, seq, gloss) VALUES (?,?,?,?)"
        )
        insert_etym = (
            "INSERT INTO etymologies (lexeme_id, lang, seq, term) VALUES (?,?,?,?)"
        )
        con.execute("BEGIN")
        for lid, path in lexeme_files:
            recs = _load_records(path)
            if recs is None:
                counts["bad"] += 1
                continue
            for rec in recs:
                lex_row, lex_glosses, etyms = parse_lexeme(rec, lid)
                cur.execute(insert_lex, lex_row)
                if lex_glosses:
                    cur.executemany(insert_lexgloss, lex_glosses)
                    counts["lexeme_glosses"] += len(lex_glosses)
                if etyms:
                    cur.executemany(insert_etym, etyms)
                    counts["etymologies"] += len(etyms)
                counts["lexemes"] += 1
        con.commit()

        # ---- roots ----
        insert_root = (
            f"INSERT OR REPLACE INTO roots ({','.join(ROOT_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(ROOT_COLUMNS))})"
        )
        con.execute("BEGIN")
        for rid, path in root_files:
            recs = _load_records(path)
            if recs is None:
                counts["bad"] += 1
                continue
            for rec in recs:
                cur.execute(insert_root, parse_root(rec, rid))
                counts["roots"] += 1
        con.commit()

        fts_name = "none"
        if fts:
            try:
                con.executescript(
                    "CREATE VIRTUAL TABLE gloss_fts USING fts5("
                    "  word_id UNINDEXED, lang, gloss);"
                    "CREATE VIRTUAL TABLE lexeme_gloss_fts USING fts5("
                    "  lexeme_id UNINDEXED, lang, gloss);"
                )
                con.execute(
                    "INSERT INTO gloss_fts (word_id, lang, gloss) "
                    "SELECT word_id, lang, gloss FROM glosses"
                )
                con.execute(
                    "INSERT INTO lexeme_gloss_fts (lexeme_id, lang, gloss) "
                    "SELECT lexeme_id, lang, gloss FROM lexeme_glosses"
                )
                con.commit()
                fts_name = "gloss_fts+lexeme_gloss_fts"
            except sqlite3.OperationalError as exc:
                print(
                    f"  note: FTS5 unavailable, skipping full-text index ({exc})",
                    file=sys.stderr,
                )

        meta = {
            "source_dir": str(src),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "word_files": str(len(word_files)),
            "lexeme_files": str(len(lexeme_files)),
            "root_files": str(len(root_files)),
            "words": str(counts["words"]),
            "glosses": str(counts["glosses"]),
            "lexemes": str(counts["lexemes"]),
            "lexeme_glosses": str(counts["lexeme_glosses"]),
            "etymologies": str(counts["etymologies"]),
            "roots": str(counts["roots"]),
            "bad_files": str(counts["bad"]),
            "fts": fts_name,
            "citation": SEDRA_CITATION,
        }
        con.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", meta.items()
        )
        con.commit()

        word_langs = con.execute(
            "SELECT lang, COUNT(*) FROM glosses GROUP BY lang ORDER BY 2 DESC"
        ).fetchall()
        etym_langs = con.execute(
            "SELECT lang, COUNT(*) FROM etymologies GROUP BY lang ORDER BY 2 DESC"
        ).fetchall()
    finally:
        con.execute("PRAGMA optimize")
        con.close()

    return {
        "word_files": len(word_files),
        "lexeme_files": len(lexeme_files),
        "root_files": len(root_files),
        "fts": fts_name,
        "word_langs": word_langs,
        "etym_langs": etym_langs,
        "db": db_path,
        **counts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="directory of scraped <id>.json files")
    ap.add_argument("--out", type=Path, default=DEFAULT_DB,
                    help="output SQLite path (git-ignored; never commit SEDRA data)")
    ap.add_argument("--no-fts", action="store_true",
                    help="skip building the FTS5 full-text gloss index")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"building SQLite DB from {args.src} -> {args.out}", file=sys.stderr)
    stats = build(args.src, args.out, fts=not args.no_fts)

    dt = time.time() - t0
    size_mb = stats["db"].stat().st_size / 1e6
    print(
        f"done in {dt:.1f}s  ->  {stats['db']}  ({size_mb:.1f} MB, fts={stats['fts']})",
        file=sys.stderr,
    )
    print(
        f"  words {stats['words']} (glosses {stats['glosses']}) | "
        f"lexemes {stats['lexemes']} (glosses {stats['lexeme_glosses']}, "
        f"etymologies {stats['etymologies']}) | roots {stats['roots']}"
        + (f" | {stats['bad']} unreadable" if stats["bad"] else ""),
        file=sys.stderr,
    )
    if stats["word_langs"]:
        langs = ", ".join(f"{lang}:{n}" for lang, n in stats["word_langs"])
        print(f"  word glosses by language: {langs}", file=sys.stderr)
    if stats["etym_langs"]:
        el = ", ".join(f"{lang}:{n}" for lang, n in stats["etym_langs"])
        print(f"  etymologies by source language: {el}", file=sys.stderr)
    print(f"  {SEDRA_CITATION}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
