#!/usr/bin/env python3
"""Deterministic Syriac transliteration for cross-script transfer.

Standard library only. Two uses:

1. **Syriac -> Hebrew script** (``syriac_to_hebrew``). Syriac and Hebrew are both
   22-letter abjads descended from Imperial Aramaic with a near 1:1 consonant
   correspondence. Mapping Syriac into Hebrew letters lets a Hebrew/Aramaic
   pretrained model's subword vocabulary apply to Syriac -- the cheapest possible
   Semitic transfer (see ``docs/DESIGN.md``, Transfer door). The reverse map
   (``hebrew_to_syriac``) makes the consonantal round-trip exact.

2. **Syriac -> Latin** (``syriac_to_latin``), a scholarly romanization for
   logging and human inspection, so readers who cannot read the script still
   follow.

We map the 22 base consonants. Combining vowel/diacritic marks are dropped by
default (the consonantal skeleton is what transfers); pass ``keep_marks=True`` to
preserve them as-is. Non-letter characters (spaces, punctuation, digits) pass
through unchanged.

    .venv/bin/python -m neural.transliterate --demo
    echo "ܡܠܟܘܬܐ" | .venv/bin/python -m neural.transliterate --to hebrew
"""

from __future__ import annotations

import argparse
import sys
import unicodedata

# --------------------------------------------------------------------------- #
# Base consonant inventory (Unicode names abbreviated in comments).
# Syriac letters live in U+0710..U+072F; Hebrew in U+05D0..U+05EA.
# --------------------------------------------------------------------------- #
# (syriac_codepoint, hebrew_nonfinal, hebrew_final_or_None, latin, cal_ascii)
#
# cal_ascii is SEDRA's CAL-style romanization (George A. Kiraz / Beth Mardutho).
# It is deliberately *non-phonetic* -- the ASCII letter is a fixed slot in the
# Aramaic alphabet order, NOT the consonant's Latin value: e.g. K=Heth, C=Kaph,
# Y=Teth, ;=Yudh, E=Ayin, I=Pe, /=Sadhe, X=Qaph, O=Waw, W=Shin. This is the
# alphabet the SEDRA vocalised lexicon (and the vocaliser trained on it) uses, so
# mapping Syriac running text into CAL is what lets the vocaliser read it. The
# assignment below was verified against the SEDRA 3 WORDS skeleton field: all 22
# consonants occur and the only extra symbol is "-" (a boundary marker).
_LETTERS: list[tuple[int, str, str | None, str, str]] = [
    (0x0710, "\u05D0", None,    "ʾ", "A"),   # Alaph   -> Alef            / CAL A
    (0x0712, "\u05D1", None,    "b", "B"),    # Beth    -> Bet             / CAL B
    (0x0713, "\u05D2", None,    "g", "G"),    # Gamal   -> Gimel           / CAL G
    (0x0715, "\u05D3", None,    "d", "D"),    # Dalath  -> Dalet           / CAL D
    (0x0717, "\u05D4", None,    "h", "H"),    # He      -> He              / CAL H
    (0x0718, "\u05D5", None,    "w", "O"),    # Waw     -> Vav             / CAL O
    (0x0719, "\u05D6", None,    "z", "Z"),    # Zain    -> Zayin           / CAL Z
    (0x071A, "\u05D7", None,    "ḥ", "K"),   # Heth    -> Het             / CAL K
    (0x071B, "\u05D8", None,    "ṭ", "Y"),   # Teth    -> Tet             / CAL Y
    (0x071D, "\u05D9", None,    "y", ";"),    # Yudh    -> Yod             / CAL ;
    (0x071F, "\u05DB", "\u05DA", "k", "C"),   # Kaph    -> Kaf (final Kaf)  / CAL C
    (0x0720, "\u05DC", None,    "l", "L"),    # Lamadh  -> Lamed           / CAL L
    (0x0721, "\u05DE", "\u05DD", "m", "M"),   # Mim     -> Mem (final Mem)  / CAL M
    (0x0722, "\u05E0", "\u05DF", "n", "N"),   # Nun     -> Nun (final Nun)  / CAL N
    (0x0723, "\u05E1", None,    "s", "S"),    # Semkath -> Samekh          / CAL S
    (0x0725, "\u05E2", None,    "ʿ", "E"),   # E       -> Ayin            / CAL E
    (0x0726, "\u05E4", "\u05E3", "p", "I"),   # Pe      -> Pe (final Pe)    / CAL I
    (0x0728, "\u05E6", "\u05E5", "ṣ", "/"),  # Sadhe   -> Tsadi (final)    / CAL /
    (0x0729, "\u05E7", None,    "q", "X"),    # Qaph    -> Qof             / CAL X
    (0x072A, "\u05E8", None,    "r", "R"),    # Rish    -> Resh            / CAL R
    (0x072B, "\u05E9", None,    "š", "W"),   # Shin    -> Shin            / CAL W
    (0x072C, "\u05EA", None,    "t", "T"),    # Taw     -> Tav             / CAL T
]

_SYRIAC_TO_HEBREW: dict[str, str] = {chr(cp): heb for cp, heb, _fin, _lat, _cal in _LETTERS}
_SYRIAC_TO_HEBREW_FINAL: dict[str, str] = {
    chr(cp): (fin or heb) for cp, heb, fin, _lat, _cal in _LETTERS
}
_SYRIAC_TO_LATIN: dict[str, str] = {chr(cp): lat for cp, _heb, _fin, lat, _cal in _LETTERS}
_SYRIAC_TO_CAL: dict[str, str] = {chr(cp): cal for cp, _heb, _fin, _lat, cal in _LETTERS}
_CAL_TO_SYRIAC: dict[str, str] = {cal: chr(cp) for cp, _heb, _fin, _lat, cal in _LETTERS}

# Reverse: both regular and final Hebrew forms map back to the one Syriac letter.
_HEBREW_TO_SYRIAC: dict[str, str] = {}
for _cp, _heb, _fin, _lat, _cal in _LETTERS:
    _HEBREW_TO_SYRIAC[_heb] = chr(_cp)
    if _fin:
        _HEBREW_TO_SYRIAC[_fin] = chr(_cp)

# Syriac combining marks (vowel points etc.) occupy U+0730..U+074A; the corpus
# also uses the generic combining block U+0300..U+036F (e.g. seyame).
def _is_combining(ch: str) -> bool:
    return unicodedata.combining(ch) != 0


def _is_syriac_letter(ch: str) -> bool:
    return ch in _SYRIAC_TO_HEBREW


def syriac_to_hebrew(text: str, *, finals: bool = True, keep_marks: bool = False) -> str:
    """Map Syriac script to Hebrew script (1:1 on the 22 consonants).

    With ``finals=True`` the word-final forms of k/m/n/p/ṣ are emitted, which is
    idiomatic Hebrew; the reverse map accepts both forms, so the consonantal
    round-trip is exact either way.
    """
    out: list[str] = []
    # Determine word-final positions: a letter is final if the next character is
    # not a Syriac letter and not a combining mark attached to this word.
    chars = list(text)
    n = len(chars)
    for i, ch in enumerate(chars):
        if _is_syriac_letter(ch):
            is_final = True
            for j in range(i + 1, n):
                nxt = chars[j]
                if _is_combining(nxt):
                    continue
                is_final = not _is_syriac_letter(nxt)
                break
            table = _SYRIAC_TO_HEBREW_FINAL if (finals and is_final) else _SYRIAC_TO_HEBREW
            out.append(table[ch])
        elif _is_combining(ch):
            if keep_marks:
                out.append(ch)
            # else: drop the mark (consonantal skeleton only)
        else:
            out.append(ch)
    return "".join(out)


def hebrew_to_syriac(text: str, *, keep_marks: bool = False) -> str:
    """Map Hebrew script back to Syriac (inverse of ``syriac_to_hebrew``)."""
    out: list[str] = []
    for ch in text:
        if ch in _HEBREW_TO_SYRIAC:
            out.append(_HEBREW_TO_SYRIAC[ch])
        elif _is_combining(ch):
            if keep_marks:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def syriac_to_latin(text: str, *, keep_marks: bool = False) -> str:
    """Scholarly romanization of the Syriac consonantal skeleton."""
    out: list[str] = []
    for ch in text:
        if ch in _SYRIAC_TO_LATIN:
            out.append(_SYRIAC_TO_LATIN[ch])
        elif _is_combining(ch):
            if keep_marks:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def syriac_to_cal(text: str) -> str:
    """Map Syriac script to SEDRA's CAL-ASCII *skeleton* alphabet.

    This is the alphabet the SEDRA vocalised lexicon uses, so it is how Syriac
    running text is fed to the vocaliser (``neural.vocalizer``). Combining vowel/
    diacritic marks are always dropped -- the CAL skeleton is consonants only, and
    the vocaliser's whole job is to *predict* the pointing -- so unlike the Hebrew
    map there is no ``keep_marks`` option. Non-letter characters (spaces, the
    SEDRA "-" boundary marker, punctuation) pass through unchanged.
    """
    out: list[str] = []
    for ch in text:
        if ch in _SYRIAC_TO_CAL:
            out.append(_SYRIAC_TO_CAL[ch])
        elif _is_combining(ch):
            continue  # skeleton only
        else:
            out.append(ch)
    return "".join(out)


def cal_to_syriac(text: str) -> str:
    """Map a SEDRA CAL-ASCII skeleton back to Syriac script (inverse of above).

    Only the 22 consonant slots are mapped; any other character (vowels a/e/i/o/u,
    diacritics, "-", spaces) passes through unchanged, so this is exact on a pure
    skeleton and lossless-by-passthrough on a vocalised CAL string.
    """
    return "".join(_CAL_TO_SYRIAC.get(ch, ch) for ch in text)


# Curated demo words (consonantal skeletons) with glosses.
_DEMO = [
    ("\u0721\u0720\u071F\u0710", "king"),            # malka
    ("\u0721\u0720\u071F\u0718\u072C\u0710", "kingdom"),  # malkuta
    ("\u071F\u072C\u0712\u0710", "book"),            # ktaba
    ("\u0710\u0712\u0710", "father"),                # aba
]


def _run_demo() -> None:
    print(f"{'Syriac':<12} {'Hebrew':<12} {'CAL':<10} {'Latin':<10} gloss")
    print("-" * 56)
    for syr, gloss in _DEMO:
        heb = syriac_to_hebrew(syr)
        cal = syriac_to_cal(syr)
        lat = syriac_to_latin(syr)
        heb_ok = "OK" if hebrew_to_syriac(heb) == syr else "MISMATCH"
        cal_ok = "OK" if cal_to_syriac(cal) == syr else "MISMATCH"
        print(f"{syr:<12} {heb:<12} {cal:<10} {lat:<10} {gloss}  "
              f"[heb {heb_ok} / cal {cal_ok}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Syriac transliteration utilities.")
    ap.add_argument("--demo", action="store_true", help="show a small demo table")
    ap.add_argument("--to", choices=["hebrew", "latin", "cal", "syriac"],
                    help="convert stdin to the given script")
    ap.add_argument("--keep-marks", action="store_true",
                    help="preserve combining diacritics instead of dropping them")
    ap.add_argument("--no-finals", action="store_true",
                    help="do not use Hebrew word-final letter forms")
    args = ap.parse_args(argv)

    if args.demo:
        _run_demo()
        return 0
    if args.to:
        data = sys.stdin.read()
        if args.to == "hebrew":
            sys.stdout.write(syriac_to_hebrew(data, finals=not args.no_finals,
                                              keep_marks=args.keep_marks))
        elif args.to == "latin":
            sys.stdout.write(syriac_to_latin(data, keep_marks=args.keep_marks))
        elif args.to == "cal":
            sys.stdout.write(syriac_to_cal(data))
        else:
            sys.stdout.write(hebrew_to_syriac(data, keep_marks=args.keep_marks))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
