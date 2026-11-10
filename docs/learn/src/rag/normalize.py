"""Arabic-aware text normalisation and the lexical tokenizer used by BM25.

Two levels of normalisation:

* ``clean_text`` keeps the text readable (used for what we store and show): it removes tatweel and
  invisible marks and collapses whitespace, but keeps tashkeel and letter forms as written.
* ``normalize_for_match`` is for matching only (BM25 tokens, answer checks): it strips tashkeel,
  unifies alef/yaa/taa-marbuta/hamza forms, maps Arabic-Indic digits to ASCII and lowercases Latin.

The same rules are ported to JavaScript in ``docs/app.js`` so the browser demo tokenises exactly
like Python; ``tests/test_normalize.py`` pins the behaviour.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (fatha .. sukun), superscript alef, Quranic marks.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_ALEF = re.compile(r"[آأإٱ]")  # آ أ إ ٱ -> ا
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[a-z]+|[ء-ي]+")

# Light prefix stripping for Arabic tokens (definite article and attached conjunction/prepositions).
# Kept deliberately conservative: only strip when at least 3 letters remain.
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")

STOPWORDS_EN = frozenset(
    "a an the and or of to in on at for from by with is are was were be been being it its this "
    "that these those as do does did can could should would will shall may might must not no "
    "i you he she we they me my your our their what which who whom when where why how any all "
    "if then than so such there here up out about into over under".split()
)
STOPWORDS_AR = frozenset(
    "في من على الى إلى عن مع هل ما ماذا متى كيف كم اين أين هو هي هم انا أنا نحن انت أنت هذا هذه "
    "ذلك تلك التي الذي الذين او أو ثم ان أن إن لا لم لن قد كان كانت يكون تكون كل اي أي بعد قبل "
    "عند حتى اذا إذا و ب ل ك ف يجب يمكن".split()
)


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) and tatweel."""
    return _TASHKEEL.sub("", text).replace(_TATWEEL, "")


def clean_text(text: str) -> str:
    """Readable cleanup: NFKC, no tatweel / invisible marks, collapsed spaces (keeps newlines)."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text).replace(_TATWEEL, "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    """Unify the letter forms that are spelled inconsistently in practice.

    alef variants (آ أ إ ٱ) -> ا, alef maqsura ى -> ي, taa marbuta ة -> ه,
    hamza on waw/yaa (ؤ ئ) -> ء, and tashkeel/tatweel removed.
    """
    text = strip_tashkeel(text)
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "ء").replace("ئ", "ء")
    return text


def normalize_for_match(text: str) -> str:
    """Normalisation used for matching only (never for display)."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = normalize_arabic(text).translate(_DIGITS)
    return text.lower()


def _strip_ar_prefix(token: str) -> str:
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            return token[len(p) :]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lexical tokens for BM25: normalised, Arabic article stripped, stopwords dropped."""
    out = []
    for tok in _TOKEN.findall(normalize_for_match(text)):
        if remove_stopwords and (tok in STOPWORDS_EN or tok in STOPWORDS_AR):
            continue
        if _ARABIC_CHAR.match(tok):
            tok = _strip_ar_prefix(tok)
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # naive English plural folding: days -> day, filters -> filter
        out.append(tok)
    return out


def detect_language(text: str) -> str:
    """'ar' if Arabic letters dominate Latin letters, else 'en'."""
    ar = len(_ARABIC_CHAR.findall(text))
    en = len(_LATIN_CHAR.findall(text))
    return "ar" if ar > en else "en"
