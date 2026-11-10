"""Query translation (Arabic <-> English) for cross-lingual retrieval.

A short Arabic question about an English document (or the reverse) is the hardest case for a small
multilingual embedder: on natural phrasings like "هل عندنا إجازة الأمومة؟" the right passage was
ranked first only 2 times out of 13 (see README "Cross-language questions"). Translating the
question and searching with both wordings lets BM25 match the exact words ("maternity") and lets the
cross-encoder compare same-language text.

Models: Helsinki-NLP opus-mt-ar-en and opus-mt-en-ar (MarianMT, ~300 MB each, CPU, Apache-2.0 /
CC-BY-4.0). Each direction is loaded on first use. The browser demo uses their ONNX exports
(Xenova/opus-mt-ar-en, Xenova/opus-mt-en-ar).
"""

from __future__ import annotations

import os
from functools import lru_cache

from .normalize import detect_language

MODELS = {
    "ar": os.getenv("RAG_MT_AR_EN", "Helsinki-NLP/opus-mt-ar-en"),  # question in Arabic -> English
    "en": os.getenv("RAG_MT_EN_AR", "Helsinki-NLP/opus-mt-en-ar"),  # question in English -> Arabic
}


class MarianTranslator:
    """Translates a question into the other language of the pair. Returns "" if it can't."""

    name = "opus-mt"

    def __init__(self, max_chars: int = 300):
        self.max_chars = max_chars

    @staticmethod
    @lru_cache(maxsize=2)
    def _load(model_name: str):
        from transformers import MarianMTModel, MarianTokenizer

        return MarianTokenizer.from_pretrained(model_name), MarianMTModel.from_pretrained(model_name).eval()

    def __call__(self, text: str) -> str:
        text = text.strip()[: self.max_chars]
        if not text:
            return ""
        tok, model = self._load(MODELS[detect_language(text)])
        batch = tok([text], return_tensors="pt", truncation=True)
        out = model.generate(**batch, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3)
        result = tok.decode(out[0], skip_special_tokens=True).strip()
        return result if plausible(text, result) else ""


def plausible(question: str, translation: str) -> bool:
    """Small MT models sometimes loop ("الدنيا : الدنيا : ..."); a looping output is worse than none."""
    words = translation.split()
    if not words or len(translation) > 3 * len(question) + 20:
        return False
    return len(words) < 6 or len(set(words)) / len(words) >= 0.5
