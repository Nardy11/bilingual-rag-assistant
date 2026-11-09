"""Optional multilingual cross-encoder re-ranking.

Default model: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` (multilingual MiniLM trained on
mMARCO, 14 languages incl. Arabic, ~118M params, CPU-friendly). Enable with ``rerank=True``.
"""

from __future__ import annotations

import os

from .retrieval import Hit

DEFAULT_RERANKER = os.getenv("RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, hits: list[Hit], k: int | None = None) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits], show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        out = [Hit(h.chunk, float(s), h.dense) for h, s in ranked]
        return out[:k] if k else out
