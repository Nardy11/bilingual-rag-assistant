"""Optional multilingual cross-encoder re-ranking.

Default model: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` (multilingual MiniLM trained on
mMARCO, 14 languages incl. Arabic, ~118M params, CPU-friendly). Enable with ``rerank=True``.

With ``alt_queries`` (the translated question), each passage is scored against every wording and
keeps the best score. A chunk often covers several topics in one language; scoring it against a
same-language wording of the question is what lifts the right passage to the top.
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

    def rerank(self, query: str, hits: list[Hit], k: int | None = None, alt_queries: tuple[str, ...] = ()) -> list[Hit]:
        if not hits:
            return hits
        queries = [query, *[q for q in alt_queries if q]]
        pairs = [(q, h.chunk.text) for q in queries for h in hits]
        flat = self.model.predict(pairs, show_progress_bar=False)
        n = len(hits)
        scores = [max(float(flat[j * n + i]) for j in range(len(queries))) for i in range(n)]
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        out = [Hit(h.chunk, s, h.dense) for h, s in ranked]
        return out[:k] if k else out
