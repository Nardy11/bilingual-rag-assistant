"""Dense (FAISS), lexical (BM25) and hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import Embedder
from .normalize import tokenize

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float  # method-specific score (cosine, BM25, RRF or rerank)
    dense: float = 0.0  # cosine similarity to the query, kept for the confidence check


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_hits(lists: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """RRF over several hit lists (e.g. the question and its translation), keyed by chunk id.

    The fused hit keeps the best cosine any of the queries reached, for the no-reranker confidence.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in best or h.dense > best[cid].dense:
                best[cid] = h
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(best[c].chunk, scores[c], best[c].dense) for c in order]


class HybridIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._faiss = faiss.IndexFlatIP(embedder.dim)  # exact inner product on unit vectors = cosine
        self._bm25: BM25Okapi | None = None

    # -- building ---------------------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        if not chunks:
            return
        if vectors is None:
            vectors = self.embedder.embed_passages([c.text for c in chunks])
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])
        self._faiss.add(vectors)
        self._bm25 = BM25Okapi([tokenize(c.text) or ["∅"] for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search -----------------------------------------------------------------------------
    def _qvec(self, query: str) -> np.ndarray:
        return self.embedder.embed_queries([query])

    def dense_search(self, query: str, k: int = 10, qvec: np.ndarray | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        qvec = self._qvec(query) if qvec is None else qvec
        scores, idx = self._faiss.search(qvec, min(k, len(self.chunks)))
        return [Hit(self.chunks[i], float(s), float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def bm25_scores(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        if not self._bm25 or not toks:
            return np.zeros(len(self.chunks))
        return self._bm25.get_scores(toks)

    def bm25_search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.bm25_scores(query)
        order = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]

    def hybrid_search(self, query: str, k: int = 10, pool: int = 30) -> list[Hit]:
        if not self.chunks:
            return []
        qvec = self._qvec(query)
        cos = (self.vectors @ qvec[0]).astype(float)
        dense_rank = list(np.argsort(-cos)[:pool])
        bm = self.bm25_scores(query)
        bm_rank = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        fused = rrf([dense_rank, bm_rank])
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(cos[i])) for i in order]

    # -- persistence ------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        meta = {"model": getattr(self.embedder, "model_name", "?"), "chunks": [c.to_dict() for c in self.chunks]}
        (d / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> HybridIndex:
        d = Path(directory)
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        idx = cls(embedder)
        idx.add([Chunk(**c) for c in meta["chunks"]], np.load(d / "vectors.npy"))
        return idx
