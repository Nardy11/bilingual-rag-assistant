"""The end-to-end pipeline: ingest -> chunk -> index -> retrieve -> (rerank) -> generate with citations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_pages, simple_token_count
from .embeddings import Embedder, SentenceTransformerEmbedder
from .generate import NOT_FOUND, Generator, get_generator
from .ingest import Page, load_directory
from .normalize import detect_language
from .retrieval import HybridIndex

# Minimum cosine similarity (query vs best retrieved chunk) to attempt an answer. Calibrated on
# eval/questions.jsonl with multilingual-e5-small (see README "Evaluation"); a different embedding
# model needs a different value.
MIN_CONFIDENCE = float(os.getenv("RAG_MIN_CONFIDENCE", "0.80"))

RERANK_POOL = 20  # hybrid candidates handed to the cross-encoder


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc: str
    page: int
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    found: bool
    confidence: float
    generator: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        rerank: bool = False,
        min_confidence: float = MIN_CONFIDENCE,
        max_tokens: int = 160,
        overlap_tokens: int = 40,
    ):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.index = HybridIndex(self.embedder)
        self.generator = generator or get_generator(self.embedder)
        self.reranker = None
        if rerank:
            from .rerank import CrossEncoderReranker

            self.reranker = CrossEncoderReranker()
        self.min_confidence = min_confidence
        self.max_tokens, self.overlap_tokens = max_tokens, overlap_tokens

    def _count(self, text: str) -> int:
        counter = getattr(self.embedder, "count_tokens", None)
        return counter(text) if counter else simple_token_count(text)

    def add_pages(self, pages: list[Page], index: HybridIndex | None = None) -> int:
        chunks = chunk_pages(pages, self.max_tokens, self.overlap_tokens, self._count)
        # explicit None check: an empty HybridIndex is falsy (__len__) and must not fall back to the shared one
        target = self.index if index is None else index
        target.add(chunks)
        return len(chunks)

    def fork_index(self) -> HybridIndex:
        """A private copy of the shared index (e.g. one per Space session), so uploads never touch it."""
        fork = HybridIndex(self.embedder)
        fork.add(list(self.index.chunks), self.index.vectors.copy())
        return fork

    def ingest_directory(self, path: str | Path) -> int:
        return self.add_pages(load_directory(path))

    def retrieve(self, question: str, k: int = 5, index: HybridIndex | None = None):
        target = self.index if index is None else index
        hits = target.hybrid_search(question, k=max(k, RERANK_POOL) if self.reranker else k)
        if self.reranker:
            hits = self.reranker.rerank(question, hits, k)
        return hits

    def ask(self, question: str, k: int = 5, index: HybridIndex | None = None) -> Answer:
        hits = self.retrieve(question, k, index)
        confidence = max((h.dense for h in hits), default=0.0)
        if not hits or confidence < self.min_confidence:
            return Answer(
                question,
                NOT_FOUND[detect_language(question)],
                False,
                confidence,
                self.generator.name,
                [],
                [h.chunk.id for h in hits],
            )
        gen = self.generator.generate(question, hits)
        found = not gen.text.strip().startswith(tuple(NOT_FOUND.values()))
        cites = [
            Citation(n, hits[n - 1].chunk.id, hits[n - 1].chunk.doc, hits[n - 1].chunk.page, hits[n - 1].chunk.text)
            for n in gen.used
        ]
        return Answer(question, gen.text, found, confidence, self.generator.name, cites, [h.chunk.id for h in hits])
