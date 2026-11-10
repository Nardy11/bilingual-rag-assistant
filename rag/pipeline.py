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
from .retrieval import HybridIndex, fuse_hits

# Confidence comes in three bands: at or above SURE -> answer; between FLOOR and SURE -> answer,
# marked "not sure" (the closest passage found); below FLOOR -> "not found".
#
# A single yes/no cut-off does not work: on 48 answerable and 28 off-topic questions no threshold on
# either signal separates the two groups (README "Confidence"), and the old 0.80 cosine gate refused
# answerable questions whose right passage was ranked first. Each floor is set so that no answerable
# question in that set is refused; the "not sure" band is where the two groups overlap.
#
# With the re-ranker the signal is the cross-encoder score of the top passage (a logit); without
# it, the top cosine of multilingual-e5-small. Another model needs re-calibrating
# (eval/calibrate_confidence.py).
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "-4.0"))  # re-ranker floor
SURE_SCORE = float(os.getenv("RAG_SURE_SCORE", "-2.0"))  # re-ranker "sure"
MIN_CONFIDENCE = float(os.getenv("RAG_MIN_CONFIDENCE", "0.76"))  # cosine floor (no re-ranker)
SURE_CONFIDENCE = float(os.getenv("RAG_SURE_CONFIDENCE", "0.82"))  # cosine "sure" (no re-ranker)

RERANK_POOL = 10  # hybrid candidates handed to the cross-encoder (10 vs 20: same quality, half the cost; README)


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
    sure: bool = True  # False in the "not sure" band: the answer is the closest passage found
    confidence_kind: str = "cosine"  # "cosine" (top dense similarity) or "rerank" (cross-encoder score)
    translated_query: str = ""  # the question in the other language, when translation is on


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        rerank: bool = False,
        translate: bool = False,
        min_confidence: float = MIN_CONFIDENCE,
        sure_confidence: float = SURE_CONFIDENCE,
        min_score: float = MIN_SCORE,
        sure_score: float = SURE_SCORE,
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
        self.translator = None
        if translate:
            from .translate import MarianTranslator

            self.translator = MarianTranslator()
        self.min_confidence, self.sure_confidence = min_confidence, sure_confidence
        self.min_score, self.sure_score = min_score, sure_score
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

    def alt_queries(self, question: str) -> tuple[str, ...]:
        """The question in the other language (empty when translation is off or adds nothing)."""
        if not self.translator:
            return ()
        t = self.translator(question)
        return (t,) if t and t.strip().lower() != question.strip().lower() else ()

    def retrieve(
        self,
        question: str,
        k: int = 5,
        index: HybridIndex | None = None,
        alt_queries: tuple[str, ...] | None = None,
    ):
        """Hybrid search with the question (and its translation), fused, then optionally re-ranked."""
        target = self.index if index is None else index
        alts = self.alt_queries(question) if alt_queries is None else alt_queries
        n = max(k, RERANK_POOL) if self.reranker else k
        lists = [target.hybrid_search(q, k=n) for q in (question, *alts)]
        hits = lists[0] if len(lists) == 1 else fuse_hits(lists, n)
        if self.reranker:
            hits = self.reranker.rerank(question, hits, k, alt_queries=alts)
        return hits[:k]

    def ask(self, question: str, k: int = 5, index: HybridIndex | None = None) -> Answer:
        alts = self.alt_queries(question)
        hits = self.retrieve(question, k, index, alts)
        if self.reranker:
            kind, floor, sure = "rerank", self.min_score, self.sure_score
            confidence = hits[0].score if hits else floor - 1.0
        else:
            kind, floor, sure = "cosine", self.min_confidence, self.sure_confidence
            confidence = max((h.dense for h in hits), default=0.0)
        extra = {"confidence_kind": kind, "translated_query": alts[0] if alts else ""}
        if not hits or confidence < floor:
            return Answer(
                question,
                NOT_FOUND[detect_language(question)],
                False,
                confidence,
                self.generator.name,
                [],
                [h.chunk.id for h in hits],
                sure=False,
                **extra,
            )
        gen = self.generator.generate(question, hits, alt_queries=alts)
        found = not gen.text.strip().startswith(tuple(NOT_FOUND.values()))
        cites = [
            Citation(n, hits[n - 1].chunk.id, hits[n - 1].chunk.doc, hits[n - 1].chunk.page, hits[n - 1].chunk.text)
            for n in gen.used
        ]
        return Answer(
            question,
            gen.text,
            found,
            confidence,
            self.generator.name,
            cites,
            [h.chunk.id for h in hits],
            sure=found and confidence >= sure,
            **extra,
        )
