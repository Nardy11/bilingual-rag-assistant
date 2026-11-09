"""Sentence-aware chunking by token count, with overlap.

Sentences are packed into a chunk until it reaches ``max_tokens``; the next chunk starts with the
last sentences of the previous one, up to ``overlap_tokens``. Token counts come from the embedding
model's own tokenizer when available (so a chunk never exceeds what the model sees), else from a
whitespace/punctuation count. A sentence longer than ``max_tokens`` is split on words.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .ingest import Page
from .normalize import detect_language

_SENT_END = re.compile(r"(?<=[.!?؟。])\s+|\n+")
_HEADING = re.compile(r"^#+\s*")

TokenCounter = Callable[[str], int]


def simple_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc>#p<page>-c<n>", stable for citations
    doc: str
    page: int
    text: str
    lang: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    parts = (_HEADING.sub("", s).strip(" >-*\t") for s in _SENT_END.split(text))
    return [p for p in parts if p]


def _split_long(sentence: str, max_tokens: int, count: TokenCounter) -> list[str]:
    words, out, cur = sentence.split(), [], []
    for w in words:
        if cur and count(" ".join(cur + [w])) > max_tokens:
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_pages(
    pages: list[Page],
    max_tokens: int = 160,
    overlap_tokens: int = 40,
    count: TokenCounter = simple_token_count,
) -> list[Chunk]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    chunks: list[Chunk] = []
    for page in pages:
        sents: list[tuple[str, int]] = []
        for s in split_sentences(page.text):
            pieces = [s] if count(s) <= max_tokens else _split_long(s, max_tokens, count)
            sents.extend((p, count(p)) for p in pieces)
        n, i = 0, 0
        while i < len(sents):
            cur, size, j = [], 0, i
            while j < len(sents) and (not cur or size + sents[j][1] <= max_tokens):
                cur.append(sents[j][0])
                size += sents[j][1]
                j += 1
            text = "\n".join(cur)
            chunks.append(Chunk(f"{page.doc}#p{page.page}-c{n}", page.doc, page.page, text, detect_language(text)))
            n += 1
            if j >= len(sents):
                break
            # step back over trailing sentences to build the overlap, but always move forward
            back, k = 0, j
            while k - 1 > i and back + sents[k - 1][1] <= overlap_tokens:
                k -= 1
                back += sents[k][1]
            i = k
    return chunks
