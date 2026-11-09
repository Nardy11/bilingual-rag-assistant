"""Answer generation behind one interface: Gemini, any OpenAI-compatible endpoint, or extractive.

``get_generator()`` picks from ``RAG_LLM`` (gemini | openai | extractive); when unset it uses Gemini
if ``GEMINI_API_KEY`` is set, an OpenAI-compatible endpoint if ``OPENAI_API_KEY`` or
``OPENAI_BASE_URL`` is set, and otherwise the extractive generator, which needs no key.
Every generator returns text whose citations are ``[n]`` markers pointing at the numbered sources.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from .chunking import split_sentences
from .embeddings import Embedder
from .normalize import detect_language, tokenize
from .retrieval import Hit

NOT_FOUND = {
    "en": "Not found in the documents.",
    "ar": "لم أجد الإجابة في المستندات.",
}

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. "
    "Answer in the same language as the question (Arabic or English), in at most three sentences. "
    "After each claim, cite the source number in square brackets, e.g. [1] or [2]. "
    "If the sources do not contain the answer, reply exactly: {not_found}"
)


@dataclass
class Generation:
    text: str
    used: list[int] = field(default_factory=list)  # 1-based indices of the sources cited


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Generation: ...


def is_answer_candidate(sentence: str) -> bool:
    """Headings (fewer than 4 words) and questions (FAQ titles) are never quoted as answers."""
    s = sentence.strip()
    return len(tokenize(s, remove_stopwords=False)) >= 4 and not s.endswith(("?", "؟"))


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.doc}, page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    lang = detect_language(question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(not_found=NOT_FOUND[lang])},
        {"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"},
    ]


def _cited(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


class OpenAICompatibleGenerator:
    """Any /chat/completions endpoint: OpenAI, Azure-compatible gateways, Ollama, vLLM, Groq..."""

    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            timeout=60,
            json={"model": self.model, "temperature": 0, "messages": build_messages(question, hits)},
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return Generation(text, _cited(text, len(hits)))


class GeminiGenerator:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        system, user = build_messages(question, hits)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        r = httpx.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            timeout=60,
            json={
                "systemInstruction": {"parts": [{"text": system["content"]}]},
                "contents": [{"role": "user", "parts": [{"text": user["content"]}]}],
                "generationConfig": {"temperature": 0},
            },
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return Generation(text, _cited(text, len(hits)))


class ExtractiveGenerator:
    """No-LLM fallback: returns the sentences from the retrieved chunks that best answer the query.

    Sentence score = 0.75 * cosine(query, sentence) + 0.25 * share of query terms in the sentence.
    """

    name = "extractive"

    def __init__(self, embedder: Embedder, max_sentences: int = 2, top_chunks: int = 3):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.top_chunks = top_chunks

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        cands: list[tuple[int, str]] = []
        seen: set[str] = set()  # overlapping chunks repeat sentences; keep the higher-ranked copy
        for i, h in enumerate(hits[: self.top_chunks], start=1):
            for s in split_sentences(h.chunk.text):
                if is_answer_candidate(s) and s not in seen:
                    seen.add(s)
                    cands.append((i, s))
        if not cands:
            return Generation(NOT_FOUND[detect_language(question)])
        q = self.embedder.embed_queries([question])[0]
        sv = self.embedder.embed_passages([s for _, s in cands])
        qtok = set(tokenize(question))
        lex = np.array([len(qtok & set(tokenize(s))) / max(len(qtok), 1) for _, s in cands])
        score = 0.75 * (sv @ q) + 0.25 * lex
        best = sorted(np.argsort(-score)[: self.max_sentences])  # keep document order
        best = [b for b in best if score[b] >= score.max() - 0.08] or [int(np.argmax(score))]
        text = " ".join(f"{cands[b][1]} [{cands[b][0]}]" for b in best)
        return Generation(text, sorted({cands[b][0] for b in best}))


def get_generator(embedder: Embedder, name: str | None = None) -> Generator:
    name = (name or os.getenv("RAG_LLM") or "").lower()
    if not name:
        if os.getenv("GEMINI_API_KEY"):
            name = "gemini"
        elif os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
            name = "openai"
        else:
            name = "extractive"
    if name == "gemini":
        return GeminiGenerator()
    if name == "openai":
        return OpenAICompatibleGenerator()
    if name == "extractive":
        return ExtractiveGenerator(embedder)
    raise ValueError(f"unknown generator {name!r}")
