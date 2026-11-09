"""Embedding models behind one small interface.

Default: ``intfloat/multilingual-e5-small`` (118M params, 384-dim, ~100 languages incl. Arabic).
It is the strongest multilingual retriever in its size class, runs on CPU, and the same weights
exist as a quantized ONNX export (``Xenova/multilingual-e5-small``) for the browser demo, so the
precomputed index and the in-browser query embeddings live in the same vector space.
E5 expects ``query: `` / ``passage: `` prefixes; this class adds them.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

DEFAULT_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")


class Embedder(Protocol):
    dim: int

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if self._e5:
            texts = [prefix + t for t in texts]
        vecs = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder: no download, used by tests and CI."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        from .normalize import tokenize

        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_passages = embed_queries
