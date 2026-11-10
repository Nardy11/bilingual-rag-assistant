"""Precompute the browser demo's index: docs/index.json.

Chunks, their BM25 tokens (Python tokenizer, so the browser does not re-tokenise documents), chunk
embeddings and the embeddings of every candidate answer sentence, all computed with the SAME model
the browser loads (intfloat/multilingual-e5-small; the browser uses its ONNX export
Xenova/multilingual-e5-small). Vectors are float32, base64-encoded, L2-normalised.

    python scripts/export_web_index.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import split_sentences  # noqa: E402
from rag.generate import is_answer_candidate  # noqa: E402
from rag.normalize import _AR_PREFIXES, STOPWORDS_AR, STOPWORDS_EN, tokenize  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.rerank import DEFAULT_RERANKER  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def main() -> None:
    p = RAGPipeline()
    p.ingest_directory(ROOT / "data" / "corpus")
    chunks = p.index.chunks
    sents = []  # (chunk index, sentence)
    for ci, c in enumerate(chunks):
        for s in split_sentences(c.text):
            if is_answer_candidate(s):
                sents.append((ci, s))
    svec = p.embedder.embed_passages([s for _, s in sents])
    out = {
        "model": "Xenova/multilingual-e5-small",
        "python_model": p.embedder.model_name,
        "dim": p.embedder.dim,
        # confidence bands (rag/pipeline.py): cosine in fast mode, cross-encoder score in full mode
        "min_confidence": p.min_confidence,
        "sure_confidence": p.sure_confidence,
        "min_score": p.min_score,
        "sure_score": p.sure_score,
        # full mode: the same cross-encoder as rag/rerank.py (its own quantized ONNX file) and the
        # ONNX exports of the rag/translate.py models
        "reranker": {"model": DEFAULT_RERANKER, "file": "model_quint8_avx2"},
        "translators": {"ar": "Xenova/opus-mt-ar-en", "en": "Xenova/opus-mt-en-ar"},
        # the browser tokenises the query with exactly these lists (rag/normalize.py)
        "stopwords_en": sorted(STOPWORDS_EN),
        "stopwords_ar": sorted(STOPWORDS_AR),
        "ar_prefixes": list(_AR_PREFIXES),
        "chunks": [
            {"id": c.id, "doc": c.doc, "page": c.page, "lang": c.lang, "text": c.text, "tokens": tokenize(c.text)}
            for c in chunks
        ],
        "chunk_vectors": b64(p.index.vectors),
        "sentences": [{"chunk": ci, "text": s} for ci, s in sents],
        "sentence_vectors": b64(svec),
    }
    path = ROOT / "docs" / "index.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB): {len(chunks)} chunks, {len(sents)} sentences")


if __name__ == "__main__":
    main()
