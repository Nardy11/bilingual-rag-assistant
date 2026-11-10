"""FastAPI service.

    uvicorn rag.api:app --port 8000

POST /ask     {"question": "...", "k": 5}     -> answer with citations
POST /ingest  (multipart files: PDF/TXT/MD)  -> chunks added; admin only, see below
GET  /health

On start-up the sample corpus in ``RAG_CORPUS_DIR`` (default ``data/corpus``) is indexed; set it to
an empty string to start with an empty index.

``/ingest`` writes to the index every user queries, so it is disabled unless ``RAG_ADMIN_KEY`` is
set, and then requires that key in the ``X-API-Key`` header. Uploads are capped at
``RAG_MAX_FILES`` files, ``RAG_MAX_FILE_BYTES`` bytes per file and ``RAG_MAX_PDF_PAGES`` PDF pages.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import ingest as limits
from .ingest import SUPPORTED, UploadTooLarge, load_bytes
from .pipeline import RAGPipeline

_READ_CHUNK = 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)


def _check_admin(key: str | None) -> None:
    expected = os.getenv("RAG_ADMIN_KEY", "")
    if not expected:
        raise HTTPException(403, "ingestion is disabled: set RAG_ADMIN_KEY on the server to enable it")
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(403, "missing or invalid X-API-Key")


async def _read_capped(f: UploadFile) -> bytes:
    """Read an upload in chunks and stop as soon as it passes the size cap."""
    buf = bytearray()
    while chunk := await f.read(_READ_CHUNK):
        buf.extend(chunk)
        if len(buf) > limits.MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename}: larger than {limits.MAX_FILE_BYTES} bytes")
    return bytes(buf)


def create_app(pipeline: RAGPipeline | None = None, corpus_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Bilingual RAG Assistant", version="1.0.0")
    state: dict = {"pipeline": pipeline}

    def get() -> RAGPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline(
                rerank=os.getenv("RAG_RERANK", "1") == "1", translate=os.getenv("RAG_TRANSLATE", "1") == "1"
            )
            corpus = os.getenv("RAG_CORPUS_DIR", "data/corpus") if corpus_dir is None else corpus_dir
            if corpus and Path(corpus).is_dir():
                state["pipeline"].ingest_directory(corpus)
        return state["pipeline"]

    @app.get("/health")
    def health():
        p = get()
        return {"status": "ok", "chunks": len(p.index), "generator": p.generator.name}

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...), x_api_key: str | None = Header(None)):
        _check_admin(x_api_key)
        if len(files) > limits.MAX_FILES:
            raise HTTPException(413, f"at most {limits.MAX_FILES} files per request")
        for f in files:
            if Path(f.filename or "").suffix.lower() not in SUPPORTED:
                raise HTTPException(415, f"{f.filename}: unsupported type, use PDF, TXT or MD")
        # parse everything first, so a rejected file leaves the index untouched
        parsed = {}
        for f in files:
            data = await _read_capped(f)
            try:
                parsed[f.filename] = load_bytes(data, f.filename)
            except UploadTooLarge as e:
                raise HTTPException(413, str(e)) from e
        p = get()
        added = {name: p.add_pages(pages) for name, pages in parsed.items()}
        return {"added": added, "total_chunks": len(p.index)}

    @app.post("/ask")
    def ask(req: AskRequest):
        return asdict(get().ask(req.question, req.k))

    return app


app = create_app()
