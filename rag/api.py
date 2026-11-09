"""FastAPI service.

    uvicorn rag.api:app --port 8000

POST /ingest  (multipart files: PDF/TXT/MD)  -> chunks added
POST /ask     {"question": "...", "k": 5}     -> answer with citations
GET  /health

On start-up the sample corpus in ``RAG_CORPUS_DIR`` (default ``data/corpus``) is indexed; set it to
an empty string to start with an empty index.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .ingest import SUPPORTED, load_bytes
from .pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)


def create_app(pipeline: RAGPipeline | None = None, corpus_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Bilingual RAG Assistant", version="1.0.0")
    state: dict = {"pipeline": pipeline}

    def get() -> RAGPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline(rerank=os.getenv("RAG_RERANK", "1") == "1")
            corpus = os.getenv("RAG_CORPUS_DIR", "data/corpus") if corpus_dir is None else corpus_dir
            if corpus and Path(corpus).is_dir():
                state["pipeline"].ingest_directory(corpus)
        return state["pipeline"]

    @app.get("/health")
    def health():
        p = get()
        return {"status": "ok", "chunks": len(p.index), "generator": p.generator.name}

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)):
        p, added = get(), {}
        for f in files:
            if Path(f.filename or "").suffix.lower() not in SUPPORTED:
                raise HTTPException(415, f"{f.filename}: unsupported type, use PDF, TXT or MD")
            added[f.filename] = p.add_pages(load_bytes(await f.read(), f.filename))
        return {"added": added, "total_chunks": len(p.index)}

    @app.post("/ask")
    def ask(req: AskRequest):
        return asdict(get().ask(req.question, req.k))

    return app


app = create_app()
