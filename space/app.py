"""Gradio UI for Hugging Face Spaces (also runs locally: python space/app.py).

The sample corpus is the shared index. Documents a visitor uploads go into a private copy of that
index kept in the visitor's session (gr.State), so one visitor can never change another's answers.
All text shown is HTML-escaped (rag/render.py) and uploads are capped like the API's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "rag").is_dir() else HERE.parent  # Space layout or repo layout
sys.path.insert(0, str(ROOT))

from rag import ingest as limits  # noqa: E402
from rag.ingest import SUPPORTED, UploadTooLarge, check_upload_size, load_file  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.render import render_answer, render_citations  # noqa: E402

PIPE = RAGPipeline(rerank=os.getenv("RAG_RERANK", "1") == "1")
PIPE.ingest_directory(ROOT / "data" / "corpus")

EXAMPLES = [
    "How many days a week can I work from home?",
    "What is the hotel limit per night in the GCC?",
    "متى يجب تغيير فلتر الكربون؟",
    "كم مدة إجازة الأمومة؟",
]


def answer(question: str, session_index):
    if not question.strip():
        return "", ""
    a = PIPE.ask(question[:1000], index=session_index)
    return render_answer(a), render_citations(a)


def add_files(files, session_index):
    files = files or []
    if len(files) > limits.MAX_FILES:
        return f"Upload at most {limits.MAX_FILES} files at a time.", session_index
    paths = [f.name if hasattr(f, "name") else f for f in files]
    try:
        pages = []
        for p in paths:
            if Path(p).suffix.lower() not in SUPPORTED:
                return f"{Path(p).name}: use PDF, TXT or MD.", session_index
            check_upload_size(p)
            pages.append(load_file(p))
    except UploadTooLarge as e:
        return f"Rejected: {e}", session_index
    index = session_index or PIPE.fork_index()
    added = sum(PIPE.add_pages(pg, index=index) for pg in pages)
    return f"Added {added} passages to your session ({len(index)} in total). Other visitors don't see them.", index


with gr.Blocks(title="Bilingual RAG Assistant") as demo:
    session_index = gr.State(None)  # None = the shared sample corpus
    gr.Markdown("# Bilingual RAG Assistant\nAsk in **Arabic or English**. Answers cite the passages they come from.")
    q = gr.Textbox(label="Question / السؤال", placeholder="How many days of annual leave do I get?", max_length=1000)
    btn = gr.Button("Ask", variant="primary")
    out = gr.HTML()
    src = gr.HTML()
    gr.Examples(EXAMPLES, inputs=q)
    with gr.Accordion("Add your own documents (PDF, TXT, MD; private to your session)", open=False):
        up = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"])
        status = gr.Markdown()
        up.upload(add_files, [up, session_index], [status, session_index])
    btn.click(answer, [q, session_index], [out, src])
    q.submit(answer, [q, session_index], [out, src])

if __name__ == "__main__":
    demo.launch(max_file_size=limits.MAX_FILE_BYTES)
