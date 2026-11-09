"""Gradio UI for Hugging Face Spaces (also runs locally: python space/app.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "rag").is_dir() else HERE.parent  # Space layout or repo layout
sys.path.insert(0, str(ROOT))

from rag.ingest import load_file  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

PIPE = RAGPipeline(rerank=os.getenv("RAG_RERANK", "1") == "1")
PIPE.ingest_directory(ROOT / "data" / "corpus")

EXAMPLES = [
    "How many days a week can I work from home?",
    "What is the hotel limit per night in the GCC?",
    "متى يجب تغيير فلتر الكربون؟",
    "كم مدة إجازة الأمومة؟",
]


def answer(question: str):
    if not question.strip():
        return "", ""
    a = PIPE.ask(question)
    rtl = any("؀" <= ch <= "ۿ" for ch in a.answer)
    text = f'<div dir="{"rtl" if rtl else "ltr"}" style="font-size:1.1em">{a.answer}</div>'
    text += f"<p style='opacity:.7'>confidence {a.confidence:.3f} · generator: {a.generator}</p>"
    sources = "\n\n".join(f"**[{c.n}] {c.doc}, page {c.page}** (`{c.chunk_id}`)\n\n> {c.text}" for c in a.citations)
    return text, sources or "_No passage cited._"


def add_files(files):
    added = sum(PIPE.add_pages(load_file(f.name if hasattr(f, "name") else f)) for f in files or [])
    return f"Added {added} passages. The index now has {len(PIPE.index)} passages."


with gr.Blocks(title="Bilingual RAG Assistant") as demo:
    gr.Markdown("# Bilingual RAG Assistant\nAsk in **Arabic or English**. Answers cite the passages they come from.")
    q = gr.Textbox(label="Question / السؤال", rtl=False, placeholder="How many days of annual leave do I get?")
    btn = gr.Button("Ask", variant="primary")
    out = gr.HTML()
    src = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=q)
    with gr.Accordion("Add your own documents (PDF, TXT, MD)", open=False):
        up = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"])
        status = gr.Markdown()
        up.upload(add_files, up, status)
    btn.click(answer, q, [out, src])
    q.submit(answer, q, [out, src])

if __name__ == "__main__":
    demo.launch()
