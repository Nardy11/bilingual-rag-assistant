"""HTML rendering of an answer for the Gradio UI.

Answers and passages can come from uploaded documents, so every piece of text is HTML-escaped; only
the fixed markup and the fixed ``dir`` values ("rtl"/"ltr") are trusted.
"""

from __future__ import annotations

from html import escape

from .normalize import detect_language
from .pipeline import Answer


def _dir(text: str) -> str:
    return "rtl" if detect_language(text) == "ar" else "ltr"


def render_answer(a: Answer) -> str:
    return (
        f'<div dir="{_dir(a.answer)}" style="font-size:1.1em">{escape(a.answer)}</div>'
        f'<p style="opacity:.7" dir="ltr">confidence {a.confidence:.3f} · generator: {escape(a.generator)}</p>'
    )


def render_citations(a: Answer) -> str:
    if not a.citations:
        return "<p><em>No passage cited.</em></p>"
    items = "".join(
        f'<li><b dir="ltr">[{c.n}] {escape(c.doc)}, page {c.page}</b> <code>{escape(c.chunk_id)}</code>'
        f'<blockquote dir="{_dir(c.text)}">{escape(c.text)}</blockquote></li>'
        for c in a.citations
    )
    return f'<ol style="list-style:none;padding:0">{items}</ol>'
