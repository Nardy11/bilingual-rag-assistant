"""HTML rendering of an answer for the Gradio UI.

Answers and passages can come from uploaded documents, so every piece of text is HTML-escaped; only
the fixed markup and the fixed ``dir`` values ("rtl"/"ltr") are trusted.
"""

from __future__ import annotations

from html import escape

from .normalize import detect_language
from .pipeline import Answer

NOT_SURE = {
    "en": "Not sure this answers it. The closest passage in the documents says:",
    "ar": "لست متأكداً أن هذا يجيب عن سؤالك. أقرب ما وجدته في المستندات:",
}


def _dir(text: str) -> str:
    return "rtl" if detect_language(text) == "ar" else "ltr"


def render_answer(a: Answer) -> str:
    note = ""
    if a.found and not a.sure:
        ql = detect_language(a.question)
        note = f'<p dir="{_dir(NOT_SURE[ql])}" style="opacity:.8"><em>{escape(NOT_SURE[ql])}</em></p>'
    via = f" · searched as: {escape(a.translated_query)}" if a.translated_query else ""
    return (
        note + f'<div dir="{_dir(a.answer)}" style="font-size:1.1em">{escape(a.answer)}</div>'
        f'<p style="opacity:.7" dir="ltr">{escape(a.confidence_kind)} score {a.confidence:.3f}'
        f" · generator: {escape(a.generator)}{via}</p>"
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
