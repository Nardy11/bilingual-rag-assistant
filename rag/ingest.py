"""Load PDF / TXT / MD files into pages of clean text, with limits for untrusted uploads."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .normalize import clean_text

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}

# Upload limits (API and Space). Local files loaded with load_directory() are trusted and only the
# page cap applies.
MAX_FILES = int(os.getenv("RAG_MAX_FILES", "5"))
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("RAG_MAX_PDF_PAGES", "200"))


class UploadTooLarge(ValueError):
    """An upload exceeds MAX_FILES, MAX_FILE_BYTES or MAX_PDF_PAGES."""


# PDF text comes with hard line wraps; a newline followed by a lowercase letter, digit or bracket
# (and not preceded by sentence punctuation) is a wrap, not a paragraph or heading break.
_WRAP = re.compile(r"(?<![.!?:؟])\n(?=[a-z0-9(])")


def unwrap_lines(text: str) -> str:
    return _WRAP.sub(" ", text)


@dataclass(frozen=True)
class Page:
    doc: str  # file name, used in citations
    page: int  # 1-based; text files are a single page
    text: str


def load_file(path: str | Path, name: str | None = None, max_pages: int | None = None) -> list[Page]:
    path = Path(path)
    name = name or path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED)}")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        max_pages = MAX_PDF_PAGES if max_pages is None else max_pages
        if len(reader.pages) > max_pages:
            raise UploadTooLarge(f"{name}: {len(reader.pages)} pages, the limit is {max_pages}")
        texts = [unwrap_lines(clean_text(p.extract_text() or "")) for p in reader.pages]
        pages = [Page(name, i + 1, t) for i, t in enumerate(texts)]
        return [p for p in pages if p.text]
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Page(name, 1, clean_text(text))] if text.strip() else []


def check_upload_size(path: str | Path) -> None:
    size = Path(path).stat().st_size
    if size > MAX_FILE_BYTES:
        raise UploadTooLarge(f"{Path(path).name}: {size} bytes, the limit is {MAX_FILE_BYTES}")


def load_bytes(data: bytes, filename: str) -> list[Page]:
    """Load an uploaded file (used by the API). The caller enforces MAX_FILE_BYTES while reading."""
    import tempfile

    if len(data) > MAX_FILE_BYTES:
        raise UploadTooLarge(f"{filename}: larger than {MAX_FILE_BYTES} bytes")
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"upload{suffix}"
        p.write_bytes(data)
        return load_file(p, name=Path(filename).name)


def load_directory(path: str | Path) -> list[Page]:
    pages: list[Page] = []
    for f in sorted(Path(path).iterdir()):
        if f.suffix.lower() in SUPPORTED:
            pages.extend(load_file(f))
    return pages
