"""Build the demo's code tour: docs/learn/manifest.json and the copies of the Python sources.

GitHub Pages serves only docs/, so the Python files the tour shows are copied to docs/learn/src/.
The line-by-line notes live in docs/learn/notes/<path with / as __>.json; this script checks that
each one covers every line of the file as it is now, so a code change without a notes update fails.

    python scripts/build_code_tour.py          # copy sources, write the manifest, check the notes
    python scripts/build_code_tour.py --check  # only check (CI): copies up to date, notes complete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LEARN = DOCS / "learn"
REPO = "https://github.com/Nardy11/bilingual-rag-assistant/blob/main/"

# The tour's table of contents, in the order a learner should read it.
GROUPS = [
    (
        "Prepare the documents",
        ["rag/ingest.py", "rag/normalize.py", "rag/chunking.py", "rag/embeddings.py", "rag/retrieval.py"],
    ),
    ("Answer a question", ["rag/translate.py", "rag/rerank.py", "rag/generate.py", "rag/pipeline.py"]),
    ("Serve it", ["rag/api.py", "rag/render.py", "rag/__main__.py", "rag/__init__.py"]),
    (
        "The browser version",
        ["scripts/export_web_index.py", "docs/js/text.js", "docs/js/ingest.js", "docs/js/engine.js", "docs/js/llm.js"],
    ),
]

PATHS = {
    "document": [
        ("rag/ingest.py", "read the file"),
        ("rag/normalize.py", "clean the text"),
        ("rag/chunking.py", "split into passages"),
        ("rag/embeddings.py", "embed each passage"),
        ("rag/retrieval.py", "index vectors and words"),
    ],
    "question": [
        ("rag/normalize.py", "normalise, tokenise"),
        ("rag/translate.py", "translate"),
        ("rag/embeddings.py", "embed"),
        ("rag/retrieval.py", "hybrid search + RRF"),
        ("rag/rerank.py", "re-rank"),
        ("rag/pipeline.py", "confidence bands"),
        ("rag/generate.py", "answer with citations"),
    ],
}


def notes_path(path: str) -> Path:
    return LEARN / "notes" / (path.replace("/", "__") + ".json")


def served_src(path: str) -> str:
    """URL of the source relative to docs/index.html."""
    return path[len("docs/") :] if path.startswith("docs/") else f"learn/src/{path}"


def check_notes(path: str, n_lines: int) -> list[str]:
    p = notes_path(path)
    if not p.exists():
        return [f"{path}: no notes ({p.relative_to(ROOT)})"]
    data = json.loads(p.read_text(encoding="utf-8"))
    errors, expect = [], 1
    for b in data.get("blocks", []):
        if b["from"] != expect:
            errors.append(f"{path}: block starts at {b['from']}, expected {expect}")
        if b["to"] < b["from"] or not b.get("note", "").strip():
            errors.append(f"{path}: bad block {b['from']}-{b['to']}")
        expect = b["to"] + 1
    if expect - 1 != n_lines:
        errors.append(f"{path}: notes cover lines 1-{expect - 1}, the file has {n_lines}")
    text = p.read_text(encoding="utf-8")
    if "—" in text or "–" in text:
        errors.append(f"{path}: notes contain an em or en dash")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    files, errors = {}, []
    for _, group in GROUPS:
        for path in group:
            src = (ROOT / path).read_text(encoding="utf-8")
            n = len(src.splitlines())
            if not path.startswith("docs/"):
                copy = LEARN / "src" / path
                if args.check:
                    if not copy.exists() or copy.read_text(encoding="utf-8") != src:
                        errors.append(f"{path}: docs/learn/src copy is stale; run scripts/build_code_tour.py")
                else:
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    copy.write_text(src, encoding="utf-8", newline="\n")
            errors += check_notes(path, n)
            files[path] = {
                "src": served_src(path),
                "notes": f"learn/notes/{notes_path(path).name}",
                "lang": "javascript" if path.endswith(".js") else "python",
                "lines": n,
                "github": REPO + path,
            }
    manifest = {
        "groups": [{"title": t, "files": fs} for t, fs in GROUPS],
        "order": [f for _, fs in GROUPS for f in fs],
        "files": files,
        "paths": {k: [{"file": f, "label": label} for f, label in v] for k, v in PATHS.items()},
    }
    out = LEARN / "manifest.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            errors.append("docs/learn/manifest.json is stale; run scripts/build_code_tour.py")
    else:
        out.write_text(text, encoding="utf-8", newline="\n")
    for e in errors:
        print(e)
    print(f"{len(files)} files, {sum(f['lines'] for f in files.values())} lines, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
