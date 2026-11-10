"""CLI: python -m rag ask "question" [--docs data/corpus] [--rerank] [--translate] [--llm extractive|gemini|openai]"""

from __future__ import annotations

import argparse

from .generate import get_generator
from .pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer one question about a folder of documents")
    ask.add_argument("question")
    ask.add_argument("--docs", default="data/corpus")
    ask.add_argument("--rerank", action="store_true")
    ask.add_argument("--translate", action="store_true", help="also search with the question translated (ar<->en)")
    ask.add_argument("--llm", default=None)
    ask.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    p = RAGPipeline(rerank=args.rerank, translate=args.translate)
    if args.llm:
        p.generator = get_generator(p.embedder, args.llm)
    p.ingest_directory(args.docs)
    a = p.ask(args.question, args.k)
    if a.found and not a.sure:
        print("(not sure: this is the closest passage in the documents)")
    print(a.answer)
    print(f"\n{a.confidence_kind} score={a.confidence:.3f} generator={a.generator}")
    if a.translated_query:
        print(f"also searched as: {a.translated_query}")
    for c in a.citations:
        print(f"[{c.n}] {c.chunk_id} (page {c.page})")


if __name__ == "__main__":
    main()
