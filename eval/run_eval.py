"""Retrieval + answer evaluation on eval/questions.jsonl (15 Arabic, 15 English, 8 cross-lingual).

    python eval/run_eval.py            # BM25, dense, hybrid, hybrid+rerank, extractive answers
    python eval/run_eval.py --no-rerank

A retrieved chunk counts as gold when it comes from the gold document and contains the gold
evidence span (both compared after ``normalize_for_match``). Writes eval/results.json and prints
markdown tables. All numbers in the README come from this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.normalize import normalize_for_match  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

KS = (1, 3, 5)
DEPTH = 10


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines() if line]


def is_gold(chunk, q) -> bool:
    return chunk.doc == q["doc"] and normalize_for_match(q["evidence"]) in normalize_for_match(chunk.text)


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr@10"] = sum(1 / r for r in ranks if r) / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    qs, nas = load("questions.jsonl"), load("unanswerable.jsonl")
    t0 = time.time()
    pipe = RAGPipeline()
    from rag.generate import ExtractiveGenerator

    pipe.generator = ExtractiveGenerator(pipe.embedder)
    n_chunks = pipe.ingest_directory(ROOT / "data" / "corpus")
    print(f"indexed {n_chunks} chunks in {time.time() - t0:.1f}s")

    # every question's evidence must sit inside one chunk, or the metric is meaningless
    for q in qs:
        assert any(is_gold(c, q) for c in pipe.index.chunks), f"{q['id']}: evidence not inside a chunk"

    idx = pipe.index
    methods = {
        "BM25": lambda q: idx.bm25_search(q, DEPTH),
        "Dense (e5-small)": lambda q: idx.dense_search(q, DEPTH),
        "Hybrid (RRF)": lambda q: idx.hybrid_search(q, DEPTH),
    }
    if not args.no_rerank:
        from rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        methods["Hybrid + rerank"] = lambda q: reranker.rerank(q, idx.hybrid_search(q, 20), DEPTH)

    results: dict = {"n_questions": len(qs), "n_chunks": n_chunks, "retrieval": {}}
    for name, search in methods.items():
        t = time.time()
        ranks = {}
        for q in qs:
            hits = search(q["question"])
            ranks[q["id"]] = next((i for i, h in enumerate(hits, 1) if is_gold(h.chunk, q)), None)
        ms = (time.time() - t) / len(qs) * 1000
        groups = {
            "all": qs,
            "en": [q for q in qs if q["lang"] == "en"],
            "ar": [q for q in qs if q["lang"] == "ar"],
            "cross-lingual": [q for q in qs if q.get("cross")],
        }
        results["retrieval"][name] = {g: metrics([ranks[q["id"]] for q in sub]) for g, sub in groups.items()} | {
            "ms_per_query": ms,
            "ranks": ranks,
        }

    # confidence calibration: top cosine for answerable vs unanswerable questions
    def top_cos(q: str) -> float:
        return max(h.dense for h in idx.hybrid_search(q, 5))

    results["confidence"] = {
        "answerable": {q["id"]: top_cos(q["question"]) for q in qs},
        "unanswerable": {q["id"]: top_cos(q["question"]) for q in nas},
        "threshold": pipe.min_confidence,
    }

    # end-to-end with the extractive generator (no LLM), k=5, without and with re-ranking
    def run_generation(label: str) -> dict:
        rows = []
        for q in qs:
            a = pipe.ask(q["question"], 5)
            rows.append(
                {
                    "id": q["id"],
                    "found": a.found,
                    "answer_has_fact": normalize_for_match(q["answer"]) in normalize_for_match(a.answer),
                    "cites_gold": any(is_gold(c, q) for c in a.citations),
                    "answer": a.answer,
                }
            )
        na = [{"id": q["id"], "abstained": not pipe.ask(q["question"], 5).found} for q in nas]
        n = len(rows)
        return {
            "answered": sum(r["found"] for r in rows) / n,
            "answer_has_fact": sum(r["answer_has_fact"] for r in rows) / n,
            "cites_gold": sum(r["cites_gold"] for r in rows) / n,
            "abstained_on_unanswerable": f"{sum(r['abstained'] for r in na)}/{len(na)}",
            "rows": rows,
            "unanswerable_rows": na,
        }

    results["generation"] = {"Extractive, hybrid": run_generation("hybrid")}
    if not args.no_rerank:
        pipe.reranker = reranker
        results["generation"]["Extractive, hybrid + rerank"] = run_generation("rerank")
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@10 | R@5 EN | R@5 AR | R@5 cross-lingual | ms/query |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in results["retrieval"].items():
        a = r["all"]
        print(
            f"| {name} | {a['recall@1']:.2f} | {a['recall@3']:.2f} | {a['recall@5']:.2f} | {a['mrr@10']:.3f} | "
            f"{r['en']['recall@5']:.2f} | {r['ar']['recall@5']:.2f} | {r['cross-lingual']['recall@5']:.2f} | "
            f"{r['ms_per_query']:.0f} |"
        )
    print("\n| Pipeline | Answered | Answer has the gold fact | Cites a gold passage | Abstained on unanswerable |")
    print("|---|---|---|---|---|")
    for name, g in results["generation"].items():
        print(
            f"| {name} | {g['answered']:.2f} | {g['answer_has_fact']:.2f} | {g['cites_gold']:.2f} | "
            f"{g['abstained_on_unanswerable']} |"
        )
    c = results["confidence"]
    print(
        f"top cosine answerable min={min(c['answerable'].values()):.3f} "
        f"unanswerable max={max(c['unanswerable'].values()):.3f} "
        f"unanswerable={ {k: round(v, 3) for k, v in c['unanswerable'].items()} }"
    )


if __name__ == "__main__":
    main()
