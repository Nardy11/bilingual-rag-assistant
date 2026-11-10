"""Retrieval, confidence and answer evaluation. All numbers in the README come from this script.

Question sets:
  eval/questions.jsonl          30 questions (15 Arabic, 15 English, 8 cross-lingual), written carefully
  eval/natural_questions.jsonl  16 short, conversational cross-language questions ("هل عندنا إجازة الأمومة؟")
  eval/unanswerable.jsonl       24 questions the documents don't answer (off-topic or missing policies)

    python eval/run_eval.py              # BM25, dense, hybrid, + rerank, + translate + rerank
    python eval/run_eval.py --no-models  # skip the re-ranker and translation models

A retrieved chunk counts as gold when it comes from the gold document and contains the gold
evidence span (both compared after ``normalize_for_match``). Writes eval/results.json and prints
markdown tables.
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
from rag.pipeline import RERANK_POOL, RAGPipeline  # noqa: E402

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


def bands(values: list[float], floor: float, sure: float) -> dict:
    return {
        "sure": sum(v >= sure for v in values),
        "not_sure": sum(floor <= v < sure for v in values),
        "refused": sum(v < floor for v in values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", "--no-rerank", action="store_true", help="skip re-ranker and translation")
    args = ap.parse_args()

    qs, nat, nas = load("questions.jsonl"), load("natural_questions.jsonl"), load("unanswerable.jsonl")
    t0 = time.time()
    pipe = RAGPipeline()
    from rag.generate import ExtractiveGenerator

    pipe.generator = ExtractiveGenerator(pipe.embedder)
    n_chunks = pipe.ingest_directory(ROOT / "data" / "corpus")
    print(f"indexed {n_chunks} chunks in {time.time() - t0:.1f}s")

    # every question's evidence must sit inside one chunk, or the metric is meaningless
    for q in qs + nat:
        assert any(is_gold(c, q) for c in pipe.index.chunks), f"{q['id']}: evidence not inside a chunk"

    idx = pipe.index
    methods = {
        "BM25": lambda q: idx.bm25_search(q, DEPTH),
        "Dense (e5-small)": lambda q: idx.dense_search(q, DEPTH),
        "Hybrid (RRF)": lambda q: idx.hybrid_search(q, DEPTH),
    }
    reranker = translator = None
    if not args.no_models:
        from rag.rerank import CrossEncoderReranker
        from rag.translate import MarianTranslator

        reranker, translator = CrossEncoderReranker(), MarianTranslator()
        methods["Hybrid + rerank"] = lambda q: reranker.rerank(q, idx.hybrid_search(q, RERANK_POOL), DEPTH)

        def translate_rerank(q: str):
            pipe.reranker, pipe.translator = reranker, translator
            try:
                return pipe.retrieve(q, DEPTH)
            finally:
                pipe.reranker = pipe.translator = None

        methods["Hybrid + translate + rerank"] = translate_rerank

    results: dict = {"n_questions": len(qs), "n_natural": len(nat), "n_unanswerable": len(nas), "n_chunks": n_chunks}
    results["retrieval"] = {}
    for name, search in methods.items():
        t = time.time()
        ranks = {}
        for q in qs + nat:
            hits = search(q["question"])
            ranks[q["id"]] = next((i for i, h in enumerate(hits, 1) if is_gold(h.chunk, q)), None)
        ms = (time.time() - t) / len(qs + nat) * 1000
        groups = {
            "all": qs,
            "en": [q for q in qs if q["lang"] == "en"],
            "ar": [q for q in qs if q["lang"] == "ar"],
            "cross-lingual": [q for q in qs if q.get("cross")],
            "natural cross-language": nat,
        }
        results["retrieval"][name] = {g: metrics([ranks[q["id"]] for q in sub]) for g, sub in groups.items()} | {
            "ms_per_query": ms,
            "ranks": ranks,
        }

    # end-to-end with the extractive generator (no LLM), k=5
    def run_generation() -> dict:
        out = {}
        for set_name, questions in (("questions", qs), ("natural", nat)):
            rows = []
            for q in questions:
                a = pipe.ask(q["question"], 5)
                rows.append(
                    {
                        "id": q["id"],
                        "found": a.found,
                        "sure": a.sure,
                        "confidence": a.confidence,
                        "answer_has_fact": normalize_for_match(q["answer"]) in normalize_for_match(a.answer),
                        "cites_gold": any(is_gold(c, q) for c in a.citations),
                        "answer": a.answer,
                    }
                )
            n = len(rows)
            out[set_name] = {
                "answered": sum(r["found"] for r in rows) / n,
                "answered_sure": sum(r["found"] and r["sure"] for r in rows) / n,
                "answer_has_fact": sum(r["answer_has_fact"] for r in rows) / n,
                "cites_gold": sum(r["cites_gold"] for r in rows) / n,
                "rows": rows,
            }
        na = []
        for q in nas:
            a = pipe.ask(q["question"], 5)
            na.append({"id": q["id"], "found": a.found, "sure": a.sure, "confidence": a.confidence})
        out["unanswerable"] = {
            "refused": sum(not r["found"] for r in na),
            "not_sure": sum(r["found"] and not r["sure"] for r in na),
            "answered_sure": sum(r["found"] and r["sure"] for r in na),
            "n": len(na),
            "rows": na,
        }
        kind = "rerank" if pipe.reranker else "cosine"
        if pipe.reranker:
            floor, sure = pipe.min_score, pipe.sure_score
        else:
            floor, sure = pipe.min_confidence, pipe.sure_confidence
        conf_ans = [r["confidence"] for s in ("questions", "natural") for r in out[s]["rows"]]
        conf_na = [r["confidence"] for r in na]
        out["confidence"] = {
            "kind": kind,
            "floor": floor,
            "sure": sure,
            "answerable": bands(conf_ans, floor, sure) | {"min": min(conf_ans)},
            "unanswerable": bands(conf_na, floor, sure) | {"max": max(conf_na)},
        }
        return out

    results["generation"] = {"Extractive, hybrid": run_generation()}
    if reranker:
        pipe.reranker = reranker
        results["generation"]["Extractive, hybrid + rerank"] = run_generation()
        pipe.translator = translator
        results["generation"]["Extractive, translate + rerank"] = run_generation()
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(
        "\n| Retriever | Recall@1 | Recall@5 | MRR@10 | R@5 EN | R@5 AR | R@5 cross-lingual |"
        " Natural R@1 | Natural R@5 | ms/query |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results["retrieval"].items():
        a, n = r["all"], r["natural cross-language"]
        print(
            f"| {name} | {a['recall@1']:.2f} | {a['recall@5']:.2f} | {a['mrr@10']:.3f} | "
            f"{r['en']['recall@5']:.2f} | {r['ar']['recall@5']:.2f} | {r['cross-lingual']['recall@5']:.2f} | "
            f"{n['recall@1']:.2f} | {n['recall@5']:.2f} | {r['ms_per_query']:.0f} |"
        )
    print(
        "\n| Pipeline | Set | Answered | Answered, sure | Answer has the gold fact | Cites a gold passage |"
        "\n|---|---|---|---|---|---|"
    )
    for name, g in results["generation"].items():
        for s in ("questions", "natural"):
            r = g[s]
            print(
                f"| {name} | {s} | {r['answered']:.2f} | {r['answered_sure']:.2f} | "
                f"{r['answer_has_fact']:.2f} | {r['cites_gold']:.2f} |"
            )
    print("\n| Pipeline | Signal | Floor / sure | Answerable (46): sure · not sure · refused | Unanswerable (24) |")
    print("|---|---|---|---|---|")
    for name, g in results["generation"].items():
        c = g["confidence"]
        a, u = c["answerable"], c["unanswerable"]
        print(
            f"| {name} | {c['kind']} | {c['floor']} / {c['sure']} | {a['sure']} · {a['not_sure']} · {a['refused']} | "
            f"{u['sure']} · {u['not_sure']} · {u['refused']} |"
        )


if __name__ == "__main__":
    main()
