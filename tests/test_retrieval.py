import numpy as np

from rag.embeddings import HashingEmbedder
from rag.retrieval import HybridIndex, rrf


def test_rrf_rewards_agreement():
    fused = rrf([[0, 1, 2], [1, 0, 3]])
    assert fused[1] == fused[0] > fused[2] > 0
    assert fused[3] == fused[2]  # same rank in one list each


def test_bm25_finds_arabic_with_spelling_variants(pipeline):
    hits = pipeline.index.bm25_search("متى أغير فلتر الكربون؟", 3)
    assert hits[0].chunk.id == "faq.ar.md#p2-c0"


def test_hybrid_ranks_gold_first(pipeline):
    assert pipeline.index.hybrid_search("How many days of paid annual leave?", 3)[0].chunk.id == "leave.en.md#p1-c0"
    assert pipeline.index.hybrid_search("كم مدة الضمان على الجهاز؟", 3)[0].chunk.id == "faq.ar.md#p1-c0"


def test_dense_scores_are_cosines(pipeline):
    hits = pipeline.index.dense_search("medical certificate sick leave", 4)
    assert hits[0].chunk.id == "leave.en.md#p2-c0"
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)


def test_save_and_load_roundtrip(pipeline, tmp_path):
    pipeline.index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, HashingEmbedder())
    assert [c.id for c in loaded.chunks] == [c.id for c in pipeline.index.chunks]
    assert np.allclose(loaded.vectors, pipeline.index.vectors)


def test_ask_cites_and_abstains(pipeline):
    a = pipeline.ask("How many days of paid annual leave do employees get?")
    assert a.found and "30 calendar days" in a.answer
    assert a.citations and a.citations[0].chunk_id == "leave.en.md#p1-c0"
    none = pipeline.ask("quantum chromodynamics lattice gauge")
    assert not none.found and none.answer == "Not found in the documents."
    assert pipeline.ask("ما هو سعر سهم البورصة اليوم؟").answer == "لم أجد الإجابة في المستندات."
