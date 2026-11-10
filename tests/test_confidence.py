"""Confidence bands and query translation, with no model download (hashing embedder, fakes)."""

from rag.chunking import Chunk
from rag.embeddings import HashingEmbedder
from rag.generate import ExtractiveGenerator
from rag.ingest import Page
from rag.pipeline import RAGPipeline
from rag.render import render_answer
from rag.retrieval import Hit, fuse_hits

HANDBOOK = Page(
    "handbook.en.md",
    1,
    "Family leave. Female employees are entitled to 70 days of maternity leave at full pay. "
    "Fathers receive 5 working days of paternity leave.",
)
FAQ = Page("faq.ar.md", 1, "الفلاتر. يجب تغيير فلتر الكربون كل ستة أشهر للحفاظ على جودة المياه.")
QUESTION = "هل عندنا إجازة الأمومة؟"


def make(min_confidence=0.1, sure_confidence=0.9, translator=None):
    emb = HashingEmbedder()
    p = RAGPipeline(
        embedder=emb,
        generator=ExtractiveGenerator(emb),
        min_confidence=min_confidence,
        sure_confidence=sure_confidence,
    )
    p.translator = translator
    p.add_pages([HANDBOOK, FAQ])
    return p


def test_fuse_hits_merges_by_chunk_and_keeps_best_cosine():
    a, b = Chunk("a", "d", 1, "x", "en"), Chunk("b", "d", 1, "y", "en")
    fused = fuse_hits([[Hit(a, 1, 0.5), Hit(b, 1, 0.9)], [Hit(b, 1, 0.7)]], 5)
    assert [h.chunk.id for h in fused] == ["b", "a"]  # b appears in both lists
    assert fused[0].dense == 0.9


def test_arabic_question_finds_english_passage_through_translation():
    # the hashing embedder shares no words across languages, so without translation nothing matches
    assert not make().ask(QUESTION).found
    p = make(translator=lambda q: "Do we have maternity leave?")
    a = p.ask(QUESTION)
    assert a.found and "70 days of maternity leave" in a.answer
    assert a.citations[0].doc == "handbook.en.md"
    assert a.translated_query == "Do we have maternity leave?"


def test_three_bands():
    t = lambda q: "Do we have maternity leave?"  # noqa: E731
    sure = make(sure_confidence=0.1, translator=t).ask(QUESTION)
    assert sure.found and sure.sure
    unsure = make(sure_confidence=0.99, translator=t).ask(QUESTION)
    assert unsure.found and not unsure.sure  # answered, but flagged as the closest passage
    assert "Not sure" in render_answer(unsure) or "لست متأكداً" in render_answer(unsure)
    refused = make(min_confidence=0.99, translator=t).ask(QUESTION)
    assert not refused.found and not refused.sure


def test_looping_translation_is_rejected():
    from rag.translate import plausible

    assert plausible("Do we have maternity leave?", "هل لدينا إجازة أمومة؟")
    assert not plausible("who won the world cup", "الدنيا : " * 30)
    assert not plausible("hi", "a b c d e f a b c d e f a b c d e f")


def test_translation_identical_to_question_is_ignored():
    p = make(translator=lambda q: q)
    assert p.alt_queries("maternity leave") == ()
