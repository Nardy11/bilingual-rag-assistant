from rag.ingest import Page
from rag.pipeline import Answer, Citation
from rag.render import render_answer, render_citations

EVIL = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


def test_answer_and_citations_are_escaped():
    a = Answer("q", f"{EVIL} [1]", True, 0.9, "extractive", [Citation(1, f"{EVIL}#p1-c0", EVIL, 1, f"text {EVIL}")])
    html = render_answer(a) + render_citations(a)
    assert "<script" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html


def test_dir_attribute_is_fixed():
    a = Answer("q", 'x" onmouseover="alert(1)', True, 0.9, "extractive", [])
    html = render_answer(a)
    assert 'dir="ltr"' in html and 'onmouseover="' not in html
    assert 'dir="rtl"' in render_answer(Answer("q", "لم أجد الإجابة", False, 0.1, "extractive", []))


def test_session_uploads_do_not_touch_the_shared_index(pipeline):
    shared_before = len(pipeline.index)
    session = pipeline.fork_index()
    pipeline.add_pages([Page("notes.md", 1, "Parking. Free parking is in basement B2 for employees.")], index=session)
    assert len(pipeline.index) == shared_before
    assert len(session) == shared_before + 1
    assert "basement B2" in pipeline.ask("Where is the free parking for employees?", index=session).answer
    assert "basement B2" not in pipeline.ask("Where is the free parking for employees?").answer


def test_empty_session_index_never_writes_to_the_shared_index(pipeline):
    from rag.retrieval import HybridIndex

    shared_before = len(pipeline.index)
    empty = HybridIndex(pipeline.embedder)
    assert not empty  # falsy: the reason the pipeline checks `is None`
    pipeline.add_pages([Page("notes.md", 1, "Parking. Free parking is in basement B2 for employees.")], index=empty)
    assert len(pipeline.index) == shared_before
    assert len(empty) == 1
    assert pipeline.retrieve("free parking basement", 1, index=empty)[0].chunk.doc == "notes.md"
