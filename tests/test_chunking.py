from rag.chunking import chunk_pages, simple_token_count, split_sentences
from rag.ingest import Page


def test_split_sentences_handles_arabic_question_mark_and_headings():
    assert split_sentences("## Title\nOne. Two? ثلاثة؟ أربعة.") == ["Title", "One.", "Two?", "ثلاثة؟", "أربعة."]


def test_chunks_respect_max_tokens_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_pages([Page("d.txt", 1, text)], max_tokens=30, overlap_tokens=10)
    assert len(chunks) > 3
    assert all(simple_token_count(c.text) <= 30 for c in chunks)
    # consecutive chunks share their boundary sentence (overlap)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split("\n")[-1] == b.text.split("\n")[0]
    # every sentence is covered
    joined = "\n".join(c.text for c in chunks)
    assert all(f"Sentence number {i} is here." in joined for i in range(40))


def test_chunk_ids_pages_and_language():
    pages = [Page("a.pdf", 1, "Hello world."), Page("a.pdf", 2, "مرحبا بالعالم.")]
    chunks = chunk_pages(pages)
    assert [c.id for c in chunks] == ["a.pdf#p1-c0", "a.pdf#p2-c0"]
    assert [c.lang for c in chunks] == ["en", "ar"]


def test_long_sentence_is_split():
    chunks = chunk_pages([Page("x.txt", 1, "word " * 100)], max_tokens=20, overlap_tokens=5)
    assert all(simple_token_count(c.text) <= 20 for c in chunks)


def test_pdf_line_wraps_are_joined():
    from rag.ingest import unwrap_lines

    assert unwrap_lines("Directors may fly business class on flights\nlonger than six hours.\nHotels\nThe limit") == (
        "Directors may fly business class on flights longer than six hours.\nHotels\nThe limit"
    )
