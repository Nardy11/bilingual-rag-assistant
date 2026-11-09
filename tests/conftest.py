import pytest

from rag.embeddings import HashingEmbedder
from rag.generate import ExtractiveGenerator
from rag.ingest import Page
from rag.pipeline import RAGPipeline

DOCS = [
    Page(
        "leave.en.md",
        1,
        "Annual leave. Employees get 30 calendar days of paid annual leave per year. "
        "Unused days can be carried over up to a maximum of 10 days.",
    ),
    Page("leave.en.md", 2, "Sick leave. A medical certificate is required for absences longer than two days."),
    Page(
        "faq.ar.md",
        1,
        "الضمان. يأتي الجهاز بضمان لمدة سنتين من تاريخ التركيب. لا يشمل الضمان الفلاتر لأنها قطع استهلاكية.",
    ),
    Page("faq.ar.md", 2, "الفلاتر. يجب تغيير فلتر الكربون كل ستة أشهر للحفاظ على جودة المياه."),
]


@pytest.fixture
def pipeline():
    """A tiny bilingual index with the download-free hashing embedder (no model, no API key)."""
    emb = HashingEmbedder()
    p = RAGPipeline(embedder=emb, generator=ExtractiveGenerator(emb), min_confidence=0.1)
    p.add_pages(DOCS)
    return p
