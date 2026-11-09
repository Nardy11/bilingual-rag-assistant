import pytest
from fastapi.testclient import TestClient

from rag import ingest as limits
from rag.api import create_app

KEY = {"X-API-Key": "test-admin-key"}
NOTE = b"Parking. Free parking is available for all employees in basement B2."


@pytest.fixture
def client(pipeline, monkeypatch):
    monkeypatch.setenv("RAG_ADMIN_KEY", "test-admin-key")
    return TestClient(create_app(pipeline, corpus_dir=""))


def test_ask_and_ingest(client, pipeline):
    assert client.get("/health").json()["chunks"] == len(pipeline.index)

    r = client.post("/ask", json={"question": "كم مدة الضمان على الجهاز؟"})
    body = r.json()
    assert r.status_code == 200 and body["found"]
    assert "سنتين" in body["answer"]
    assert body["citations"][0]["doc"] == "faq.ar.md"

    r = client.post("/ingest", files={"files": ("notes.md", NOTE, "text/markdown")}, headers=KEY)
    assert r.status_code == 200 and r.json()["added"]["notes.md"] == 1
    body = client.post("/ask", json={"question": "Where is the free parking for employees?"}).json()
    assert "basement B2" in body["answer"]


def test_ingest_requires_the_admin_key(client, pipeline):
    before = len(pipeline.index)
    files = {"files": ("notes.md", NOTE, "text/markdown")}
    assert client.post("/ingest", files=files).status_code == 403
    assert client.post("/ingest", files=files, headers={"X-API-Key": "wrong"}).status_code == 403
    assert len(pipeline.index) == before


def test_ingest_disabled_without_admin_key_env(pipeline, monkeypatch):
    monkeypatch.delenv("RAG_ADMIN_KEY", raising=False)
    client = TestClient(create_app(pipeline, corpus_dir=""))
    r = client.post("/ingest", files={"files": ("notes.md", NOTE, "text/markdown")}, headers=KEY)
    assert r.status_code == 403 and "disabled" in r.json()["detail"]


def test_upload_caps(client, pipeline, monkeypatch):
    monkeypatch.setattr(limits, "MAX_FILE_BYTES", 1000)
    big = {"files": ("big.txt", b"a " * 1000, "text/plain")}
    assert client.post("/ingest", files=big, headers=KEY).status_code == 413

    many = [("files", (f"n{i}.md", NOTE, "text/markdown")) for i in range(limits.MAX_FILES + 1)]
    assert client.post("/ingest", files=many, headers=KEY).status_code == 413

    monkeypatch.setattr(limits, "MAX_PDF_PAGES", 1)
    pdf = open("data/corpus/travel-policy.en.pdf", "rb").read()  # 2 pages
    monkeypatch.setattr(limits, "MAX_FILE_BYTES", 10_000_000)
    r = client.post("/ingest", files={"files": ("t.pdf", pdf, "application/pdf")}, headers=KEY)
    assert r.status_code == 413 and "pages" in r.json()["detail"]


def test_rejects_bad_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    exe = {"files": ("x.exe", b"MZ", "application/octet-stream")}
    assert client.post("/ingest", files=exe, headers=KEY).status_code == 415
