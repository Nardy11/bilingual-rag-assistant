from fastapi.testclient import TestClient

from rag.api import create_app


def test_ask_and_ingest(pipeline):
    client = TestClient(create_app(pipeline, corpus_dir=""))
    assert client.get("/health").json()["chunks"] == len(pipeline.index)

    r = client.post("/ask", json={"question": "كم مدة الضمان على الجهاز؟"})
    body = r.json()
    assert r.status_code == 200 and body["found"]
    assert "سنتين" in body["answer"]
    assert body["citations"][0]["doc"] == "faq.ar.md"

    note = b"Parking. Free parking is available for all employees in basement B2."
    r = client.post("/ingest", files={"files": ("notes.md", note, "text/markdown")})
    assert r.status_code == 200 and r.json()["added"]["notes.md"] == 1
    body = client.post("/ask", json={"question": "Where is the free parking for employees?"}).json()
    assert "basement B2" in body["answer"]


def test_rejects_bad_input(pipeline):
    client = TestClient(create_app(pipeline, corpus_dir=""))
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ingest", files={"files": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
