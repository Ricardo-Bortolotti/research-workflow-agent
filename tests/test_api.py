"""Tests for the FastAPI analysis endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.routes import get_analysis_service
from api.service import AnalysisService
from api.storage import AnalysisStorage
from app.main import create_app


@pytest.fixture
def storage(tmp_path):
    return AnalysisStorage(data_dir=tmp_path)


@pytest.fixture
def service(storage):
    return AnalysisService(storage=storage)


@pytest.fixture
def client(service):
    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf(client):
    files = {"file": ("sample.pdf", b"%PDF-1.4 test content", "application/pdf")}
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert data["document_id"]


def test_upload_rejects_non_pdf(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400


@patch("api.service.run_workflow")
@patch("api.service.ChunkEmbedder")
@patch("api.service.split_documents")
@patch("api.service.load_pdf")
def test_analyze_and_get_results(
    mock_load_pdf,
    mock_split_documents,
    mock_embedder_cls,
    mock_run_workflow,
    client,
    service,
):
    upload = service.upload_pdf("sample.pdf", b"%PDF-1.4 test")
    mock_load_pdf.return_value = [MagicMock(page_content="page", metadata={"page": 0})]
    mock_split_documents.return_value = [MagicMock(page_content="chunk", metadata={"page": 0})]

    embedder = MagicMock()
    embedder.embed_chunks.return_value = [[0.1, 0.2]]
    embedder.embedding_dimension = 2
    embedder.embed_query.return_value = [0.1, 0.2]
    mock_embedder_cls.return_value = embedder

    mock_run_workflow.return_value = {
        "summary": {
            "executive_summary": "Summary text.",
            "key_insights": ["Insight 1"],
        },
        "concepts": {
            "concepts": [
                {
                    "concept": "Attention",
                    "definition": "A weighting mechanism.",
                    "relevance": "Core idea.",
                }
            ]
        },
        "quiz": {
            "questions": [
                {
                    "question": "What is attention?",
                    "answer": "A mechanism.",
                    "difficulty": "easy",
                }
            ]
        },
        "flashcards": {"flashcards": [{"front": "Attention", "back": "Mechanism."}]},
        "mindmap": {
            "title": "Main Topic",
            "nodes": [{"label": "Attention", "children": []}],
            "text": "Main Topic\n└── Attention",
        },
    }

    analyze_response = client.post(
        "/analyze",
        json={
            "document_id": upload.document_id,
            "question": "What is this document about?",
            "top_k": 3,
        },
    )
    assert analyze_response.status_code == 200
    analyze_data = analyze_response.json()
    assert analyze_data["status"] == "completed"

    results_response = client.get(f"/results/{analyze_data['analysis_id']}")
    assert results_response.status_code == 200
    results = results_response.json()
    assert results["summary"]["executive_summary"] == "Summary text."
    assert results["concepts"][0]["concept"] == "Attention"
    assert results["quiz"][0]["difficulty"] == "easy"
    assert results["flashcards"][0]["front"] == "Attention"
    assert "Attention" in results["mindmap"]["text"]


def test_get_results_not_found(client):
    response = client.get("/results/nonexistent")
    assert response.status_code == 404
