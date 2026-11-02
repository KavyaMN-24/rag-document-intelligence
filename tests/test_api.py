
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.main as api_module
from app.api.main import create_app


class FakePipeline:
    """A lightweight pipeline for API tests."""

    def index_document(self, file_path):
        return {
            "document_id": "test-document-id",
            "filename": Path(file_path).name,
            "page_count": 2,
            "chunk_count": 4,
        }

    def ask(self, question, top_k=5):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return {
            "question": question,
            "answer": "ChromaDB is a vector database. [Page 2]",
            "sources": [
                {
                    "chunk_id": "test-chunk-1",
                    "page_number": 2,
                    "similarity": 0.85,
                    "text": "ChromaDB is a vector database.",
                }
            ],
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with a fake pipeline."""

    monkeypatch.setattr(
        api_module,
        "UPLOAD_DIRECTORY",
        tmp_path / "uploads",
    )

    application = create_app(pipeline=FakePipeline())

    with TestClient(application) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf_success(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"fake PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "sample.pdf"
    assert response.json()["page_count"] == 2


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"Some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]


def test_ask_question_success(client):
    response = client.post(
        "/questions/ask",
        json={
            "question": "What is ChromaDB?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert "ChromaDB" in response.json()["answer"]
    assert response.json()["sources"][0]["page_number"] == 2


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/questions/ask",
        json={
            "question": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 400


def test_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/questions/ask",
        json={
            "question": "What is RAG?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
