import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


class FakeEmbeddingEngine:
    """Provide deterministic embeddings for testing."""

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def retriever(tmp_path):
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="retriever_tests",
    )

    chunks = [
        TextChunk(
            chunk_id="doc1:page-1:chunk-0",
            text="Artificial intelligence enables machines to perform tasks.",
            page_number=1,
            chunk_index=0,
            document_id="test-document",
            filename="test.pdf",
        ),
        TextChunk(
            chunk_id="doc1:page-2:chunk-0",
            text="Retrieval augmented generation uses external knowledge.",
            page_number=2,
            chunk_index=0,
            document_id="test-document",
            filename="test.pdf",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    return Retriever(
        embedding_engine=FakeEmbeddingEngine(),
        vector_store=store,
    )


def test_retrieve_returns_relevant_chunks(retriever):
    results = retriever.retrieve("What is artificial intelligence?")

    assert len(results) == 2
    assert "Artificial intelligence" in results[0]["text"]


def test_retrieve_returns_similarity_score(retriever):
    results = retriever.retrieve("What is artificial intelligence?")

    assert "similarity" in results[0]
    assert 0 <= results[0]["similarity"] <= 1


def test_retrieve_preserves_page_metadata(retriever):
    results = retriever.retrieve("What is artificial intelligence?")

    assert results[0]["metadata"]["page_number"] == 1


def test_retrieve_respects_top_k(retriever):
    results = retriever.retrieve(
        "What is artificial intelligence?",
        top_k=1,
    )

    assert len(results) == 1


def test_empty_question_raises_error(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_invalid_top_k_raises_error(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve(
            "What is artificial intelligence?",
            top_k=0,
        )