
import pymupdf
import pytest

from app.ingestion.indexer import DocumentIndexer
from app.retrieval.vector_store import VectorStore


class FakeEmbeddingEngine:
    """Deterministic embedding engine for testing."""

    def embed_texts(self, texts):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a two-page sample PDF."""

    path = tmp_path / "sample.pdf"

    document = pymupdf.open()

    page1 = document.new_page()
    page1.insert_text(
        (72, 72),
        "Artificial intelligence is transforming businesses."
    )

    page2 = document.new_page()
    page2.insert_text(
        (72, 72),
        "Retrieval augmented generation uses external documents."
    )

    document.save(path)
    document.close()

    return path


@pytest.fixture
def indexer(tmp_path):
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="indexer_tests",
    )

    return DocumentIndexer(
        embedding_engine=FakeEmbeddingEngine(),
        vector_store=store,
        chunk_size=500,
        chunk_overlap=100,
    )


def test_index_pdf_returns_summary(indexer, sample_pdf):
    result = indexer.index_pdf(sample_pdf)

    assert result.filename == "sample.pdf"
    assert result.page_count == 2
    assert result.chunk_count == 2
    assert len(result.document_id) == 64


def test_index_pdf_stores_chunks(indexer, sample_pdf):
    indexer.index_pdf(sample_pdf)

    results = indexer.vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=5,
    )

    assert len(results) == 2


def test_index_pdf_preserves_page_numbers(indexer, sample_pdf):
    indexer.index_pdf(sample_pdf)

    results = indexer.vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=5,
    )

    page_numbers = {
        result["metadata"]["page_number"]
        for result in results
    }

    assert page_numbers == {1, 2}


def test_same_pdf_generates_same_document_id(
    indexer,
    sample_pdf,
):
    first = indexer.index_pdf(sample_pdf)
    second = indexer.index_pdf(sample_pdf)

    assert first.document_id == second.document_id


def test_missing_pdf_raises_error(indexer, tmp_path):
    with pytest.raises(FileNotFoundError):
        indexer.index_pdf(tmp_path / "missing.pdf")
