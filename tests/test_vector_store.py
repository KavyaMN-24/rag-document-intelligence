
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(
        persist_directory=tmp_path / "chroma_test",
        collection_name="test_documents",
    )


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="doc1:page-1:chunk-0",
            text="Python is a programming language.",
            page_number=1,
            chunk_index=0,
            document_id="test-document",
            filename="test.pdf",
        ),
        TextChunk(
            chunk_id="doc1:page-2:chunk-0",
            text="Machine learning uses data to train models.",
            page_number=2,
            chunk_index=0,
            document_id="test-document",
            filename="test.pdf",
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_add_chunks_and_retrieve(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == sample_chunks[0].text


def test_page_metadata_is_preserved(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
    )

    assert results[0]["metadata"]["page_number"] == 1


def test_empty_store_returns_empty_results(vector_store):
    results = vector_store.search([1.0, 0.0, 0.0])

    assert results == []


def test_mismatched_chunks_and_embeddings_raise_error(
    vector_store,
    sample_chunks,
):
    with pytest.raises(ValueError):
        vector_store.add_chunks(
            sample_chunks,
            [[1.0, 0.0, 0.0]],
        )


def test_empty_embedding_raises_error(
    vector_store,
    sample_chunks,
):
    with pytest.raises(ValueError):
        vector_store.add_chunks(
            sample_chunks,
            [[], [1.0, 0.0, 0.0]],
        )


def test_empty_query_raises_error(vector_store):
    with pytest.raises(ValueError):
        vector_store.search([])


def test_invalid_result_count_raises_error(vector_store):
    with pytest.raises(ValueError):
        vector_store.search(
            [1.0, 0.0, 0.0],
            n_results=0,
        )


def test_upsert_updates_existing_chunk(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    updated_chunk = TextChunk(
        chunk_id="doc1:page-1:chunk-0",
        text="Python is widely used in AI engineering.",
        page_number=1,
        chunk_index=0,
        document_id="test-document",
        filename="test.pdf",
    )

    vector_store.add_chunks(
        [updated_chunk],
        [[1.0, 0.0, 0.0]],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == updated_chunk.text
