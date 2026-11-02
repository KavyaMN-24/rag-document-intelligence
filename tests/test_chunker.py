
import pytest

from app.ingestion.pdf_loader import PageText
from app.ingestion.chunker import chunk_pages


def test_chunking_creates_multiple_chunks():
    pages = [
        PageText(
            page_number=1,
            text="A" * 1200,
        )
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=100,
        document_id="test-document",
        filename="test.pdf",
    )

    assert len(chunks) == 3


def test_chunk_overlap_is_preserved():
    text = "A" * 1000

    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=100,
        document_id="test-document",
        filename="test.pdf",
    )

    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_page_metadata_is_preserved():
    pages = [
        PageText(page_number=1, text="First page content."),
        PageText(page_number=2, text="Second page content."),
    ]

    chunks = chunk_pages(pages)

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_ids_are_unique():
    pages = [
        PageText(page_number=1, text="A" * 1200),
    ]

    chunks = chunk_pages(pages)

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_pages_are_skipped():
    pages = [
        PageText(page_number=1, text=""),
        PageText(page_number=2, text="   "),
        PageText(page_number=3, text="Valid content."),
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_pages(
            [],
            chunk_size=0,
        )


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_pages(
            [],
            chunk_size=100,
            chunk_overlap=100,
        )


def test_document_id_cannot_be_empty():
    with pytest.raises(ValueError):
        chunk_pages(
            [],
            document_id=" ",
        )
