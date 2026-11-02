
from dataclasses import dataclass

from app.ingestion.pdf_loader import PageText


@dataclass
class TextChunk:
    """A chunk of document text with source metadata."""

    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    document_id: str
    filename: str


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    document_id: str = "document",
    filename: str = "document.pdf",
) -> list[TextChunk]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Chunking is performed independently for each page so that
    every chunk retains its original page number.

    Args:
        pages: Extracted PDF pages.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters.
        document_id: Identifier for the source document.

    Returns:
        A list of TextChunk objects.

    Raises:
        ValueError: If chunking parameters are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    chunks = []

    step_size = chunk_size - chunk_overlap

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = TextChunk(
                    chunk_id=(
                        f"{document_id}:"
                        f"page-{page.page_number}:"
                        f"chunk-{chunk_index}"
                    ),
                    text=chunk_text,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    filename=filename,
                )

                chunks.append(chunk)

            start += step_size
            chunk_index += 1

    return chunks
