
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore


@dataclass
class IndexingResult:
    """Summary of a successfully indexed PDF document."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentIndexer:
    """Orchestrate PDF extraction, chunking, embedding and storage."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _generate_document_id(file_path: str | Path) -> str:
        """Generate a stable ID based on the PDF content."""
        content = Path(file_path).read_bytes()
        return hashlib.sha256(content).hexdigest()

    def index_pdf(self, file_path: str | Path) -> IndexingResult:
        """Extract, chunk, embed and store a PDF document."""

        path = Path(file_path)

        # Step 1: Extract PDF text.
        pages = extract_pdf(path)

        # Step 2: Generate a stable document identifier.
        document_id = self._generate_document_id(path)

        # Step 3: Split extracted text into chunks.
        chunks = chunk_pages(
            pages=pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            document_id=document_id,
            filename=path.name,
        )

        # Step 4: Generate embeddings and store chunks.
        if chunks:
            texts = [chunk.text for chunk in chunks]

            embeddings = self.embedding_engine.embed_texts(texts)

            self.vector_store.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
            )

        # Step 5: Return an indexing summary.
        return IndexingResult(
            document_id=document_id,
            filename=path.name,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
