from pathlib import Path

from app.ingestion.indexer import DocumentIndexer
from app.llm.answer_generator import AnswerGenerator
from app.llm.groq_provider import GroqProvider
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


class RAGPipeline:
    """End-to-end document question-answering pipeline."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
    ):
        self.embedding_engine = EmbeddingEngine()

        self.vector_store = VectorStore(
            persist_directory=persist_directory,
        )

        self.indexer = DocumentIndexer(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
        )

        self.retriever = Retriever(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
        )

        self.llm = GroqProvider()

        self.answer_generator = AnswerGenerator(
            retriever=self.retriever,
            llm=self.llm,
        )

    def index_document(self, file_path: str | Path) -> dict:
        """Index a PDF document."""

        result = self.indexer.index_pdf(file_path)

        return {
            "document_id": result.document_id,
            "filename": result.filename,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        }

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """Answer a question using indexed documents."""

        return self.answer_generator.answer(
            question=question,
            top_k=top_k,
        )