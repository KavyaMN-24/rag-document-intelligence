
from pathlib import Path

import chromadb

from app.ingestion.chunker import TextChunk


class VectorStore:
    """Persistent vector storage and semantic retrieval using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks and their corresponding embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return

        if any(not embedding for embedding in embeddings):
            raise ValueError("Embeddings cannot be empty.")

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                
                }
                for chunk in chunks
            ],
        )

    
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant document chunks."""

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection_count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
            where=where,
        )

        retrieved_chunks = []

        for index, chunk_id in enumerate(results["ids"][0]):
            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        return retrieved_chunks
