from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore


class Retriever:
    """Retrieve relevant document chunks for a user question."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store

    
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Find the most relevant chunks for a question."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_engine.embed_query(question)

        where = (
            {"document_id": document_id}
            if document_id is not None
            else None
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        for result in results:
            result["similarity"] = 1 - result["distance"]

        return results
