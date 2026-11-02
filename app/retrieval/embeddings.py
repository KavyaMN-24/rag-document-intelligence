
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Generate semantic embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple text strings into embeddings."""

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Input texts cannot contain empty strings.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a single user question into an embedding."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.embed_texts([query])[0]
