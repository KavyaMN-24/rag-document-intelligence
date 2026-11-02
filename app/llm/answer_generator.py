from app.llm.base import LLMProvider
from app.retrieval.retriever import Retriever


class AnswerGenerator:
    """Generate document-grounded answers with source citations."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
    ):
        self.retriever = retriever
        self.llm = llm

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> dict:
        """Answer a question using retrieved document passages."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.retriever.retrieve(
            question=question,
            top_k=top_k,
            document_id=document_id,
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the indexed documents."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for result in results:
            metadata = result["metadata"]
            page_number = metadata["page_number"]

            context_parts.append(
                f"[Source: Page {page_number}]\n"
                f"{result['text']}"
            )

            sources.append(
                {
                    "chunk_id": result["chunk_id"],
                    "page_number": page_number,
                    "similarity": result["similarity"],
                    "text": result["text"],
                }
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a document question-answering assistant.

Answer questions using ONLY the provided document context.

Rules:
1. Do not invent facts that are not in the context.
2. If the context does not contain the answer, say so.
3. Cite supporting pages using [Page X].
4. Be clear and concise.
5. Distinguish facts from uncertainty.
"""

        user_prompt = f"""
Document context:

{context}

User question:

{question}

Provide an answer with page citations.
"""

        generated_answer = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return {
            "question": question,
            "answer": generated_answer,
            "sources": sources,
        }