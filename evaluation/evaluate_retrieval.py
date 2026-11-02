
import json
from pathlib import Path

from app.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"


def evaluate_retrieval():
    """Measure retrieval hit rate at different K values."""

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline()
    sample_pdf = PROJECT_ROOT / "sample_rag_document.pdf"

    indexing_result = pipeline.index_document(sample_pdf)

    document_id = indexing_result["document_id"]

    print(f"Evaluating document: {indexing_result['filename']}")
    print(f"Document ID: {document_id}")

    results = []

    for item in questions:
        question = item["question"]
        expected_page = item["expected_page"]

        retrieved_chunks = pipeline.retriever.retrieve(
            question,
            top_k=5,
            document_id=document_id,
        )
        
        retrieved_pages = [
            chunk["metadata"]["page_number"]
            for chunk in retrieved_chunks
        ]
        precision_at_1 = (
            sum(page == expected_page for page in retrieved_pages[:1])
            / len(retrieved_pages[:1])
        )

        precision_at_3 = (
            sum(page == expected_page for page in retrieved_pages[:3])
            / len(retrieved_pages[:3])
        )

        precision_at_5 = (
            sum(page == expected_page for page in retrieved_pages[:5])
            / len(retrieved_pages[:5])
        )

        result = {
            "question": question,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "hit_at_1": expected_page in retrieved_pages[:1],
            "hit_at_3": expected_page in retrieved_pages[:3],
            "hit_at_5": expected_page in retrieved_pages[:5],
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
        }

        results.append(result)

        print(f"\nQuestion: {question}")
        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Hit@1: {result['hit_at_1']}")
        print(f"Hit@3: {result['hit_at_3']}")
        print(f"Hit@5: {result['hit_at_5']}")
        total = len(results)

        for metric in (
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
        ):
            score = sum(result[metric] for result in results) / total
            print(f"{metric.upper()}: {score:.2%}")

    total = len(results)

    for metric in ("hit_at_1", "hit_at_3", "hit_at_5"):
        score = sum(result[metric] for result in results) / total
        print(f"\n{metric.upper()}: {score:.2%}")

    return results


if __name__ == "__main__":
    evaluate_retrieval()
