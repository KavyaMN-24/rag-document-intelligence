
import json
import re
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from app.llm.answer_generator import AnswerGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
SAMPLE_PDF = PROJECT_ROOT / "sample_rag_document.pdf"


def extract_cited_pages(answer: str) -> set[int]:
    """Extract page numbers explicitly cited in the generated answer."""
    matches = re.findall(
        r"[\[【]\s*Page\s+(\d+)\s*[\]】]",
        answer,
        re.IGNORECASE,
    )
    return {int(page) for page in matches}


def contains_expected_keywords(
    answer: str,
    expected_keywords: list[str],
) -> bool:
    """Check whether all expected keyword phrases appear in the answer."""
    normalized_answer = answer.lower()
    return all(
        keyword.lower() in normalized_answer
        for keyword in expected_keywords
    )


def is_refusal(answer: str) -> bool:
    """Identify common refusal phrases for unanswerable questions."""
    refusal_phrases = [
        "does not include",
        "doesn't include",
        "not contain",
        "could not find",
        "cannot find",
        "don't know",
        "do not know",
        "not specified",
        "not provide",
        "unable to answer",
        "does not state",
        "doesn't state",
        "document does not specify",
        "document doesn't specify",
        "not mentioned",
    ]

    normalized_answer = answer.lower()
    return any(
        phrase in normalized_answer
        for phrase in refusal_phrases
    )


def evaluate_answers():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    pipeline = RAGPipeline()

    indexing_result = pipeline.index_document(SAMPLE_PDF)
    document_id = indexing_result["document_id"]

    filtered_generator = AnswerGenerator(
        retriever=pipeline.retriever,
        llm=pipeline.answer_generator.llm,
    )

    print(f"Indexed: {indexing_result['filename']}")
    print(f"Evaluating {len(questions)} questions\n")

    results = []

    for item in questions:
        question = item["question"]
        expected_page = item["expected_page"]
        answerable = item["answerable"]
        expected_keywords = item.get("expected_keywords", [])

        result = filtered_generator.answer(
            question=question,
            top_k=5,
            document_id=document_id,
        )

        answer = result["answer"]
        sources = result["sources"]

        cited_pages = extract_cited_pages(answer)

        if answerable:
            answerability_correct = bool(answer.strip()) and not is_refusal(answer)

            answer_relevant = contains_expected_keywords(
                answer,
                expected_keywords,
            )

            citation_correct = (
                expected_page in cited_pages
                and expected_page in {
                    source["page_number"] for source in sources
                }
            )
        else:
            answerability_correct = is_refusal(answer)
            answer_relevant = None
            citation_correct = len(cited_pages) == 0

        results.append(
            {
                "question": question,
                "answerable": answerable,
                "answerability_correct": answerability_correct,
                "answer_relevant": answer_relevant,
                "citation_correct": citation_correct,
            }
        )

        retrieved_pages = sorted({
            source["page_number"] for source in sources
        })

        print(f"Question: {question}")
        print(f"Generated answer: {answer}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Pages cited in answer: {sorted(cited_pages)}")
        print(f"Answerability correct: {answerability_correct}")
        print(f"Answer relevance: {answer_relevant}")
        print(f"Citation correct: {citation_correct}")
        print("-" * 60)

    total = len(results)

    answerability_score = (
        sum(r["answerability_correct"] for r in results) / total
    )

    answerable_results = [
        r for r in results if r["answerable"]
    ]

    unanswerable_results = [
        r for r in results if not r["answerable"]
    ]

    relevance_score = (
        sum(r["answer_relevant"] for r in answerable_results)
        / len(answerable_results)
    )

    citation_score = (
        sum(r["citation_correct"] for r in answerable_results)
        / len(answerable_results)
    )

    abstention_score = (
        sum(r["answerability_correct"] for r in unanswerable_results)
        / len(unanswerable_results)
    )

    print("\nFINAL EVALUATION")
    print(f"Answerability accuracy: {answerability_score:.2%}")
    print(f"Answer relevance (keyword match): {relevance_score:.2%}")
    print(f"Citation correctness: {citation_score:.2%}")
    print(f"Unanswerable question abstention: {abstention_score:.2%}")


if __name__ == "__main__":
    evaluate_answers()
