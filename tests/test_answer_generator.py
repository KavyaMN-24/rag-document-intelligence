import pytest

from app.llm.answer_generator import AnswerGenerator


class FakeRetriever:
    """Simulate document retrieval."""

    def __init__(self, results=None):
        self.results = results or []
        self.last_question = None
        self.last_top_k = None

    def retrieve(self, question, top_k=5, document_id=None):
        self.last_question = question
        self.last_top_k = top_k
        self.last_document_id = document_id
        return self.results


class FakeLLM:
    """Simulate LLM answer generation."""

    def __init__(self, response="AI is transforming businesses. [Page 1]"):
        self.response = response
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt, system_prompt):
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response


@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "doc1:page-1:chunk-0",
            "text": "Artificial intelligence is transforming businesses.",
            "metadata": {"page_number": 1},
            "distance": 0.1,
            "similarity": 0.9,
        },
        {
            "chunk_id": "doc1:page-2:chunk-0",
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"page_number": 2},
            "distance": 0.2,
            "similarity": 0.8,
        },
    ]


@pytest.fixture
def generator(sample_results):
    retriever = FakeRetriever(sample_results)
    llm = FakeLLM()

    return AnswerGenerator(retriever, llm), retriever, llm


def test_answer_returns_generated_response(generator):
    answer_generator, _, _ = generator

    result = answer_generator.answer(
        "What is artificial intelligence?"
    )

    assert result["answer"] == (
        "AI is transforming businesses. [Page 1]"
    )


def test_answer_returns_question(generator):
    answer_generator, _, _ = generator

    result = answer_generator.answer("What is AI?")

    assert result["question"] == "What is AI?"


def test_answer_includes_source_citations(generator):
    answer_generator, _, _ = generator

    result = answer_generator.answer("What is AI?")

    assert len(result["sources"]) == 2
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][1]["page_number"] == 2


def test_answer_includes_source_text(generator):
    answer_generator, _, _ = generator

    result = answer_generator.answer("What is AI?")

    assert (
        result["sources"][0]["text"]
        == "Artificial intelligence is transforming businesses."
    )


def test_answer_passes_context_to_llm(generator):
    answer_generator, _, llm = generator

    answer_generator.answer("What is AI?")

    assert "Artificial intelligence" in llm.last_prompt
    assert "[Source: Page 1]" in llm.last_prompt
    assert "[Source: Page 2]" in llm.last_prompt


def test_answer_passes_question_to_retriever(generator):
    answer_generator, retriever, _ = generator

    answer_generator.answer("What is machine learning?")

    assert retriever.last_question == "What is machine learning?"


def test_answer_respects_top_k(generator):
    answer_generator, retriever, _ = generator

    answer_generator.answer("What is AI?", top_k=3)

    assert retriever.last_top_k == 3


def test_empty_question_raises_error(generator):
    answer_generator, _, _ = generator

    with pytest.raises(ValueError):
        answer_generator.answer("")


def test_answer_when_no_documents_are_found():
    retriever = FakeRetriever([])
    llm = FakeLLM()

    answer_generator = AnswerGenerator(retriever, llm)

    result = answer_generator.answer("What is AI?")

    assert result["sources"] == []
    assert "could not find relevant information" in result["answer"]

    # The LLM should not be called when retrieval returns nothing.
    assert llm.last_prompt is None


def test_system_prompt_requires_document_grounded_answers(generator):
    answer_generator, _, llm = generator

    answer_generator.answer("What is AI?")

    assert "ONLY the provided document context" in (
        llm.last_system_prompt
    )
    assert "Do not invent facts" in llm.last_system_prompt


def test_system_prompt_requires_page_citations(generator):
    answer_generator, _, llm = generator

    answer_generator.answer("What is AI?")

    assert "Cite supporting pages using [Page X]" in (
        llm.last_system_prompt
    )


def test_answer_can_report_missing_information(generator):
    answer_generator, _, llm = generator

    llm.response = (
        "The document does not contain this information."
    )

    result = answer_generator.answer(
        "What is the company's annual revenue?"
    )

    assert result["answer"] == (
        "The document does not contain this information."
    )