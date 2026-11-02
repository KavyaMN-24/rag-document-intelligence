from types import SimpleNamespace

import pytest

from app.llm.groq_provider import GroqProvider


class FakeGroqClient:
    """Simulate the Groq API without making network requests."""

    def __init__(self, response_text="Test response"):
        self.response_text = response_text
        self.last_request = None

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create_completion
            )
        )

    def create_completion(self, **kwargs):
        self.last_request = kwargs

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.response_text
                    )
                )
            ]
        )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    client = FakeGroqClient()

    return GroqProvider(client=client), client


def test_generate_returns_response(provider):
    llm, _ = provider

    result = llm.generate("Explain artificial intelligence.")

    assert result == "Test response"


def test_generate_strips_whitespace(provider):
    llm, client = provider
    client.response_text = "  Hello world!  "

    result = llm.generate("Say hello.")

    assert result == "Hello world!"


def test_generate_uses_correct_model(provider):
    llm, client = provider

    llm.generate("Explain RAG.")

    assert client.last_request["model"] == "openai/gpt-oss-20b"


def test_generate_includes_system_prompt(provider):
    llm, client = provider

    llm.generate(
        prompt="Explain RAG.",
        system_prompt="You are a technical assistant.",
    )

    assert client.last_request["messages"][0] == {
        "role": "system",
        "content": "You are a technical assistant.",
    }


def test_generate_rejects_empty_prompt(provider):
    llm, _ = provider

    with pytest.raises(ValueError):
        llm.generate("")


def test_generate_rejects_empty_response(provider):
    llm, client = provider
    client.response_text = ""

    with pytest.raises(RuntimeError):
        llm.generate("Explain RAG.")
        