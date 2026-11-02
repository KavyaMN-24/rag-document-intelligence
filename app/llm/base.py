from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for language model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate a text response from a prompt."""
        raise NotImplementedError