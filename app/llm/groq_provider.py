import os

from dotenv import load_dotenv
from groq import Groq

from app.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    """Generate text using the Groq API."""

    def __init__(self, client=None):
        load_dotenv()

        self.model = os.getenv("GROQ_MODEL")

        if not self.model:
            raise ValueError("GROQ_MODEL is not configured.")

        if client is None:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise ValueError("GROQ_API_KEY is not configured.")

            client = Groq(api_key=api_key)

        self.client = client

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate a response using Groq."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()