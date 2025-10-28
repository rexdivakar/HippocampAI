"""Base LLM adapter interface."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    def chat(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Chat completion."""
        pass
