"""OpenAI LLM adapter."""

import logging
from typing import Any, Optional, cast

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.utils.retry import get_llm_retry_decorator

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM adapter."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI: {model}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, max_tokens, temperature)

    @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    def chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Chat completion (with automatic retry on transient failures)."""
        try:
            # Cast to proper type for OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast("Any", messages),  # Type-safe cast for dict compatibility
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            return ""
