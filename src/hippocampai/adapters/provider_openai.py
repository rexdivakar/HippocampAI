"""OpenAI LLM adapter."""

import logging
from typing import Any, Optional, cast

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.telemetry import (
    UpstreamMetadata,
    classify_llm_error,
    get_telemetry,
    llm_attempt,
    llm_invocation,
    sanitize_error_message,
)
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

        result: str = self.chat(messages, max_tokens, temperature)
        return result

    def chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Chat completion (with automatic retry on transient failures).

        Each retry attempt is recorded as a separate upstream attempt under
        one logical invocation; on total failure this still returns "" to
        preserve existing caller behavior.
        """
        with llm_invocation(
            provider="openai", requested_model=self.model, operation="chat"
        ) as invocation_id:
            try:
                result: str = self._chat_with_retry(messages, max_tokens, temperature)
                return result
            except Exception as e:
                logger.error(f"OpenAI chat failed: {e}")
                error_type, _ = classify_llm_error(e)
                get_telemetry().end_llm_invocation(
                    invocation_id,
                    status="error",
                    error_type=error_type,
                    error_message=sanitize_error_message(str(e)),
                )
                return ""

    @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    def _chat_with_retry(
        self, messages: list[dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """One upstream attempt at a chat completion. Tenacity re-invokes this
        whole function on each retry, so each call is exactly one attempt."""
        with llm_attempt(provider="openai", requested_model=self.model) as attempt:
            # Cast to proper type for OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=cast("Any", messages),  # Type-safe cast for dict compatibility
                max_tokens=max_tokens,
                temperature=temperature,
            )
            attempt.apply_metadata(UpstreamMetadata.from_openai_response(response))
            if response.choices:
                attempt.finish_reason = getattr(response.choices[0], "finish_reason", None)
            content = response.choices[0].message.content
            return content if content is not None else ""
