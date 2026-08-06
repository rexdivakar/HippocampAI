"""Anthropic (Claude) LLM adapter."""

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


class AnthropicLLM(BaseLLM):
    """Anthropic (Claude) LLM adapter."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic LLM client.

        Args:
            api_key: Anthropic API key
            model: Claude model name. Currently supported models include:
                   - claude-3-5-sonnet-20241022 (recommended for quality/speed balance)
                   - claude-3-opus-20240229 (highest quality)
                   - claude-3-haiku-20240307 (fastest, most economical)
                   See https://docs.anthropic.com/claude/docs/models-overview for latest models
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required for Claude: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Anthropic: {model}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result: str = self.chat(messages, max_tokens, temperature, system)
        return result

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> str:
        """Chat completion (with automatic retry on transient failures).

        Each retry attempt is recorded as a separate upstream attempt under
        one logical invocation; on total failure this still returns "" to
        preserve existing caller behavior.
        """
        with llm_invocation(
            provider="anthropic", requested_model=self.model, operation="chat"
        ) as invocation_id:
            try:
                result: str = self._chat_with_retry(messages, max_tokens, temperature, system)
                return result
            except Exception as e:
                logger.error(f"Anthropic chat failed: {e}")
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
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> str:
        """One upstream attempt at a chat completion. Tenacity re-invokes this
        whole function on each retry, so each call is exactly one attempt."""
        with llm_attempt(provider="anthropic", requested_model=self.model) as attempt:
            # Anthropic requires system messages to be passed separately
            message_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": cast("Any", messages),  # Type-safe cast for dict compatibility
            }

            if system:
                message_kwargs["system"] = system

            response = self.client.messages.create(**message_kwargs)
            attempt.apply_metadata(UpstreamMetadata.from_anthropic_response(response))
            attempt.finish_reason = getattr(response, "stop_reason", None)

            # Extract text content from response
            content = response.content
            if content and len(content) > 0:
                # Anthropic returns a list of content blocks
                return content[0].text if hasattr(content[0], "text") else str(content[0])
            return ""
