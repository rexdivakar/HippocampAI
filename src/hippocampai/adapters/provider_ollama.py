"""Ollama LLM adapter."""

import logging
from typing import Optional, cast

import httpx

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


class OllamaLLM(BaseLLM):
    """Ollama local LLM adapter."""

    def __init__(
        self, model: str = "qwen2.5:7b-instruct", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info(f"Initialized Ollama: {model} at {base_url}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion (with automatic retry on transient failures).

        Each retry attempt is recorded as a separate upstream attempt under
        one logical invocation; on total failure this still returns "" to
        preserve existing caller behavior.
        """
        with llm_invocation(
            provider="ollama", requested_model=self.model, operation="generate"
        ) as invocation_id:
            try:
                result: str = self._generate_with_retry(prompt, system, max_tokens, temperature)
                return result
            except Exception as e:
                logger.error(f"Ollama generate failed: {e}")
                error_type, _ = classify_llm_error(e)
                get_telemetry().end_llm_invocation(
                    invocation_id,
                    status="error",
                    error_type=error_type,
                    error_message=sanitize_error_message(str(e)),
                )
                return ""

    @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    def _generate_with_retry(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """One upstream attempt at a generate call. Tenacity re-invokes this
        whole function on each retry, so each call is exactly one attempt."""
        with llm_attempt(provider="ollama", requested_model=self.model) as attempt:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            attempt.apply_metadata(UpstreamMetadata.from_ollama_response(data))
            attempt.finish_reason = data.get("done_reason")
            return cast(str, data["response"])

    def chat(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Chat completion (with automatic retry on transient failures).

        Each retry attempt is recorded as a separate upstream attempt under
        one logical invocation; on total failure this still returns "" to
        preserve existing caller behavior.
        """
        with llm_invocation(
            provider="ollama", requested_model=self.model, operation="chat"
        ) as invocation_id:
            try:
                result: str = self._chat_with_retry(messages, max_tokens, temperature)
                return result
            except Exception as e:
                logger.error(f"Ollama chat failed: {e}")
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
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """One upstream attempt at a chat completion. Tenacity re-invokes this
        whole function on each retry, so each call is exactly one attempt."""
        with llm_attempt(provider="ollama", requested_model=self.model) as attempt:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            attempt.apply_metadata(UpstreamMetadata.from_ollama_response(data))
            attempt.finish_reason = data.get("done_reason")
            return cast(str, data["message"]["content"])
