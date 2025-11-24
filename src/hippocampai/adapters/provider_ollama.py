"""Ollama LLM adapter."""

import logging
from typing import Optional, cast

import httpx

from hippocampai.adapters.llm_base import BaseLLM
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

    @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion (with automatic retry on transient failures)."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return cast(str, response.json()["response"])
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            return ""

    @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    def chat(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        """Chat completion (with automatic retry on transient failures)."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return cast(str, response.json()["message"]["content"])
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            return ""
