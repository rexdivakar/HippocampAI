"""Base class for components using LLM functionality with multi-provider support."""

import json
import logging
import time
from typing import Optional, Dict, Any

from src.llm_provider import get_llm_client, BaseLLMClient


logger = logging.getLogger(__name__)


class LLMBaseMixin:
    """
    Mixin class providing LLM functionality with multi-provider support.

    Components that need LLM capabilities should inherit from this class.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider (anthropic, openai, groq)
            api_key: API key for the provider
            model: Model name
            **kwargs: Additional provider-specific settings
        """
        self.llm_client: BaseLLMClient = get_llm_client(
            provider=provider,
            api_key=api_key,
            model=model,
            **kwargs
        )
        self.provider = provider or self.llm_client.__class__.__name__.replace('Client', '').lower()
        self.model = model or self.llm_client.model

        logger.info(f"Initialized LLM: provider={self.provider}, model={self.model}")

    def _call_llm(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 2
    ) -> str:
        """
        Call LLM with retry logic.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts

        Returns:
            Generated text

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response

            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")

                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts") from e

        raise RuntimeError("LLM call failed")

    def _call_llm_json(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts

        Returns:
            Parsed JSON dict

        Raises:
            RuntimeError: If parsing fails
        """
        response = self._call_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries
        )

        # Extract JSON from response
        try:
            # Try to find JSON in response
            response = response.strip()

            # Look for JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")

            json_text = response[start_idx:end_idx]
            return json.loads(json_text)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response text: {response[:500]}...")
            raise RuntimeError("Failed to parse JSON from LLM response") from e

    def _call_llm_chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 2
    ) -> str:
        """
        Call LLM with chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts

        Returns:
            Assistant response

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response

            except Exception as e:
                logger.error(f"LLM chat failed (attempt {attempt + 1}/{max_retries + 1}): {e}")

                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM chat failed after {max_retries + 1} attempts") from e

        raise RuntimeError("LLM chat failed")
