"""Multi-provider LLM client supporting Anthropic, OpenAI, and Groq."""

import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the provider
            model: Model name/identifier
            **kwargs: Additional provider-specific settings
        """
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self._last_request_time = 0
        self._min_request_interval = kwargs.get("rate_limit_interval", 0.1)

    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            Assistant response text
        """
        pass


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) client."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(api_key, model, **kwargs)
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.anthropic = anthropic
        logger.info(f"Initialized Anthropic client with model: {model}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using Anthropic."""
        self._rate_limit()

        max_tokens = max_tokens or self.kwargs.get("max_tokens", 4096)
        temperature = (
            temperature if temperature is not None else self.kwargs.get("temperature", 0.0)
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return message.content[0].text

        except self.anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit: {e}")
            raise
        except self.anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Chat completion using Anthropic.

        Args:
            messages: List of message dicts
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            tools: Optional list of tool schemas for function calling
            **kwargs: Additional parameters

        Returns:
            Either str (text response) or dict (with tool calls)
        """
        self._rate_limit()

        max_tokens = max_tokens or self.kwargs.get("max_tokens", 4096)
        temperature = (
            temperature if temperature is not None else self.kwargs.get("temperature", 0.0)
        )

        # Anthropic requires system message as separate parameter
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                user_messages.append(msg)

        try:
            api_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages,
                **kwargs,
            }

            # Add system parameter if system message exists
            if system_message:
                api_params["system"] = system_message

            # Add tools if provided
            if tools:
                api_params["tools"] = tools

            message = self.client.messages.create(**api_params)

            # Check if tool was called
            if message.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = []
                for content in message.content:
                    if content.type == "tool_use":
                        tool_calls.append(
                            {"id": content.id, "name": content.name, "arguments": content.input}
                        )

                return {"type": "tool_calls", "tool_calls": tool_calls, "message": message}

            # Regular text response
            return message.content[0].text

        except self.anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit: {e}")
            raise
        except self.anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """OpenAI client."""

    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs):
        super().__init__(api_key, model, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Chat completion using OpenAI."""
        self._rate_limit()

        max_tokens = max_tokens or self.kwargs.get("max_tokens", 4096)
        temperature = (
            temperature if temperature is not None else self.kwargs.get("temperature", 0.0)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GroqClient(BaseLLMClient):
    """Groq client."""

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile", **kwargs):
        super().__init__(api_key, model, **kwargs)
        from groq import Groq

        self.client = Groq(api_key=api_key)
        logger.info(f"Initialized Groq client with model: {model}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using Groq."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Chat completion using Groq."""
        self._rate_limit()

        max_tokens = max_tokens or self.kwargs.get("max_tokens", 8192)
        temperature = (
            temperature if temperature is not None else self.kwargs.get("temperature", 0.0)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(
        provider: str, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs
    ) -> BaseLLMClient:
        """
        Create LLM client for specified provider.

        Args:
            provider: Provider name (anthropic, openai, groq)
            api_key: API key (uses env var if not provided)
            model: Model name (uses default if not provided)
            **kwargs: Additional provider-specific settings

        Returns:
            BaseLLMClient instance

        Raises:
            ValueError: If provider is unsupported or API key missing
        """
        provider = provider.lower()

        # Get API key from environment if not provided
        if api_key is None:
            env_key_map = {
                LLMProvider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
                LLMProvider.OPENAI.value: "OPENAI_API_KEY",
                LLMProvider.GROQ.value: "GROQ_API_KEY",
            }
            env_key = env_key_map.get(provider)
            if env_key:
                api_key = os.getenv(env_key)

            if not api_key:
                raise ValueError(f"API key required for provider '{provider}'")

        # Get default model if not provided
        if model is None:
            default_models = {
                LLMProvider.ANTHROPIC.value: "claude-3-5-sonnet-20241022",
                LLMProvider.OPENAI.value: "gpt-4o",
                LLMProvider.GROQ.value: "llama-3.1-70b-versatile",
            }
            model = default_models.get(provider)

        # Create client
        if provider == LLMProvider.ANTHROPIC.value:
            return AnthropicClient(api_key=api_key, model=model, **kwargs)
        elif provider == LLMProvider.OPENAI.value:
            return OpenAIClient(api_key=api_key, model=model, **kwargs)
        elif provider == LLMProvider.GROQ.value:
            return GroqClient(api_key=api_key, model=model, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {[p.value for p in LLMProvider]}"
            )


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Get LLM client using settings from environment or parameters.

    Args:
        provider: Provider name (defaults to LLM_PROVIDER env var or 'anthropic')
        api_key: API key (defaults to provider-specific env var)
        model: Model name (defaults to provider-specific env var or default)
        **kwargs: Additional settings

    Returns:
        BaseLLMClient instance

    Example:
        # Use default provider from environment
        client = get_llm_client()

        # Specify provider
        client = get_llm_client(provider="openai")

        # Full customization
        client = get_llm_client(
            provider="groq",
            model="mixtral-8x7b-32768",
            temperature=0.7
        )
    """
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", LLMProvider.ANTHROPIC.value)

    # Get model from env if not specified
    if model is None:
        model_env_map = {
            LLMProvider.ANTHROPIC.value: "ANTHROPIC_MODEL",
            LLMProvider.OPENAI.value: "OPENAI_MODEL",
            LLMProvider.GROQ.value: "GROQ_MODEL",
        }
        model_env = model_env_map.get(provider.lower())
        if model_env:
            model = os.getenv(model_env)

    return LLMClientFactory.create_client(provider=provider, api_key=api_key, model=model, **kwargs)
