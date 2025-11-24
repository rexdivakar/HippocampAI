"""Retry utilities for resilient operations.

This module provides retry decorators and configurations for handling
transient failures in external services (Qdrant, LLM providers, etc).
"""

import logging
from typing import Any, Callable, Literal, Optional, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MIN_WAIT = 1  # seconds
DEFAULT_MAX_WAIT = 10  # seconds
DEFAULT_MULTIPLIER = 2


def get_qdrant_retry_decorator(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
) -> Callable[[F], F]:
    """Get retry decorator for Qdrant operations.

    Retries on common transient failures:
    - Connection errors
    - Timeout errors
    - 500-series HTTP errors
    - Rate limit errors (429)

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Returns:
        Retry decorator configured for Qdrant
    """
    return retry(
        retry=retry_if_exception_type(
            (
                ConnectionError,
                TimeoutError,
                OSError,
                # Qdrant-specific exceptions would go here
            )
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=DEFAULT_MULTIPLIER,
            min=min_wait,
            max=max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def get_llm_retry_decorator(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
) -> Callable[[F], F]:
    """Get retry decorator for LLM operations.

    Retries on common transient failures:
    - Connection errors
    - Timeout errors
    - Rate limit errors
    - Server errors (500-series)

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Returns:
        Retry decorator configured for LLM calls
    """
    return retry(
        retry=retry_if_exception_type(
            (
                ConnectionError,
                TimeoutError,
                OSError,
                # LLM-specific exceptions would go here
                # e.g., openai.error.RateLimitError, anthropic.RateLimitError
            )
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=DEFAULT_MULTIPLIER,
            min=min_wait,
            max=max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def retry_on_exception(
    exception_types: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
    log_level: int = logging.WARNING,
) -> Callable[[F], F]:
    """Generic retry decorator for any operation.

    Args:
        exception_types: Tuple of exception types to retry on
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        log_level: Logging level for retry attempts

    Returns:
        Retry decorator

    Example:
        @retry_on_exception((ConnectionError, TimeoutError), max_attempts=5)
        def risky_operation():
            # ... operation that might fail transiently
            pass
    """
    return retry(
        retry=retry_if_exception_type(exception_types),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=DEFAULT_MULTIPLIER,
            min=min_wait,
            max=max_wait,
        ),
        before_sleep=before_sleep_log(logger, log_level),
        reraise=True,
    )


class RetryContext:
    """Context manager for tracking retry attempts.

    Example:
        with RetryContext("qdrant_upsert") as ctx:
            # Operation that might be retried
            qdrant.upsert(...)
            ctx.success()
    """

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.attempt = 0
        self.succeeded = False

    def __enter__(self) -> "RetryContext":
        self.attempt += 1
        logger.debug(f"Starting {self.operation_name} (attempt {self.attempt})")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> Literal[False]:
        if exc_type is None and self.succeeded:
            logger.debug(f"{self.operation_name} succeeded on attempt {self.attempt}")
        elif exc_type is not None:
            logger.warning(f"{self.operation_name} failed on attempt {self.attempt}: {exc_val}")
        return False  # Don't suppress exceptions

    def success(self) -> None:
        """Mark operation as successful."""
        self.succeeded = True
