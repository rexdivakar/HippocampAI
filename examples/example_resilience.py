"""Example demonstrating retry logic and structured logging.

This example shows how HippocampAI handles transient failures gracefully
using automatic retry logic and provides production-grade observability
through structured JSON logging with request ID tracking.
"""

import logging

from hippocampai.memory.client import MemoryClient

from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.utils.structured_logging import (
    RequestContext,
    get_logger,
    setup_structured_logging,
)

# Setup structured logging with JSON format
setup_structured_logging(level=logging.INFO, format_json=True)

logger = get_logger(__name__)


def example_retry_logic():
    """Demonstrate automatic retry on transient failures.

    All Qdrant operations and LLM calls automatically retry on:
    - Connection errors
    - Timeout errors
    - Network issues
    - 500-series HTTP errors

    Default configuration:
    - Max attempts: 3
    - Exponential backoff: 1-10 seconds
    - Automatic logging of retry attempts
    """
    logger.info("=== Retry Logic Example ===")

    # Initialize client - all operations are automatically retryable
    client = MemoryClient(
        qdrant_url="http://localhost:6333", llm=OllamaLLM(model="qwen2.5:7b-instruct")
    )

    # These operations will automatically retry on transient failures:
    # 1. Store a memory (Qdrant upsert with retry)
    client.store(
        user_id="user123",
        content="Python is my favorite programming language",
        memory_type="prefs",
        tags=["programming", "languages"],
    )
    logger.info("Memory stored successfully (with automatic retry support)")

    # 2. Search memories (Qdrant search with retry)
    results = client.search(
        user_id="user123", query="What programming languages does the user like?", limit=5
    )
    logger.info(f"Search completed (with automatic retry support): {len(results)} results")

    # 3. LLM generation (with retry)
    # Note: If Ollama is temporarily unavailable, it will retry automatically
    response = client.llm.generate(
        prompt="List 3 benefits of Python",
        system="You are a helpful programming assistant",
        max_tokens=256,
    )
    logger.info(f"LLM generation completed (with automatic retry support): {response[:50]}...")


def example_structured_logging():
    """Demonstrate structured JSON logging with request ID tracking.

    Benefits:
    - All logs are JSON formatted for easy parsing
    - Request IDs automatically tracked across operations
    - Rich metadata: timestamps, levels, modules, functions, line numbers
    - Easy integration with log aggregation tools (ELK, Splunk, etc.)
    """
    logger.info("=== Structured Logging Example ===")

    # Use RequestContext for automatic request tracking
    with RequestContext(operation="user_query_processing") as ctx:
        logger.info(
            "Processing user query",
            extra={"user_id": "user123", "query": "What languages do I know?", "source": "api"},
        )

        client = MemoryClient(
            qdrant_url="http://localhost:6333", llm=OllamaLLM(model="qwen2.5:7b-instruct")
        )

        # All logs within this context will have the same request_id
        results = client.search(user_id="user123", query="What languages do I know?", limit=5)

        logger.info(
            "Query processing completed",
            extra={
                "user_id": "user123",
                "result_count": len(results),
                "request_id": ctx.request_id,
            },
        )

    # Request ID is automatically cleared after context exit
    logger.info("Request context completed")


def example_combined():
    """Demonstrate retry logic and structured logging working together.

    This is the recommended pattern for production deployments:
    - Wrap operations in RequestContext for tracing
    - Let retry decorators handle transient failures automatically
    - Monitor logs for retry attempts and request completion
    """
    logger.info("=== Combined Example ===")

    with RequestContext(operation="memory_management"):
        logger.info("Starting memory management operation", extra={"operation_type": "batch_store"})

        client = MemoryClient(
            qdrant_url="http://localhost:6333", llm=OllamaLLM(model="qwen2.5:7b-instruct")
        )

        # Store multiple memories - each with automatic retry
        memories = [
            {"content": "I live in San Francisco", "type": "facts", "tags": ["location"]},
            {"content": "I prefer dark mode interfaces", "type": "prefs", "tags": ["ui"]},
            {"content": "I speak English and Spanish", "type": "facts", "tags": ["languages"]},
        ]

        for i, memory in enumerate(memories):
            try:
                client.store(
                    user_id="user123",
                    content=memory["content"],
                    memory_type=memory["type"],
                    tags=memory["tags"],
                )
                logger.info(
                    f"Memory {i + 1}/{len(memories)} stored",
                    extra={"memory_index": i, "memory_type": memory["type"]},
                )
            except Exception as e:
                # Even after retries, if it fails, we log it properly
                logger.error(
                    f"Failed to store memory {i + 1}",
                    extra={"memory_index": i, "error": str(e)},
                    exc_info=True,
                )

        logger.info(
            "Memory management operation completed", extra={"total_memories": len(memories)}
        )


def example_retry_configuration():
    """Show how retry configuration works.

    Retry decorators are configured with:
    - max_attempts: Number of retry attempts (default: 3)
    - min_wait: Minimum wait time between retries in seconds (default: 1)
    - max_wait: Maximum wait time between retries in seconds (default: 10)
    - Exponential backoff with 2x multiplier

    For Qdrant operations:
    - @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)

    For LLM operations:
    - @get_llm_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
    """
    logger.info("=== Retry Configuration Example ===")

    logger.info("Default Qdrant retry config: max_attempts=3, min_wait=1s, max_wait=5s")
    logger.info("Default LLM retry config: max_attempts=3, min_wait=2s, max_wait=10s")
    logger.info("Retry strategy: Exponential backoff with 2x multiplier")
    logger.info("Retryable exceptions: ConnectionError, TimeoutError, OSError")

    logger.info(
        "Example retry sequence:",
        extra={
            "attempt_1": "Immediate",
            "attempt_2": "Wait 1-2s (exponential backoff)",
            "attempt_3": "Wait 2-4s (exponential backoff)",
            "final_failure": "Raise exception after max attempts",
        },
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HippocampAI - Resilience & Observability Examples")
    print("=" * 70 + "\n")

    # Run examples
    example_retry_logic()
    print()

    example_structured_logging()
    print()

    example_combined()
    print()

    example_retry_configuration()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("Check your logs for JSON-formatted output with request IDs")
    print("=" * 70 + "\n")
