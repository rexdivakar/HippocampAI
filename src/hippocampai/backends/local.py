"""Local backend - wrapper for the existing MemoryClient implementation."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from hippocampai.backends.base import BaseBackend
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class LocalBackend(BaseBackend):
    """Local backend that wraps the existing MemoryClient implementation."""

    def __init__(self, **kwargs: Any):
        """
        Initialize local backend with MemoryClient.

        Args:
            **kwargs: Arguments passed to MemoryClient constructor
        """
        # Import here to avoid circular dependency
        try:
            from hippocampai.client import MemoryClient
        except ImportError as e:
            raise ImportError(
                "Failed to import MemoryClient. Make sure all dependencies are installed.\n"
                "Install with: pip install -e ."
            ) from e

        try:
            self._client = MemoryClient(**kwargs)
            logger.info("Initialized LocalBackend with MemoryClient")

            # Log connection details if available
            if hasattr(self._client, "config"):
                config = self._client.config
                logger.info(f"  Qdrant: {config.qdrant_url}")
                logger.info(f"  Redis: {config.redis_url}")
                logger.info(f"  LLM Provider: {config.llm_provider}")

        except Exception as e:
            error_msg = f"Failed to initialize local backend: {e}\n\n"
            error_msg += "Common issues:\n"
            error_msg += "1. Qdrant not running: docker run -p 6333:6333 qdrant/qdrant\n"
            error_msg += "2. Redis not running: docker run -p 6379:6379 redis:alpine\n"
            error_msg += "3. Ollama not running: docker run -p 11434:11434 ollama/ollama\n"
            error_msg += "4. Missing .env file: Create .env with QDRANT_URL, REDIS_URL, etc.\n"
            error_msg += "5. Start all services: docker-compose up -d\n"

            raise ConnectionError(error_msg) from e

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        expires_at: Optional[datetime] = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        extract_relationships: bool = False,
    ) -> Memory:
        """Store a memory using MemoryClient."""
        # Convert expires_at to ttl_days for MemoryClient
        ttl_days = None
        if expires_at:
            now = datetime.now(expires_at.tzinfo if expires_at.tzinfo else timezone.utc)
            delta = expires_at - now
            ttl_days = max(1, int(delta.total_seconds() / (24 * 3600)))

        return self._client.remember(
            text=text,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            importance=importance,
            ttl_days=ttl_days,
        )

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories using MemoryClient."""
        return self._client.recall(
            query=query,
            user_id=user_id,
            session_id=session_id,
            k=limit,  # MemoryClient uses 'k' parameter instead of 'limit'
            filters=filters,
        )

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID using MemoryClient."""
        # MemoryClient doesn't have get_memory, so we'll use get_memories with a filter
        try:
            # This is a workaround - we don't have a direct way to get by ID
            # For now, return None as we can't efficiently implement this
            return None
        except Exception:
            return None

    def get_memories(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[dict[str, Any]] = None,
        min_importance: Optional[float] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> list[Memory]:
        """Get memories using MemoryClient."""
        # Build filters for MemoryClient
        client_filters = {}
        if filters:
            client_filters.update(filters)
        if session_id:
            client_filters["session_id"] = session_id
        if min_importance is not None:
            client_filters["min_importance"] = min_importance

        return self._client.get_memories(
            user_id=user_id,
            filters=client_filters,
            limit=limit,
        )

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update a memory using MemoryClient."""
        return self._client.update_memory(
            memory_id=memory_id,
            text=text,
            metadata=metadata,
            tags=tags,
            importance=importance,
            expires_at=expires_at,
        )

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory using MemoryClient."""
        return self._client.delete_memory(memory_id)

    # Additional abstract methods from BaseBackend
    def batch_remember(self, memories: list[dict[str, Any]]) -> list[Memory]:
        """Store multiple memories in batch."""
        results = []
        for memory_data in memories:
            result = self.remember(**memory_data)
            results.append(result)
        return results

    def batch_get_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by IDs."""
        results = []
        for memory_id in memory_ids:
            memory = self.get_memory(memory_id)
            if memory:
                results.append(memory)
        return results

    def batch_delete_memories(self, memory_ids: list[str]) -> bool:
        """Delete multiple memories."""
        try:
            for memory_id in memory_ids:
                self.delete_memory(memory_id)
            return True
        except Exception:
            return False

    def consolidate_memories(
        self, user_id: str, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Consolidate related memories."""
        # This would require implementing complex logic, for now return empty list
        return []

    def cleanup_expired_memories(self) -> int:
        """Remove expired memories."""
        # MemoryClient should handle this automatically, return 0 as placeholder
        return 0

    def get_memory_analytics(self, user_id: str) -> dict[str, Any]:
        """Get analytics for user's memories."""
        try:
            return self._client.get_memory_statistics(user_id)
        except Exception:
            return {}
