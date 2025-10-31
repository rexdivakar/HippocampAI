"""Base backend interface for memory operations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from hippocampai.models.memory import Memory, RetrievalResult


class BaseBackend(ABC):
    """Abstract base class for memory backends (local or remote)."""

    @abstractmethod
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
        """Store a memory."""

    @abstractmethod
    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""

    @abstractmethod
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
        """Get all memories for a user with optional filters."""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update a memory."""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""

    @abstractmethod
    def batch_remember(self, memories: list[dict[str, Any]]) -> list[Memory]:
        """Store multiple memories in batch."""

    @abstractmethod
    def batch_get_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by IDs."""

    @abstractmethod
    def batch_delete_memories(self, memory_ids: list[str]) -> bool:
        """Delete multiple memories."""

    @abstractmethod
    def consolidate_memories(
        self, user_id: str, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Consolidate related memories."""

    @abstractmethod
    def cleanup_expired_memories(self) -> int:
        """Remove expired memories."""

    @abstractmethod
    def get_memory_analytics(self, user_id: str) -> dict[str, Any]:
        """Get analytics for user's memories."""
