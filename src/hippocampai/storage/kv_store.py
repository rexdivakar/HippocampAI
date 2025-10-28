"""Key-Value store for fast memory lookups using Redis-like interface."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InMemoryKVStore:
    """In-memory key-value store with TTL support."""

    def __init__(self):
        """Initialize in-memory store."""
        self._store: dict[str, Any] = {}
        self._ttl: dict[str, datetime] = {}

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set a key-value pair with optional TTL."""
        self._store[key] = value
        if ttl_seconds:
            self._ttl[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        logger.debug(f"Set key: {key}")

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        # Check TTL
        if key in self._ttl:
            if datetime.now(timezone.utc) > self._ttl[key]:
                self.delete(key)
                return None

        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if key in self._store:
            del self._store[key]
            if key in self._ttl:
                del self._ttl[key]
            logger.debug(f"Deleted key: {key}")
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        value = self.get(key)  # This handles TTL check
        return value is not None

    def keys(self, pattern: Optional[str] = None) -> list[str]:
        """Get all keys, optionally matching pattern."""
        all_keys = list(self._store.keys())

        if pattern:
            # Simple wildcard pattern matching
            import fnmatch

            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

        return all_keys

    def clear(self):
        """Clear all keys."""
        self._store.clear()
        self._ttl.clear()
        logger.debug("Cleared KV store")

    def size(self) -> int:
        """Get number of keys."""
        return len(self._store)


class MemoryKVStore:
    """
    Key-Value store optimized for memory lookups.

    Provides fast O(1) access to memories by ID, with caching layer.
    """

    def __init__(self, backend: Optional[InMemoryKVStore] = None, cache_ttl: int = 300):
        """
        Initialize memory KV store.

        Args:
            backend: Optional KV backend (defaults to in-memory)
            cache_ttl: Cache TTL in seconds (default: 5 minutes)
        """
        self.backend = backend or InMemoryKVStore()
        self.cache_ttl = cache_ttl

    def _memory_key(self, memory_id: str) -> str:
        """Generate key for memory."""
        return f"memory:{memory_id}"

    def _user_memories_key(self, user_id: str) -> str:
        """Generate key for user's memory list."""
        return f"user:{user_id}:memories"

    def _tag_key(self, tag: str) -> str:
        """Generate key for tag index."""
        return f"tag:{tag}:memories"

    def set_memory(self, memory_id: str, memory_data: dict):
        """Store memory data."""
        key = self._memory_key(memory_id)
        self.backend.set(key, memory_data, ttl_seconds=self.cache_ttl)

        # Index by user
        user_id = memory_data.get("user_id")
        if user_id:
            self._add_to_index(self._user_memories_key(user_id), memory_id)

        # Index by tags
        tags = memory_data.get("tags", [])
        for tag in tags:
            self._add_to_index(self._tag_key(tag), memory_id)

    def get_memory(self, memory_id: str) -> Optional[dict]:
        """Retrieve memory data by ID."""
        key = self._memory_key(memory_id)
        return self.backend.get(key)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from store."""
        # Get memory data first to clean up indices
        memory_data = self.get_memory(memory_id)

        # Delete main key
        key = self._memory_key(memory_id)
        deleted = self.backend.delete(key)

        # Clean up indices
        if memory_data:
            user_id = memory_data.get("user_id")
            if user_id:
                self._remove_from_index(self._user_memories_key(user_id), memory_id)

            tags = memory_data.get("tags", [])
            for tag in tags:
                self._remove_from_index(self._tag_key(tag), memory_id)

        return deleted

    def get_user_memories(self, user_id: str) -> list[str]:
        """Get all memory IDs for a user."""
        key = self._user_memories_key(user_id)
        memory_ids = self.backend.get(key)
        return memory_ids if memory_ids else []

    def get_memories_by_tag(self, tag: str) -> list[str]:
        """Get all memory IDs with a specific tag."""
        key = self._tag_key(tag)
        memory_ids = self.backend.get(key)
        return memory_ids if memory_ids else []

    def _add_to_index(self, index_key: str, value: str):
        """Add value to an index (set)."""
        current = self.backend.get(index_key)
        if current is None:
            current = set()
        elif not isinstance(current, set):
            current = set(current)

        current.add(value)
        self.backend.set(index_key, current)

    def _remove_from_index(self, index_key: str, value: str):
        """Remove value from an index (set)."""
        current = self.backend.get(index_key)
        if current:
            if isinstance(current, set):
                current.discard(value)
            elif isinstance(current, list):
                if value in current:
                    current.remove(value)

            if current:
                self.backend.set(index_key, current)
            else:
                self.backend.delete(index_key)

    def clear_user_cache(self, user_id: str):
        """Clear all cached memories for a user."""
        memory_ids = self.get_user_memories(user_id)
        for memory_id in memory_ids:
            self.delete_memory(memory_id)

    def get_stats(self) -> dict:
        """Get store statistics."""
        return {
            "total_keys": self.backend.size(),
            "memory_keys": len([k for k in self.backend.keys("memory:*")]),
            "user_indices": len([k for k in self.backend.keys("user:*")]),
            "tag_indices": len([k for k in self.backend.keys("tag:*")]),
        }
