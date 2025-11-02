"""Redis-based key-value store for fast memory lookups with async support."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Constants for error messages
_REDIS_NOT_CONNECTED_ERROR = "Redis client not connected"


class AsyncRedisKVStore:
    """Async Redis key-value store with TTL support and connection pooling."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        max_connections: int = 100,
        min_idle: int = 10,
    ):
        """
        Initialize async Redis store with connection pooling.

        Args:
            redis_url: Redis connection URL
            db: Redis database number
            max_connections: Maximum connections in pool (default: 100)
            min_idle: Minimum idle connections to maintain (default: 10)
        """
        self.redis_url = redis_url
        self.db = db
        self.max_connections = max_connections
        self.min_idle = min_idle
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None

    async def connect(self) -> None:
        """Establish connection to Redis with connection pooling."""
        if self._client is None:
            # Create connection pool for better performance
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                decode_responses=True,
                encoding="utf-8",
                max_connections=self.max_connections,
                # Connection pool tuning
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            # Test connection with an async operation
            await self._client.ping()
            logger.info(
                f"Connected to Redis at {self.redis_url} with pool size {self.max_connections}"
            )

    async def close(self) -> None:
        """Close Redis connection and pool."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection and pool closed")

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a key-value pair with optional TTL."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        serialized = json.dumps(value)
        if ttl_seconds:
            await self._client.setex(key, ttl_seconds, serialized)
        else:
            await self._client.set(key, serialized)
        logger.debug(f"Set key: {key} with TTL: {ttl_seconds}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        value = await self._client.get(key)
        if value:
            return json.loads(value)
        return None

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        result = await self._client.delete(key)
        logger.debug(f"Deleted key: {key}, result: {result}")
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        return await self._client.exists(key) > 0

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        keys = await self._client.keys(pattern)
        return [k.decode() if isinstance(k, bytes) else k for k in keys]

    async def clear(self) -> None:
        """Clear all keys in the current database."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        await self._client.flushdb()
        logger.debug("Cleared Redis database")

    async def size(self) -> int:
        """Get number of keys."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        return await self._client.dbsize()

    async def sadd(self, key: str, *values: str) -> int:
        """Add values to a set."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        result = self._client.sadd(key, *values)
        # Handle both sync and async returns from redis client
        if isinstance(result, int):
            return result
        return int(await result) if result is not None else 0

    async def smembers(self, key: str) -> set[str]:  # type: ignore[valid-type]
        """Get all members of a set."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        # smembers is synchronous in redis-py async client, returns a Set directly
        members = self._client.smembers(key)
        if isinstance(members, set):
            return {str(m.decode() if isinstance(m, bytes) else m) for m in members}
        # If it's a coroutine, await it
        members_result = await members
        return {str(m.decode() if isinstance(m, bytes) else m) for m in members_result}

    async def srem(self, key: str, *values: str) -> int:
        """Remove values from a set."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        # srem might return int directly or a coroutine
        result = self._client.srem(key, *values)
        if isinstance(result, int):
            return result
        return int(await result) if result is not None else 0

    async def pipeline(self) -> Any:
        """Create a pipeline for batch operations."""
        await self.connect()
        if self._client is None:
            raise RuntimeError(_REDIS_NOT_CONNECTED_ERROR)
        return self._client.pipeline()


class AsyncMemoryKVStore:
    """
    Key-Value store optimized for memory lookups with Redis backend.

    Provides fast O(1) access to memories by ID, with caching layer and connection pooling.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 300,
        max_connections: int = 100,
        min_idle: int = 10,
    ):
        """
        Initialize async memory KV store with connection pooling.

        Args:
            redis_url: Redis connection URL
            cache_ttl: Cache TTL in seconds (default: 5 minutes)
            max_connections: Maximum connections in pool (default: 100)
            min_idle: Minimum idle connections to maintain (default: 10)
        """
        self.store = AsyncRedisKVStore(
            redis_url=redis_url, max_connections=max_connections, min_idle=min_idle
        )
        self.cache_ttl = cache_ttl

    @property
    def backend(self):
        """Backward compatibility - alias for store."""
        return self.store

    async def connect(self):
        """Connect to Redis."""
        await self.backend.connect()

    async def close(self):
        """Close Redis connection."""
        await self.backend.close()

    def _memory_key(self, memory_id: str) -> str:
        """Generate key for memory."""
        return f"memory:{memory_id}"

    def _user_memories_key(self, user_id: str) -> str:
        """Generate key for user's memory list."""
        return f"user:{user_id}:memories"

    def _tag_key(self, tag: str) -> str:
        """Generate key for tag index."""
        return f"tag:{tag}:memories"

    async def set_memory(self, memory_id: str, memory_data: dict):
        """Store memory data."""
        key = self._memory_key(memory_id)
        await self.backend.set(key, memory_data, ttl_seconds=self.cache_ttl)

        # Index by user
        user_id = memory_data.get("user_id")
        if user_id:
            await self.backend.sadd(self._user_memories_key(user_id), memory_id)

        # Index by tags
        tags = memory_data.get("tags", [])
        for tag in tags:
            await self.backend.sadd(self._tag_key(tag), memory_id)

    async def get_memory(self, memory_id: str) -> Optional[dict]:
        """Retrieve memory data by ID."""
        key = self._memory_key(memory_id)
        return await self.backend.get(key)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from store."""
        # Get memory data first to clean up indices
        memory_data = await self.get_memory(memory_id)

        # Delete main key
        key = self._memory_key(memory_id)
        deleted = await self.backend.delete(key)

        # Clean up indices
        if memory_data:
            user_id = memory_data.get("user_id")
            if user_id:
                await self.backend.srem(self._user_memories_key(user_id), memory_id)

            tags = memory_data.get("tags", [])
            for tag in tags:
                await self.backend.srem(self._tag_key(tag), memory_id)

        return deleted

    async def get_user_memories(self, user_id: str) -> list[str]:
        """Get all memory IDs for a user."""
        key = self._user_memories_key(user_id)
        memory_ids = await self.backend.smembers(key)
        return list(memory_ids)

    async def get_memories_by_tag(self, tag: str) -> list[str]:
        """Get all memory IDs with a specific tag."""
        key = self._tag_key(tag)
        memory_ids = await self.backend.smembers(key)
        return list(memory_ids)

    async def batch_set_memories(self, memories: list[tuple[str, dict]]):
        """Batch store multiple memories."""
        pipe = await self.backend.pipeline()
        for memory_id, memory_data in memories:
            key = self._memory_key(memory_id)
            serialized = json.dumps(memory_data)
            pipe.setex(key, self.cache_ttl, serialized)

            # Index by user
            user_id = memory_data.get("user_id")
            if user_id:
                pipe.sadd(self._user_memories_key(user_id), memory_id)

            # Index by tags
            tags = memory_data.get("tags", [])
            for tag in tags:
                pipe.sadd(self._tag_key(tag), memory_id)

        await pipe.execute()

    async def batch_delete_memories(self, memory_ids: list[str]):
        """Batch delete multiple memories."""
        pipe = await self.backend.pipeline()
        for memory_id in memory_ids:
            # Get memory data for cleanup
            memory_data = await self.get_memory(memory_id)

            # Delete main key
            key = self._memory_key(memory_id)
            pipe.delete(key)

            # Clean up indices
            if memory_data:
                user_id = memory_data.get("user_id")
                if user_id:
                    pipe.srem(self._user_memories_key(user_id), memory_id)

                tags = memory_data.get("tags", [])
                for tag in tags:
                    pipe.srem(self._tag_key(tag), memory_id)

        await pipe.execute()

    async def clear_user_cache(self, user_id: str):
        """Clear all cached memories for a user."""
        memory_ids = await self.get_user_memories(user_id)
        await self.batch_delete_memories(memory_ids)

    async def get_stats(self) -> dict:
        """Get store statistics."""
        all_keys = await self.backend.keys("*")
        return {
            "total_keys": len(all_keys),
            "memory_keys": len([k for k in all_keys if k.startswith("memory:")]),
            "user_indices": len([k for k in all_keys if k.startswith("user:")]),
            "tag_indices": len([k for k in all_keys if k.startswith("tag:")]),
        }
