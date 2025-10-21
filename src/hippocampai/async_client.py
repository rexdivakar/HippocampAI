"""Async variants of MemoryClient operations.

This module provides asynchronous versions of core memory operations
for use in async applications and frameworks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class AsyncMemoryClient(MemoryClient):
    """Async-enabled MemoryClient with async variants of core operations."""

    async def remember_async(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        ttl_days: Optional[int] = None,
    ) -> Memory:
        """Store a memory asynchronously.

        This runs the synchronous remember() method in a thread pool executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.remember(
                text=text,
                user_id=user_id,
                session_id=session_id,
                type=type,
                importance=importance,
                tags=tags,
                ttl_days=ttl_days,
            ),
        )

    async def recall_async(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recall(
                query=query,
                user_id=user_id,
                session_id=session_id,
                k=k,
                filters=filters,
            ),
        )

    async def update_memory_async(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update a memory asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.update_memory(
                memory_id=memory_id,
                text=text,
                importance=importance,
                tags=tags,
                metadata=metadata,
                expires_at=expires_at,
            ),
        )

    async def delete_memory_async(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a memory asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.delete_memory(memory_id, user_id))

    async def get_memories_async(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Get memories asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_memories(user_id, filters, limit))

    async def extract_from_conversation_async(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[Memory]:
        """Extract memories from conversation asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.extract_from_conversation(conversation, user_id, session_id),
        )

    async def add_memories_async(
        self,
        memories: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        """Batch add memories asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_memories(memories, user_id, session_id)
        )

    async def delete_memories_async(
        self, memory_ids: List[str], user_id: Optional[str] = None
    ) -> int:
        """Batch delete memories asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.delete_memories(memory_ids, user_id))

    async def get_memory_statistics_async(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_memory_statistics(user_id))

    async def inject_context_async(
        self,
        prompt: str,
        query: str,
        user_id: str,
        k: int = 5,
        template: str = "default",
    ) -> str:
        """Inject context asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.inject_context(prompt, query, user_id, k, template),
        )
