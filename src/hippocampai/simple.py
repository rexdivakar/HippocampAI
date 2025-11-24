"""
Simplified API for HippocampAI - As easy as mem0 and zep!

This module provides a simplified interface similar to mem0 and zep,
making it extremely easy to get started with HippocampAI.

Example (mem0-style):
    >>> from hippocampai import Memory
    >>> m = Memory()
    >>> m.add("I love Python", user_id="alice")
    >>> results = m.search("programming", user_id="alice")

Example (zep-style):
    >>> from hippocampai import MemoryClient
    >>> client = MemoryClient()
    >>> client.add("I love Python", session_id="alice")
    >>> results = client.search("programming", session_id="alice")

Example (HippocampAI native):
    >>> from hippocampai import Memory
    >>> m = Memory()
    >>> m.remember("I love Python", user_id="alice")
    >>> results = m.recall("programming", user_id="alice")
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from hippocampai.client import MemoryClient as _MemoryClient
from hippocampai.models.memory import Memory as _Memory
from hippocampai.models.memory import RetrievalResult

if TYPE_CHECKING:
    from hippocampai.unified_client import UnifiedMemoryClient


class Memory:
    """
    Simplified Memory interface - compatible with mem0 API patterns.

    This class provides a simplified, mem0-compatible interface to HippocampAI.
    Perfect for quick prototyping and simple use cases.

    Example:
        >>> m = Memory()
        >>> m.add("I prefer dark mode", user_id="alice")
        >>> results = m.search("preferences", user_id="alice")
        >>> for result in results:
        ...     print(result.text, result.score)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize Memory client.

        Args:
            config: Optional configuration dict (auto-detected if not provided)
            api_key: Optional API key for remote mode
            api_url: Optional API URL for remote mode (default: local mode)

        Example:
            >>> # Local mode (default)
            >>> m = Memory()

            >>> # Remote mode
            >>> m = Memory(api_url="http://localhost:8000", api_key="your-key")

            >>> # Custom config
            >>> m = Memory(config={"llm_provider": "groq"})
        """
        # Determine mode
        if api_url:
            from hippocampai import UnifiedMemoryClient

            self._client: Union[_MemoryClient, "UnifiedMemoryClient"] = UnifiedMemoryClient(
                mode="remote", api_url=api_url, api_key=api_key
            )
        else:
            self._client = _MemoryClient(**(config or {}))

    def add(
        self, text: str, user_id: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> _Memory:
        """
        Add a memory (mem0-compatible API).

        Args:
            text: The memory text
            user_id: User identifier
            metadata: Optional metadata dict (ignored - use kwargs for parameters)
            **kwargs: Additional parameters (type, importance, tags, etc.)

        Returns:
            Memory object

        Example:
            >>> m.add("I prefer oat milk", user_id="alice")
            >>> m.add("Paris is in France", user_id="bob", type="fact")
        """
        # Note: metadata parameter is ignored to maintain API compatibility
        result: _Memory = self._client.remember(text=text, user_id=user_id, **kwargs)
        return result

    def search(
        self, query: str, user_id: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search memories (mem0-compatible API).

        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results to return
            filters: Optional filters dict

        Returns:
            List of search results with scores

        Example:
            >>> results = m.search("coffee", user_id="alice", limit=3)
            >>> for result in results:
            ...     print(f"{result.score:.2f}: {result.memory.text}")
        """
        # Handle both MemoryClient (k parameter) and UnifiedMemoryClient (limit parameter)
        if isinstance(self._client, _MemoryClient):
            results: List[RetrievalResult] = self._client.recall(query=query, user_id=user_id, k=limit, filters=filters or {})
            return results
        else:
            # UnifiedMemoryClient uses limit instead of k
            results = self._client.recall(query=query, user_id=user_id, limit=limit, filters=filters or {})
            return results

    def get(self, memory_id: str) -> Optional[_Memory]:
        """
        Get memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory object or None

        Example:
            >>> memory = m.get("mem_123")
            >>> print(memory.text)
        """
        return self._client.get_memory(memory_id)

    def get_all(self, user_id: str, limit: Optional[int] = None) -> List[_Memory]:
        """
        Get all memories for a user.

        Args:
            user_id: User identifier
            limit: Optional limit on number of memories

        Returns:
            List of memories

        Example:
            >>> memories = m.get_all(user_id="alice", limit=10)
        """
        memories: List[_Memory] = self._client.get_memories(user_id=user_id, limit=limit or 100)
        return memories

    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[_Memory]:
        """
        Update a memory.

        Args:
            memory_id: Memory identifier
            text: New text (optional)
            metadata: New metadata (optional)
            **kwargs: Additional fields to update

        Returns:
            Updated memory or None

        Example:
            >>> m.update("mem_123", text="Updated text")
        """
        return self._client.update_memory(
            memory_id=memory_id, text=text, metadata=metadata, **kwargs
        )

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False otherwise

        Example:
            >>> m.delete("mem_123")
        """
        result: bool = self._client.delete_memory(memory_id)
        return result

    def delete_all(self, user_id: str) -> int:
        """
        Delete all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted

        Example:
            >>> count = m.delete_all(user_id="alice")
            >>> print(f"Deleted {count} memories")
        """
        memories = self._client.get_memories(user_id=user_id, limit=1000)
        ids = [m.id for m in memories]
        if isinstance(self._client, _MemoryClient):
            count: int = self._client.delete_memories(ids, user_id=user_id)
            return count
        else:
            # UnifiedMemoryClient uses batch_delete_memories
            self._client.batch_delete_memories(ids)
            return len(ids)

    # HippocampAI native methods (keep for backward compatibility)

    def remember(self, *args: Any, **kwargs: Any) -> _Memory:
        """Alias for add() - HippocampAI native API."""
        result: _Memory = self._client.remember(*args, **kwargs)
        return result

    def recall(self, *args: Any, **kwargs: Any) -> List[RetrievalResult]:
        """Alias for search() - HippocampAI native API."""
        results: List[RetrievalResult] = self._client.recall(*args, **kwargs)
        return results


class Session:
    """
    Simplified Session interface - compatible with zep API patterns.

    This class provides a session-based interface similar to zep,
    with automatic conversation management.

    Example:
        >>> session = Session(session_id="conversation_123")
        >>> session.add_message("user", "Hello!")
        >>> session.add_message("assistant", "Hi there!")
        >>> summary = session.get_summary()
    """

    def __init__(
        self, session_id: str, user_id: Optional[str] = None, client: Optional[_MemoryClient] = None
    ):
        """
        Initialize Session.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier (defaults to session_id)
            client: Optional MemoryClient instance (creates one if not provided)

        Example:
            >>> session = Session(session_id="conv_123", user_id="alice")
        """
        self.session_id = session_id
        self.user_id = user_id or session_id
        self._client = client or _MemoryClient()

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> _Memory:
        """
        Add a message to the session.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata (ignored - kept for API compatibility)

        Returns:
            Memory object

        Example:
            >>> session.add_message("user", "What's the weather?")
        """
        # Note: metadata is ignored to maintain compatibility with MemoryClient API
        return self._client.remember(
            text=content, user_id=self.user_id, session_id=self.session_id, type="context"
        )

    def get_messages(self, limit: Optional[int] = None) -> List[_Memory]:
        """
        Get all messages in the session.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of memories

        Example:
            >>> messages = session.get_messages(limit=10)
        """
        messages: List[_Memory] = self._client.get_memories(
            user_id=self.user_id, filters={"session_id": self.session_id}, limit=limit or 100
        )
        return messages

    def search(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        """
        Search within session messages.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results

        Example:
            >>> results = session.search("weather", limit=3)
        """
        results: List[RetrievalResult] = self._client.recall(
            query=query, user_id=self.user_id, k=limit, filters={"session_id": self.session_id}
        )
        return results

    def get_summary(self) -> str:
        """
        Get session summary.

        Returns:
            Summary text

        Example:
            >>> summary = session.get_summary()
            >>> print(summary)
        """
        messages = self.get_messages()
        if not messages:
            return "No messages in session"

        # Convert to conversation format
        conversation = [
            {"role": m.metadata.get("role", "user"), "content": m.text} for m in messages
        ]

        summary_obj = self._client.summarize_conversation(
            messages=conversation, session_id=self.session_id
        )

        return summary_obj.summary if summary_obj else "Summary not available"

    def clear(self) -> int:
        """
        Clear all messages in the session.

        Returns:
            Number of messages deleted

        Example:
            >>> count = session.clear()
        """
        memories = self.get_messages()
        ids = [m.id for m in memories]
        count: int = self._client.delete_memories(ids, user_id=self.user_id)
        return count


# Convenience aliases for different naming preferences
MemoryStore = Memory  # Alternative name
MemoryManager = Memory  # Alternative name


__all__ = [
    "Memory",
    "Session",
    "MemoryStore",
    "MemoryManager",
]
