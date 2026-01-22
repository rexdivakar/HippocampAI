"""Unified Memory Client supporting both local and remote modes."""

import logging
from datetime import datetime
from typing import Any, Literal, Optional, cast

from hippocampai.backends.remote import RemoteBackend
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)

ConnectionMode = Literal["local", "remote"]


class UnifiedMemoryClient:
    """
    Unified Memory Client that supports both local and remote modes.

    **Local Mode** (default):
    - Connects directly to Qdrant/Redis/Ollama
    - Maximum performance (5-15ms latency)
    - Python only
    - Example: client = UnifiedMemoryClient(mode="local")

    **Remote Mode**:
    - Connects to HippocampAI SaaS API via HTTP
    - Multi-language support (any HTTP client)
    - Network latency (20-50ms)
    - Example: client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

    Usage:
        # Local mode (direct connection)
        client = UnifiedMemoryClient(mode="local")

        # Remote mode (via SaaS API)
        client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

        # Either way, same API
        memory = client.remember("User prefers dark mode", user_id="user123")
        results = client.recall("UI preferences", user_id="user123")
    """

    def __init__(
        self,
        mode: ConnectionMode = "local",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        **local_kwargs: Any,
    ):
        """
        Initialize unified memory client.

        Args:
            mode: Connection mode - "local" or "remote"
            api_url: API URL for remote mode (e.g., "http://localhost:8000")
            api_key: Optional API key for remote mode authentication
            timeout: Request timeout for remote mode (seconds)
            **local_kwargs: Additional kwargs for local mode (passed to MemoryClient)
                Available kwargs:
                - qdrant_url: Qdrant server URL (default: http://localhost:6333)
                - collection_facts: Qdrant collection for facts
                - collection_prefs: Qdrant collection for preferences
                - embed_model: Embedding model name
                - llm_provider: LLM provider (ollama, openai, groq)
                - llm_model: LLM model name
                - hnsw_M, ef_construction, ef_search: HNSW optimization params

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            ConnectionError: If unable to connect to required services
        """
        self.mode = mode
        self._backend: Any = None

        if mode == "remote":
            if not api_url:
                raise ValueError(
                    "api_url is required for remote mode. "
                    "Example: UnifiedMemoryClient(mode='remote', api_url='http://localhost:8000')"
                )
            try:
                self._backend = RemoteBackend(api_url=api_url, api_key=api_key, timeout=timeout)
                logger.info(f"Initialized in REMOTE mode: {api_url}")
            except Exception as e:
                raise ConnectionError(
                    f"Failed to initialize remote backend: {e}\n"
                    f"Make sure the API server is running at {api_url}\n"
                    f"Start server with: uvicorn hippocampai.api.async_app:app --port 8000"
                ) from e

        elif mode == "local":
            # Import here to avoid circular dependency
            try:
                from hippocampai.backends.local import LocalBackend
            except ImportError as e:
                raise ImportError(
                    "Failed to import LocalBackend. Make sure all dependencies are installed.\n"
                    "Install with: pip install -e ."
                ) from e

            try:
                self._backend = LocalBackend(**local_kwargs)
                logger.info("Initialized in LOCAL mode")

                # Log connection details if available
                if hasattr(self._backend, "_client") and hasattr(self._backend._client, "config"):
                    config = self._backend._client.config
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
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'local' or 'remote'.\n"
                f"Examples:\n"
                f"  - UnifiedMemoryClient(mode='local')\n"
                f"  - UnifiedMemoryClient(mode='remote', api_url='http://localhost:8000')"
            )

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
        """
        Store a memory.

        Args:
            text: The memory text
            user_id: User identifier
            session_id: Optional session identifier
            metadata: Additional metadata
            tags: List of tags for categorization
            importance: Importance score (0.0-1.0)
            expires_at: Optional expiration timestamp
            extract_entities: Extract entities (people, places, etc.)
            extract_facts: Extract factual statements
            extract_relationships: Extract entity relationships

        Returns:
            Created Memory object
        """
        return cast(
            Memory,
            self._backend.remember(
                text=text,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                tags=tags,
                importance=importance,
                expires_at=expires_at,
                extract_entities=extract_entities,
                extract_facts=extract_facts,
                extract_relationships=extract_relationships,
            ),
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
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum number of results
            filters: Optional filters (tags, importance, etc.)
            min_score: Minimum relevance score

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        return cast(
            list[RetrievalResult],
            self._backend.recall(
                query=query,
                user_id=user_id,
                session_id=session_id,
                limit=limit,
                filters=filters,
                min_score=min_score,
            ),
        )

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory object or None if not found
        """
        return cast(Optional[Memory], self._backend.get_memory(memory_id))

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
        """
        Get all memories for a user with optional filters.

        Args:
            user_id: User identifier
            session_id: Optional session filter
            limit: Maximum number of results
            filters: Optional filters (tags, type, etc.)
            min_importance: Minimum importance score
            after: Get memories created after this timestamp
            before: Get memories created before this timestamp

        Returns:
            List of Memory objects
        """
        return cast(
            list[Memory],
            self._backend.get_memories(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
                filters=filters,
                min_importance=min_importance,
                after=after,
                before=before,
            ),
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
        """
        Update a memory.

        Args:
            memory_id: Memory identifier
            text: New text (optional)
            metadata: New metadata (optional)
            tags: New tags (optional)
            importance: New importance score (optional)
            expires_at: New expiration (optional)

        Returns:
            Updated Memory object or None if not found
        """
        return cast(
            Optional[Memory],
            self._backend.update_memory(
                memory_id=memory_id,
                text=text,
                metadata=metadata,
                tags=tags,
                importance=importance,
                expires_at=expires_at,
            ),
        )

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        return cast(bool, self._backend.delete_memory(memory_id))

    def batch_remember(self, memories: list[dict[str, Any]]) -> list[Memory]:
        """
        Store multiple memories in batch (3-5x faster than individual calls).

        Args:
            memories: List of memory data dictionaries

        Returns:
            List of created Memory objects
        """
        return cast(list[Any], self._backend.batch_remember(memories))

    def batch_get_memories(self, memory_ids: list[str]) -> list[Memory]:
        """
        Get multiple memories by IDs.

        Args:
            memory_ids: List of memory identifiers

        Returns:
            List of Memory objects
        """
        return cast(list[Any], self._backend.batch_get_memories(memory_ids))

    def batch_delete_memories(self, memory_ids: list[str]) -> bool:
        """
        Delete multiple memories.

        Args:
            memory_ids: List of memory identifiers

        Returns:
            True if successful
        """
        return cast(bool, self._backend.batch_delete_memories(memory_ids))

    def consolidate_memories(
        self, user_id: str, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Consolidate related memories into summaries.

        Args:
            user_id: User identifier
            session_id: Optional session filter

        Returns:
            List of consolidated memory groups
        """
        return cast(
            list[dict[str, Any]],
            self._backend.consolidate_memories(user_id=user_id, session_id=session_id),
        )

    def cleanup_expired_memories(self) -> int:
        """
        Remove expired memories.

        Returns:
            Number of memories deleted
        """
        return cast(int, self._backend.cleanup_expired_memories())

    def get_memory_analytics(self, user_id: str) -> dict[str, Any]:
        """
        Get analytics for user's memories.

        Args:
            user_id: User identifier

        Returns:
            Analytics dictionary with metrics
        """
        return cast(dict[str, Any], self._backend.get_memory_analytics(user_id))

    def health_check(self) -> dict[str, Any]:
        """
        Check backend health (remote mode only).

        Returns:
            Health status dictionary

        Raises:
            AttributeError: If called in local mode
        """
        if self.mode == "remote":
            return cast(dict[str, Any], self._backend.health_check())
        raise AttributeError("health_check() only available in remote mode")

    def compact_conversations(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        lookback_hours: int = 168,
        min_memories: int = 5,
        dry_run: bool = False,
        memory_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compact and consolidate conversation memories.

        This analyzes conversation memories, clusters them by topic/time,
        and creates concise summaries while preserving key facts.

        Args:
            user_id: User identifier
            session_id: Optional session filter
            lookback_hours: How far back to look (default 1 week)
            min_memories: Minimum memories to trigger compaction
            dry_run: If True, preview without making changes
            memory_types: Optional list of memory types to compact
                         (e.g., ["fact", "event", "context"])

        Returns:
            CompactionResult dictionary with:
            - metrics: Token counts, compression ratio, storage saved
            - insights: Human-readable insights about the compaction
            - preserved_facts: Key facts that were preserved
            - preserved_entities: Entities that were preserved
            - actions: Detailed list of actions taken
        """
        from hippocampai.consolidation.compactor import ConversationCompactor

        compactor = ConversationCompactor(qdrant_url=self._backend.qdrant_url)
        result = compactor.compact_conversations(
            user_id=user_id,
            session_id=session_id,
            lookback_hours=lookback_hours,
            min_memories=min_memories,
            dry_run=dry_run,
            memory_types=memory_types,
        )
        return result.to_dict()
