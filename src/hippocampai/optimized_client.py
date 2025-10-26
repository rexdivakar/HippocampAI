"""Optimized Memory Client with async support and performance improvements.

This client provides:
- Async/await support for all I/O operations
- Batch operations for better performance
- LRU caching for frequently accessed data
- Parallel processing where possible
- Optimized imports and reduced overhead
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class OptimizedMemoryClient:
    """Optimized memory client with async support and caching."""

    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        enable_caching: bool = True,
        cache_size: int = 128,
        **kwargs,
    ):
        """Initialize optimized memory client.

        Args:
            provider: LLM provider ("groq", "openai", "ollama")
            model: Model name (auto-selected if None)
            api_key: API key (reads from environment if None)
            qdrant_url: Qdrant vector database URL
            enable_caching: Enable LRU caching for recall operations
            cache_size: Size of LRU cache
            **kwargs: Additional arguments for MemoryClient
        """
        self.provider = provider
        self.enable_caching = enable_caching

        # Auto-select model if not provided
        if model is None:
            model = self._get_default_model(provider)

        # Get API key from environment if needed
        if api_key is None:
            api_key = self._get_api_key(provider)

        # Set API key in environment
        if api_key:
            self._set_api_key_env(provider, api_key)

        # Initialize base client
        logger.info(f"Initializing OptimizedMemoryClient with {provider}")
        self.client = MemoryClient(
            llm_provider=provider,
            llm_model=model,
            qdrant_url=qdrant_url,
            allow_cloud=True,
            enable_telemetry=True,
            **kwargs,
        )

        if not self.client.llm:
            raise RuntimeError(f"Failed to initialize {provider} LLM")

        # Setup caching if enabled
        if enable_caching:
            self._setup_cache(cache_size)

        logger.info(f"✓ OptimizedMemoryClient ready (caching: {enable_caching})")

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_default_model(provider: str) -> str:
        """Get default model (cached)."""
        return {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "ollama": "qwen2.5:7b-instruct",
        }.get(provider, "gpt-4o-mini")

    @staticmethod
    def _get_api_key(provider: str) -> Optional[str]:
        """Get API key from environment."""
        import os

        env_vars = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}
        env_var = env_vars.get(provider)
        return os.getenv(env_var) if env_var else None

    @staticmethod
    def _set_api_key_env(provider: str, api_key: str):
        """Set API key in environment."""
        import os

        env_vars = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}
        env_var = env_vars.get(provider)
        if env_var:
            os.environ[env_var] = api_key

    def _setup_cache(self, cache_size: int):
        """Setup LRU cache for recall operations."""
        # Wrap recall method with caching
        original_recall = self.client.recall

        @lru_cache(maxsize=cache_size)
        def cached_recall(query: str, user_id: str, k: int = 5):
            return original_recall(query=query, user_id=user_id, k=k)

        self._cached_recall = cached_recall

    # Synchronous methods (delegate to base client)
    def remember(self, text: str, user_id: str, **kwargs) -> Memory:
        """Store a memory (sync)."""
        return self.client.remember(text=text, user_id=user_id, **kwargs)

    def recall(self, query: str, user_id: str, k: int = 5, **kwargs) -> List[RetrievalResult]:
        """Retrieve relevant memories (sync, with optional caching)."""
        if self.enable_caching and not kwargs:  # Only cache simple queries
            return self._cached_recall(query, user_id, k)
        return self.client.recall(query=query, user_id=user_id, k=k, **kwargs)

    def extract_from_conversation(self, conversation: str, user_id: str, **kwargs) -> List[Memory]:
        """Extract memories from conversation (sync)."""
        return self.client.extract_from_conversation(
            conversation=conversation, user_id=user_id, **kwargs
        )

    # Async methods for I/O-bound operations
    async def remember_async(self, text: str, user_id: str, **kwargs) -> Memory:
        """Store a memory (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.remember(text=text, user_id=user_id, **kwargs)
        )

    async def recall_async(
        self, query: str, user_id: str, k: int = 5, **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant memories (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.recall(query=query, user_id=user_id, k=k, **kwargs)
        )

    async def extract_from_conversation_async(
        self, conversation: str, user_id: str, **kwargs
    ) -> List[Memory]:
        """Extract memories from conversation (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.extract_from_conversation(
                conversation=conversation, user_id=user_id, **kwargs
            ),
        )

    # Batch operations for performance
    async def remember_batch_async(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[Memory]:
        """Store multiple memories in parallel (async).

        Args:
            memories: List of memory dicts with 'text' and optional params
            user_id: User ID

        Returns:
            List of created Memory objects

        Example:
            memories = [
                {"text": "I love Python", "type": "preference"},
                {"text": "I work at Google", "type": "fact"},
            ]
            results = await client.remember_batch_async(memories, "user123")
        """
        tasks = [
            self.remember_async(
                text=mem["text"], user_id=user_id, **{k: v for k, v in mem.items() if k != "text"}
            )
            for mem in memories
        ]
        return await asyncio.gather(*tasks)

    def remember_batch(self, memories: List[Dict[str, Any]], user_id: str) -> List[Memory]:
        """Store multiple memories (sync wrapper for async batch)."""
        return asyncio.run(self.remember_batch_async(memories, user_id))

    async def recall_batch_async(
        self, queries: List[str], user_id: str, k: int = 5
    ) -> List[List[RetrievalResult]]:
        """Recall multiple queries in parallel (async).

        Args:
            queries: List of query strings
            user_id: User ID
            k: Results per query

        Returns:
            List of result lists (one per query)
        """
        tasks = [self.recall_async(query=q, user_id=user_id, k=k) for q in queries]
        return await asyncio.gather(*tasks)

    def recall_batch(
        self, queries: List[str], user_id: str, k: int = 5
    ) -> List[List[RetrievalResult]]:
        """Recall multiple queries (sync wrapper)."""
        return asyncio.run(self.recall_batch_async(queries, user_id, k))

    # Optimized conversation processing
    async def process_conversation_async(
        self, user_message: str, user_id: str, k: int = 5
    ) -> tuple:
        """Process a conversation turn (recall + extract) in parallel.

        Args:
            user_message: User's message
            user_id: User ID
            k: Number of memories to recall

        Returns:
            tuple: (recall_results, extracted_memories)
        """
        # Run recall and extraction in parallel
        recall_task = self.recall_async(user_message, user_id, k)
        extract_task = self.extract_from_conversation_async(f"User: {user_message}", user_id)

        recall_results, extracted = await asyncio.gather(recall_task, extract_task)
        return recall_results, extracted

    # Utility methods
    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics (sync)."""
        return self.client.get_memory_statistics(user_id=user_id)

    def get_memories(self, user_id: str, limit: int = 100, **kwargs) -> List[Memory]:
        """Get all memories (sync)."""
        return self.client.get_memories(user_id=user_id, limit=limit, **kwargs)

    async def get_memories_async(self, user_id: str, limit: int = 100, **kwargs) -> List[Memory]:
        """Get all memories (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.get_memories(user_id=user_id, limit=limit, **kwargs)
        )

    # Session management (delegate to base client)
    def create_session(self, user_id: str, **kwargs):
        """Create a session (sync)."""
        return self.client.create_session(user_id=user_id, **kwargs)

    def track_session_message(self, session_id: str, text: str, user_id: str, **kwargs):
        """Track a session message (sync)."""
        return self.client.track_session_message(
            session_id=session_id, text=text, user_id=user_id, **kwargs
        )

    def complete_session(self, session_id: str, **kwargs):
        """Complete a session (sync)."""
        return self.client.complete_session(session_id=session_id, **kwargs)

    # Cache management
    def clear_cache(self):
        """Clear the recall cache."""
        if hasattr(self, "_cached_recall"):
            self._cached_recall.cache_clear()
            logger.info("Recall cache cleared")

    def get_cache_info(self):
        """Get cache statistics."""
        if hasattr(self, "_cached_recall"):
            return self._cached_recall.cache_info()
        return None

    @property
    def llm(self):
        """Access underlying LLM."""
        return self.client.llm

    # Smart memory updates and clustering
    def reconcile_user_memories(self, user_id: str):
        """Reconcile and resolve conflicts in user's memories."""
        return self.client.reconcile_user_memories(user_id=user_id)

    def cluster_user_memories(self, user_id: str, max_clusters: int = 10):
        """Cluster user's memories by topics."""
        return self.client.cluster_user_memories(user_id=user_id, max_clusters=max_clusters)

    def suggest_memory_tags(self, memory, max_tags: int = 5):
        """Suggest tags for a memory."""
        return self.client.suggest_memory_tags(memory=memory, max_tags=max_tags)

    def refine_memory_quality(self, memory_id: str, context: Optional[str] = None):
        """Refine a memory's text quality using LLM."""
        return self.client.refine_memory_quality(memory_id=memory_id, context=context)

    def detect_topic_shift(self, user_id: str, window_size: int = 10):
        """Detect if there's been a shift in conversation topics."""
        return self.client.detect_topic_shift(user_id=user_id, window_size=window_size)

    # Multi-agent support
    def create_agent(self, name: str, user_id: str, role=None, description=None, metadata=None):
        """Create a new agent with its own memory space."""
        return self.client.create_agent(name, user_id, role, description, metadata)

    def get_agent(self, agent_id: str):
        """Get agent by ID."""
        return self.client.get_agent(agent_id)

    def list_agents(self, user_id=None):
        """List all agents."""
        return self.client.list_agents(user_id)

    def create_run(self, agent_id: str, user_id: str, name=None, metadata=None):
        """Create a new run for an agent."""
        return self.client.create_run(agent_id, user_id, name, metadata)

    def grant_agent_permission(
        self,
        granter_agent_id: str,
        grantee_agent_id: str,
        permissions,
        memory_filters=None,
        expires_at=None,
    ):
        """Grant permission for one agent to access another's memories."""
        return self.client.grant_agent_permission(
            granter_agent_id, grantee_agent_id, permissions, memory_filters, expires_at
        )

    def get_agent_memories(
        self, agent_id: str, requesting_agent_id=None, filters=None, limit: int = 100
    ):
        """Get memories for an agent, respecting permissions."""
        return self.client.get_agent_memories(agent_id, requesting_agent_id, filters, limit)

    def transfer_memory(
        self,
        memory_id: str,
        source_agent_id: str,
        target_agent_id: str,
        transfer_type: str = "copy",
    ):
        """Transfer a memory from one agent to another."""
        return self.client.transfer_memory(
            memory_id, source_agent_id, target_agent_id, transfer_type
        )
