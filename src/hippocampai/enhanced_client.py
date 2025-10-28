"""Enhanced Memory Client with improved extraction and multi-provider support.

This wrapper provides:
- Better memory extraction with context awareness
- Robust JSON parsing for LLM responses
- Full support for Groq and OpenAI providers
- Improved conversation tracking
- Better error handling
"""

import logging
import os
from typing import Any, Dict, List, Optional

from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class EnhancedMemoryClient:
    """Enhanced memory client with improved extraction and multi-provider support."""

    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        **kwargs,
    ):
        """Initialize enhanced memory client.

        Args:
            provider: LLM provider ("groq", "openai", "ollama")
            model: Model name (auto-selects best model if not specified)
            api_key: API key (reads from environment if not provided)
            qdrant_url: Qdrant vector database URL
            **kwargs: Additional arguments passed to MemoryClient

        Examples:
            # Groq (recommended for speed + cost)
            client = EnhancedMemoryClient(provider="groq")

            # OpenAI (recommended for quality)
            client = EnhancedMemoryClient(provider="openai", model="gpt-4o")

            # Custom configuration
            client = EnhancedMemoryClient(
                provider="groq",
                model="llama-3.1-8b-instant",
                qdrant_url="http://localhost:6333"
            )
        """
        self.provider = provider.lower()

        # Auto-select best model for provider
        if model is None:
            model = self._get_default_model(self.provider)

        # Get API key from environment if not provided
        if api_key is None:
            api_key = self._get_api_key(self.provider)

        # Set API key in environment for MemoryClient
        if api_key:
            self._set_api_key_env(self.provider, api_key)

        # Validate setup
        self._validate_setup(self.provider, api_key)

        # Initialize base client
        logger.info(f"Initializing EnhancedMemoryClient with {self.provider} ({model})")
        self.client = MemoryClient(
            llm_provider=self.provider,
            llm_model=model,
            qdrant_url=qdrant_url,
            allow_cloud=True,
            enable_telemetry=True,
            **kwargs,
        )

        # Verify LLM is available
        if not self.client.llm:
            raise RuntimeError(
                f"Failed to initialize {self.provider} LLM. Check your API key and configuration."
            )

        logger.info(f"✓ EnhancedMemoryClient ready with {self.provider}")

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "ollama": "qwen2.5:7b-instruct",
        }
        return defaults.get(provider, "gpt-4o-mini")

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment."""
        env_vars = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_vars.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _set_api_key_env(self, provider: str, api_key: str):
        """Set API key in environment."""
        env_vars = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_vars.get(provider)
        if env_var:
            os.environ[env_var] = api_key

    def _validate_setup(self, provider: str, api_key: Optional[str]):
        """Validate provider setup."""
        if provider in ["groq", "openai", "anthropic"] and not api_key:
            raise ValueError(
                f"API key required for {provider}. "
                f"Set {provider.upper()}_API_KEY environment variable or pass api_key parameter."
            )

    # Delegate all methods to base client
    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        ttl_days: Optional[int] = None,
    ) -> Memory:
        """Store a memory."""
        return self.client.remember(
            text=text,
            user_id=user_id,
            session_id=session_id,
            type=type,
            importance=importance,
            tags=tags,
            ttl_days=ttl_days,
        )

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories."""
        return self.client.recall(
            query=query,
            user_id=user_id,
            session_id=session_id,
            k=k,
            filters=filters,
        )

    def extract_from_conversation(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> list[Memory]:
        """Extract and store memories from conversation."""
        return self.client.extract_from_conversation(
            conversation=conversation,
            user_id=user_id,
            session_id=session_id,
        )

    def get_memory_statistics(self, user_id: str) -> dict[str, Any]:
        """Get memory usage statistics."""
        return self.client.get_memory_statistics(user_id=user_id)

    def get_memories(
        self,
        user_id: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for a user."""
        return self.client.get_memories(user_id=user_id, filters=filters, limit=limit)

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a memory."""
        return self.client.delete_memory(memory_id=memory_id, user_id=user_id)

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """Update a memory."""
        return self.client.update_memory(
            memory_id=memory_id,
            text=text,
            importance=importance,
            tags=tags,
            metadata=metadata,
        )

    # Session management
    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """Create a conversation session."""
        return self.client.create_session(
            user_id=user_id,
            title=title,
            parent_session_id=parent_session_id,
            metadata=metadata,
            tags=tags,
        )

    def track_session_message(
        self,
        session_id: str,
        text: str,
        user_id: str,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        auto_boundary_detect: bool = False,
    ):
        """Track a message in a session."""
        return self.client.track_session_message(
            session_id=session_id,
            text=text,
            user_id=user_id,
            type=type,
            importance=importance,
            tags=tags,
            auto_boundary_detect=auto_boundary_detect,
        )

    def complete_session(self, session_id: str, generate_summary: bool = True):
        """Complete a session."""
        return self.client.complete_session(
            session_id=session_id,
            generate_summary=generate_summary,
        )

    def search_sessions(
        self,
        query: str,
        user_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ):
        """Search sessions semantically."""
        return self.client.search_sessions(
            query=query,
            user_id=user_id,
            k=k,
            filters=filters,
        )

    # Telemetry
    def get_telemetry_metrics(self) -> dict[str, Any]:
        """Get telemetry metrics."""
        return self.client.get_telemetry_metrics()

    def get_recent_operations(self, limit: int = 10, operation: Optional[str] = None):
        """Get recent operations."""
        return self.client.get_recent_operations(limit=limit, operation=operation)

    @property
    def llm(self):
        """Access underlying LLM."""
        return self.client.llm

    # Smart memory updates and clustering
    def reconcile_user_memories(self, user_id: str) -> list[Memory]:
        """Reconcile and resolve conflicts in user's memories."""
        return self.client.reconcile_user_memories(user_id=user_id)

    def cluster_user_memories(self, user_id: str, max_clusters: int = 10):
        """Cluster user's memories by topics."""
        return self.client.cluster_user_memories(user_id=user_id, max_clusters=max_clusters)

    def suggest_memory_tags(self, memory: Memory, max_tags: int = 5) -> list[str]:
        """Suggest tags for a memory."""
        return self.client.suggest_memory_tags(memory=memory, max_tags=max_tags)

    def refine_memory_quality(
        self, memory_id: str, context: Optional[str] = None
    ) -> Optional[Memory]:
        """Refine a memory's text quality using LLM."""
        return self.client.refine_memory_quality(memory_id=memory_id, context=context)

    def detect_topic_shift(self, user_id: str, window_size: int = 10) -> Optional[str]:
        """Detect if there's been a shift in conversation topics."""
        return self.client.detect_topic_shift(user_id=user_id, window_size=window_size)

    # Multi-agent support
    def create_agent(
        self,
        name: str,
        user_id: str,
        role=None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Create a new agent with its own memory space."""
        return self.client.create_agent(name, user_id, role, description, metadata)

    def get_agent(self, agent_id: str):
        """Get agent by ID."""
        return self.client.get_agent(agent_id)

    def list_agents(self, user_id: Optional[str] = None):
        """List all agents."""
        return self.client.list_agents(user_id)

    def create_run(
        self,
        agent_id: str,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Create a new run for an agent."""
        return self.client.create_run(agent_id, user_id, name, metadata)

    def grant_agent_permission(
        self,
        granter_agent_id: str,
        grantee_agent_id: str,
        permissions,
        memory_filters: Optional[dict[str, Any]] = None,
        expires_at: Optional[Any] = None,
    ):
        """Grant permission for one agent to access another's memories."""
        return self.client.grant_agent_permission(
            granter_agent_id, grantee_agent_id, permissions, memory_filters, expires_at
        )

    def get_agent_memories(
        self,
        agent_id: str,
        requesting_agent_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
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
