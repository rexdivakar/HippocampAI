"""Base classes for HippocampAI plugin system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from hippocampai.models.memory import Memory, RetrievalResult


class PluginPriority(IntEnum):
    """Plugin execution priority (lower = earlier)."""

    FIRST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LAST = 100


@dataclass
class PluginContext:
    """Context passed to plugins during execution."""

    user_id: str
    session_id: Optional[str] = None
    namespace: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata or config."""
        return self.metadata.get(key, self.config.get(key, default))


class BasePlugin(ABC):
    """Base class for all plugins."""

    name: str = "base_plugin"
    version: str = "1.0.0"
    priority: PluginPriority = PluginPriority.NORMAL
    enabled: bool = True

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}

    def initialize(self) -> None:
        """Called when plugin is registered. Override for setup."""
        pass

    def shutdown(self) -> None:
        """Called when plugin is unregistered. Override for cleanup."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, priority={self.priority})>"


class MemoryProcessor(BasePlugin):
    """Plugin for processing memories before storage or after retrieval.

    Processors can modify memory content, add metadata, enrich with
    external data, or perform validation.

    Example:
        >>> class TranslationProcessor(MemoryProcessor):
        ...     name = "translator"
        ...     def process(self, memory, context=None):
        ...         if context and context.get("target_language"):
        ...             memory.metadata["translated"] = translate(
        ...                 memory.text, context.get("target_language")
        ...             )
        ...         return memory
    """

    @abstractmethod
    def process(self, memory: Memory, context: Optional[PluginContext] = None) -> Memory:
        """Process a memory and return the modified version.

        Args:
            memory: The memory to process
            context: Optional context with user/session info

        Returns:
            The processed memory (can be the same object, modified)
        """
        pass

    def process_batch(
        self, memories: list[Memory], context: Optional[PluginContext] = None
    ) -> list[Memory]:
        """Process multiple memories. Override for batch optimization."""
        return [self.process(m, context) for m in memories]


class MemoryScorer(BasePlugin):
    """Plugin for custom memory scoring.

    Scorers contribute to the final relevance score during retrieval.
    Multiple scorers can be combined with configurable weights.

    Example:
        >>> class RecencyScorer(MemoryScorer):
        ...     name = "recency"
        ...     weight = 0.2
        ...     def score(self, memory, query, context=None):
        ...         age_days = (datetime.now() - memory.created_at).days
        ...         return max(0, 1 - age_days / 365)
    """

    weight: float = 1.0  # Weight in final score fusion

    @abstractmethod
    def score(self, memory: Memory, query: str, context: Optional[PluginContext] = None) -> float:
        """Score a memory's relevance to a query.

        Args:
            memory: The memory to score
            query: The search query
            context: Optional context

        Returns:
            Score between 0.0 and 1.0
        """
        pass

    def score_batch(
        self,
        memories: list[Memory],
        query: str,
        context: Optional[PluginContext] = None,
    ) -> list[float]:
        """Score multiple memories. Override for batch optimization."""
        return [self.score(m, query, context) for m in memories]


class MemoryRetriever(BasePlugin):
    """Plugin for custom retrieval strategies.

    Retrievers can implement alternative search methods (graph traversal,
    temporal queries, etc.) that complement the default hybrid search.

    Example:
        >>> class GraphRetriever(MemoryRetriever):
        ...     name = "graph"
        ...     def retrieve(self, query, context, k=10):
        ...         # Find related memories via graph relationships
        ...         return graph.find_related(query, k=k)
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        context: PluginContext,
        k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query.

        Args:
            query: Search query
            context: Context with user/session info
            k: Number of results to return
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of retrieval results
        """
        pass


class MemoryFilter(BasePlugin):
    """Plugin for filtering memories during retrieval.

    Filters can exclude memories based on custom criteria (permissions,
    content policies, user preferences, etc.).

    Example:
        >>> class ContentPolicyFilter(MemoryFilter):
        ...     name = "content_policy"
        ...     def filter(self, memory, context=None):
        ...         return not contains_pii(memory.text)
    """

    @abstractmethod
    def filter(self, memory: Memory, context: Optional[PluginContext] = None) -> bool:
        """Determine if a memory should be included.

        Args:
            memory: The memory to check
            context: Optional context

        Returns:
            True to include, False to exclude
        """
        pass

    def filter_batch(
        self, memories: list[Memory], context: Optional[PluginContext] = None
    ) -> list[Memory]:
        """Filter multiple memories. Override for batch optimization."""
        return [m for m in memories if self.filter(m, context)]
