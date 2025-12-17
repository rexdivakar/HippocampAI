"""Plugin registry for managing and executing plugins."""

import logging
from typing import Any, Optional

from hippocampai.models.memory import Memory, RetrievalResult
from hippocampai.plugins.base import (
    BasePlugin,
    MemoryFilter,
    MemoryProcessor,
    MemoryRetriever,
    MemoryScorer,
    PluginContext,
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for managing HippocampAI plugins.

    The registry handles plugin lifecycle, execution ordering, and
    provides methods to run plugins at various pipeline stages.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_processor(SentimentProcessor())
        >>> registry.register_scorer(RecencyScorer(weight=0.2))
        >>>
        >>> # Process a memory through all registered processors
        >>> processed = registry.run_processors(memory, context)
        >>>
        >>> # Get combined scores from all scorers
        >>> score = registry.run_scorers(memory, query, context)
    """

    def __init__(self):
        self._processors: list[MemoryProcessor] = []
        self._scorers: list[MemoryScorer] = []
        self._retrievers: dict[str, MemoryRetriever] = {}
        self._filters: list[MemoryFilter] = []
        self._all_plugins: dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin based on its type."""
        if isinstance(plugin, MemoryProcessor):
            self.register_processor(plugin)
        elif isinstance(plugin, MemoryScorer):
            self.register_scorer(plugin)
        elif isinstance(plugin, MemoryRetriever):
            self.register_retriever(plugin)
        elif isinstance(plugin, MemoryFilter):
            self.register_filter(plugin)
        else:
            raise TypeError(f"Unknown plugin type: {type(plugin)}")

    def register_processor(self, processor: MemoryProcessor) -> None:
        """Register a memory processor."""
        if processor.name in self._all_plugins:
            logger.warning(f"Replacing existing plugin: {processor.name}")
            self.unregister(processor.name)

        processor.initialize()
        self._processors.append(processor)
        self._processors.sort(key=lambda p: p.priority)
        self._all_plugins[processor.name] = processor
        logger.info(f"Registered processor: {processor.name} (priority={processor.priority})")

    def register_scorer(self, scorer: MemoryScorer) -> None:
        """Register a memory scorer."""
        if scorer.name in self._all_plugins:
            logger.warning(f"Replacing existing plugin: {scorer.name}")
            self.unregister(scorer.name)

        scorer.initialize()
        self._scorers.append(scorer)
        self._scorers.sort(key=lambda s: s.priority)
        self._all_plugins[scorer.name] = scorer
        logger.info(f"Registered scorer: {scorer.name} (weight={scorer.weight})")

    def register_retriever(self, retriever: MemoryRetriever) -> None:
        """Register a memory retriever."""
        if retriever.name in self._all_plugins:
            logger.warning(f"Replacing existing plugin: {retriever.name}")
            self.unregister(retriever.name)

        retriever.initialize()
        self._retrievers[retriever.name] = retriever
        self._all_plugins[retriever.name] = retriever
        logger.info(f"Registered retriever: {retriever.name}")

    def register_filter(self, filter_plugin: MemoryFilter) -> None:
        """Register a memory filter."""
        if filter_plugin.name in self._all_plugins:
            logger.warning(f"Replacing existing plugin: {filter_plugin.name}")
            self.unregister(filter_plugin.name)

        filter_plugin.initialize()
        self._filters.append(filter_plugin)
        self._filters.sort(key=lambda f: f.priority)
        self._all_plugins[filter_plugin.name] = filter_plugin
        logger.info(f"Registered filter: {filter_plugin.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name not in self._all_plugins:
            return False

        plugin = self._all_plugins.pop(name)
        plugin.shutdown()

        if isinstance(plugin, MemoryProcessor):
            self._processors = [p for p in self._processors if p.name != name]
        elif isinstance(plugin, MemoryScorer):
            self._scorers = [s for s in self._scorers if s.name != name]
        elif isinstance(plugin, MemoryRetriever):
            self._retrievers.pop(name, None)
        elif isinstance(plugin, MemoryFilter):
            self._filters = [f for f in self._filters if f.name != name]

        logger.info(f"Unregistered plugin: {name}")
        return True

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self._all_plugins.get(name)

    def list_plugins(self) -> dict[str, list[str]]:
        """List all registered plugins by type."""
        return {
            "processors": [p.name for p in self._processors],
            "scorers": [s.name for s in self._scorers],
            "retrievers": list(self._retrievers.keys()),
            "filters": [f.name for f in self._filters],
        }

    # Execution methods

    def run_processors(
        self,
        memory: Memory,
        context: Optional[PluginContext] = None,
        stage: str = "pre_store",
    ) -> Memory:
        """Run all enabled processors on a memory.

        Args:
            memory: Memory to process
            context: Plugin context
            stage: Pipeline stage (pre_store, post_retrieve, etc.)

        Returns:
            Processed memory
        """
        for processor in self._processors:
            if not processor.enabled:
                continue
            try:
                memory = processor.process(memory, context)
            except Exception as e:
                logger.error(f"Processor {processor.name} failed: {e}")
                if processor.config.get("fail_fast", False):
                    raise
        return memory

    def run_processors_batch(
        self,
        memories: list[Memory],
        context: Optional[PluginContext] = None,
    ) -> list[Memory]:
        """Run all enabled processors on multiple memories."""
        for processor in self._processors:
            if not processor.enabled:
                continue
            try:
                memories = processor.process_batch(memories, context)
            except Exception as e:
                logger.error(f"Processor {processor.name} batch failed: {e}")
                if processor.config.get("fail_fast", False):
                    raise
        return memories

    def run_scorers(
        self,
        memory: Memory,
        query: str,
        context: Optional[PluginContext] = None,
        base_score: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        """Run all enabled scorers and combine scores.

        Args:
            memory: Memory to score
            query: Search query
            context: Plugin context
            base_score: Base score from default retrieval

        Returns:
            Tuple of (combined_score, breakdown_dict)
        """
        if not self._scorers:
            return base_score, {"base": base_score}

        breakdown = {"base": base_score}
        total_weight = 1.0  # Base score weight

        for scorer in self._scorers:
            if not scorer.enabled:
                continue
            try:
                score = scorer.score(memory, query, context)
                breakdown[scorer.name] = score
                total_weight += scorer.weight
            except Exception as e:
                logger.error(f"Scorer {scorer.name} failed: {e}")
                breakdown[scorer.name] = 0.0

        # Weighted average
        combined = base_score
        for scorer in self._scorers:
            if scorer.enabled and scorer.name in breakdown:
                combined += breakdown[scorer.name] * scorer.weight

        combined /= total_weight
        breakdown["combined"] = combined

        return combined, breakdown

    def run_retriever(
        self,
        name: str,
        query: str,
        context: PluginContext,
        k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Run a specific retriever by name."""
        retriever = self._retrievers.get(name)
        if not retriever:
            raise ValueError(f"Retriever not found: {name}")
        if not retriever.enabled:
            return []
        return retriever.retrieve(query, context, k, **kwargs)

    def run_filters(
        self,
        memories: list[Memory],
        context: Optional[PluginContext] = None,
    ) -> list[Memory]:
        """Run all enabled filters on memories."""
        for filter_plugin in self._filters:
            if not filter_plugin.enabled:
                continue
            try:
                memories = filter_plugin.filter_batch(memories, context)
            except Exception as e:
                logger.error(f"Filter {filter_plugin.name} failed: {e}")
                if filter_plugin.config.get("fail_fast", False):
                    raise
        return memories

    def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        for plugin in self._all_plugins.values():
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {plugin.name}: {e}")
        self._processors.clear()
        self._scorers.clear()
        self._retrievers.clear()
        self._filters.clear()
        self._all_plugins.clear()


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry
