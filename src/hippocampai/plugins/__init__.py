"""HippocampAI Plugin System.

Allows custom memory processors, scorers, and retrievers to be registered
and used throughout the memory pipeline.

Example:
    >>> from hippocampai.plugins import PluginRegistry, MemoryProcessor
    >>>
    >>> class SentimentProcessor(MemoryProcessor):
    ...     name = "sentiment"
    ...     def process(self, memory, context=None):
    ...         memory.metadata["sentiment"] = analyze_sentiment(memory.text)
    ...         return memory
    >>>
    >>> registry = PluginRegistry()
    >>> registry.register_processor(SentimentProcessor())
"""

from hippocampai.plugins.base import (
    BasePlugin,
    MemoryFilter,
    MemoryProcessor,
    MemoryRetriever,
    MemoryScorer,
    PluginContext,
    PluginPriority,
)
from hippocampai.plugins.registry import PluginRegistry

__all__ = [
    "BasePlugin",
    "MemoryProcessor",
    "MemoryScorer",
    "MemoryRetriever",
    "MemoryFilter",
    "PluginContext",
    "PluginPriority",
    "PluginRegistry",
]
