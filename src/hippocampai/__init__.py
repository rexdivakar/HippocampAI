"""HippocampAI: Production-ready long-term memory engine with hybrid retrieval."""

from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, MemoryType

__version__ = "0.1.0"
__all__ = ["MemoryClient", "Memory", "MemoryType"]
