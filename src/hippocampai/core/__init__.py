"""HippocampAI Core Library.

This module contains the core memory engine functionality that works
independently of any SaaS infrastructure. It requires only:
- Qdrant (vector database)
- An LLM provider (Ollama, OpenAI, Groq, Anthropic)

The core library provides:
- Memory storage and retrieval (remember/recall)
- Hybrid search (vector + BM25 + reranking)
- Memory types (facts, preferences, goals, habits, events)
- Session management
- Multi-agent support
- Temporal reasoning
- Graph relationships
- Version control and audit trails

For SaaS features (API server, auth, background tasks, Celery),
use `hippocampai.platform`.

Example:
    >>> from hippocampai.core import MemoryClient
    >>> client = MemoryClient()
    >>> memory = client.remember("I love coffee", user_id="alice")
    >>> results = client.recall("beverages", user_id="alice")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hippocampai.graph import MemoryGraph, RelationType
from hippocampai.models.agent import (
    Agent,
    AgentPermission,
    AgentRole,
    MemoryVisibility,
    PermissionType,
    Run,
)
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.models.session import (
    Entity,
    Session,
    SessionFact,
    SessionSearchResult,
    SessionStatus,
)
from hippocampai.pipeline.insights import (
    BehaviorChange,
    HabitScore,
    Pattern,
    PreferenceDrift,
    Trend,
)
from hippocampai.pipeline.insights import (
    ChangeType as InsightChangeType,
)
from hippocampai.pipeline.temporal import ScheduledMemory, TemporalEvent, Timeline, TimeRange
from hippocampai.storage import MemoryKVStore
from hippocampai.utils.context_injection import ContextInjector, inject_context
from hippocampai.versioning import AuditEntry, ChangeType, MemoryVersion, MemoryVersionControl

__version__ = "0.3.0"

__all__ = [
    # Main client
    "MemoryClient",
    "AsyncMemoryClient",
    "UnifiedMemoryClient",
    "EnhancedMemoryClient",
    "OptimizedMemoryClient",
    # Configuration
    "Config",
    "get_config",
    # Core models
    "Memory",
    "MemoryType",
    "RetrievalResult",
    # Session management
    "Session",
    "SessionStatus",
    "SessionSearchResult",
    "SessionFact",
    "Entity",
    "SessionManager",
    # Multi-agent support
    "Agent",
    "AgentRole",
    "Run",
    "AgentPermission",
    "PermissionType",
    "MemoryVisibility",
    "MultiAgentManager",
    # Temporal reasoning
    "TimeRange",
    "ScheduledMemory",
    "Timeline",
    "TemporalEvent",
    # Cross-session insights
    "Pattern",
    "BehaviorChange",
    "PreferenceDrift",
    "HabitScore",
    "Trend",
    "InsightChangeType",
    # Graph
    "MemoryGraph",
    "RelationType",
    # Versioning
    "MemoryVersionControl",
    "MemoryVersion",
    "AuditEntry",
    "ChangeType",
    # Storage
    "MemoryKVStore",
    # Utils
    "ContextInjector",
    "inject_context",
    # Telemetry
    "get_telemetry",
    "OperationType",
    # Simplified API
    "SimpleMemory",
    "SimpleSession",
    "MemoryStore",
    "MemoryManager",
]

if TYPE_CHECKING:
    from hippocampai.async_client import AsyncMemoryClient
    from hippocampai.client import MemoryClient
    from hippocampai.config import Config, get_config
    from hippocampai.enhanced_client import EnhancedMemoryClient
    from hippocampai.multiagent import MultiAgentManager
    from hippocampai.optimized_client import OptimizedMemoryClient
    from hippocampai.session import SessionManager
    from hippocampai.telemetry import OperationType, get_telemetry
    from hippocampai.unified_client import UnifiedMemoryClient


_lazy_imports = {
    "MemoryClient": "hippocampai.client",
    "AsyncMemoryClient": "hippocampai.async_client",
    "UnifiedMemoryClient": "hippocampai.unified_client",
    "EnhancedMemoryClient": "hippocampai.enhanced_client",
    "OptimizedMemoryClient": "hippocampai.optimized_client",
    "Config": "hippocampai.config",
    "get_config": "hippocampai.config",
    "SessionManager": "hippocampai.session",
    "MultiAgentManager": "hippocampai.multiagent",
    "get_telemetry": "hippocampai.telemetry",
    "OperationType": "hippocampai.telemetry",
}

_simple_imports = {
    "SimpleMemory": ("hippocampai.simple", "Memory"),
    "SimpleSession": ("hippocampai.simple", "Session"),
    "MemoryStore": ("hippocampai.simple", "MemoryStore"),
    "MemoryManager": ("hippocampai.simple", "MemoryManager"),
}


def __getattr__(name: str):
    """Lazy loading for heavy imports."""
    import importlib
    
    if name in _lazy_imports:
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    
    if name in _simple_imports:
        module_name, attr_name = _simple_imports[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    
    raise AttributeError(f"module 'hippocampai.core' has no attribute {name!r}")
