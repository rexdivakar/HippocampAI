"""HippocampAI: Production-ready long-term memory engine with hybrid retrieval.

HippocampAI is an enterprise-grade memory engine that transforms how AI systems
remember, reason, and learn from interactions. It provides persistent, intelligent
memory capabilities that enable AI agents to maintain context across sessions.

Package Structure:
    hippocampai          - Main package (backward compatible, includes everything)
    hippocampai.core     - Core library (memory engine, no SaaS dependencies)
    hippocampai.platform - SaaS platform (API, auth, Celery, monitoring)

Quick Start:
    >>> from hippocampai import MemoryClient
    >>> client = MemoryClient()
    >>> memory = client.remember("I love coffee", user_id="alice")
    >>> results = client.recall("beverages", user_id="alice")

For core library only (no SaaS dependencies):
    >>> from hippocampai.core import MemoryClient

For SaaS platform features:
    >>> from hippocampai.platform import run_api_server, AutomationController

Installation:
    pip install hippocampai           # Core library only
    pip install hippocampai[saas]     # With SaaS features
    pip install hippocampai[all]      # Everything
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Core models (always available, no heavy dependencies)
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

# Pipeline components (insights, temporal)
from hippocampai.pipeline.insights import (
    BehaviorChange,
    HabitScore,
    Pattern,
    PreferenceDrift,
    Trend,
)
from hippocampai.pipeline.insights import ChangeType as InsightChangeType
from hippocampai.pipeline.temporal import ScheduledMemory, TemporalEvent, Timeline, TimeRange

__version__ = "0.3.0"

__all__ = [
    # Version
    "__version__",
    # Submodules
    "core",
    "platform",
    "plugins",
    "namespaces",
    "portability",
    "offline",
    "tiered",
    "integrations",
    # Main clients
    "MemoryClient",
    "UnifiedMemoryClient",
    "EnhancedMemoryClient",
    "OptimizedMemoryClient",
    "AsyncMemoryClient",
    # Core models
    "Memory",
    "MemoryType",
    "RetrievalResult",
    # Simplified API (mem0/zep compatible)
    "SimpleMemory",
    "SimpleSession",
    "MemoryStore",
    "MemoryManager",
    # Configuration
    "get_config",
    "Config",
    # Telemetry
    "get_telemetry",
    "OperationType",
    # Session management
    "Session",
    "SessionStatus",
    "SessionSearchResult",
    "SessionFact",
    "Entity",
    "SessionManager",
    # Advanced features
    "MemoryGraph",
    "RelationType",
    "MemoryKVStore",
    "MemoryVersionControl",
    "MemoryVersion",
    "AuditEntry",
    "ChangeType",
    "ContextInjector",
    "inject_context",
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
    # SaaS automation (platform features)
    "AutomationController",
    "AutomationPolicy",
    "AutomationSchedule",
    "PolicyType",
    "TaskManager",
    "TaskPriority",
    "TaskStatus",
    "BackgroundTask",
]

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from hippocampai import core as core
    from hippocampai import integrations as integrations
    from hippocampai import namespaces as namespaces
    from hippocampai import offline as offline
    from hippocampai import platform as platform
    from hippocampai import plugins as plugins
    from hippocampai import portability as portability
    from hippocampai import tiered as tiered
    from hippocampai.async_client import AsyncMemoryClient as AsyncMemoryClient
    from hippocampai.client import MemoryClient as MemoryClient
    from hippocampai.config import Config as Config
    from hippocampai.config import get_config as get_config
    from hippocampai.enhanced_client import EnhancedMemoryClient as EnhancedMemoryClient
    from hippocampai.graph import MemoryGraph as MemoryGraph
    from hippocampai.graph import RelationType as RelationType
    from hippocampai.multiagent import MultiAgentManager as MultiAgentManager
    from hippocampai.optimized_client import OptimizedMemoryClient as OptimizedMemoryClient
    from hippocampai.session import SessionManager as SessionManager
    from hippocampai.storage import MemoryKVStore as MemoryKVStore
    from hippocampai.telemetry import OperationType as OperationType
    from hippocampai.telemetry import get_telemetry as get_telemetry
    from hippocampai.unified_client import UnifiedMemoryClient as UnifiedMemoryClient
    from hippocampai.utils.context_injection import ContextInjector as ContextInjector
    from hippocampai.utils.context_injection import inject_context as inject_context
    from hippocampai.versioning import AuditEntry as AuditEntry
    from hippocampai.versioning import ChangeType as ChangeType
    from hippocampai.versioning import MemoryVersion as MemoryVersion
    from hippocampai.versioning import MemoryVersionControl as MemoryVersionControl


# Lazy loading mappings for reducing cognitive complexity
_SUBMODULE_MAP: dict[str, str] = {
    "core": "hippocampai.core",
    "platform": "hippocampai.platform",
    "plugins": "hippocampai.plugins",
    "namespaces": "hippocampai.namespaces",
    "portability": "hippocampai.portability",
    "offline": "hippocampai.offline",
    "tiered": "hippocampai.tiered",
    "integrations": "hippocampai.integrations",
}

_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # Configuration
    "Config": ("hippocampai.config", "Config"),
    "get_config": ("hippocampai.config", "get_config"),
    # Telemetry
    "get_telemetry": ("hippocampai.telemetry", "get_telemetry"),
    "OperationType": ("hippocampai.telemetry", "OperationType"),
    # Session management
    "SessionManager": ("hippocampai.session", "SessionManager"),
    # Graph
    "MemoryGraph": ("hippocampai.graph", "MemoryGraph"),
    "RelationType": ("hippocampai.graph", "RelationType"),
    # Storage
    "MemoryKVStore": ("hippocampai.storage", "MemoryKVStore"),
    # Versioning
    "MemoryVersionControl": ("hippocampai.versioning", "MemoryVersionControl"),
    "MemoryVersion": ("hippocampai.versioning", "MemoryVersion"),
    "AuditEntry": ("hippocampai.versioning", "AuditEntry"),
    "ChangeType": ("hippocampai.versioning", "ChangeType"),
    # Context injection
    "ContextInjector": ("hippocampai.utils.context_injection", "ContextInjector"),
    "inject_context": ("hippocampai.utils.context_injection", "inject_context"),
    # Multi-agent
    "MultiAgentManager": ("hippocampai.multiagent", "MultiAgentManager"),
    # SaaS automation
    "AutomationController": ("hippocampai.saas.automation", "AutomationController"),
    "AutomationPolicy": ("hippocampai.saas.automation", "AutomationPolicy"),
    "AutomationSchedule": ("hippocampai.saas.automation", "AutomationSchedule"),
    "PolicyType": ("hippocampai.saas.automation", "PolicyType"),
    "TaskManager": ("hippocampai.saas.tasks", "TaskManager"),
    "TaskPriority": ("hippocampai.saas.tasks", "TaskPriority"),
    "TaskStatus": ("hippocampai.saas.tasks", "TaskStatus"),
    "BackgroundTask": ("hippocampai.saas.tasks", "BackgroundTask"),
    # Simplified API (mem0/zep compatible)
    "MemoryStore": ("hippocampai.simple", "MemoryStore"),
    "MemoryManager": ("hippocampai.simple", "MemoryManager"),
    # Unified client
    "UnifiedMemoryClient": ("hippocampai.unified_client", "UnifiedMemoryClient"),
}

# Special imports with renamed exports
_RENAMED_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "SimpleMemory": ("hippocampai.simple", "Memory"),
    "SimpleSession": ("hippocampai.simple", "Session"),
}

# Clients that require optional dependencies
_OPTIONAL_CLIENT_MAP: dict[str, tuple[str, str]] = {
    "MemoryClient": ("hippocampai.client", "MemoryClient"),
    "EnhancedMemoryClient": ("hippocampai.enhanced_client", "EnhancedMemoryClient"),
    "OptimizedMemoryClient": ("hippocampai.optimized_client", "OptimizedMemoryClient"),
    "AsyncMemoryClient": ("hippocampai.async_client", "AsyncMemoryClient"),
}


def _import_optional_client(name: str, module_path: str, class_name: str) -> Any:
    """Import a client that requires optional dependencies."""
    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"hippocampai.{name} requires optional dependencies. "
            "Install HippocampAI with: pip install hippocampai"
        ) from exc


def __getattr__(name: str) -> Any:
    """Lazy loading for heavy imports to improve startup time."""
    import importlib

    # Submodules
    if name in _SUBMODULE_MAP:
        return importlib.import_module(_SUBMODULE_MAP[name])

    # Standard imports
    if name in _IMPORT_MAP:
        module_path, attr_name = _IMPORT_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    # Renamed imports
    if name in _RENAMED_IMPORT_MAP:
        module_path, attr_name = _RENAMED_IMPORT_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    # Optional clients with dependencies
    if name in _OPTIONAL_CLIENT_MAP:
        module_path, class_name = _OPTIONAL_CLIENT_MAP[name]
        return _import_optional_client(name, module_path, class_name)

    raise AttributeError(f"module 'hippocampai' has no attribute {name!r}")
