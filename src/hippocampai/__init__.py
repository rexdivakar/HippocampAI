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


def __getattr__(name: str) -> Any:
    """Lazy loading for heavy imports to improve startup time."""
    import importlib
    
    # Submodules - use importlib to avoid circular imports
    if name == "core":
        return importlib.import_module("hippocampai.core")
    
    if name == "platform":
        return importlib.import_module("hippocampai.platform")
    
    if name == "plugins":
        return importlib.import_module("hippocampai.plugins")
    
    if name == "namespaces":
        return importlib.import_module("hippocampai.namespaces")
    
    if name == "portability":
        return importlib.import_module("hippocampai.portability")
    
    if name == "offline":
        return importlib.import_module("hippocampai.offline")
    
    if name == "tiered":
        return importlib.import_module("hippocampai.tiered")
    
    if name == "integrations":
        return importlib.import_module("hippocampai.integrations")
    
    # Main clients
    if name == "MemoryClient":
        try:
            from hippocampai.client import MemoryClient
            return MemoryClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hippocampai.MemoryClient requires optional dependencies (qdrant-client, sentence-transformers). "
                "Install HippocampAI with: pip install hippocampai"
            ) from exc

    if name == "UnifiedMemoryClient":
        from hippocampai.unified_client import UnifiedMemoryClient
        return UnifiedMemoryClient

    if name == "EnhancedMemoryClient":
        try:
            from hippocampai.enhanced_client import EnhancedMemoryClient
            return EnhancedMemoryClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hippocampai.EnhancedMemoryClient requires optional dependencies. "
                "Install HippocampAI with: pip install hippocampai"
            ) from exc

    if name == "OptimizedMemoryClient":
        try:
            from hippocampai.optimized_client import OptimizedMemoryClient
            return OptimizedMemoryClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hippocampai.OptimizedMemoryClient requires optional dependencies. "
                "Install HippocampAI with: pip install hippocampai"
            ) from exc

    if name == "AsyncMemoryClient":
        try:
            from hippocampai.async_client import AsyncMemoryClient
            return AsyncMemoryClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hippocampai.AsyncMemoryClient requires optional dependencies. "
                "Install HippocampAI with: pip install hippocampai"
            ) from exc

    # Configuration
    if name == "Config":
        from hippocampai.config import Config
        return Config

    if name == "get_config":
        from hippocampai.config import get_config
        return get_config

    # Telemetry
    if name == "get_telemetry":
        from hippocampai.telemetry import get_telemetry
        return get_telemetry

    if name == "OperationType":
        from hippocampai.telemetry import OperationType
        return OperationType

    # Session management
    if name == "SessionManager":
        from hippocampai.session import SessionManager
        return SessionManager

    # Graph
    if name == "MemoryGraph":
        from hippocampai.graph import MemoryGraph
        return MemoryGraph

    if name == "RelationType":
        from hippocampai.graph import RelationType
        return RelationType

    # Storage
    if name == "MemoryKVStore":
        from hippocampai.storage import MemoryKVStore
        return MemoryKVStore

    # Versioning
    if name == "MemoryVersionControl":
        from hippocampai.versioning import MemoryVersionControl
        return MemoryVersionControl

    if name == "MemoryVersion":
        from hippocampai.versioning import MemoryVersion
        return MemoryVersion

    if name == "AuditEntry":
        from hippocampai.versioning import AuditEntry
        return AuditEntry

    if name == "ChangeType":
        from hippocampai.versioning import ChangeType
        return ChangeType

    # Context injection
    if name == "ContextInjector":
        from hippocampai.utils.context_injection import ContextInjector
        return ContextInjector

    if name == "inject_context":
        from hippocampai.utils.context_injection import inject_context
        return inject_context

    # Multi-agent
    if name == "MultiAgentManager":
        from hippocampai.multiagent import MultiAgentManager
        return MultiAgentManager

    # SaaS automation (platform features)
    if name == "AutomationController":
        from hippocampai.saas.automation import AutomationController
        return AutomationController

    if name == "AutomationPolicy":
        from hippocampai.saas.automation import AutomationPolicy
        return AutomationPolicy

    if name == "AutomationSchedule":
        from hippocampai.saas.automation import AutomationSchedule
        return AutomationSchedule

    if name == "PolicyType":
        from hippocampai.saas.automation import PolicyType
        return PolicyType

    if name == "TaskManager":
        from hippocampai.saas.tasks import TaskManager
        return TaskManager

    if name == "TaskPriority":
        from hippocampai.saas.tasks import TaskPriority
        return TaskPriority

    if name == "TaskStatus":
        from hippocampai.saas.tasks import TaskStatus
        return TaskStatus

    if name == "BackgroundTask":
        from hippocampai.saas.tasks import BackgroundTask
        return BackgroundTask

    # Simplified API (mem0/zep compatible)
    if name == "SimpleMemory":
        from hippocampai.simple import Memory
        return Memory

    if name == "SimpleSession":
        from hippocampai.simple import Session
        return Session

    if name == "MemoryStore":
        from hippocampai.simple import MemoryStore
        return MemoryStore

    if name == "MemoryManager":
        from hippocampai.simple import MemoryManager
        return MemoryManager

    raise AttributeError(f"module 'hippocampai' has no attribute {name!r}")
