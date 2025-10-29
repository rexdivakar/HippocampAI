"""HippocampAI: Production-ready long-term memory engine with hybrid retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

__version__ = "1.0.0"
__all__ = [
    "MemoryClient",
    "UnifiedMemoryClient",  # New: Supports both local and remote modes
    "EnhancedMemoryClient",
    "OptimizedMemoryClient",
    "AsyncMemoryClient",
    "Memory",
    "MemoryType",
    "RetrievalResult",
    "get_config",
    "Config",
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
]

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from hippocampai.async_client import AsyncMemoryClient as AsyncMemoryClient
    from hippocampai.client import MemoryClient as MemoryClient
    from hippocampai.config import Config as Config
    from hippocampai.config import get_config as get_config
    from hippocampai.enhanced_client import EnhancedMemoryClient as EnhancedMemoryClient
    from hippocampai.graph import MemoryGraph as MemoryGraph
    from hippocampai.graph import RelationType as RelationType
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

_MEMORY_CLIENT: Any | None = None
_UNIFIED_MEMORY_CLIENT: Any | None = None
_ENHANCED_MEMORY_CLIENT: Any | None = None
_OPTIMIZED_MEMORY_CLIENT: Any | None = None
_ASYNC_MEMORY_CLIENT: Any | None = None
_CONFIG: Any | None = None
_GET_CONFIG: Any | None = None
_GET_TELEMETRY: Any | None = None
_OPERATION_TYPE: Any | None = None
_SESSION_MANAGER: Any | None = None
_MEMORY_GRAPH: Any | None = None
_RELATION_TYPE: Any | None = None
_MEMORY_KV_STORE: Any | None = None
_MEMORY_VERSION_CONTROL: Any | None = None
_MEMORY_VERSION: Any | None = None
_AUDIT_ENTRY: Any | None = None
_CHANGE_TYPE: Any | None = None
_CONTEXT_INJECTOR: Any | None = None
_INJECT_CONTEXT: Any | None = None


def __getattr__(name: str) -> Any:
    global \
        _MEMORY_CLIENT, \
        _UNIFIED_MEMORY_CLIENT, \
        _ENHANCED_MEMORY_CLIENT, \
        _OPTIMIZED_MEMORY_CLIENT, \
        _ASYNC_MEMORY_CLIENT, \
        _CONFIG, \
        _GET_CONFIG, \
        _GET_TELEMETRY, \
        _OPERATION_TYPE, \
        _SESSION_MANAGER
    global _MEMORY_GRAPH, _RELATION_TYPE, _MEMORY_KV_STORE, _MEMORY_VERSION_CONTROL
    global _MEMORY_VERSION, _AUDIT_ENTRY, _CHANGE_TYPE, _CONTEXT_INJECTOR, _INJECT_CONTEXT

    if name == "UnifiedMemoryClient":
        if _UNIFIED_MEMORY_CLIENT is None:
            from hippocampai.unified_client import (
                UnifiedMemoryClient as _ImportedUnifiedMemoryClient,
            )

            _UNIFIED_MEMORY_CLIENT = _ImportedUnifiedMemoryClient
        return _UNIFIED_MEMORY_CLIENT

    if name == "MemoryClient":
        if _MEMORY_CLIENT is None:
            try:
                from hippocampai.client import MemoryClient as _ImportedMemoryClient

                _MEMORY_CLIENT = _ImportedMemoryClient
            except ModuleNotFoundError as exc:  # pragma: no cover - configuration dependent
                raise ModuleNotFoundError(
                    "hippocampai.MemoryClient requires optional dependencies (qdrant-client, sentence-transformers). "
                    "Install HippocampAI with the appropriate extras, e.g. `pip install -e '.[core]'`."
                ) from exc

        return _MEMORY_CLIENT

    if name == "EnhancedMemoryClient":
        if _ENHANCED_MEMORY_CLIENT is None:
            try:
                from hippocampai.enhanced_client import (
                    EnhancedMemoryClient as _ImportedEnhancedMemoryClient,
                )

                _ENHANCED_MEMORY_CLIENT = _ImportedEnhancedMemoryClient
            except ModuleNotFoundError as exc:  # pragma: no cover - configuration dependent
                raise ModuleNotFoundError(
                    "hippocampai.EnhancedMemoryClient requires optional dependencies. "
                    "Install HippocampAI with the appropriate extras, e.g. `pip install -e '.[core]'`."
                ) from exc

        return _ENHANCED_MEMORY_CLIENT

    if name == "OptimizedMemoryClient":
        if _OPTIMIZED_MEMORY_CLIENT is None:
            try:
                from hippocampai.optimized_client import (
                    OptimizedMemoryClient as _ImportedOptimizedMemoryClient,
                )

                _OPTIMIZED_MEMORY_CLIENT = _ImportedOptimizedMemoryClient
            except ModuleNotFoundError as exc:  # pragma: no cover - configuration dependent
                raise ModuleNotFoundError(
                    "hippocampai.OptimizedMemoryClient requires optional dependencies. "
                    "Install HippocampAI with the appropriate extras, e.g. `pip install -e '.[core]'`."
                ) from exc

        return _OPTIMIZED_MEMORY_CLIENT

    if name == "AsyncMemoryClient":
        if _ASYNC_MEMORY_CLIENT is None:
            try:
                from hippocampai.async_client import AsyncMemoryClient as _ImportedAsyncMemoryClient

                _ASYNC_MEMORY_CLIENT = _ImportedAsyncMemoryClient
            except ModuleNotFoundError as exc:  # pragma: no cover - configuration dependent
                raise ModuleNotFoundError(
                    "hippocampai.AsyncMemoryClient requires optional dependencies (qdrant-client, sentence-transformers). "
                    "Install HippocampAI with the appropriate extras, e.g. `pip install -e '.[core]'`."
                ) from exc

        return _ASYNC_MEMORY_CLIENT

    if name == "Config":
        if _CONFIG is None:
            from hippocampai.config import Config as _ImportedConfig

            _CONFIG = _ImportedConfig
        return _CONFIG

    if name == "get_config":
        if _GET_CONFIG is None:
            from hippocampai.config import get_config as _ImportedGetConfig

            _GET_CONFIG = _ImportedGetConfig
        return _GET_CONFIG

    if name == "get_telemetry":
        if _GET_TELEMETRY is None:
            from hippocampai.telemetry import get_telemetry as _ImportedGetTelemetry

            _GET_TELEMETRY = _ImportedGetTelemetry
        return _GET_TELEMETRY

    if name == "OperationType":
        if _OPERATION_TYPE is None:
            from hippocampai.telemetry import OperationType as _ImportedOperationType

            _OPERATION_TYPE = _ImportedOperationType
        return _OPERATION_TYPE

    # Session management
    if name == "SessionManager":
        if _SESSION_MANAGER is None:
            from hippocampai.session import SessionManager as _ImportedSessionManager

            _SESSION_MANAGER = _ImportedSessionManager
        return _SESSION_MANAGER

    # Advanced features
    if name == "MemoryGraph":
        if _MEMORY_GRAPH is None:
            from hippocampai.graph import MemoryGraph as _ImportedMemoryGraph

            _MEMORY_GRAPH = _ImportedMemoryGraph
        return _MEMORY_GRAPH

    if name == "RelationType":
        if _RELATION_TYPE is None:
            from hippocampai.graph import RelationType as _ImportedRelationType

            _RELATION_TYPE = _ImportedRelationType
        return _RELATION_TYPE

    if name == "MemoryKVStore":
        if _MEMORY_KV_STORE is None:
            from hippocampai.storage import MemoryKVStore as _ImportedMemoryKVStore

            _MEMORY_KV_STORE = _ImportedMemoryKVStore
        return _MEMORY_KV_STORE

    if name == "MemoryVersionControl":
        if _MEMORY_VERSION_CONTROL is None:
            from hippocampai.versioning import MemoryVersionControl as _ImportedMemoryVersionControl

            _MEMORY_VERSION_CONTROL = _ImportedMemoryVersionControl
        return _MEMORY_VERSION_CONTROL

    if name == "MemoryVersion":
        if _MEMORY_VERSION is None:
            from hippocampai.versioning import MemoryVersion as _ImportedMemoryVersion

            _MEMORY_VERSION = _ImportedMemoryVersion
        return _MEMORY_VERSION

    if name == "AuditEntry":
        if _AUDIT_ENTRY is None:
            from hippocampai.versioning import AuditEntry as _ImportedAuditEntry

            _AUDIT_ENTRY = _ImportedAuditEntry
        return _AUDIT_ENTRY

    if name == "ChangeType":
        if _CHANGE_TYPE is None:
            from hippocampai.versioning import ChangeType as _ImportedChangeType

            _CHANGE_TYPE = _ImportedChangeType
        return _CHANGE_TYPE

    if name == "ContextInjector":
        if _CONTEXT_INJECTOR is None:
            from hippocampai.utils.context_injection import (
                ContextInjector as _ImportedContextInjector,
            )

            _CONTEXT_INJECTOR = _ImportedContextInjector
        return _CONTEXT_INJECTOR

    if name == "inject_context":
        if _INJECT_CONTEXT is None:
            from hippocampai.utils.context_injection import inject_context as _ImportedInjectContext

            _INJECT_CONTEXT = _ImportedInjectContext
        return _INJECT_CONTEXT

    # Multi-agent manager
    if name == "MultiAgentManager":
        from hippocampai.multiagent import MultiAgentManager as _ImportedMultiAgentManager

        return _ImportedMultiAgentManager

    raise AttributeError(f"module 'hippocampai' has no attribute {name!r}")
