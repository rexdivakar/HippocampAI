"""HippocampAI: Production-ready long-term memory engine with hybrid retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hippocampai.models.memory import Memory, MemoryType, RetrievalResult

__version__ = "0.1.0"
__all__ = [
    "MemoryClient",
    "Memory",
    "MemoryType",
    "RetrievalResult",
    "get_config",
    "Config",
    "get_telemetry",
    "OperationType",
]

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from hippocampai.client import MemoryClient as MemoryClient
    from hippocampai.config import Config as Config
    from hippocampai.config import get_config as get_config
    from hippocampai.telemetry import OperationType as OperationType
    from hippocampai.telemetry import get_telemetry as get_telemetry

_MEMORY_CLIENT: Any | None = None
_CONFIG: Any | None = None
_GET_CONFIG: Any | None = None
_GET_TELEMETRY: Any | None = None
_OPERATION_TYPE: Any | None = None


def __getattr__(name: str) -> Any:
    global _MEMORY_CLIENT, _CONFIG, _GET_CONFIG, _GET_TELEMETRY, _OPERATION_TYPE

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

    raise AttributeError(f"module 'hippocampai' has no attribute {name!r}")
