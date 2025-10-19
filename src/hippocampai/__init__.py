"""HippocampAI: Production-ready long-term memory engine with hybrid retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hippocampai.models.memory import Memory, MemoryType

__version__ = "0.1.0"
__all__ = ["MemoryClient", "Memory", "MemoryType"]

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from hippocampai.client import MemoryClient as MemoryClient

_MEMORY_CLIENT: Any | None = None


def __getattr__(name: str) -> Any:
    global _MEMORY_CLIENT

    if name == "MemoryClient":
        if _MEMORY_CLIENT is None:
            try:
                from hippocampai.client import MemoryClient as _ImportedMemoryClient

                _MEMORY_CLIENT = _ImportedMemoryClient
            except ModuleNotFoundError as exc:  # pragma: no cover - configuration dependent
                raise ModuleNotFoundError(
                    "hippocampai.MemoryClient requires optional dependencies (qdrant-client, sentence-transformers). "
                    "Install HippocampAI with the appropriate extras, e.g. `pip install hippocampai[core]`."
                ) from exc

        return _MEMORY_CLIENT

    raise AttributeError(f"module 'hippocampai' has no attribute {name!r}")
