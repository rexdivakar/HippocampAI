"""Storage Module - KV Store, User/Session Management, and Bi-temporal Storage."""

from typing import Any

# Original KV Store exports
# Bi-temporal Storage
from hippocampai.storage.bitemporal_store import BiTemporalStore
from hippocampai.storage.kv_store import InMemoryKVStore, MemoryKVStore

# User/Session models (no extra deps); UserStore/get_user_store are lazy below
# since they require the optional `duckdb` dependency (saas extra only).
from hippocampai.storage.models import Session, SoftDeleteRecord, User

__all__ = [
    # KV Store
    "InMemoryKVStore",
    "MemoryKVStore",
    # User/Session Storage
    "UserStore",
    "get_user_store",
    "User",
    "Session",
    "SoftDeleteRecord",
    # Bi-temporal Storage
    "BiTemporalStore",
]

_LAZY_DUCKDB_EXPORTS = ("UserStore", "get_user_store")


def __getattr__(name: str) -> Any:
    """Lazily import UserStore/get_user_store so core imports don't require duckdb."""
    if name in _LAZY_DUCKDB_EXPORTS:
        from hippocampai.storage import user_store

        return getattr(user_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
