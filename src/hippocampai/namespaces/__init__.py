"""Memory Namespaces for logical grouping beyond user_id.

Namespaces provide hierarchical organization of memories into projects,
contexts, or any logical grouping. They support:
- Isolation: Memories in different namespaces don't interfere
- Inheritance: Child namespaces can inherit from parents
- Permissions: Fine-grained access control per namespace
- Quotas: Storage limits per namespace

Example:
    >>> from hippocampai.namespaces import NamespaceManager, Namespace
    >>>
    >>> ns_manager = NamespaceManager()
    >>> work_ns = ns_manager.create("work", user_id="alice")
    >>> project_ns = ns_manager.create("work/project-x", user_id="alice", parent="work")
    >>>
    >>> # Store memory in namespace
    >>> client.remember("API key is xyz", user_id="alice", namespace="work/project-x")
    >>>
    >>> # Query within namespace
    >>> results = client.recall("API key", user_id="alice", namespace="work/project-x")
"""

from hippocampai.namespaces.manager import (
    NamespaceError,
    NamespaceManager,
    NamespaceNotFoundError,
    NamespacePermissionError,
    NamespaceQuotaError,
)
from hippocampai.namespaces.models import (
    Namespace,
    NamespacePermission,
    NamespaceQuota,
    NamespaceStats,
)

__all__ = [
    "NamespaceManager",
    "NamespaceError",
    "NamespaceNotFoundError",
    "NamespacePermissionError",
    "NamespaceQuotaError",
    "Namespace",
    "NamespacePermission",
    "NamespaceQuota",
    "NamespaceStats",
]
