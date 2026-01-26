"""Models for memory namespaces."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class NamespacePermission(str, Enum):
    """Permission levels for namespace access."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


class NamespaceQuota(BaseModel):
    """Quota limits for a namespace."""

    max_memories: int = Field(default=-1, description="Max memories (-1 = unlimited)")
    max_storage_bytes: int = Field(default=-1, description="Max storage in bytes (-1 = unlimited)")
    max_children: int = Field(default=100, description="Max child namespaces")
    retention_days: Optional[int] = Field(default=None, description="Auto-delete after N days")

    def is_unlimited(self) -> bool:
        """Check if quota is unlimited."""
        return self.max_memories == -1 and self.max_storage_bytes == -1


class NamespaceStats(BaseModel):
    """Statistics for a namespace."""

    memory_count: int = 0
    storage_bytes: int = 0
    child_count: int = 0
    last_access: Optional[datetime] = None
    last_write: Optional[datetime] = None
    access_count: int = 0
    write_count: int = 0


class Namespace(BaseModel):
    """A logical grouping for memories.

    Namespaces are hierarchical (like folders) and provide isolation,
    permissions, and quotas for memory organization.

    Attributes:
        id: Unique namespace identifier
        name: Human-readable name
        path: Full path (e.g., "work/project-x/docs")
        owner_id: User who owns this namespace
        parent_path: Parent namespace path (None for root)
        description: Optional description
        metadata: Custom metadata
        quota: Storage/count limits
        stats: Usage statistics
        permissions: User permissions map
        inherit_permissions: Whether to inherit parent permissions
        is_public: Whether namespace is publicly readable
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    path: str
    owner_id: str
    parent_path: Optional[str] = None
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)
    stats: NamespaceStats = Field(default_factory=NamespaceStats)
    permissions: dict[str, NamespacePermission] = Field(default_factory=dict)
    inherit_permissions: bool = True
    is_public: bool = False
    is_archived: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_permission(self, user_id: str) -> NamespacePermission:
        """Get effective permission for a user."""
        if user_id == self.owner_id:
            return NamespacePermission.OWNER
        if user_id in self.permissions:
            return self.permissions[user_id]
        if self.is_public:
            return NamespacePermission.READ
        return NamespacePermission.NONE

    def can_read(self, user_id: str) -> bool:
        """Check if user can read from namespace."""
        perm = self.get_permission(user_id)
        return perm in [
            NamespacePermission.READ,
            NamespacePermission.WRITE,
            NamespacePermission.ADMIN,
            NamespacePermission.OWNER,
        ]

    def can_write(self, user_id: str) -> bool:
        """Check if user can write to namespace."""
        perm = self.get_permission(user_id)
        return perm in [
            NamespacePermission.WRITE,
            NamespacePermission.ADMIN,
            NamespacePermission.OWNER,
        ]

    def can_admin(self, user_id: str) -> bool:
        """Check if user can administer namespace."""
        perm = self.get_permission(user_id)
        return perm in [NamespacePermission.ADMIN, NamespacePermission.OWNER]

    def is_within_quota(self, additional_memories: int = 1, additional_bytes: int = 0) -> bool:
        """Check if operation would exceed quota."""
        if self.quota.max_memories != -1:
            if self.stats.memory_count + additional_memories > self.quota.max_memories:
                return False
        if self.quota.max_storage_bytes != -1:
            if self.stats.storage_bytes + additional_bytes > self.quota.max_storage_bytes:
                return False
        return True

    def get_depth(self) -> int:
        """Get namespace depth (root = 0)."""
        return self.path.count("/")

    def is_child_of(self, parent_path: str) -> bool:
        """Check if this namespace is a child of another."""
        return self.path.startswith(parent_path + "/")

    def get_ancestors(self) -> list[str]:
        """Get all ancestor namespace paths."""
        parts = self.path.split("/")
        ancestors = []
        for i in range(1, len(parts)):
            ancestors.append("/".join(parts[:i]))
        return ancestors
