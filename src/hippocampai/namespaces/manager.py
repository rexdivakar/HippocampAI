"""Namespace manager for memory organization."""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from hippocampai.namespaces.models import (
    Namespace,
    NamespacePermission,
    NamespaceQuota,
)

logger = logging.getLogger(__name__)


class NamespaceError(Exception):
    """Base exception for namespace operations."""

    pass


class NamespaceNotFoundError(NamespaceError):
    """Namespace does not exist."""

    pass


class NamespacePermissionError(NamespaceError):
    """User lacks permission for operation."""

    pass


class NamespaceQuotaError(NamespaceError):
    """Operation would exceed quota."""

    pass


class NamespaceManager:
    """Manages memory namespaces for logical grouping.

    Provides CRUD operations for namespaces with permission checking,
    quota enforcement, and hierarchical organization.

    Example:
        >>> manager = NamespaceManager()
        >>> ns = manager.create("work", user_id="alice", description="Work memories")
        >>> child = manager.create("work/project-x", user_id="alice")
        >>> manager.grant_permission("work", "bob", NamespacePermission.READ)
    """

    # Valid namespace path pattern
    PATH_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*(/[a-zA-Z0-9][a-zA-Z0-9_-]*)*$")
    MAX_PATH_LENGTH = 256
    MAX_DEPTH = 10

    def __init__(self, storage: Optional[Any] = None):
        """Initialize namespace manager.

        Args:
            storage: Optional storage backend. If None, uses in-memory storage.
        """
        self._storage = storage
        self._namespaces: dict[str, Namespace] = {}  # path -> Namespace
        self._user_namespaces: dict[str, set[str]] = {}  # user_id -> set of paths

    def _validate_path(self, path: str) -> None:
        """Validate namespace path format."""
        if not path:
            raise NamespaceError("Namespace path cannot be empty")
        if len(path) > self.MAX_PATH_LENGTH:
            raise NamespaceError(f"Path exceeds max length of {self.MAX_PATH_LENGTH}")
        if not self.PATH_PATTERN.match(path):
            raise NamespaceError(
                f"Invalid path format: {path}. Use alphanumeric, hyphens, underscores, "
                "separated by slashes."
            )
        if path.count("/") >= self.MAX_DEPTH:
            raise NamespaceError(f"Path exceeds max depth of {self.MAX_DEPTH}")

    def create(
        self,
        path: str,
        user_id: str,
        description: Optional[str] = None,
        quota: Optional[NamespaceQuota] = None,
        metadata: Optional[dict[str, Any]] = None,
        is_public: bool = False,
        inherit_permissions: bool = True,
    ) -> Namespace:
        """Create a new namespace.

        Args:
            path: Namespace path (e.g., "work/project-x")
            user_id: Owner user ID
            description: Optional description
            quota: Optional quota limits
            metadata: Optional custom metadata
            is_public: Whether namespace is publicly readable
            inherit_permissions: Whether to inherit parent permissions

        Returns:
            Created Namespace object

        Raises:
            NamespaceError: If path is invalid or already exists
            NamespacePermissionError: If user can't create under parent
        """
        self._validate_path(path)

        if path in self._namespaces:
            raise NamespaceError(f"Namespace already exists: {path}")

        # Check parent exists and user has permission
        parent_path = None
        if "/" in path:
            parent_path = "/".join(path.split("/")[:-1])
            parent = self._namespaces.get(parent_path)
            if not parent:
                raise NamespaceError(f"Parent namespace does not exist: {parent_path}")
            if not parent.can_write(user_id):
                raise NamespacePermissionError(
                    f"User {user_id} cannot create namespaces under {parent_path}"
                )
            # Check parent quota for children
            if parent.stats.child_count >= parent.quota.max_children:
                raise NamespaceQuotaError(
                    f"Parent namespace {parent_path} has reached max children"
                )

        name = path.split("/")[-1]
        namespace = Namespace(
            name=name,
            path=path,
            owner_id=user_id,
            parent_path=parent_path,
            description=description,
            quota=quota or NamespaceQuota(),
            metadata=metadata or {},
            is_public=is_public,
            inherit_permissions=inherit_permissions,
        )

        self._namespaces[path] = namespace

        # Track user ownership
        if user_id not in self._user_namespaces:
            self._user_namespaces[user_id] = set()
        self._user_namespaces[user_id].add(path)

        # Update parent stats
        if parent_path:
            parent = self._namespaces[parent_path]
            parent.stats.child_count += 1
            parent.updated_at = datetime.now(timezone.utc)

        logger.info(f"Created namespace: {path} (owner={user_id})")
        return namespace

    def get(self, path: str, user_id: Optional[str] = None) -> Namespace:
        """Get a namespace by path.

        Args:
            path: Namespace path
            user_id: Optional user ID for permission check

        Returns:
            Namespace object

        Raises:
            NamespaceNotFoundError: If namespace doesn't exist
            NamespacePermissionError: If user can't read namespace
        """
        namespace = self._namespaces.get(path)
        if not namespace:
            raise NamespaceNotFoundError(f"Namespace not found: {path}")

        if user_id and not namespace.can_read(user_id):
            # Check inherited permissions
            if namespace.inherit_permissions:
                for ancestor_path in namespace.get_ancestors():
                    ancestor = self._namespaces.get(ancestor_path)
                    if ancestor and ancestor.can_read(user_id):
                        return namespace
            raise NamespacePermissionError(f"User {user_id} cannot access namespace {path}")

        return namespace

    def list(
        self,
        user_id: str,
        parent_path: Optional[str] = None,
        include_public: bool = True,
    ) -> list[Namespace]:
        """List namespaces accessible to a user.

        Args:
            user_id: User ID
            parent_path: Optional parent to list children of
            include_public: Whether to include public namespaces

        Returns:
            List of accessible namespaces
        """
        results = []
        for path, ns in self._namespaces.items():
            # Filter by parent
            if parent_path:
                if ns.parent_path != parent_path:
                    continue
            # Check access
            if ns.can_read(user_id):
                results.append(ns)
            elif include_public and ns.is_public:
                results.append(ns)
        return sorted(results, key=lambda n: n.path)

    def update(
        self,
        path: str,
        user_id: str,
        description: Optional[str] = None,
        quota: Optional[NamespaceQuota] = None,
        metadata: Optional[dict[str, Any]] = None,
        is_public: Optional[bool] = None,
    ) -> Namespace:
        """Update namespace properties.

        Args:
            path: Namespace path
            user_id: User performing update
            description: New description
            quota: New quota
            metadata: Metadata to merge
            is_public: New public status

        Returns:
            Updated Namespace

        Raises:
            NamespaceNotFoundError: If namespace doesn't exist
            NamespacePermissionError: If user can't admin namespace
        """
        namespace = self.get(path)
        if not namespace.can_admin(user_id):
            raise NamespacePermissionError(f"User {user_id} cannot modify namespace {path}")

        if description is not None:
            namespace.description = description
        if quota is not None:
            namespace.quota = quota
        if metadata is not None:
            namespace.metadata.update(metadata)
        if is_public is not None:
            namespace.is_public = is_public

        namespace.updated_at = datetime.now(timezone.utc)
        logger.info(f"Updated namespace: {path}")
        return namespace

    def delete(self, path: str, user_id: str, recursive: bool = False) -> bool:
        """Delete a namespace.

        Args:
            path: Namespace path
            user_id: User performing deletion
            recursive: Whether to delete children

        Returns:
            True if deleted

        Raises:
            NamespaceNotFoundError: If namespace doesn't exist
            NamespacePermissionError: If user can't delete
            NamespaceError: If has children and not recursive
        """
        namespace = self.get(path)
        if not namespace.can_admin(user_id):
            raise NamespacePermissionError(f"User {user_id} cannot delete namespace {path}")

        # Check for children
        children = [p for p in self._namespaces if p.startswith(path + "/")]
        if children and not recursive:
            raise NamespaceError(
                f"Namespace {path} has {len(children)} children. Use recursive=True to delete."
            )

        # Delete children first
        for child_path in sorted(children, reverse=True):
            del self._namespaces[child_path]
            logger.info(f"Deleted child namespace: {child_path}")

        # Delete namespace
        del self._namespaces[path]

        # Update user tracking
        if namespace.owner_id in self._user_namespaces:
            self._user_namespaces[namespace.owner_id].discard(path)

        # Update parent stats
        if namespace.parent_path:
            parent = self._namespaces.get(namespace.parent_path)
            if parent:
                parent.stats.child_count = max(0, parent.stats.child_count - 1)

        logger.info(f"Deleted namespace: {path}")
        return True

    def grant_permission(
        self,
        path: str,
        user_id: str,
        permission: NamespacePermission,
        granter_id: str,
    ) -> None:
        """Grant permission to a user.

        Args:
            path: Namespace path
            user_id: User to grant permission to
            permission: Permission level
            granter_id: User granting permission

        Raises:
            NamespacePermissionError: If granter can't admin namespace
        """
        namespace = self.get(path)
        if not namespace.can_admin(granter_id):
            raise NamespacePermissionError(
                f"User {granter_id} cannot modify permissions on {path}"
            )

        namespace.permissions[user_id] = permission
        namespace.updated_at = datetime.now(timezone.utc)
        logger.info(f"Granted {permission.value} to {user_id} on {path}")

    def revoke_permission(self, path: str, user_id: str, revoker_id: str) -> None:
        """Revoke a user's permission.

        Args:
            path: Namespace path
            user_id: User to revoke permission from
            revoker_id: User revoking permission
        """
        namespace = self.get(path)
        if not namespace.can_admin(revoker_id):
            raise NamespacePermissionError(
                f"User {revoker_id} cannot modify permissions on {path}"
            )

        namespace.permissions.pop(user_id, None)
        namespace.updated_at = datetime.now(timezone.utc)
        logger.info(f"Revoked permission from {user_id} on {path}")

    def update_stats(
        self,
        path: str,
        memory_delta: int = 0,
        storage_delta: int = 0,
        is_write: bool = False,
    ) -> None:
        """Update namespace statistics.

        Args:
            path: Namespace path
            memory_delta: Change in memory count
            storage_delta: Change in storage bytes
            is_write: Whether this was a write operation
        """
        namespace = self._namespaces.get(path)
        if not namespace:
            return

        namespace.stats.memory_count += memory_delta
        namespace.stats.storage_bytes += storage_delta
        namespace.stats.access_count += 1
        namespace.stats.last_access = datetime.now(timezone.utc)

        if is_write:
            namespace.stats.write_count += 1
            namespace.stats.last_write = datetime.now(timezone.utc)

    def check_quota(self, path: str, additional_memories: int = 1, additional_bytes: int = 0) -> bool:
        """Check if operation would exceed quota.

        Args:
            path: Namespace path
            additional_memories: Memories to add
            additional_bytes: Bytes to add

        Returns:
            True if within quota
        """
        namespace = self._namespaces.get(path)
        if not namespace:
            return True
        return namespace.is_within_quota(additional_memories, additional_bytes)

    def get_effective_permission(self, path: str, user_id: str) -> NamespacePermission:
        """Get effective permission considering inheritance.

        Args:
            path: Namespace path
            user_id: User ID

        Returns:
            Effective permission level
        """
        namespace = self._namespaces.get(path)
        if not namespace:
            return NamespacePermission.NONE

        # Direct permission
        perm = namespace.get_permission(user_id)
        if perm != NamespacePermission.NONE:
            return perm

        # Check inherited permissions
        if namespace.inherit_permissions:
            for ancestor_path in namespace.get_ancestors():
                ancestor = self._namespaces.get(ancestor_path)
                if ancestor:
                    ancestor_perm = ancestor.get_permission(user_id)
                    if ancestor_perm != NamespacePermission.NONE:
                        return ancestor_perm

        return NamespacePermission.NONE
