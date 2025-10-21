"""Memory versioning and audit trail system."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes to memories."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"
    RELATIONSHIP_ADDED = "relationship_added"
    RELATIONSHIP_REMOVED = "relationship_removed"


@dataclass
class MemoryVersion:
    """A single version of a memory."""

    version_id: str
    memory_id: str
    version_number: int
    data: Dict[str, Any]
    created_at: datetime
    created_by: Optional[str] = None  # User/system that created this version
    change_summary: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "memory_id": self.memory_id,
            "version_number": self.version_number,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_summary": self.change_summary,
        }


@dataclass
class AuditEntry:
    """Audit trail entry."""

    audit_id: str = field(default_factory=lambda: str(uuid4()))
    memory_id: str = ""
    change_type: ChangeType = ChangeType.UPDATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "audit_id": self.audit_id,
            "memory_id": self.memory_id,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "changes": self.changes,
            "metadata": self.metadata,
        }


class MemoryVersionControl:
    """Version control system for memories."""

    def __init__(self, max_versions_per_memory: int = 10):
        """
        Initialize version control.

        Args:
            max_versions_per_memory: Maximum versions to keep per memory
        """
        self.max_versions = max_versions_per_memory
        self._versions: Dict[str, List[MemoryVersion]] = {}  # memory_id -> versions
        self._audit_trail: List[AuditEntry] = []

    def create_version(
        self,
        memory_id: str,
        data: Dict[str, Any],
        created_by: Optional[str] = None,
        change_summary: Optional[str] = None,
    ) -> MemoryVersion:
        """
        Create a new version of a memory.

        Args:
            memory_id: Memory ID
            data: Memory data snapshot
            created_by: Who created this version
            change_summary: Summary of changes

        Returns:
            MemoryVersion object
        """
        if memory_id not in self._versions:
            self._versions[memory_id] = []

        version_number = len(self._versions[memory_id]) + 1
        version = MemoryVersion(
            version_id=str(uuid4()),
            memory_id=memory_id,
            version_number=version_number,
            data=data.copy(),
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            change_summary=change_summary,
        )

        self._versions[memory_id].append(version)

        # Trim old versions if needed
        if len(self._versions[memory_id]) > self.max_versions:
            self._versions[memory_id] = self._versions[memory_id][-self.max_versions :]

        logger.debug(f"Created version {version_number} for memory {memory_id}")
        return version

    def get_version(
        self, memory_id: str, version_number: Optional[int] = None
    ) -> Optional[MemoryVersion]:
        """
        Get a specific version of a memory.

        Args:
            memory_id: Memory ID
            version_number: Version number (None = latest)

        Returns:
            MemoryVersion or None
        """
        versions = self._versions.get(memory_id, [])
        if not versions:
            return None

        if version_number is None:
            return versions[-1]  # Latest version

        # Find version by number
        for v in versions:
            if v.version_number == version_number:
                return v

        return None

    def get_version_history(self, memory_id: str) -> List[MemoryVersion]:
        """Get all versions of a memory."""
        return self._versions.get(memory_id, [])

    def compare_versions(
        self, memory_id: str, version1: int, version2: int
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two versions of a memory.

        Returns:
            Dictionary with added, removed, and changed fields
        """
        v1 = self.get_version(memory_id, version1)
        v2 = self.get_version(memory_id, version2)

        if not v1 or not v2:
            return None

        diff = {"added": {}, "removed": {}, "changed": {}}

        # Find added and changed
        for key, value in v2.data.items():
            if key not in v1.data:
                diff["added"][key] = value
            elif v1.data[key] != value:
                diff["changed"][key] = {"old": v1.data[key], "new": value}

        # Find removed
        for key in v1.data:
            if key not in v2.data:
                diff["removed"][key] = v1.data[key]

        return diff

    def rollback(self, memory_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """
        Rollback to a previous version.

        Returns:
            Data from the specified version
        """
        version = self.get_version(memory_id, version_number)
        if version:
            return version.data.copy()
        return None

    def add_audit_entry(
        self,
        memory_id: str,
        change_type: ChangeType,
        user_id: Optional[str] = None,
        changes: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> AuditEntry:
        """
        Add an audit trail entry.

        Args:
            memory_id: Memory ID
            change_type: Type of change
            user_id: User who made the change
            changes: Dictionary of changes
            metadata: Additional metadata

        Returns:
            AuditEntry object
        """
        entry = AuditEntry(
            memory_id=memory_id,
            change_type=change_type,
            user_id=user_id,
            changes=changes or {},
            metadata=metadata or {},
        )

        self._audit_trail.append(entry)
        logger.debug(f"Added audit entry: {change_type.value} for memory {memory_id}")
        return entry

    def get_audit_trail(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Get audit trail entries with optional filtering.

        Args:
            memory_id: Filter by memory ID
            user_id: Filter by user ID
            change_type: Filter by change type
            limit: Maximum entries to return

        Returns:
            List of AuditEntry objects
        """
        filtered = self._audit_trail

        if memory_id:
            filtered = [e for e in filtered if e.memory_id == memory_id]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if change_type:
            filtered = [e for e in filtered if e.change_type == change_type]

        # Sort by timestamp descending (most recent first)
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    def clear_old_audit_entries(self, days_old: int = 90) -> int:
        """
        Clear audit entries older than specified days.

        Returns:
            Number of entries cleared
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days_old * 24 * 3600)
        original_count = len(self._audit_trail)

        self._audit_trail = [e for e in self._audit_trail if e.timestamp.timestamp() >= cutoff]

        cleared = original_count - len(self._audit_trail)
        logger.info(f"Cleared {cleared} old audit entries")
        return cleared

    def get_statistics(self) -> Dict:
        """Get version control statistics."""
        total_versions = sum(len(versions) for versions in self._versions.values())

        return {
            "total_memories_tracked": len(self._versions),
            "total_versions": total_versions,
            "avg_versions_per_memory": total_versions / max(len(self._versions), 1),
            "total_audit_entries": len(self._audit_trail),
        }
