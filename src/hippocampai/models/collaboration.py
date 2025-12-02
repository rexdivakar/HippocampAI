"""Models for multi-agent collaboration features."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.models.agent import PermissionType


class SharedMemorySpace(BaseModel):
    """Shared memory space for agent collaboration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    owner_agent_id: str
    collaborator_agent_ids: list[str] = Field(default_factory=list)
    permissions: dict[str, list[str]] = Field(
        default_factory=dict
    )  # agent_id -> list of permission names
    memory_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    def add_collaborator(self, agent_id: str, permissions: list[PermissionType]) -> None:
        """Add a collaborator with specific permissions."""
        if agent_id not in self.collaborator_agent_ids:
            self.collaborator_agent_ids.append(agent_id)
        self.permissions[agent_id] = [p.value for p in permissions]
        self.updated_at = datetime.now(timezone.utc)

    def remove_collaborator(self, agent_id: str) -> None:
        """Remove a collaborator from the space."""
        if agent_id in self.collaborator_agent_ids:
            self.collaborator_agent_ids.remove(agent_id)
        if agent_id in self.permissions:
            del self.permissions[agent_id]
        self.updated_at = datetime.now(timezone.utc)

    def has_permission(self, agent_id: str, permission: PermissionType) -> bool:
        """Check if an agent has a specific permission."""
        # Owner has all permissions
        if agent_id == self.owner_agent_id:
            return True
        # Check collaborator permissions
        return permission.value in self.permissions.get(agent_id, [])

    def add_memory(self, memory_id: str) -> None:
        """Add a memory to the shared space."""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.updated_at = datetime.now(timezone.utc)

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the shared space."""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            self.updated_at = datetime.now(timezone.utc)


class CollaborationEventType(str, Enum):
    """Types of collaboration events."""

    MEMORY_ADDED = "memory_added"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"
    PERMISSION_CHANGED = "permission_changed"
    SPACE_UPDATED = "space_updated"


class CollaborationEvent(BaseModel):
    """Event in a shared memory space."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    space_id: str
    agent_id: str
    event_type: CollaborationEventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictType(str, Enum):
    """Types of memory conflicts."""

    CONCURRENT_UPDATE = "concurrent_update"  # Two agents updated same memory simultaneously
    DIVERGENT_CONTENT = "divergent_content"  # Different versions of same memory
    PERMISSION_CONFLICT = "permission_conflict"  # Permission denied during operation
    MERGE_CONFLICT = "merge_conflict"  # Automatic merge failed


class ResolutionStrategy(str, Enum):
    """Strategies for conflict resolution."""

    LATEST_WINS = "latest_wins"  # Most recent change wins
    MERGE_CHANGES = "merge_changes"  # Attempt automatic merge
    MANUAL = "manual"  # Requires manual resolution
    OWNER_WINS = "owner_wins"  # Owner's version takes precedence
    HIGHEST_IMPORTANCE = "highest_importance"  # Most important version wins


class ConflictResolution(BaseModel):
    """Resolution of a memory conflict."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str
    space_id: str
    conflict_type: ConflictType
    conflicting_versions: list[dict[str, Any]] = Field(
        default_factory=list
    )  # List of conflicting memory data
    resolution_strategy: ResolutionStrategy
    resolved_version: Optional[dict[str, Any]] = None
    resolved_by: Optional[str] = None  # agent_id
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def resolve(
        self, resolved_version: dict[str, Any], resolved_by: str, strategy: ResolutionStrategy
    ) -> None:
        """Mark conflict as resolved."""
        self.resolved_version = resolved_version
        self.resolved_by = resolved_by
        self.resolution_strategy = strategy
        self.resolved_at = datetime.now(timezone.utc)
        self.is_resolved = True


class NotificationType(str, Enum):
    """Types of notifications."""

    MEMORY_CHANGE = "memory_change"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    CONFLICT_DETECTED = "conflict_detected"
    SPACE_INVITATION = "space_invitation"
    MENTION = "mention"
    SYSTEM_ALERT = "system_alert"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Notification(BaseModel):
    """Notification for agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    recipient_agent_id: str
    sender_agent_id: Optional[str] = None
    notification_type: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM
    title: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    is_read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def mark_as_read(self) -> None:
        """Mark notification as read."""
        self.is_read = True
        self.read_at = datetime.now(timezone.utc)
