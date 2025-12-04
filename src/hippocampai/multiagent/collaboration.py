"""Multi-agent collaboration manager for shared memory spaces."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from hippocampai.models.agent import PermissionType
from hippocampai.models.collaboration import (
    CollaborationEvent,
    CollaborationEventType,
    ConflictResolution,
    ConflictType,
    Notification,
    NotificationPriority,
    NotificationType,
    ResolutionStrategy,
    SharedMemorySpace,
)

logger = logging.getLogger(__name__)


class CollaborationManager:
    """
    Manages collaboration between agents in shared memory spaces.

    Features:
    - Create and manage shared memory spaces
    - Track collaboration events
    - Detect and resolve conflicts
    - Send notifications to collaborators
    """

    def __init__(self) -> None:
        """Initialize collaboration manager."""
        self.spaces: dict[str, SharedMemorySpace] = {}
        self.events: list[CollaborationEvent] = []
        self.conflicts: dict[str, ConflictResolution] = {}
        self.notifications: dict[str, list[Notification]] = {}  # agent_id -> notifications

    # === SPACE MANAGEMENT ===

    def create_space(
        self,
        name: str,
        owner_agent_id: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SharedMemorySpace:
        """
        Create a new shared memory space.

        Args:
            name: Name of the space
            owner_agent_id: Agent who owns the space
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Created SharedMemorySpace
        """
        space = SharedMemorySpace(
            name=name,
            owner_agent_id=owner_agent_id,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.spaces[space.id] = space
        logger.info(f"Created shared space: {name} ({space.id}) owned by {owner_agent_id}")

        # Create event
        self._add_event(
            space_id=space.id,
            agent_id=owner_agent_id,
            event_type=CollaborationEventType.SPACE_UPDATED,
            data={"action": "space_created", "space_name": name},
        )

        return space

    def get_space(self, space_id: str) -> Optional[SharedMemorySpace]:
        """Get space by ID."""
        return self.spaces.get(space_id)

    def list_spaces(
        self, agent_id: Optional[str] = None, include_inactive: bool = False
    ) -> list[SharedMemorySpace]:
        """
        List all spaces, optionally filtered by agent.

        Args:
            agent_id: Filter spaces where this agent is owner or collaborator
            include_inactive: Include inactive spaces

        Returns:
            List of shared memory spaces
        """
        spaces = list(self.spaces.values())

        if not include_inactive:
            spaces = [s for s in spaces if s.is_active]

        if agent_id:
            spaces = [
                s
                for s in spaces
                if s.owner_agent_id == agent_id or agent_id in s.collaborator_agent_ids
            ]

        return spaces

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[SharedMemorySpace]:
        """Update space properties."""
        space = self.spaces.get(space_id)
        if not space:
            return None

        if name is not None:
            space.name = name
        if description is not None:
            space.description = description
        if tags is not None:
            space.tags = tags
        if metadata is not None:
            space.metadata.update(metadata)

        space.updated_at = datetime.now(timezone.utc)
        return space

    def delete_space(self, space_id: str) -> bool:
        """Delete a shared memory space."""
        if space_id in self.spaces:
            space = self.spaces[space_id]
            space.is_active = False
            logger.info(f"Deleted shared space: {space_id}")
            return True
        return False

    # === COLLABORATOR MANAGEMENT ===

    def add_collaborator(
        self, space_id: str, agent_id: str, permissions: list[PermissionType], inviter_id: str
    ) -> bool:
        """
        Add a collaborator to a shared space.

        Args:
            space_id: Space to add collaborator to
            agent_id: Agent to add
            permissions: Permissions to grant
            inviter_id: Agent who is inviting

        Returns:
            True if successful
        """
        space = self.spaces.get(space_id)
        if not space:
            return False

        # Check if inviter has permission to add collaborators
        if not space.has_permission(inviter_id, PermissionType.SHARE):
            logger.warning(f"Agent {inviter_id} lacks permission to add collaborators to {space_id}")
            return False

        space.add_collaborator(agent_id, permissions)

        # Create event
        self._add_event(
            space_id=space_id,
            agent_id=inviter_id,
            event_type=CollaborationEventType.AGENT_JOINED,
            data={
                "new_agent_id": agent_id,
                "permissions": [p.value for p in permissions],
                "invited_by": inviter_id,
            },
        )

        # Notify the new collaborator
        self._send_notification(
            recipient_id=agent_id,
            sender_id=inviter_id,
            notification_type=NotificationType.SPACE_INVITATION,
            priority=NotificationPriority.HIGH,
            title=f"Invited to '{space.name}'",
            message=f"You've been invited to collaborate on '{space.name}'",
            data={"space_id": space_id, "permissions": [p.value for p in permissions]},
        )

        logger.info(f"Added agent {agent_id} to space {space_id} with permissions {permissions}")
        return True

    def remove_collaborator(self, space_id: str, agent_id: str, remover_id: str) -> bool:
        """Remove a collaborator from a space."""
        space = self.spaces.get(space_id)
        if not space:
            return False

        # Check if remover is owner or the agent themselves
        if remover_id != space.owner_agent_id and remover_id != agent_id:
            logger.warning(f"Agent {remover_id} lacks permission to remove collaborators")
            return False

        space.remove_collaborator(agent_id)

        # Create event
        self._add_event(
            space_id=space_id,
            agent_id=remover_id,
            event_type=CollaborationEventType.AGENT_LEFT,
            data={"removed_agent_id": agent_id, "removed_by": remover_id},
        )

        logger.info(f"Removed agent {agent_id} from space {space_id}")
        return True

    def update_permissions(
        self, space_id: str, agent_id: str, permissions: list[PermissionType], updater_id: str
    ) -> bool:
        """Update permissions for a collaborator."""
        space = self.spaces.get(space_id)
        if not space:
            return False

        # Only owner can update permissions
        if updater_id != space.owner_agent_id:
            logger.warning(f"Agent {updater_id} lacks permission to update permissions")
            return False

        if agent_id in space.collaborator_agent_ids:
            space.permissions[agent_id] = [p.value for p in permissions]
            space.updated_at = datetime.now(timezone.utc)

            # Create event
            self._add_event(
                space_id=space_id,
                agent_id=updater_id,
                event_type=CollaborationEventType.PERMISSION_CHANGED,
                data={
                    "target_agent_id": agent_id,
                    "new_permissions": [p.value for p in permissions],
                },
            )

            # Notify affected agent
            self._send_notification(
                recipient_id=agent_id,
                sender_id=updater_id,
                notification_type=NotificationType.PERMISSION_GRANTED,
                priority=NotificationPriority.MEDIUM,
                title=f"Permissions updated in '{space.name}'",
                message="Your permissions have been updated",
                data={"space_id": space_id, "permissions": [p.value for p in permissions]},
            )

            return True

        return False

    # === MEMORY MANAGEMENT IN SPACES ===

    def add_memory_to_space(
        self, space_id: str, memory_id: str, agent_id: str
    ) -> bool:
        """Add a memory to a shared space."""
        space = self.spaces.get(space_id)
        if not space:
            return False

        # Check if agent has WRITE permission
        if not space.has_permission(agent_id, PermissionType.WRITE):
            logger.warning(f"Agent {agent_id} lacks WRITE permission for space {space_id}")
            return False

        space.add_memory(memory_id)

        # Create event
        self._add_event(
            space_id=space_id,
            agent_id=agent_id,
            event_type=CollaborationEventType.MEMORY_ADDED,
            data={"memory_id": memory_id},
        )

        # Notify collaborators
        self._notify_collaborators(
            space=space,
            exclude_agent_id=agent_id,
            notification_type=NotificationType.MEMORY_CHANGE,
            title=f"New memory in '{space.name}'",
            message=f"A memory was added to '{space.name}'",
            data={"memory_id": memory_id, "action": "added"},
        )

        return True

    def update_memory_in_space(
        self, space_id: str, memory_id: str, agent_id: str
    ) -> bool:
        """Update a memory in a shared space."""
        space = self.spaces.get(space_id)
        if not space or memory_id not in space.memory_ids:
            return False

        # Check if agent has WRITE permission
        if not space.has_permission(agent_id, PermissionType.WRITE):
            logger.warning(f"Agent {agent_id} lacks WRITE permission for space {space_id}")
            return False

        # Create event
        self._add_event(
            space_id=space_id,
            agent_id=agent_id,
            event_type=CollaborationEventType.MEMORY_UPDATED,
            data={"memory_id": memory_id},
        )

        # Notify collaborators
        self._notify_collaborators(
            space=space,
            exclude_agent_id=agent_id,
            notification_type=NotificationType.MEMORY_CHANGE,
            title=f"Memory updated in '{space.name}'",
            message=f"A memory was updated in '{space.name}'",
            data={"memory_id": memory_id, "action": "updated"},
        )

        return True

    def remove_memory_from_space(
        self, space_id: str, memory_id: str, agent_id: str
    ) -> bool:
        """Remove a memory from a shared space."""
        space = self.spaces.get(space_id)
        if not space or memory_id not in space.memory_ids:
            return False

        # Check if agent has DELETE permission
        if not space.has_permission(agent_id, PermissionType.DELETE):
            logger.warning(f"Agent {agent_id} lacks DELETE permission for space {space_id}")
            return False

        space.remove_memory(memory_id)

        # Create event
        self._add_event(
            space_id=space_id,
            agent_id=agent_id,
            event_type=CollaborationEventType.MEMORY_DELETED,
            data={"memory_id": memory_id},
        )

        # Notify collaborators
        self._notify_collaborators(
            space=space,
            exclude_agent_id=agent_id,
            notification_type=NotificationType.MEMORY_CHANGE,
            title=f"Memory removed from '{space.name}'",
            message=f"A memory was removed from '{space.name}'",
            data={"memory_id": memory_id, "action": "deleted"},
        )

        return True

    # === CONFLICT DETECTION & RESOLUTION ===

    def detect_conflict(
        self,
        space_id: str,
        memory_id: str,
        conflicting_versions: list[dict[str, Any]],
        conflict_type: ConflictType,
    ) -> ConflictResolution:
        """Detect and record a conflict."""
        conflict = ConflictResolution(
            memory_id=memory_id,
            space_id=space_id,
            conflict_type=conflict_type,
            conflicting_versions=conflicting_versions,
            resolution_strategy=ResolutionStrategy.MANUAL,  # Default to manual
        )

        self.conflicts[conflict.id] = conflict
        logger.warning(
            f"Conflict detected in space {space_id} for memory {memory_id}: {conflict_type}"
        )

        # Notify collaborators about conflict
        space = self.spaces.get(space_id)
        if space:
            self._notify_collaborators(
                space=space,
                notification_type=NotificationType.CONFLICT_DETECTED,
                priority=NotificationPriority.HIGH,
                title=f"Conflict detected in '{space.name}'",
                message="A memory conflict needs resolution",
                data={
                    "conflict_id": conflict.id,
                    "memory_id": memory_id,
                    "conflict_type": conflict_type.value,
                },
            )

        return conflict

    def resolve_conflict(
        self,
        conflict_id: str,
        resolved_version: dict[str, Any],
        resolved_by: str,
        strategy: ResolutionStrategy,
    ) -> bool:
        """Resolve a conflict."""
        conflict = self.conflicts.get(conflict_id)
        if not conflict:
            return False

        conflict.resolve(resolved_version, resolved_by, strategy)
        logger.info(f"Conflict {conflict_id} resolved by {resolved_by} using {strategy}")

        return True

    # === EVENT MANAGEMENT ===

    def _add_event(
        self,
        space_id: str,
        agent_id: str,
        event_type: CollaborationEventType,
        data: dict[str, Any],
    ) -> CollaborationEvent:
        """Add a collaboration event."""
        event = CollaborationEvent(
            space_id=space_id, agent_id=agent_id, event_type=event_type, data=data
        )
        self.events.append(event)
        return event

    def get_space_events(
        self, space_id: str, limit: int = 100, event_type: Optional[CollaborationEventType] = None
    ) -> list[CollaborationEvent]:
        """Get events for a space."""
        events = [e for e in self.events if e.space_id == space_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[:limit]

    # === NOTIFICATION MANAGEMENT ===

    def _send_notification(
        self,
        recipient_id: str,
        notification_type: NotificationType,
        priority: NotificationPriority,
        title: str,
        message: str,
        data: dict[str, Any],
        sender_id: Optional[str] = None,
    ) -> Notification:
        """Send a notification to an agent."""
        notification = Notification(
            recipient_agent_id=recipient_id,
            sender_agent_id=sender_id,
            notification_type=notification_type,
            priority=priority,
            title=title,
            message=message,
            data=data,
        )

        if recipient_id not in self.notifications:
            self.notifications[recipient_id] = []
        self.notifications[recipient_id].append(notification)

        logger.debug(f"Sent {notification_type} notification to {recipient_id}")
        return notification

    def _notify_collaborators(
        self,
        space: SharedMemorySpace,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        exclude_agent_id: Optional[str] = None,
    ) -> None:
        """Notify all collaborators in a space."""
        for agent_id in [space.owner_agent_id] + space.collaborator_agent_ids:
            if agent_id != exclude_agent_id:
                self._send_notification(
                    recipient_id=agent_id,
                    notification_type=notification_type,
                    priority=priority,
                    title=title,
                    message=message,
                    data=data,
                )

    def get_notifications(
        self, agent_id: str, unread_only: bool = False, limit: int = 50
    ) -> list[Notification]:
        """Get notifications for an agent."""
        notifications = self.notifications.get(agent_id, [])

        if unread_only:
            notifications = [n for n in notifications if not n.is_read]

        # Sort by creation time descending
        notifications.sort(key=lambda n: n.created_at, reverse=True)

        return notifications[:limit]

    def mark_notification_read(self, agent_id: str, notification_id: str) -> bool:
        """Mark a notification as read."""
        notifications = self.notifications.get(agent_id, [])
        for notification in notifications:
            if notification.id == notification_id:
                notification.mark_as_read()
                return True
        return False
