"""REST API routes for collaboration features."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippocampai.models.agent import PermissionType
from hippocampai.models.collaboration import (
    CollaborationEventType,
    ResolutionStrategy,
)
from hippocampai.multiagent.collaboration import CollaborationManager

router = APIRouter(prefix="/v1/collaboration", tags=["collaboration"])

# Global collaboration manager (in production, use dependency injection)
collab_manager = CollaborationManager()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateSpaceRequest(BaseModel):
    name: str
    owner_agent_id: str
    description: Optional[str] = None
    tags: list[str] = []


class AddCollaboratorRequest(BaseModel):
    agent_id: str
    permissions: list[str]  # ["READ", "WRITE", etc.]
    inviter_id: str


class AddMemoryToSpaceRequest(BaseModel):
    memory_id: str
    agent_id: str


class UpdatePermissionsRequest(BaseModel):
    agent_id: str
    permissions: list[str]
    updater_id: str


class ResolveConflictRequest(BaseModel):
    resolved_version: dict
    resolved_by: str
    strategy: str


# ============================================================================
# SPACE ENDPOINTS
# ============================================================================

@router.post("/spaces")
async def create_space(request: CreateSpaceRequest):
    """Create a new shared memory space."""
    try:
        space = collab_manager.create_space(
            name=request.name,
            owner_agent_id=request.owner_agent_id,
            description=request.description,
            tags=request.tags
        )
        return {
            "success": True,
            "space": {
                "id": space.id,
                "name": space.name,
                "owner_agent_id": space.owner_agent_id,
                "collaborator_count": len(space.collaborator_agent_ids),
                "memory_count": len(space.memory_ids),
                "created_at": space.created_at.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spaces/{space_id}")
async def get_space(space_id: str):
    """Get space details."""
    space = collab_manager.get_space(space_id)
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")

    return {
        "id": space.id,
        "name": space.name,
        "description": space.description,
        "owner_agent_id": space.owner_agent_id,
        "collaborators": space.collaborator_agent_ids,
        "permissions": space.permissions,
        "memory_ids": space.memory_ids,
        "tags": space.tags,
        "is_active": space.is_active,
        "created_at": space.created_at.isoformat(),
        "updated_at": space.updated_at.isoformat()
    }


@router.get("/spaces")
async def list_spaces(agent_id: Optional[str] = None, include_inactive: bool = False):
    """List all spaces, optionally filtered by agent."""
    spaces = collab_manager.list_spaces(agent_id=agent_id, include_inactive=include_inactive)

    return {
        "spaces": [
            {
                "id": space.id,
                "name": space.name,
                "owner_agent_id": space.owner_agent_id,
                "collaborator_count": len(space.collaborator_agent_ids),
                "memory_count": len(space.memory_ids),
                "is_active": space.is_active,
                "created_at": space.created_at.isoformat()
            }
            for space in spaces
        ],
        "count": len(spaces)
    }


@router.delete("/spaces/{space_id}")
async def delete_space(space_id: str):
    """Delete a space."""
    success = collab_manager.delete_space(space_id)
    if not success:
        raise HTTPException(status_code=404, detail="Space not found")

    return {"success": True, "message": "Space deleted"}


# ============================================================================
# COLLABORATOR ENDPOINTS
# ============================================================================

@router.post("/spaces/{space_id}/collaborators")
async def add_collaborator(space_id: str, request: AddCollaboratorRequest):
    """Add a collaborator to a space."""
    try:
        permissions = [PermissionType(p) for p in request.permissions]
        success = collab_manager.add_collaborator(
            space_id=space_id,
            agent_id=request.agent_id,
            permissions=permissions,
            inviter_id=request.inviter_id
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add collaborator")

        return {
            "success": True,
            "message": f"Added collaborator {request.agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/spaces/{space_id}/collaborators/{agent_id}")
async def remove_collaborator(space_id: str, agent_id: str, remover_id: str):
    """Remove a collaborator from a space."""
    success = collab_manager.remove_collaborator(space_id, agent_id, remover_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to remove collaborator")

    return {"success": True, "message": "Collaborator removed"}


@router.put("/spaces/{space_id}/collaborators/{agent_id}/permissions")
async def update_permissions(space_id: str, agent_id: str, request: UpdatePermissionsRequest):
    """Update permissions for a collaborator."""
    try:
        permissions = [PermissionType(p) for p in request.permissions]
        success = collab_manager.update_permissions(
            space_id=space_id,
            agent_id=agent_id,
            permissions=permissions,
            updater_id=request.updater_id
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update permissions")

        return {"success": True, "message": "Permissions updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MEMORY MANAGEMENT IN SPACES
# ============================================================================

@router.post("/spaces/{space_id}/memories")
async def add_memory_to_space(space_id: str, request: AddMemoryToSpaceRequest):
    """Add a memory to a shared space."""
    success = collab_manager.add_memory_to_space(
        space_id=space_id,
        memory_id=request.memory_id,
        agent_id=request.agent_id
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to add memory to space")

    return {"success": True, "message": "Memory added to space"}


@router.delete("/spaces/{space_id}/memories/{memory_id}")
async def remove_memory_from_space(space_id: str, memory_id: str, agent_id: str):
    """Remove a memory from a space."""
    success = collab_manager.remove_memory_from_space(space_id, memory_id, agent_id)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to remove memory")

    return {"success": True, "message": "Memory removed from space"}


# ============================================================================
# EVENTS ENDPOINTS
# ============================================================================

@router.get("/spaces/{space_id}/events")
async def get_space_events(
    space_id: str,
    limit: int = 100,
    event_type: Optional[str] = None
):
    """Get events for a space."""
    event_type_enum = CollaborationEventType(event_type) if event_type else None

    events = collab_manager.get_space_events(
        space_id=space_id,
        limit=limit,
        event_type=event_type_enum
    )

    return {
        "events": [
            {
                "id": event.id,
                "space_id": event.space_id,
                "agent_id": event.agent_id,
                "event_type": event.event_type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat()
            }
            for event in events
        ],
        "count": len(events)
    }


# ============================================================================
# NOTIFICATIONS ENDPOINTS
# ============================================================================

@router.get("/notifications/{agent_id}")
async def get_notifications(agent_id: str, unread_only: bool = False, limit: int = 50):
    """Get notifications for an agent."""
    notifications = collab_manager.get_notifications(
        agent_id=agent_id,
        unread_only=unread_only,
        limit=limit
    )

    return {
        "notifications": [
            {
                "id": notif.id,
                "type": notif.notification_type.value,
                "priority": notif.priority.value,
                "title": notif.title,
                "message": notif.message,
                "data": notif.data,
                "is_read": notif.is_read,
                "created_at": notif.created_at.isoformat()
            }
            for notif in notifications
        ],
        "count": len(notifications)
    }


@router.post("/notifications/{agent_id}/{notification_id}/read")
async def mark_notification_read(agent_id: str, notification_id: str):
    """Mark a notification as read."""
    success = collab_manager.mark_notification_read(agent_id, notification_id)

    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")

    return {"success": True, "message": "Notification marked as read"}


# ============================================================================
# CONFLICT ENDPOINTS
# ============================================================================

@router.get("/conflicts")
async def get_conflicts(space_id: Optional[str] = None):
    """Get all conflicts, optionally filtered by space."""
    conflicts = list(collab_manager.conflicts.values())

    if space_id:
        conflicts = [c for c in conflicts if c.space_id == space_id]

    return {
        "conflicts": [
            {
                "id": conflict.id,
                "memory_id": conflict.memory_id,
                "space_id": conflict.space_id,
                "conflict_type": conflict.conflict_type.value,
                "is_resolved": conflict.is_resolved,
                "created_at": conflict.created_at.isoformat()
            }
            for conflict in conflicts
        ],
        "count": len(conflicts)
    }


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: str, request: ResolveConflictRequest):
    """Resolve a conflict."""
    try:
        strategy = ResolutionStrategy(request.strategy)
        success = collab_manager.resolve_conflict(
            conflict_id=conflict_id,
            resolved_version=request.resolved_version,
            resolved_by=request.resolved_by,
            strategy=strategy
        )

        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found")

        return {"success": True, "message": "Conflict resolved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
