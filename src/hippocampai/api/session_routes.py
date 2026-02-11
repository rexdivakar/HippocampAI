"""API routes for session and user management."""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hippocampai.storage import get_user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================


class SoftDeleteRequest(BaseModel):
    """Request to soft delete a session."""

    session_id: str
    reason: Optional[str] = None
    admin_user_id: str  # Who is performing the delete


class WipeUserDataRequest(BaseModel):
    """Request to wipe all user data."""

    user_id: str
    reason: Optional[str] = None
    admin_user_id: str
    confirm: bool = False  # Must be True to proceed


class RestoreSessionRequest(BaseModel):
    """Request to restore a soft-deleted session."""

    record_id: str
    admin_user_id: str


class SessionResponse(BaseModel):
    """Session response."""

    session_id: str
    user_id: str
    created_at: datetime
    last_active_at: datetime
    is_active: bool
    is_deleted: bool
    memory_count: int


class SoftDeleteResponse(BaseModel):
    """Soft delete record response."""

    id: str
    entity_type: str
    entity_id: str
    user_id: str
    session_id: Optional[str]
    deleted_at: datetime
    deleted_by: str
    reason: Optional[str]
    is_restored: bool


# ============================================
# ENDPOINTS
# ============================================


@router.get("/list")
async def list_sessions(
    user_id: str = Query(..., description="User ID to list sessions for"),
    include_deleted: bool = Query(False, description="Include soft-deleted sessions (admin only)"),
) -> list[SessionResponse]:
    """List all sessions for a user."""
    store = get_user_store()
    sessions = store.get_user_sessions(user_id, include_deleted=include_deleted)

    return [
        SessionResponse(
            session_id=s.session_id,
            user_id=s.user_id,
            created_at=s.created_at,
            last_active_at=s.last_active_at,
            is_active=s.is_active,
            is_deleted=s.is_deleted,
            memory_count=s.memory_count,
        )
        for s in sessions
    ]


@router.get("/stats")
async def get_session_stats() -> dict:
    """Get session storage statistics."""
    try:
        store = get_user_store()
        return store.get_stats()
    except Exception as e:
        logger.warning(f"Could not get session stats: {e}")
        return {"total_users": 0, "total_sessions": 0, "deleted_sessions": 0, "total_memories": 0}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    include_deleted: bool = Query(False, description="Include if soft-deleted (admin only)"),
) -> SessionResponse:
    """Get a specific session."""
    store = get_user_store()
    session = store.get_session(session_id, include_deleted=include_deleted)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        created_at=session.created_at,
        last_active_at=session.last_active_at,
        is_active=session.is_active,
        is_deleted=session.is_deleted,
        memory_count=session.memory_count,
    )


@router.post("/soft-delete")
async def soft_delete_session(request: SoftDeleteRequest) -> SoftDeleteResponse:
    """
    Soft delete a session.

    This marks the session as deleted but preserves the data.
    Memories will no longer be visible to agents or dashboard.
    Admins can still view and restore the data.
    """
    store = get_user_store()

    try:
        # Also soft-delete memories in Qdrant
        await _soft_delete_qdrant_memories(request.session_id)

        record = store.soft_delete_session(
            session_id=request.session_id, deleted_by=request.admin_user_id, reason=request.reason
        )

        logger.info(f"Soft deleted session {request.session_id} by {request.admin_user_id}")

        return SoftDeleteResponse(
            id=record.id,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            user_id=record.user_id,
            session_id=record.session_id,
            deleted_at=record.deleted_at,
            deleted_by=record.deleted_by,
            reason=record.reason,
            is_restored=record.is_restored,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to soft delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wipe-user-data")
async def wipe_user_data(request: WipeUserDataRequest) -> dict:
    """
    Soft delete ALL data for a user.

    This is a destructive operation that marks all sessions and memories as deleted.
    Requires confirmation flag to be True.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to wipe user data. This action cannot be easily undone.",
        )

    store = get_user_store()

    try:
        # Get all sessions first
        sessions = store.get_user_sessions(request.user_id, include_deleted=False)

        # Soft delete memories in Qdrant for each session
        for session in sessions:
            await _soft_delete_qdrant_memories(session.session_id)

        # Soft delete all sessions in DuckDB
        records = store.soft_delete_user_data(
            user_id=request.user_id, deleted_by=request.admin_user_id, reason=request.reason
        )

        logger.warning(f"Wiped all data for user {request.user_id} by {request.admin_user_id}")

        return {
            "success": True,
            "user_id": request.user_id,
            "sessions_deleted": len(records),
            "deleted_by": request.admin_user_id,
            "reason": request.reason,
            "message": f"Successfully soft-deleted {len(records)} sessions. Data can be restored by admin.",
        }
    except Exception as e:
        logger.exception(f"Failed to wipe user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deleted/list")
async def list_soft_deletes(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    include_restored: bool = Query(False, description="Include restored records"),
    limit: int = Query(100, ge=1, le=500),
) -> list[SoftDeleteResponse]:
    """List soft-deleted records (admin only)."""
    store = get_user_store()
    records = store.get_soft_deletes(
        user_id=user_id, include_restored=include_restored, limit=limit
    )

    return [
        SoftDeleteResponse(
            id=r.id,
            entity_type=r.entity_type,
            entity_id=r.entity_id,
            user_id=r.user_id,
            session_id=r.session_id,
            deleted_at=r.deleted_at,
            deleted_by=r.deleted_by,
            reason=r.reason,
            is_restored=r.is_restored,
        )
        for r in records
    ]


@router.post("/restore")
async def restore_session(request: RestoreSessionRequest) -> SessionResponse:
    """
    Restore a soft-deleted session (admin only).

    This will make the session and its memories visible again.
    """
    store = get_user_store()

    try:
        # Get the soft delete record first
        records = store.get_soft_deletes(include_restored=False)
        record = next((r for r in records if r.id == request.record_id), None)

        if not record:
            raise HTTPException(status_code=404, detail="Soft delete record not found")

        if not record.session_id:
            raise HTTPException(status_code=400, detail="Record has no session_id")

        # Restore memories in Qdrant
        await _restore_qdrant_memories(record.session_id)

        # Restore session in DuckDB
        session = store.restore_session(request.record_id, request.admin_user_id)

        logger.info(f"Restored session {session.session_id} by {request.admin_user_id}")

        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at,
            last_active_at=session.last_active_at,
            is_active=session.is_active,
            is_deleted=session.is_deleted,
            memory_count=session.memory_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to restore session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-from-qdrant")
async def sync_from_qdrant() -> dict:
    """Sync sessions from Qdrant to DuckDB (admin only)."""
    from qdrant_client import QdrantClient

    from hippocampai.storage.models import Session, User

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    store = get_user_store()

    collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts"]
    users_created = 0
    sessions_created = 0
    users_seen = set()
    sessions_seen = set()

    for collection in collections:
        try:
            if not client.collection_exists(collection):
                continue

            offset = None
            while True:
                results, offset = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )

                if not results:
                    break

                for point in results:
                    payload = point.payload or {}
                    user_id = payload.get("user_id")
                    session_id = payload.get("session_id")

                    if user_id and user_id not in users_seen:
                        users_seen.add(user_id)
                        try:
                            existing = store.get_user(user_id, include_deleted=True)
                            if not existing:
                                from datetime import datetime

                                created_str = payload.get(
                                    "created_at", datetime.utcnow().isoformat()
                                )
                                if isinstance(created_str, str):
                                    created_str = created_str.replace("Z", "+00:00")
                                    created_at = datetime.fromisoformat(created_str)
                                else:
                                    created_at = created_str
                                user = User(
                                    user_id=user_id,
                                    username=payload.get("metadata", {}).get("username"),
                                    created_at=created_at,
                                )
                                store.create_user(user)
                                users_created += 1
                        except Exception as e:
                            logger.debug(f"Could not create user {user_id}: {e}")

                    if session_id and session_id not in sessions_seen:
                        sessions_seen.add(session_id)
                        try:
                            existing = store.get_session(session_id, include_deleted=True)
                            if not existing:
                                from datetime import datetime

                                created_str = payload.get(
                                    "created_at", datetime.utcnow().isoformat()
                                )
                                if isinstance(created_str, str):
                                    created_str = created_str.replace("Z", "+00:00")
                                    created_at = datetime.fromisoformat(created_str)
                                else:
                                    created_at = created_str
                                session = Session(
                                    session_id=session_id,
                                    user_id=user_id or "unknown",
                                    created_at=created_at,
                                    memory_count=1,
                                )
                                store.create_session(session)
                                sessions_created += 1
                            else:
                                store.update_session_memory_count(
                                    session_id, existing.memory_count + 1
                                )
                        except Exception as e:
                            logger.debug(f"Could not process session {session_id}: {e}")

                if offset is None:
                    break
        except Exception as e:
            logger.warning(f"Error scanning {collection}: {e}")

    return {
        "success": True,
        "users_created": users_created,
        "sessions_created": sessions_created,
        "total_users_seen": len(users_seen),
        "total_sessions_seen": len(sessions_seen),
    }


# ============================================
# QDRANT SOFT DELETE HELPERS
# ============================================


async def _soft_delete_qdrant_memories(session_id: str) -> int:
    """Mark memories in Qdrant as soft-deleted by adding is_deleted flag."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts"]
    total_updated = 0

    for collection in collections:
        try:
            if not client.collection_exists(collection):
                continue

            # Find all memories with this session_id
            session_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )

            # Get all matching points
            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=session_filter,
                limit=10000,
                with_payload=False,
            )

            if results:
                point_ids = [p.id for p in results]

                # Set is_deleted flag on all matching points
                client.set_payload(
                    collection_name=collection,
                    payload={"is_deleted": True, "deleted_at": datetime.utcnow().isoformat()},
                    points=PointIdsList(points=point_ids),
                )

                total_updated += len(point_ids)
                logger.info(f"Soft deleted {len(point_ids)} memories in {collection}")

        except Exception as e:
            logger.warning(f"Could not soft delete from {collection}: {e}")

    return total_updated


async def _restore_qdrant_memories(session_id: str) -> int:
    """Remove soft-delete flag from memories in Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts"]
    total_updated = 0

    for collection in collections:
        try:
            if not client.collection_exists(collection):
                continue

            # Find all soft-deleted memories with this session_id
            session_filter = Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="is_deleted", match=MatchValue(value=True)),
                ]
            )

            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=session_filter,
                limit=10000,
                with_payload=False,
            )

            if results:
                point_ids = [p.id for p in results]

                # Remove is_deleted flag
                client.set_payload(
                    collection_name=collection,
                    payload={"is_deleted": False, "deleted_at": None},
                    points=PointIdsList(points=point_ids),
                )

                total_updated += len(point_ids)
                logger.info(f"Restored {len(point_ids)} memories in {collection}")

        except Exception as e:
            logger.warning(f"Could not restore from {collection}: {e}")

    return total_updated
