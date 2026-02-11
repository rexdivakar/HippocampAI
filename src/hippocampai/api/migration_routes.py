"""REST endpoints for embedding model migration (admin-only)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin/embeddings", tags=["migration"])


class MigrationStartRequest(BaseModel):
    new_model: str
    new_dimension: int = 384


class MigrationResponse(BaseModel):
    id: str
    old_model: str
    new_model: str
    status: str
    total_memories: int
    migrated_count: int
    failed_count: int


@router.post("/migrate", response_model=MigrationResponse)
def start_migration(
    request: MigrationStartRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> MigrationResponse:
    """Start an embedding model migration."""
    migration = client.migration_manager.start_migration(
        new_model=request.new_model,
        new_dimension=request.new_dimension,
    )

    # Dispatch to Celery if available
    try:
        from hippocampai.tasks import migrate_embeddings_task

        migrate_embeddings_task.delay(migration.id)
        logger.info(f"Migration {migration.id} dispatched to Celery")
    except Exception as e:
        logger.warning(f"Could not dispatch to Celery, running inline: {e}")
        client.migration_manager.run_migration(migration.id)

    return MigrationResponse(
        id=migration.id,
        old_model=migration.old_model,
        new_model=migration.new_model,
        status=migration.status.value,
        total_memories=migration.total_memories,
        migrated_count=migration.migrated_count,
        failed_count=migration.failed_count,
    )


@router.get("/migration/{migration_id}", response_model=MigrationResponse)
def get_migration_status(
    migration_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> MigrationResponse:
    """Check status of a migration."""
    migration = client.migration_manager.get_migration(migration_id)
    if migration is None:
        raise HTTPException(status_code=404, detail="Migration not found")

    return MigrationResponse(
        id=migration.id,
        old_model=migration.old_model,
        new_model=migration.new_model,
        status=migration.status.value,
        total_memories=migration.total_memories,
        migrated_count=migration.migrated_count,
        failed_count=migration.failed_count,
    )


@router.post("/migration/{migration_id}/cancel")
def cancel_migration(
    migration_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Cancel a running migration."""
    cancelled = client.migration_manager.cancel_migration(migration_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail="Migration not found or already completed/cancelled",
        )
    return {"success": True, "migration_id": migration_id, "status": "cancelled"}
