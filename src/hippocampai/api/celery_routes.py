"""FastAPI routes for Celery task management and async operations."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippocampai.tasks import (
    batch_create_memories_task,
    create_memory_task,
    delete_memory_task,
    recall_memories_task,
    update_memory_task,
)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


# ============================================================================
# Request/Response Models
# ============================================================================


class TaskSubmitResponse(BaseModel):
    """Response when a task is submitted."""

    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response for task status check."""

    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[dict[str, Any]] = None


class MemoryCreateRequest(BaseModel):
    """Request to create a memory via Celery."""

    text: str
    user_id: str
    memory_type: str = "fact"
    importance: Optional[float] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class BatchMemoryCreateRequest(BaseModel):
    """Request to batch create memories via Celery."""

    memories: list[dict[str, Any]]
    check_duplicates: bool = True


class MemoryRecallRequest(BaseModel):
    """Request to recall memories via Celery."""

    query: str
    user_id: str
    k: int = 5
    filters: Optional[dict[str, Any]] = None


class MemoryUpdateRequest(BaseModel):
    """Request to update a memory via Celery."""

    memory_id: str
    user_id: str
    updates: dict[str, Any]


class MemoryDeleteRequest(BaseModel):
    """Request to delete a memory via Celery."""

    memory_id: str
    user_id: str


# ============================================================================
# Task Submission Endpoints
# ============================================================================


@router.post("/memory/create", response_model=TaskSubmitResponse)
async def submit_create_memory_task(request: MemoryCreateRequest):
    """
    Submit a memory creation task to Celery queue.

    This allows for asynchronous memory creation without blocking the API.

    Returns:
        task_id to track the task status
    """
    task = create_memory_task.delay(
        text=request.text,
        user_id=request.user_id,
        memory_type=request.memory_type,
        importance=request.importance,
        tags=request.tags,
        metadata=request.metadata,
    )

    return TaskSubmitResponse(
        task_id=task.id,
        status="submitted",
        message=f"Memory creation task submitted. Track with task_id: {task.id}",
    )


@router.post("/memory/batch-create", response_model=TaskSubmitResponse)
async def submit_batch_create_memories_task(request: BatchMemoryCreateRequest):
    """
    Submit a batch memory creation task to Celery queue.

    Efficient for creating multiple memories at once.

    Returns:
        task_id to track the task status
    """
    task = batch_create_memories_task.delay(
        memories=request.memories,
        check_duplicates=request.check_duplicates,
    )

    return TaskSubmitResponse(
        task_id=task.id,
        status="submitted",
        message=f"Batch memory creation task submitted for {len(request.memories)} memories. Track with task_id: {task.id}",
    )


@router.post("/memory/recall", response_model=TaskSubmitResponse)
async def submit_recall_memories_task(request: MemoryRecallRequest):
    """
    Submit a memory recall task to Celery queue.

    Useful for heavy recall operations that might take time.

    Returns:
        task_id to track the task status
    """
    task = recall_memories_task.delay(
        query=request.query,
        user_id=request.user_id,
        k=request.k,
        filters=request.filters,
    )

    return TaskSubmitResponse(
        task_id=task.id,
        status="submitted",
        message=f"Memory recall task submitted. Track with task_id: {task.id}",
    )


@router.post("/memory/update", response_model=TaskSubmitResponse)
async def submit_update_memory_task(request: MemoryUpdateRequest):
    """
    Submit a memory update task to Celery queue.

    Returns:
        task_id to track the task status
    """
    task = update_memory_task.delay(
        memory_id=request.memory_id,
        user_id=request.user_id,
        updates=request.updates,
    )

    return TaskSubmitResponse(
        task_id=task.id,
        status="submitted",
        message=f"Memory update task submitted. Track with task_id: {task.id}",
    )


@router.post("/memory/delete", response_model=TaskSubmitResponse)
async def submit_delete_memory_task(request: MemoryDeleteRequest):
    """
    Submit a memory deletion task to Celery queue.

    Returns:
        task_id to track the task status
    """
    task = delete_memory_task.delay(
        memory_id=request.memory_id,
        user_id=request.user_id,
    )

    return TaskSubmitResponse(
        task_id=task.id,
        status="submitted",
        message=f"Memory deletion task submitted. Track with task_id: {task.id}",
    )


# ============================================================================
# Task Status & Management Endpoints
# ============================================================================


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Check the status of a Celery task.

    Args:
        task_id: The task ID returned when submitting the task

    Returns:
        Current status and result (if completed)
    """
    from celery.result import AsyncResult

    from hippocampai.celery_app import celery_app

    task = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        status=task.status,
    )

    if task.ready():
        if task.successful():
            response.result = task.result
        elif task.failed():
            response.error = str(task.result)

    # Get task info if available
    if hasattr(task, "info") and task.info:
        response.progress = task.info

    return response


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running or pending Celery task.

    Args:
        task_id: The task ID to cancel

    Returns:
        Cancellation status
    """
    from celery.result import AsyncResult

    from hippocampai.celery_app import celery_app

    task = AsyncResult(task_id, app=celery_app)

    if task.state in ["PENDING", "STARTED", "RETRY"]:
        task.revoke(terminate=True)
        return {"message": f"Task {task_id} cancelled", "status": "cancelled"}
    else:
        return {"message": f"Task {task_id} is already {task.state}", "status": task.state}


@router.get("/inspect/stats")
async def get_worker_stats():
    """
    Get Celery worker statistics.

    Returns:
        Worker stats including active tasks, queues, etc.
    """
    from hippocampai.celery_app import celery_app

    inspector = celery_app.control.inspect()

    stats = {
        "active_tasks": inspector.active(),
        "scheduled_tasks": inspector.scheduled(),
        "registered_tasks": inspector.registered(),
        "stats": inspector.stats(),
    }

    return stats


@router.get("/inspect/queues")
async def get_queue_info():
    """
    Get information about Celery queues.

    Returns:
        Queue information including message counts
    """
    from hippocampai.celery_app import celery_app

    inspector = celery_app.control.inspect()

    return {
        "active_queues": inspector.active_queues(),
    }


# ============================================================================
# Scheduled Task Management
# ============================================================================


@router.get("/scheduled")
async def list_scheduled_tasks():
    """
    List all scheduled (periodic) tasks configured in Celery Beat.

    Returns:
        List of scheduled tasks with their schedules
    """
    from hippocampai.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule

    return {
        "scheduled_tasks": {
            name: {
                "task": config["task"],
                "schedule": str(config["schedule"]),
                "options": config.get("options", {}),
            }
            for name, config in schedule.items()
        }
    }


@router.post("/scheduled/{task_name}/run")
async def run_scheduled_task_now(task_name: str):
    """
    Manually trigger a scheduled task to run immediately.

    Args:
        task_name: Name of the scheduled task

    Returns:
        Task ID of the triggered task
    """
    from hippocampai.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule

    if task_name not in schedule:
        raise HTTPException(status_code=404, detail=f"Scheduled task '{task_name}' not found")

    task_path = schedule[task_name]["task"]

    # Get the task function
    task_func = celery_app.tasks.get(task_path)
    if not task_func:
        raise HTTPException(status_code=500, detail=f"Task '{task_path}' not registered")

    # Run the task
    result = task_func.delay()

    return TaskSubmitResponse(
        task_id=result.id,
        status="submitted",
        message=f"Scheduled task '{task_name}' triggered manually. Track with task_id: {result.id}",
    )
