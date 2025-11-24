"""Background task management for SaaS deployment.

Provides task queue integration that works with Celery, Redis Queue, or any task backend.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundTask(BaseModel):
    """Background task model."""

    task_id: str = Field(
        default_factory=lambda: f"task_{int(datetime.now(timezone.utc).timestamp())}"
    )
    user_id: str
    task_type: str  # summarization, consolidation, compression, etc.
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # Execution details
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    # Retry logic
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskManager:
    """
    Task manager for background job execution.

    Provides abstraction over different task backends (Celery, RQ, etc.)
    """

    def __init__(self, automation_controller: Any, backend: Optional[str] = None) -> None:
        """
        Initialize task manager.

        Args:
            automation_controller: AutomationController instance
            backend: Task backend ('celery', 'rq', 'inline', None)
        """
        self.automation_controller = automation_controller
        self.backend = backend or "inline"
        self.tasks: dict[str, Any] = {}  # In-memory task store

        # Try to import task backend
        self.celery_app = None
        self.rq_queue = None

        if self.backend == "celery":
            try:
                from celery import Celery

                self.celery_app = Celery("hippocampai")
                logger.info("Celery backend initialized")
            except ImportError:
                logger.warning("Celery not installed, falling back to inline execution")
                self.backend = "inline"

        elif self.backend == "rq":
            try:
                from redis import Redis
                from rq import Queue

                redis_conn = Redis()
                self.rq_queue = Queue(connection=redis_conn)
                logger.info("Redis Queue backend initialized")
            except ImportError:
                logger.warning("RQ not installed, falling back to inline execution")
                self.backend = "inline"

    def create_task(
        self,
        user_id: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[dict] = None,
    ) -> BackgroundTask:
        """
        Create a new background task.

        Args:
            user_id: User identifier
            task_type: Type of task (summarization, consolidation, etc.)
            priority: Task priority
            metadata: Additional metadata

        Returns:
            Created task
        """
        task = BackgroundTask(
            user_id=user_id,
            task_type=task_type,
            priority=priority,
            metadata=metadata or {},
        )

        self.tasks[task.task_id] = task
        logger.info(f"Created task {task.task_id} for user {user_id}: {task_type}")

        return task

    def execute_task(self, task: BackgroundTask) -> BackgroundTask:
        """
        Execute a task synchronously.

        Args:
            task: Task to execute

        Returns:
            Updated task with results
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)

        try:
            # Route to appropriate handler
            if task.task_type == "summarization":
                result = self.automation_controller.run_summarization(task.user_id, force=True)
            elif task.task_type == "consolidation":
                result = self.automation_controller.run_consolidation(task.user_id, force=True)
            elif task.task_type == "compression":
                result = self.automation_controller.run_compression(task.user_id, force=True)
            elif task.task_type == "decay":
                result = self.automation_controller.run_decay(task.user_id, force=True)
            elif task.task_type == "health_check":
                result = self.automation_controller.run_health_check(task.user_id, force=True)
            elif task.task_type == "all_optimizations":
                result = self.automation_controller.run_all_optimizations(task.user_id, force=True)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            task.status = TaskStatus.COMPLETED
            task.result = result
            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(
                    f"Task {task.task_id} will be retried ({task.retry_count}/{task.max_retries})"
                )

        task.completed_at = datetime.now(timezone.utc)
        self.tasks[task.task_id] = task

        return task

    def submit_task(
        self,
        user_id: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[dict] = None,
        async_execution: bool = True,
    ) -> BackgroundTask:
        """
        Submit a task for execution.

        Args:
            user_id: User identifier
            task_type: Type of task
            priority: Task priority
            metadata: Additional metadata
            async_execution: Execute asynchronously if backend supports it

        Returns:
            Created task
        """
        task = self.create_task(user_id, task_type, priority, metadata)

        if not async_execution or self.backend == "inline":
            # Execute immediately
            task = self.execute_task(task)

        elif self.backend == "celery" and self.celery_app:
            # Submit to Celery
            self.celery_app.send_task(
                "hippocampai.tasks.execute_task",
                args=[task.task_id],
                priority=self._priority_to_int(priority),
            )

        elif self.backend == "rq" and self.rq_queue:
            # Submit to Redis Queue
            self.rq_queue.enqueue(
                self.execute_task,
                task,
                job_id=task.task_id,
            )

        return task

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_user_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> list[BackgroundTask]:
        """
        Get tasks for a user.

        Args:
            user_id: User identifier
            status: Filter by status
            limit: Maximum number of tasks

        Returns:
            List of tasks
        """
        tasks = [t for t in self.tasks.values() if t.user_id == user_id]

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            self.tasks[task_id] = task
            logger.info(f"Cancelled task {task_id}")
            return True
        return False

    def _priority_to_int(self, priority: TaskPriority) -> int:
        """Convert priority enum to integer for Celery."""
        priority_map = {
            TaskPriority.LOW: 3,
            TaskPriority.NORMAL: 5,
            TaskPriority.HIGH: 7,
            TaskPriority.CRITICAL: 9,
        }
        return priority_map.get(priority, 5)

    def run_scheduled_tasks(self, user_ids: Optional[list[str]] = None) -> None:
        """
        Run scheduled tasks for users based on their policies.

        This should be called by a cron job or periodic task scheduler.

        Args:
            user_ids: List of user IDs to process (None = all users)
        """
        if user_ids is None:
            # Get all users with policies
            user_ids = list(self.automation_controller.policies.keys())

        logger.info(f"Running scheduled tasks for {len(user_ids)} users")

        for user_id in user_ids:
            policy = self.automation_controller.get_policy(user_id)

            if not policy or not policy.enabled:
                continue

            # Check each feature and submit tasks if needed
            if policy.auto_summarization and self.automation_controller.should_run_summarization(
                user_id
            ):
                self.submit_task(user_id, "summarization", priority=TaskPriority.NORMAL)

            if policy.auto_consolidation and self.automation_controller.should_run_consolidation(
                user_id
            ):
                self.submit_task(user_id, "consolidation", priority=TaskPriority.NORMAL)

            if policy.auto_compression and self.automation_controller.should_run_compression(
                user_id
            ):
                self.submit_task(user_id, "compression", priority=TaskPriority.LOW)

            if policy.importance_decay:
                self.submit_task(user_id, "decay", priority=TaskPriority.LOW)

            if policy.health_monitoring:
                self.submit_task(user_id, "health_check", priority=TaskPriority.NORMAL)

        logger.info("Scheduled tasks submission completed")
