"""Models for auto-healing and maintenance."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class HealingActionType(str, Enum):
    """Types of healing actions."""

    DELETE = "delete"  # Delete memory
    ARCHIVE = "archive"  # Archive memory
    MERGE = "merge"  # Merge duplicate memories
    UPDATE_TAGS = "update_tags"  # Update or add tags
    ADJUST_IMPORTANCE = "adjust_importance"  # Adjust importance score
    UPDATE_CONFIDENCE = "update_confidence"  # Adjust confidence
    CONSOLIDATE = "consolidate"  # Consolidate similar memories
    REFRESH = "refresh"  # Refresh stale memory


class HealingAction(BaseModel):
    """Action to heal/maintain memory health."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: HealingActionType
    memory_ids: list[str] = Field(description="Memories affected by this action")
    reason: str = Field(description="Why this action was recommended")
    impact: str = Field(description="Expected impact of this action")
    reversible: bool = Field(description="Whether this action can be undone")
    auto_applicable: bool = Field(
        default=False, description="Whether this can be auto-applied without user approval"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None  # user_id or "system"
    is_applied: bool = False
    was_successful: Optional[bool] = None
    error_message: Optional[str] = None

    def apply(self, applied_by: str = "system") -> None:
        """Mark action as applied."""
        self.is_applied = True
        self.applied_at = datetime.now(timezone.utc)
        self.applied_by = applied_by


class MaintenanceTaskType(str, Enum):
    """Types of maintenance tasks."""

    DEDUP = "dedup"  # Remove duplicates
    CLEANUP = "cleanup"  # Clean stale memories
    REINDEX = "reindex"  # Reindex for search
    HEALTH_CHECK = "health_check"  # Run health assessment
    CONSOLIDATE = "consolidate"  # Consolidate memories
    BACKUP = "backup"  # Backup memories
    OPTIMIZE = "optimize"  # Optimize storage/performance


class MaintenanceStatus(str, Enum):
    """Status of maintenance task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MaintenanceTask(BaseModel):
    """Scheduled maintenance task."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    task_type: MaintenanceTaskType
    schedule: str = Field(description="Cron expression (e.g., '0 2 * * *' for daily at 2 AM)")
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: Optional[MaintenanceStatus] = None
    next_run: datetime
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration_seconds: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict, description="Task-specific configuration")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MaintenanceRun(BaseModel):
    """Single execution of a maintenance task."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    user_id: str
    status: MaintenanceStatus = MaintenanceStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    actions_taken: list[str] = Field(
        default_factory=list, description="IDs of healing actions taken"
    )
    memories_affected: int = 0
    results: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        """Mark run as started."""
        self.status = MaintenanceStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self, success: bool = True) -> None:
        """Mark run as completed."""
        self.status = MaintenanceStatus.COMPLETED if success else MaintenanceStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class HealingReport(BaseModel):
    """Report from auto-healing run."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    run_type: Literal["full_health_check", "cleanup", "dedup", "consolidate", "custom"]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    actions_recommended: list[HealingAction] = Field(default_factory=list)
    actions_applied: list[HealingAction] = Field(default_factory=list)
    memories_analyzed: int
    memories_affected: int
    health_before: float = Field(ge=0.0, le=100.0)
    health_after: float = Field(ge=0.0, le=100.0)
    improvements: dict[str, float] = Field(default_factory=dict, description="Metrics before/after")
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationStrategy(str, Enum):
    """Strategies for memory consolidation."""

    SEMANTIC_MERGE = "semantic_merge"  # Merge semantically similar
    TIME_BASED = "time_based"  # Consolidate by time period
    TYPE_BASED = "type_based"  # Consolidate by type
    TAG_BASED = "tag_based"  # Consolidate by tags
    IMPORTANCE_WEIGHTED = "importance_weighted"  # Keep most important


class ConsolidationResult(BaseModel):
    """Result of memory consolidation."""

    consolidation_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    strategy: ConsolidationStrategy
    original_memory_ids: list[str] = Field(description="Memories that were consolidated")
    consolidated_memory_id: Optional[str] = Field(description="New consolidated memory")
    summary: str
    content_preserved: float = Field(
        ge=0.0, le=1.0, description="How much original content was preserved"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reversible: bool = True
    original_data: Optional[dict[str, Any]] = Field(None, description="Original data for reversal")


class AutoHealingConfig(BaseModel):
    """Configuration for auto-healing system."""

    user_id: str
    enabled: bool = True
    auto_cleanup_enabled: bool = True
    auto_dedup_enabled: bool = True
    auto_consolidate_enabled: bool = False  # More aggressive, off by default
    auto_tag_enabled: bool = True
    cleanup_threshold_days: int = 90  # Archive memories older than this with no access
    dedup_similarity_threshold: float = 0.90  # Merge memories more similar than this
    consolidate_threshold: int = 5  # Consolidate if >5 very similar memories
    require_user_approval: bool = True  # Require approval for major changes
    max_actions_per_run: int = 50  # Limit actions per run
    schedule_cleanup: Optional[str] = "0 2 * * 0"  # Weekly at 2 AM Sunday
    schedule_health_check: Optional[str] = "0 3 * * *"  # Daily at 3 AM
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthImprovement(BaseModel):
    """Improvement in health metrics."""

    metric_name: str
    value_before: float
    value_after: float
    improvement_percent: float
    is_improvement: bool = Field(description="True if change is positive (depends on metric)")
