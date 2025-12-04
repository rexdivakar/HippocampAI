"""Data models for memory consolidation and sleep phase processing."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConsolidationStatus(str, Enum):
    """Status of a consolidation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConsolidationRun(BaseModel):
    """Tracks a single consolidation run (nightly dream cycle)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: Optional[str] = None

    # Execution metadata
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ConsolidationStatus = ConsolidationStatus.PENDING
    duration_seconds: float = 0.0

    # Input parameters
    lookback_hours: int = 24
    min_importance: float = 3.0
    dry_run: bool = False

    # Processing statistics
    memories_reviewed: int = 0
    memories_deleted: int = 0
    memories_archived: int = 0
    memories_promoted: int = 0
    memories_updated: int = 0
    memories_synthesized: int = 0
    clusters_created: int = 0
    llm_calls_made: int = 0

    # Error tracking
    error_message: Optional[str] = None
    error_stacktrace: Optional[str] = None

    # Metadata and debugging
    metadata: dict[str, Any] = Field(default_factory=dict)
    dream_report: Optional[str] = None  # Human-readable summary

    class Config:
        json_schema_extra = {
            "example": {
                "id": "run-abc123",
                "user_id": "user-xyz",
                "status": "completed",
                "memories_reviewed": 50,
                "memories_deleted": 5,
                "memories_promoted": 10,
                "memories_synthesized": 3,
                "duration_seconds": 12.5,
                "dream_report": "Consolidated 50 memories: promoted important meetings, archived transient events.",
            }
        }


class ConsolidationDecision(BaseModel):
    """LLM's decision about how to consolidate memories."""

    # Memories to promote (increase importance)
    promoted_facts: list[dict[str, Any]] = Field(default_factory=list)

    # Memories to delete or archive (low value)
    low_value_memory_ids: list[str] = Field(default_factory=list)

    # Existing memories to update
    updated_memories: list[dict[str, Any]] = Field(default_factory=list)

    # New synthetic/summary memories to create
    synthetic_memories: list[dict[str, Any]] = Field(default_factory=list)

    # Optional: reasoning/explanation
    reasoning: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "promoted_facts": [
                    {"id": "mem-123", "reason": "Important strategic decision", "new_importance": 8.5}
                ],
                "low_value_memory_ids": ["mem-456", "mem-789"],
                "updated_memories": [
                    {
                        "id": "mem-111",
                        "new_text": "Q4 roadmap meeting: focus on AI-driven features",
                        "new_importance": 8.0,
                        "merge_from_ids": ["mem-112", "mem-113"],
                    }
                ],
                "synthetic_memories": [
                    {
                        "text": "Morning productive session: completed project planning and Q4 roadmap review",
                        "type": "context",
                        "importance": 7.0,
                        "tags": ["work", "planning", "Q4"],
                        "source_ids": ["mem-111", "mem-123"],
                    }
                ],
                "reasoning": "User had productive morning focused on strategic planning. Low-value transient events archived.",
            }
        }


class MemoryCluster(BaseModel):
    """A group of related memories for consolidation."""

    cluster_id: str = Field(default_factory=lambda: str(uuid4()))
    memories: list[str]  # Memory IDs
    theme: Optional[str] = None  # Auto-generated or LLM-suggested theme
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    avg_importance: float = 0.0
    tags: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster-morning",
                "memories": ["mem-1", "mem-2", "mem-3"],
                "theme": "Morning work routine",
                "avg_importance": 5.5,
                "tags": ["work", "morning", "routine"],
            }
        }


class ConsolidationStats(BaseModel):
    """Summary statistics for observability."""

    total_runs: int = 0
    total_users_processed: int = 0
    total_memories_reviewed: int = 0
    total_memories_deleted: int = 0
    total_memories_archived: int = 0
    total_memories_promoted: int = 0
    total_memories_synthesized: int = 0
    avg_duration_seconds: float = 0.0
    success_rate: float = 1.0  # Percentage of successful runs

    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None


# Extended Memory fields for consolidation
# (These would be added to the existing Memory model in memory.py)
class MemoryConsolidationFields(BaseModel):
    """Fields to add to the existing Memory model for consolidation support."""

    # Archival
    is_archived: bool = False
    archived_at: Optional[datetime] = None

    # Consolidation tracking
    source_memory_ids: Optional[list[str]] = None  # For synthetic memories
    consolidation_run_id: Optional[str] = None
    last_consolidated_at: Optional[datetime] = None

    # Importance decay
    decay_factor: float = 0.95  # Daily decay multiplier (0.0-1.0)
    base_importance: Optional[float] = None  # Original importance before decay
    promotion_count: int = 0  # Times promoted by consolidation

    # Metadata
    consolidation_metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "is_archived": False,
                "source_memory_ids": ["mem-101", "mem-102"],
                "consolidation_run_id": "run-abc123",
                "decay_factor": 0.95,
                "promotion_count": 2,
                "consolidation_metadata": {"promoted_reason": "Strategic importance", "cluster_id": "cluster-work"},
            }
        }
