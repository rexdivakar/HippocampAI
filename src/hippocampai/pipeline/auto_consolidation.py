"""Automatic memory consolidation scheduler and coordinator.

This module provides:
- Automatic consolidation scheduling
- Consolidation strategy management
- Batch consolidation operations
- Consolidation history and tracking
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class ConsolidationTrigger(str, Enum):
    """Triggers for automatic consolidation."""

    SCHEDULED = "scheduled"  # Time-based trigger
    THRESHOLD = "threshold"  # Memory count threshold
    TOKEN_BUDGET = "token_budget"  # Token limit reached
    MANUAL = "manual"  # Manually triggered
    SIMILARITY_DETECTED = "similarity_detected"  # Similar memories detected


class ConsolidationStatus(str, Enum):
    """Status of consolidation operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConsolidationResult(BaseModel):
    """Result of a consolidation operation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    trigger: ConsolidationTrigger
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ConsolidationStatus = ConsolidationStatus.PENDING
    memories_analyzed: int = 0
    memories_consolidated: int = 0
    consolidation_groups: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def calculate_efficiency(self) -> float:
        """Calculate consolidation efficiency (tokens saved / tokens before)."""
        if self.tokens_before == 0:
            return 0.0
        return (self.tokens_before - self.tokens_after) / self.tokens_before


class ConsolidationSchedule(BaseModel):
    """Consolidation schedule configuration."""

    enabled: bool = True
    interval_hours: int = 168  # Default: weekly
    min_memories_threshold: int = 50  # Minimum memories before consolidation
    similarity_threshold: float = 0.85  # Similarity threshold for consolidation
    max_batch_size: int = 100  # Maximum memories per consolidation run
    token_budget_threshold: int = 10000  # Trigger if tokens exceed this
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class AutoConsolidator:
    """Automatic memory consolidation manager."""

    def __init__(
        self,
        schedule: Optional[ConsolidationSchedule] = None,
        consolidator: Optional[Any] = None,  # MemoryConsolidator instance
        similarity_calculator: Optional[Callable] = None,
    ):
        """Initialize auto consolidator.

        Args:
            schedule: Consolidation schedule configuration
            consolidator: Memory consolidator instance
            similarity_calculator: Function to calculate similarity between memories
        """
        self.schedule = schedule or ConsolidationSchedule()
        self.consolidator = consolidator
        self.similarity_calculator = similarity_calculator
        self.history: list[ConsolidationResult] = []

    def should_consolidate(
        self, memories: list[Memory], current_time: Optional[datetime] = None
    ) -> tuple[bool, Optional[ConsolidationTrigger]]:
        """Determine if consolidation should be triggered.

        Args:
            memories: List of memories to evaluate
            current_time: Current timestamp

        Returns:
            Tuple of (should_consolidate, trigger_reason)
        """
        if not self.schedule.enabled:
            return False, None

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check scheduled trigger
        if self.schedule.last_run:
            hours_since_last = (current_time - self.schedule.last_run).total_seconds() / 3600
            if hours_since_last >= self.schedule.interval_hours:
                if len(memories) >= self.schedule.min_memories_threshold:
                    return True, ConsolidationTrigger.SCHEDULED

        # Check threshold trigger
        if len(memories) >= self.schedule.min_memories_threshold * 2:
            return True, ConsolidationTrigger.THRESHOLD

        # Check token budget trigger
        total_tokens = sum(Memory.estimate_tokens(m.text) for m in memories)
        if total_tokens >= self.schedule.token_budget_threshold:
            return True, ConsolidationTrigger.TOKEN_BUDGET

        return False, None

    def find_consolidation_groups(
        self, memories: list[Memory], similarity_threshold: Optional[float] = None
    ) -> list[list[Memory]]:
        """Find groups of similar memories that should be consolidated.

        Args:
            memories: List of memories to analyze
            similarity_threshold: Similarity threshold (uses config if not provided)

        Returns:
            List of memory groups to consolidate
        """
        if similarity_threshold is None:
            similarity_threshold = self.schedule.similarity_threshold

        if not self.similarity_calculator:
            # Fallback: group by tags and type
            return self._group_by_heuristics(memories)

        # Find similar memory pairs
        groups = []
        processed = set()

        for i, mem1 in enumerate(memories):
            if mem1.id in processed:
                continue

            group = [mem1]
            processed.add(mem1.id)

            for mem2 in memories[i + 1 :]:
                if mem2.id in processed:
                    continue

                # Calculate similarity
                try:
                    similarity = self.similarity_calculator(mem1, mem2)
                    if similarity >= similarity_threshold:
                        group.append(mem2)
                        processed.add(mem2.id)
                except Exception as e:
                    logger.warning(f"Similarity calculation failed: {e}")
                    continue

            # Only add groups with 2+ memories
            if len(group) >= 2:
                groups.append(group)

        return groups

    def _group_by_heuristics(self, memories: list[Memory]) -> list[list[Memory]]:
        """Group memories using simple heuristics (fallback).

        Args:
            memories: Memories to group

        Returns:
            List of memory groups
        """
        # Group by type and overlapping tags
        groups_dict: dict[str, list[Memory]] = {}

        for memory in memories:
            # Create key from type and first tag
            key = memory.type.value
            if memory.tags:
                key += "_" + memory.tags[0]

            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(memory)

        # Filter groups with 2+ members
        return [group for group in groups_dict.values() if len(group) >= 2]

    def consolidate_batch(
        self,
        memories: list[Memory],
        trigger: ConsolidationTrigger = ConsolidationTrigger.MANUAL,
        current_time: Optional[datetime] = None,
    ) -> ConsolidationResult:
        """Consolidate a batch of memories.

        Args:
            memories: Memories to consolidate
            trigger: What triggered the consolidation
            current_time: Current timestamp

        Returns:
            ConsolidationResult with statistics
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        result = ConsolidationResult(
            trigger=trigger,
            started_at=current_time,
            status=ConsolidationStatus.RUNNING,
            memories_analyzed=len(memories),
            tokens_before=sum(Memory.estimate_tokens(m.text) for m in memories),
        )

        try:
            # Find consolidation groups
            groups = self.find_consolidation_groups(memories)
            result.consolidation_groups = len(groups)

            consolidated_memories = []
            tokens_after = 0

            # Consolidate each group
            for group in groups:
                if self.consolidator:
                    consolidated = self.consolidator.consolidate(group)
                    if consolidated:
                        consolidated_memories.append(consolidated)
                        tokens_after += Memory.estimate_tokens(consolidated.text)
                    else:
                        # Keep originals if consolidation failed
                        for mem in group:
                            tokens_after += Memory.estimate_tokens(mem.text)
                else:
                    # No consolidator - keep originals
                    for mem in group:
                        tokens_after += Memory.estimate_tokens(mem.text)

            # Add ungrouped memories to token count
            grouped_ids = set()
            for group in groups:
                for mem in group:
                    grouped_ids.add(mem.id)

            for memory in memories:
                if memory.id not in grouped_ids:
                    tokens_after += Memory.estimate_tokens(memory.text)

            result.memories_consolidated = sum(len(group) for group in groups)
            result.tokens_after = tokens_after
            result.completed_at = datetime.now(timezone.utc)
            result.status = ConsolidationStatus.COMPLETED

            # Update schedule
            self.schedule.last_run = current_time
            self.schedule.next_run = current_time + timedelta(hours=self.schedule.interval_hours)

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            result.status = ConsolidationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now(timezone.utc)

        # Add to history
        self.history.append(result)

        return result

    def auto_consolidate(
        self, memories: list[Memory], current_time: Optional[datetime] = None
    ) -> Optional[ConsolidationResult]:
        """Automatically consolidate if conditions are met.

        Args:
            memories: Memories to potentially consolidate
            current_time: Current timestamp

        Returns:
            ConsolidationResult if consolidation was triggered, None otherwise
        """
        should_run, trigger = self.should_consolidate(memories, current_time)

        if not should_run or trigger is None:
            return None

        logger.info(f"Auto-consolidation triggered: {trigger.value} ({len(memories)} memories)")

        return self.consolidate_batch(memories, trigger, current_time)

    def get_consolidation_stats(self) -> dict[str, Any]:
        """Get statistics from consolidation history.

        Returns:
            Dictionary with consolidation statistics
        """
        if not self.history:
            return {
                "total_runs": 0,
                "message": "No consolidation history available",
            }

        completed = [r for r in self.history if r.status == ConsolidationStatus.COMPLETED]
        failed = [r for r in self.history if r.status == ConsolidationStatus.FAILED]

        total_memories_analyzed = sum(r.memories_analyzed for r in completed)
        total_memories_consolidated = sum(r.memories_consolidated for r in completed)
        total_tokens_saved = sum(r.tokens_before - r.tokens_after for r in completed)

        avg_efficiency = (
            sum(r.calculate_efficiency() for r in completed) / len(completed) if completed else 0.0
        )

        return {
            "total_runs": len(self.history),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "total_memories_analyzed": total_memories_analyzed,
            "total_memories_consolidated": total_memories_consolidated,
            "total_tokens_saved": total_tokens_saved,
            "average_efficiency": avg_efficiency,
            "last_run": self.schedule.last_run.isoformat() if self.schedule.last_run else None,
            "next_scheduled_run": (
                self.schedule.next_run.isoformat() if self.schedule.next_run else None
            ),
        }

    def get_recent_results(self, limit: int = 10) -> list[ConsolidationResult]:
        """Get recent consolidation results.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent results
        """
        return sorted(self.history, key=lambda r: r.started_at, reverse=True)[:limit]

    def estimate_consolidation_impact(self, memories: list[Memory]) -> dict[str, Any]:
        """Estimate potential impact of consolidation without actually consolidating.

        Args:
            memories: Memories to analyze

        Returns:
            Estimated impact metrics
        """
        # Find potential groups
        groups = self.find_consolidation_groups(memories)

        if not groups:
            return {
                "consolidation_possible": False,
                "estimated_groups": 0,
                "message": "No similar memories found for consolidation",
            }

        # Estimate token savings
        current_tokens = sum(Memory.estimate_tokens(m.text) for m in memories)

        # Estimate consolidated tokens (assume 60% of original for consolidated groups)
        estimated_consolidated_tokens = 0
        grouped_ids = set()

        for group in groups:
            group_tokens = sum(Memory.estimate_tokens(m.text) for m in group)
            # Assume consolidation results in 60% of combined tokens
            estimated_consolidated_tokens += int(group_tokens * 0.6)
            for mem in group:
                grouped_ids.add(mem.id)

        # Add ungrouped memory tokens
        for memory in memories:
            if memory.id not in grouped_ids:
                estimated_consolidated_tokens += Memory.estimate_tokens(memory.text)

        estimated_tokens_saved = current_tokens - estimated_consolidated_tokens
        estimated_efficiency = (
            estimated_tokens_saved / current_tokens if current_tokens > 0 else 0.0
        )

        return {
            "consolidation_possible": True,
            "total_memories": len(memories),
            "estimated_groups": len(groups),
            "memories_in_groups": sum(len(g) for g in groups),
            "memories_ungrouped": len(memories) - sum(len(g) for g in groups),
            "current_tokens": current_tokens,
            "estimated_tokens_after": estimated_consolidated_tokens,
            "estimated_tokens_saved": estimated_tokens_saved,
            "estimated_efficiency": estimated_efficiency,
            "recommendation": (
                "Consolidation recommended" if estimated_tokens_saved > 500 else "Minor benefit"
            ),
        }

    def cancel_pending_consolidation(self) -> bool:
        """Cancel any pending consolidation operations.

        Returns:
            True if a consolidation was cancelled
        """
        for result in self.history:
            if result.status == ConsolidationStatus.PENDING:
                result.status = ConsolidationStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc)
                return True
        return False

    def update_schedule(self, **kwargs) -> None:
        """Update consolidation schedule parameters.

        Args:
            **kwargs: Schedule parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.schedule, key):
                setattr(self.schedule, key, value)
                logger.info(f"Updated consolidation schedule: {key} = {value}")

        # Recalculate next run if interval changed
        if "interval_hours" in kwargs and self.schedule.last_run:
            self.schedule.next_run = self.schedule.last_run + timedelta(
                hours=self.schedule.interval_hours
            )
