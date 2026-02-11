"""Feedback manager for memory relevance scoring."""

import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback on retrieved memories."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    OUTDATED = "outdated"


# Map feedback types to numeric scores
_FEEDBACK_SCORES: dict[FeedbackType, float] = {
    FeedbackType.RELEVANT: 1.0,
    FeedbackType.PARTIALLY_RELEVANT: 0.6,
    FeedbackType.NOT_RELEVANT: 0.0,
    FeedbackType.OUTDATED: 0.1,
}


class FeedbackEvent(BaseModel):
    """A single feedback event on a memory."""

    memory_id: str
    user_id: str
    query: str = ""
    feedback_type: FeedbackType
    score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackManager:
    """Manages feedback events and computes rolling feedback scores.

    Feedback scores use an exponentially-weighted rolling average with a
    configurable half-life (default 30 days). Memories with no feedback
    return a neutral score of 0.5.
    """

    def __init__(
        self,
        feedback_window_days: int = 90,
        half_life_days: int = 30,
    ) -> None:
        self.feedback_window_days = feedback_window_days
        self.half_life_days = half_life_days
        self._feedback: dict[str, list[FeedbackEvent]] = {}

    def record_feedback(
        self,
        memory_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        query: str = "",
    ) -> FeedbackEvent:
        """Record a feedback event for a memory.

        Args:
            memory_id: The memory being rated.
            user_id: The user providing feedback.
            feedback_type: Type of feedback.
            query: The query that produced this memory in results.

        Returns:
            The recorded FeedbackEvent.
        """
        score = _FEEDBACK_SCORES.get(feedback_type, 0.5)
        event = FeedbackEvent(
            memory_id=memory_id,
            user_id=user_id,
            query=query,
            feedback_type=feedback_type,
            score=score,
        )

        if memory_id not in self._feedback:
            self._feedback[memory_id] = []
        self._feedback[memory_id].append(event)

        logger.debug(
            f"Recorded feedback for memory {memory_id}: "
            f"{feedback_type.value} (score={score:.2f})"
        )
        return event

    def get_memory_feedback_score(self, memory_id: str) -> float:
        """Compute the exponentially-weighted rolling average feedback score.

        Recent feedback is weighted more heavily using exponential decay
        with the configured half-life.

        Args:
            memory_id: The memory to score.

        Returns:
            Score in [0, 1]. Returns 0.5 (neutral) if no feedback exists.
        """
        events = self._feedback.get(memory_id, [])
        if not events:
            return 0.5

        now = datetime.now(timezone.utc)
        cutoff_seconds = self.feedback_window_days * 86400
        decay_lambda = 0.693 / self.half_life_days  # ln(2) / half_life

        weighted_sum = 0.0
        weight_total = 0.0

        for event in events:
            age_days = (now - event.timestamp).total_seconds() / 86400
            if age_days * 86400 > cutoff_seconds:
                continue
            weight = math.exp(-decay_lambda * age_days)
            weighted_sum += event.score * weight
            weight_total += weight

        if weight_total <= 0:
            return 0.5

        return weighted_sum / weight_total

    def get_feedback_events(
        self,
        memory_id: str,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Get feedback events for a memory.

        Args:
            memory_id: The memory ID.
            limit: Max events to return.

        Returns:
            List of FeedbackEvent objects, newest first.
        """
        events = self._feedback.get(memory_id, [])
        sorted_events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return sorted_events[:limit]

    def get_user_feedback_stats(self, user_id: str) -> dict[str, int]:
        """Get feedback statistics for a user.

        Args:
            user_id: The user ID.

        Returns:
            Dict with counts per feedback type and total.
        """
        stats: dict[str, int] = {"total": 0}
        for ft in FeedbackType:
            stats[ft.value] = 0

        for events in self._feedback.values():
            for event in events:
                if event.user_id == user_id:
                    stats["total"] += 1
                    stats[event.feedback_type.value] += 1

        return stats

    def get_aggregated_score(self, memory_id: str) -> Optional[dict[str, float]]:
        """Get aggregated feedback info for a memory.

        Returns:
            Dict with score, event_count, and breakdown, or None if no feedback.
        """
        events = self._feedback.get(memory_id, [])
        if not events:
            return None

        score = self.get_memory_feedback_score(memory_id)
        breakdown: dict[str, int] = {}
        for ft in FeedbackType:
            breakdown[ft.value] = sum(
                1 for e in events if e.feedback_type == ft
            )

        return {
            "score": score,
            "event_count": float(len(events)),
            **{k: float(v) for k, v in breakdown.items()},
        }
