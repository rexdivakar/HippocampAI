"""Importance decay and intelligent memory pruning system.

This module provides:
- Time-based importance decay with configurable curves
- Intelligent pruning based on multiple factors
- Memory health scoring
- Automatic cleanup recommendations
"""

import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class DecayFunction(str, Enum):
    """Types of decay functions."""

    LINEAR = "linear"  # Linear decay over time
    EXPONENTIAL = "exponential"  # Exponential decay (faster)
    LOGARITHMIC = "logarithmic"  # Logarithmic decay (slower)
    STEP = "step"  # Step-based decay (discrete intervals)
    HYBRID = "hybrid"  # Combination of exponential and access-based


class PruningStrategy(str, Enum):
    """Memory pruning strategies."""

    IMPORTANCE_ONLY = "importance_only"  # Prune based on importance score only
    AGE_BASED = "age_based"  # Prune based on age
    ACCESS_BASED = "access_based"  # Prune based on access frequency
    COMPREHENSIVE = "comprehensive"  # Multi-factor analysis
    CONSERVATIVE = "conservative"  # Only prune very low value memories


class MemoryHealth(BaseModel):
    """Health score for a memory."""

    memory_id: str
    health_score: float = Field(ge=0.0, le=10.0)
    importance_score: float
    recency_score: float
    access_score: float
    confidence_score: float
    recommendation: str  # "keep", "decay", "prune", "archive"
    factors: dict[str, Any] = Field(default_factory=dict)


class DecayConfig(BaseModel):
    """Configuration for importance decay."""

    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    half_life_days: dict[str, int] = Field(
        default_factory=lambda: {
            "preference": 90,
            "goal": 60,
            "fact": 30,
            "event": 14,
            "context": 30,
            "habit": 90,
        }
    )
    min_importance: float = 1.0
    access_boost_factor: float = 0.5  # Boost for frequently accessed memories
    confidence_weight: float = 0.3  # Weight for confidence in decay calculation


class ImportanceDecayEngine:
    """Manages importance decay and memory pruning."""

    def __init__(self, config: Optional[DecayConfig] = None):
        """Initialize decay engine.

        Args:
            config: Decay configuration
        """
        self.config = config or DecayConfig()

    def calculate_decayed_importance(
        self, memory: Memory, current_time: Optional[datetime] = None
    ) -> float:
        """Calculate current importance after decay.

        Args:
            memory: Memory to evaluate
            current_time: Current timestamp (defaults to now)

        Returns:
            Decayed importance score
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate age in days
        age_days = (current_time - memory.created_at).total_seconds() / 86400

        # Get half-life for memory type
        half_life = self.config.half_life_days.get(memory.type.value, 30)

        # Apply decay function
        if self.config.decay_function == DecayFunction.LINEAR:
            decay_factor = self._linear_decay(age_days, half_life)
        elif self.config.decay_function == DecayFunction.EXPONENTIAL:
            decay_factor = self._exponential_decay(age_days, half_life)
        elif self.config.decay_function == DecayFunction.LOGARITHMIC:
            decay_factor = self._logarithmic_decay(age_days, half_life)
        elif self.config.decay_function == DecayFunction.STEP:
            decay_factor = self._step_decay(age_days, half_life)
        else:  # HYBRID
            decay_factor = self._hybrid_decay(memory, age_days, half_life)

        # Apply access boost
        if memory.access_count > 0:
            access_boost = min(memory.access_count * self.config.access_boost_factor, 3.0)
            decay_factor = min(decay_factor + (access_boost * 0.1), 1.0)

        # Calculate decayed importance
        decayed = memory.importance * decay_factor

        # Apply confidence weight
        confidence_factor = 1.0 + ((memory.confidence - 0.5) * self.config.confidence_weight * 2)
        decayed *= confidence_factor

        # Enforce minimum
        return max(decayed, self.config.min_importance)

    def _linear_decay(self, age_days: float, half_life: int) -> float:
        """Linear decay function.

        Args:
            age_days: Age in days
            half_life: Half-life in days

        Returns:
            Decay factor (0.0-1.0)
        """
        if age_days >= half_life * 2:
            return 0.0
        return max(0.0, 1.0 - (age_days / (half_life * 2)))

    def _exponential_decay(self, age_days: float, half_life: int) -> float:
        """Exponential decay function (classic radioactive decay).

        Args:
            age_days: Age in days
            half_life: Half-life in days

        Returns:
            Decay factor (0.0-1.0)
        """
        return math.pow(0.5, age_days / half_life)

    def _logarithmic_decay(self, age_days: float, half_life: int) -> float:
        """Logarithmic decay function (slower decay).

        Args:
            age_days: Age in days
            half_life: Half-life in days

        Returns:
            Decay factor (0.0-1.0)
        """
        if age_days <= 0:
            return 1.0
        # Logarithmic decay: slower than exponential
        decay = 1.0 - (math.log(1 + age_days) / math.log(1 + half_life * 2))
        return max(0.0, min(1.0, decay))

    def _step_decay(self, age_days: float, half_life: int) -> float:
        """Step-based decay function.

        Args:
            age_days: Age in days
            half_life: Half-life in days

        Returns:
            Decay factor (0.0-1.0)
        """
        # Decay in discrete steps
        if age_days < half_life:
            return 1.0
        elif age_days < half_life * 2:
            return 0.75
        elif age_days < half_life * 3:
            return 0.5
        elif age_days < half_life * 4:
            return 0.25
        else:
            return 0.1

    def _hybrid_decay(self, memory: Memory, age_days: float, half_life: int) -> float:
        """Hybrid decay combining exponential decay and access patterns.

        Args:
            memory: Memory object
            age_days: Age in days
            half_life: Half-life in days

        Returns:
            Decay factor (0.0-1.0)
        """
        # Base exponential decay
        base_decay = self._exponential_decay(age_days, half_life)

        # Access-based modification
        if memory.access_count > 0:
            # Recent access reduces decay
            days_since_access = (
                datetime.now(timezone.utc) - memory.updated_at
            ).total_seconds() / 86400
            access_factor = math.exp(-days_since_access / (half_life / 2))
            # Weight by access frequency
            access_weight = min(memory.access_count / 10.0, 1.0)
            modified_decay = base_decay + (access_factor * access_weight * 0.3)
            return min(modified_decay, 1.0)

        return base_decay

    def calculate_memory_health(
        self, memory: Memory, current_time: Optional[datetime] = None
    ) -> MemoryHealth:
        """Calculate comprehensive health score for memory.

        Args:
            memory: Memory to evaluate
            current_time: Current timestamp

        Returns:
            MemoryHealth assessment
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate component scores
        importance_score = memory.importance / 10.0  # Normalize to 0-1

        # Recency score (newer = higher)
        age_days = (current_time - memory.created_at).total_seconds() / 86400
        recency_score = math.exp(-age_days / 30.0)  # 30-day half-life

        # Access score (more accessed = higher)
        access_score = min(memory.access_count / 20.0, 1.0)  # Cap at 20 accesses

        # Confidence score
        confidence_score = memory.confidence

        # Calculate overall health (weighted average)
        health_score = (
            importance_score * 0.4
            + recency_score * 0.25
            + access_score * 0.20
            + confidence_score * 0.15
        ) * 10.0  # Scale to 0-10

        # Determine recommendation
        if health_score >= 7.0:
            recommendation = "keep"
        elif health_score >= 5.0:
            recommendation = "decay"
        elif health_score >= 3.0:
            recommendation = "archive"
        else:
            recommendation = "prune"

        return MemoryHealth(
            memory_id=memory.id,
            health_score=health_score,
            importance_score=importance_score * 10.0,
            recency_score=recency_score * 10.0,
            access_score=access_score * 10.0,
            confidence_score=confidence_score * 10.0,
            recommendation=recommendation,
            factors={
                "age_days": age_days,
                "access_count": memory.access_count,
                "original_importance": memory.importance,
                "decayed_importance": self.calculate_decayed_importance(memory, current_time),
            },
        )

    def identify_pruning_candidates(
        self,
        memories: list[Memory],
        strategy: PruningStrategy = PruningStrategy.COMPREHENSIVE,
        target_count: Optional[int] = None,
        min_health_threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Identify memories that are candidates for pruning.

        Args:
            memories: List of memories to evaluate
            strategy: Pruning strategy to use
            target_count: Target number of memories to keep (optional)
            min_health_threshold: Minimum health score to keep

        Returns:
            Dictionary with pruning recommendations
        """
        if not memories:
            return {"candidates": [], "stats": {}}

        # Calculate health for all memories
        health_scores = []
        for memory in memories:
            health = self.calculate_memory_health(memory)
            health_scores.append(health)

        # Apply pruning strategy
        if strategy == PruningStrategy.IMPORTANCE_ONLY:
            candidates = [
                h
                for h in health_scores
                if h.factors["decayed_importance"] < self.config.min_importance * 2
            ]
        elif strategy == PruningStrategy.AGE_BASED:
            candidates = [h for h in health_scores if h.factors["age_days"] > 90]
        elif strategy == PruningStrategy.ACCESS_BASED:
            candidates = [
                h
                for h in health_scores
                if h.factors["access_count"] == 0 and h.factors["age_days"] > 30
            ]
        elif strategy == PruningStrategy.COMPREHENSIVE:
            candidates = [h for h in health_scores if h.health_score < min_health_threshold]
        else:  # CONSERVATIVE
            candidates = [h for h in health_scores if h.health_score < 2.0]

        # If target count specified, ensure we meet it
        if target_count and len(memories) - len(candidates) > target_count:
            # Need to prune more - sort by health and take lowest
            sorted_health = sorted(health_scores, key=lambda h: h.health_score)
            prune_count = len(memories) - target_count
            candidates = sorted_health[:prune_count]

        # Calculate statistics
        total_memories = len(memories)
        candidates_count = len(candidates)
        keep_count = total_memories - candidates_count

        avg_health_all = sum(h.health_score for h in health_scores) / len(health_scores)
        avg_health_candidates = (
            sum(h.health_score for h in candidates) / len(candidates) if candidates else 0.0
        )

        # Group by recommendation
        recommendations = {}
        for health in health_scores:
            rec = health.recommendation
            if rec not in recommendations:
                recommendations[rec] = []
            recommendations[rec].append(health.memory_id)

        return {
            "candidates": candidates,
            "stats": {
                "total_memories": total_memories,
                "prune_candidates": candidates_count,
                "keep_count": keep_count,
                "prune_percentage": (candidates_count / total_memories * 100)
                if total_memories > 0
                else 0,
                "avg_health_all": avg_health_all,
                "avg_health_candidates": avg_health_candidates,
                "strategy": strategy.value,
            },
            "recommendations": recommendations,
            "summary": {
                "action": (
                    f"Prune {candidates_count} memories"
                    if candidates_count > 0
                    else "No pruning needed"
                ),
                "rationale": f"Using {strategy.value} strategy with threshold {min_health_threshold}",
            },
        }

    def apply_decay_batch(
        self, memories: list[Memory], current_time: Optional[datetime] = None
    ) -> dict[str, Any]:
        """Apply importance decay to a batch of memories.

        Args:
            memories: List of memories to decay
            current_time: Current timestamp

        Returns:
            Dictionary with updated importance values and statistics
        """
        if not memories:
            return {
                "updates": {},
                "stats": {
                    "total_memories": 0,
                    "updated_count": 0,
                    "unchanged_count": 0,
                    "total_decay": 0.0,
                    "avg_decay": 0.0,
                    "significant_decay_count": 0,
                },
            }

        updates = {}
        total_decay = 0.0
        significant_decay_count = 0  # Count memories with >10% decay

        for memory in memories:
            original_importance = memory.importance
            new_importance = self.calculate_decayed_importance(memory, current_time)

            if new_importance != original_importance:
                decay_amount = original_importance - new_importance
                decay_percentage = (
                    (decay_amount / original_importance * 100) if original_importance > 0 else 0
                )

                updates[memory.id] = {
                    "original_importance": original_importance,
                    "new_importance": new_importance,
                    "decay_amount": decay_amount,
                    "decay_percentage": decay_percentage,
                }

                total_decay += decay_amount
                if decay_percentage > 10:
                    significant_decay_count += 1

        avg_decay = total_decay / len(memories) if memories else 0.0

        return {
            "updates": updates,
            "stats": {
                "total_memories": len(memories),
                "updated_count": len(updates),
                "unchanged_count": len(memories) - len(updates),
                "total_decay": total_decay,
                "avg_decay": avg_decay,
                "significant_decay_count": significant_decay_count,
            },
        }

    def generate_maintenance_report(
        self, memories: list[Memory], current_time: Optional[datetime] = None
    ) -> dict[str, Any]:
        """Generate comprehensive maintenance report for memories.

        Args:
            memories: List of memories to analyze
            current_time: Current timestamp

        Returns:
            Detailed maintenance report
        """
        if not memories:
            return {"message": "No memories to analyze"}

        # Calculate health for all
        health_scores = [self.calculate_memory_health(m, current_time) for m in memories]

        # Group by recommendation
        by_recommendation = {}
        for health in health_scores:
            rec = health.recommendation
            if rec not in by_recommendation:
                by_recommendation[rec] = []
            by_recommendation[rec].append(health)

        # Calculate statistics
        total = len(memories)
        avg_health = sum(h.health_score for h in health_scores) / total

        # Analyze decay potential
        decay_results = self.apply_decay_batch(memories, current_time)

        # Pruning analysis
        pruning_analysis = self.identify_pruning_candidates(
            memories, strategy=PruningStrategy.COMPREHENSIVE
        )

        return {
            "summary": {
                "total_memories": total,
                "average_health": avg_health,
                "keep_count": len(by_recommendation.get("keep", [])),
                "decay_count": len(by_recommendation.get("decay", [])),
                "archive_count": len(by_recommendation.get("archive", [])),
                "prune_count": len(by_recommendation.get("prune", [])),
            },
            "recommendations_by_category": {
                rec: len(items) for rec, items in by_recommendation.items()
            },
            "decay_analysis": decay_results["stats"],
            "pruning_analysis": pruning_analysis["stats"],
            "actions": {
                "immediate": [
                    f"Prune {len(by_recommendation.get('prune', []))} low-health memories"
                    if by_recommendation.get("prune")
                    else "No immediate pruning needed"
                ],
                "recommended": [
                    f"Archive {len(by_recommendation.get('archive', []))} older memories"
                    if by_recommendation.get("archive")
                    else "No archival needed",
                    f"Apply decay to {decay_results['stats']['significant_decay_count']} memories",
                ],
                "monitoring": [
                    f"Monitor {len(by_recommendation.get('decay', []))} memories for future decay"
                ],
            },
            "health_distribution": {
                "excellent (8-10)": len([h for h in health_scores if h.health_score >= 8]),
                "good (6-8)": len([h for h in health_scores if 6 <= h.health_score < 8]),
                "fair (4-6)": len([h for h in health_scores if 4 <= h.health_score < 6]),
                "poor (2-4)": len([h for h in health_scores if 2 <= h.health_score < 4]),
                "critical (0-2)": len([h for h in health_scores if h.health_score < 2]),
            },
        }
