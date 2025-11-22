"""
Memory Quality & Health Monitoring.

This module provides comprehensive memory quality assessment including:
- Health scoring for individual memories and entire memory store
- Soft duplicate clustering and near-duplicate detection
- Stale memory detection with temporal tracking
- Memory coverage analysis for topic representation
- Quality metrics and recommendations

Features:
- Quality scores based on completeness, clarity, consistency
- Duplicate detection with similarity thresholds
- Temporal staleness tracking
- Topic coverage analysis with clustering
- Health recommendations and warnings
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall health status levels."""

    EXCELLENT = "excellent"  # Score >= 90
    GOOD = "good"  # Score >= 75
    FAIR = "fair"  # Score >= 60
    POOR = "poor"  # Score >= 40
    CRITICAL = "critical"  # Score < 40


class QualityIssue(str, Enum):
    """Types of quality issues."""

    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    STALE = "stale"
    LOW_CONFIDENCE = "low_confidence"
    LOW_IMPORTANCE = "low_importance"
    INCOMPLETE = "incomplete"
    CONTRADICTORY = "contradictory"
    ORPHANED = "orphaned"


class MemoryHealthScore(BaseModel):
    """Health score for a single memory."""

    memory_id: str
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall quality score")
    completeness_score: float = Field(
        ge=0.0, le=100.0, description="Completeness of information"
    )
    clarity_score: float = Field(ge=0.0, le=100.0, description="Clarity of content")
    freshness_score: float = Field(
        ge=0.0, le=100.0, description="How recent/up-to-date"
    )
    confidence_score: float = Field(ge=0.0, le=100.0, description="Confidence level")
    importance_score: float = Field(ge=0.0, le=100.0, description="Importance weight")
    issues: list[QualityIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    is_stale: bool = False
    days_since_update: float = 0.0
    duplicate_candidates: list[str] = Field(
        default_factory=list, description="IDs of potential duplicates"
    )

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        self.overall_score = (
            0.25 * self.completeness_score
            + 0.20 * self.clarity_score
            + 0.20 * self.freshness_score
            + 0.20 * self.confidence_score
            + 0.15 * self.importance_score
        )
        return self.overall_score


class DuplicateCluster(BaseModel):
    """Cluster of similar/duplicate memories."""

    cluster_id: str
    memory_ids: list[str]
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Average similarity within cluster"
    )
    suggested_action: str  # "merge", "keep_all", "review"
    canonical_memory_id: Optional[str] = None  # Best representative


class TopicCoverage(BaseModel):
    """Coverage analysis for a topic."""

    topic: str
    memory_count: int
    avg_quality: float
    avg_recency: float  # Days since last update
    keywords: list[str] = Field(default_factory=list)
    coverage_score: float = Field(
        ge=0.0, le=100.0, description="How well-represented this topic is"
    )


class MemoryStoreHealth(BaseModel):
    """Overall health assessment of memory store."""

    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_health_score: float = Field(ge=0.0, le=100.0)
    health_status: HealthStatus
    total_memories: int = 0
    healthy_memories: int = 0
    problematic_memories: int = 0
    stale_memories: int = 0
    duplicate_clusters: int = 0
    avg_memory_quality: float = 0.0
    issues_by_type: dict[str, int] = Field(default_factory=dict)
    topic_coverage: list[TopicCoverage] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def determine_status(self) -> HealthStatus:
        """Determine health status from score."""
        if self.overall_health_score >= 90:
            self.health_status = HealthStatus.EXCELLENT
        elif self.overall_health_score >= 75:
            self.health_status = HealthStatus.GOOD
        elif self.overall_health_score >= 60:
            self.health_status = HealthStatus.FAIR
        elif self.overall_health_score >= 40:
            self.health_status = HealthStatus.POOR
        else:
            self.health_status = HealthStatus.CRITICAL
        return self.health_status


class MemoryQualityMonitor:
    """Monitor and analyze memory quality and health."""

    def __init__(
        self,
        stale_threshold_days: int = 90,
        duplicate_threshold: float = 0.85,
        near_duplicate_threshold: float = 0.70,
        min_text_length: int = 10,
        min_confidence: float = 0.5,
    ):
        """
        Initialize quality monitor.

        Args:
            stale_threshold_days: Days after which memory is considered stale
            duplicate_threshold: Similarity threshold for duplicates (0.85)
            near_duplicate_threshold: Similarity for near-duplicates (0.70)
            min_text_length: Minimum acceptable text length
            min_confidence: Minimum acceptable confidence score
        """
        self.stale_threshold_days = stale_threshold_days
        self.duplicate_threshold = duplicate_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.min_text_length = min_text_length
        self.min_confidence = min_confidence

    def assess_memory_health(
        self,
        memory_id: str,
        text: str,
        confidence: float,
        importance: float,
        created_at: datetime,
        updated_at: datetime,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> MemoryHealthScore:
        """
        Assess health of a single memory.

        Args:
            memory_id: Memory identifier
            text: Memory text content
            confidence: Confidence score (0-1)
            importance: Importance score (0-10)
            created_at: Creation timestamp
            updated_at: Last update timestamp
            tags: Associated tags
            metadata: Additional metadata

        Returns:
            MemoryHealthScore with detailed assessment
        """
        now = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        days_since_update = (now - updated_at).total_seconds() / 86400

        # Calculate component scores
        completeness_score = self._assess_completeness(text, tags, metadata)
        clarity_score = self._assess_clarity(text)
        freshness_score = self._assess_freshness(days_since_update)
        confidence_score = confidence * 100
        importance_score = (importance / 10.0) * 100

        # Create health score
        overall_score = (
            0.25 * completeness_score
            + 0.20 * clarity_score
            + 0.20 * freshness_score
            + 0.20 * confidence_score
            + 0.15 * importance_score
        )
        health = MemoryHealthScore(
            memory_id=memory_id,
            overall_score=overall_score,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            freshness_score=freshness_score,
            confidence_score=confidence_score,
            importance_score=importance_score,
            days_since_update=days_since_update,
            is_stale=days_since_update > self.stale_threshold_days,
        )

        # Identify issues and recommendations
        if days_since_update > self.stale_threshold_days:
            health.issues.append(QualityIssue.STALE)
            health.recommendations.append(
                f"Memory is {days_since_update:.0f} days old. Consider reviewing for accuracy."
            )

        if confidence < self.min_confidence:
            health.issues.append(QualityIssue.LOW_CONFIDENCE)
            health.recommendations.append(
                f"Low confidence ({confidence:.2f}). Consider validation or removal."
            )

        if importance < 3.0:
            health.issues.append(QualityIssue.LOW_IMPORTANCE)
            health.recommendations.append(
                "Low importance score. Consider if this memory should be retained."
            )

        if len(text) < self.min_text_length:
            health.issues.append(QualityIssue.INCOMPLETE)
            health.recommendations.append(
                "Very short text. May lack sufficient detail."
            )

        if not tags and not metadata:
            health.issues.append(QualityIssue.ORPHANED)
            health.recommendations.append(
                "No tags or metadata. Consider adding context."
            )

        # Calculate overall score
        health.calculate_overall_score()

        return health

    def _assess_completeness(
        self, text: str, tags: list[str], metadata: dict[str, Any]
    ) -> float:
        """Assess completeness of memory (0-100)."""
        score = 0.0

        # Text length (40 points)
        if len(text) >= 100:
            score += 40
        elif len(text) >= 50:
            score += 30
        elif len(text) >= self.min_text_length:
            score += 20
        else:
            score += 10

        # Tags present (30 points)
        if len(tags) >= 3:
            score += 30
        elif len(tags) >= 1:
            score += 20
        else:
            score += 0

        # Metadata present (30 points)
        if len(metadata) >= 3:
            score += 30
        elif len(metadata) >= 1:
            score += 20
        else:
            score += 10

        return min(score, 100.0)

    def _assess_clarity(self, text: str) -> float:
        """Assess clarity of text (0-100)."""
        score = 80.0  # Base score

        # Penalize very short text
        if len(text) < 20:
            score -= 30
        elif len(text) < 50:
            score -= 15

        # Reward proper sentence structure (simple heuristic)
        sentences = text.count(".") + text.count("!") + text.count("?")
        if sentences > 0:
            score += 10
        if sentences > 2:
            score += 10

        # Penalize excessive repetition (simple check)
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 20

        return max(0.0, min(score, 100.0))

    def _assess_freshness(self, days_since_update: float) -> float:
        """Assess freshness based on age (0-100)."""
        # Exponential decay: 100 at day 0, ~50 at 30 days, ~25 at 60 days
        if days_since_update < 0:
            return 100.0

        # Use exponential decay with half-life of 30 days
        half_life = 30.0
        freshness = 100.0 * (0.5 ** (days_since_update / half_life))

        return cast(float, max(0.0, min(freshness, 100.0)))

    def detect_duplicate_clusters(
        self,
        memories: list[dict[str, Any]],
        similarity_matrix: dict[tuple[str, str], float],
    ) -> list[DuplicateCluster]:
        """
        Detect clusters of duplicate/similar memories.

        Args:
            memories: List of memory dictionaries with 'id' and 'text'
            similarity_matrix: Precomputed similarity scores between memory pairs

        Returns:
            List of DuplicateCluster objects
        """
        clusters: list[DuplicateCluster] = []
        processed: set[str] = set()

        # Build adjacency list for near-duplicates
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (id1, id2), similarity in similarity_matrix.items():
            if similarity >= self.near_duplicate_threshold:
                adjacency[id1].append((id2, similarity))
                adjacency[id2].append((id1, similarity))

        # Find clusters using simple graph traversal
        cluster_id_counter = 0
        for memory in memories:
            memory_id = memory["id"]
            if memory_id in processed:
                continue

            # Start new cluster
            cluster_members = {memory_id}
            to_visit = [memory_id]
            similarities = []

            while to_visit:
                current = to_visit.pop()
                if current in processed:
                    continue
                processed.add(current)

                for neighbor, sim in adjacency.get(current, []):
                    if neighbor not in processed:
                        cluster_members.add(neighbor)
                        to_visit.append(neighbor)
                        similarities.append(sim)

            # Only create cluster if > 1 member
            if len(cluster_members) > 1:
                avg_similarity = (
                    sum(similarities) / len(similarities) if similarities else 0.0
                )

                # Determine suggested action
                if avg_similarity >= self.duplicate_threshold:
                    action = "merge"
                elif avg_similarity >= self.near_duplicate_threshold:
                    action = "review"
                else:
                    action = "keep_all"

                # Find canonical (best quality) memory
                canonical = max(
                    cluster_members,
                    key=lambda mid: similarity_matrix.get(
                        (mid, list(cluster_members)[0]), 0.0
                    ),
                )

                cluster = DuplicateCluster(
                    cluster_id=f"cluster_{cluster_id_counter}",
                    memory_ids=list(cluster_members),
                    similarity_score=avg_similarity,
                    suggested_action=action,
                    canonical_memory_id=canonical,
                )
                clusters.append(cluster)
                cluster_id_counter += 1

        return clusters

    def analyze_topic_coverage(
        self, memories: list[dict[str, Any]], topics: Optional[list[str]] = None
    ) -> list[TopicCoverage]:
        """
        Analyze memory coverage across topics.

        Args:
            memories: List of memories with tags and metadata
            topics: Optional predefined topics to analyze

        Returns:
            List of TopicCoverage objects
        """
        if not topics:
            # Extract topics from tags
            topic_memories: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for memory in memories:
                for tag in memory.get("tags", []):
                    topic_memories[tag].append(memory)
        else:
            # Use predefined topics
            topic_memories = {topic: [] for topic in topics}
            for memory in memories:
                for tag in memory.get("tags", []):
                    if tag in topics:
                        topic_memories[tag].append(memory)

        # Analyze each topic
        coverage_list: list[TopicCoverage] = []
        now = datetime.now(timezone.utc)

        for topic, topic_mems in topic_memories.items():
            if not topic_mems:
                continue

            # Calculate metrics
            memory_count = len(topic_mems)

            # Average quality (using confidence as proxy)
            avg_quality = (
                sum(m.get("confidence", 0.5) for m in topic_mems) / memory_count * 100
            )

            # Average recency
            total_days = 0.0
            for m in topic_mems:
                updated_at = m.get("updated_at")
                if isinstance(updated_at, datetime):
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    days = (now - updated_at).total_seconds() / 86400
                    total_days += days
            avg_recency = total_days / memory_count if memory_count > 0 else 0

            # Extract keywords (top words from text)
            all_text = " ".join(m.get("text", "") for m in topic_mems)
            words = all_text.lower().split()
            word_freq: dict[str, int] = defaultdict(int)
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] += 1
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            keywords = [word for word, _ in top_keywords]

            # Coverage score (based on count, quality, recency)
            count_score = min(memory_count / 10.0 * 50, 50)  # Max 50 points
            quality_score = avg_quality * 0.3  # Max 30 points
            recency_score = max(
                0, 20 - (avg_recency / 30)
            )  # Max 20 points, decay over 30 days
            coverage_score = min(count_score + quality_score + recency_score, 100)

            coverage = TopicCoverage(
                topic=topic,
                memory_count=memory_count,
                avg_quality=avg_quality,
                avg_recency=avg_recency,
                keywords=keywords,
                coverage_score=coverage_score,
            )
            coverage_list.append(coverage)

        # Sort by coverage score descending
        coverage_list.sort(key=lambda x: x.coverage_score, reverse=True)

        return coverage_list

    def assess_memory_store_health(
        self,
        user_id: str,
        memory_health_scores: list[MemoryHealthScore],
        duplicate_clusters: list[DuplicateCluster],
        topic_coverage: list[TopicCoverage],
    ) -> MemoryStoreHealth:
        """
        Assess overall health of memory store.

        Args:
            user_id: User identifier
            memory_health_scores: Individual memory health assessments
            duplicate_clusters: Detected duplicate clusters
            topic_coverage: Topic coverage analysis

        Returns:
            MemoryStoreHealth with overall assessment
        """
        total_memories = len(memory_health_scores)
        if total_memories == 0:
            return MemoryStoreHealth(
                user_id=user_id,
                overall_health_score=0.0,
                health_status=HealthStatus.CRITICAL,
                recommendations=["No memories found in store."],
            )

        # Count issues
        issues_by_type: dict[str, int] = defaultdict(int)
        stale_count = 0
        healthy_count = 0
        problematic_count = 0

        for health in memory_health_scores:
            if health.overall_score >= 70:
                healthy_count += 1
            else:
                problematic_count += 1

            if health.is_stale:
                stale_count += 1

            for issue in health.issues:
                issues_by_type[issue.value] += 1

        # Calculate averages
        avg_quality = sum(h.overall_score for h in memory_health_scores) / total_memories

        # Calculate overall health score
        # Factors: avg quality (50%), duplicate ratio (20%), stale ratio (20%), coverage (10%)
        duplicate_penalty = min(len(duplicate_clusters) / total_memories * 100, 20)
        stale_penalty = min(stale_count / total_memories * 100, 20)
        coverage_bonus = (
            min(len(topic_coverage) / 10.0 * 10, 10) if topic_coverage else 0
        )

        overall_score = max(
            0, avg_quality * 0.5 - duplicate_penalty - stale_penalty + coverage_bonus
        )

        # Create health assessment
        store_health = MemoryStoreHealth(
            user_id=user_id,
            overall_health_score=overall_score,
            health_status=HealthStatus.GOOD,  # Will be set by determine_status
            total_memories=total_memories,
            healthy_memories=healthy_count,
            problematic_memories=problematic_count,
            stale_memories=stale_count,
            duplicate_clusters=len(duplicate_clusters),
            avg_memory_quality=avg_quality,
            issues_by_type=dict(issues_by_type),
            topic_coverage=topic_coverage,
        )

        # Determine status
        store_health.determine_status()

        # Generate recommendations
        if stale_count > total_memories * 0.3:
            store_health.recommendations.append(
                f"High stale memory count ({stale_count}/{total_memories}). "
                "Consider reviewing old memories for accuracy."
            )

        if len(duplicate_clusters) > total_memories * 0.1:
            store_health.recommendations.append(
                f"Found {len(duplicate_clusters)} duplicate clusters. "
                "Consider consolidating similar memories."
            )

        if avg_quality < 60:
            store_health.recommendations.append(
                f"Average memory quality is low ({avg_quality:.1f}/100). "
                "Focus on improving confidence and completeness."
            )

        if len(topic_coverage) < 3:
            store_health.recommendations.append(
                "Limited topic diversity. Consider adding tags to improve organization."
            )

        if problematic_count > total_memories * 0.4:
            store_health.recommendations.append(
                f"Many problematic memories ({problematic_count}/{total_memories}). "
                "Review and clean up low-quality entries."
            )

        if not store_health.recommendations:
            store_health.recommendations.append(
                "Memory store health is good. Continue maintaining quality standards."
            )

        return store_health
