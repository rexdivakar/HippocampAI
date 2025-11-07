"""Memory quality and health monitoring system."""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Memory health status levels."""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 75-89%
    FAIR = "fair"  # 60-74%
    POOR = "poor"  # 40-59%
    CRITICAL = "critical"  # <40%


class DuplicateClusterType(str, Enum):
    """Type of duplicate clustering."""

    EXACT = "exact"  # Identical or near-identical text
    SOFT = "soft"  # Semantically similar
    PARAPHRASE = "paraphrase"  # Same meaning, different wording
    VARIANT = "variant"  # Similar but with variations


class StaleReason(str, Enum):
    """Reason why memory is considered stale."""

    OUTDATED = "outdated"  # Very old with no recent access
    REPLACED = "replaced"  # Better/newer version exists
    LOW_CONFIDENCE = "low_confidence"  # Confidence has decayed
    NO_ACTIVITY = "no_activity"  # Never accessed
    TEMPORAL_CONTEXT = "temporal_context"  # Time-sensitive info that expired


class CoverageLevel(str, Enum):
    """Coverage level for topics."""

    COMPREHENSIVE = "comprehensive"  # 10+ memories
    ADEQUATE = "adequate"  # 5-9 memories
    SPARSE = "sparse"  # 2-4 memories
    MINIMAL = "minimal"  # 1 memory
    MISSING = "missing"  # 0 memories


class MemoryHealthScore(BaseModel):
    """Overall health score for memory store."""

    overall_score: float = Field(ge=0.0, le=100.0)
    status: HealthStatus
    total_memories: int
    healthy_memories: int
    stale_memories: int
    duplicate_clusters: int
    low_quality_memories: int
    coverage_score: float = Field(ge=0.0, le=100.0)
    freshness_score: float = Field(ge=0.0, le=100.0)
    diversity_score: float = Field(ge=0.0, le=100.0)
    consistency_score: float = Field(ge=0.0, le=100.0)
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recommendations: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class DuplicateCluster(BaseModel):
    """Group of duplicate or near-duplicate memories."""

    cluster_id: str
    cluster_type: DuplicateClusterType
    memories: list[Memory]
    similarity_scores: list[float] = Field(default_factory=list)
    representative_memory_id: Optional[str] = None  # Best memory in cluster
    merge_suggestion: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class StaleMemory(BaseModel):
    """Memory flagged as potentially stale."""

    memory: Memory
    reason: StaleReason
    staleness_score: float = Field(ge=0.0, le=1.0)  # 1.0 = very stale
    days_since_access: int
    days_since_creation: int
    recommendation: str
    should_archive: bool = False
    should_delete: bool = False


class TopicCoverage(BaseModel):
    """Coverage analysis for a specific topic."""

    topic: str
    memory_count: int
    coverage_level: CoverageLevel
    representative_memories: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=100.0)


class MemoryQualityReport(BaseModel):
    """Comprehensive quality and health report."""

    user_id: str
    health_score: MemoryHealthScore
    duplicate_clusters: list[DuplicateCluster] = Field(default_factory=list)
    stale_memories: list[StaleMemory] = Field(default_factory=list)
    topic_coverage: list[TopicCoverage] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryHealthMonitor:
    """
    Comprehensive memory quality and health monitoring system.

    Features:
    - Memory health scoring
    - Duplicate clustering and detection
    - Stale memory identification
    - Topic coverage analysis
    - Quality metrics and recommendations
    """

    def __init__(
        self,
        embedder: Embedder,
        exact_duplicate_threshold: float = 0.98,
        soft_duplicate_threshold: float = 0.85,
        stale_days_threshold: int = 90,
        no_access_days_threshold: int = 60,
        min_confidence_threshold: float = 0.3,
    ):
        """
        Initialize memory health monitor.

        Args:
            embedder: Embedding model for similarity calculations
            exact_duplicate_threshold: Threshold for exact duplicates (0-1)
            soft_duplicate_threshold: Threshold for soft duplicates (0-1)
            stale_days_threshold: Days until memory considered stale
            no_access_days_threshold: Days without access to flag
            min_confidence_threshold: Minimum confidence for healthy memory
        """
        self.embedder = embedder
        self.exact_duplicate_threshold = exact_duplicate_threshold
        self.soft_duplicate_threshold = soft_duplicate_threshold
        self.stale_days_threshold = stale_days_threshold
        self.no_access_days_threshold = no_access_days_threshold
        self.min_confidence_threshold = min_confidence_threshold

    def calculate_health_score(
        self, memories: list[Memory], detailed: bool = True
    ) -> MemoryHealthScore:
        """
        Calculate overall health score for memory store.

        Args:
            memories: List of memories to analyze
            detailed: Whether to include detailed metrics

        Returns:
            MemoryHealthScore with overall health assessment
        """
        if not memories:
            return MemoryHealthScore(
                overall_score=0.0,
                status=HealthStatus.CRITICAL,
                total_memories=0,
                healthy_memories=0,
                stale_memories=0,
                duplicate_clusters=0,
                low_quality_memories=0,
                coverage_score=0.0,
                freshness_score=0.0,
                diversity_score=0.0,
                consistency_score=0.0,
                recommendations=["No memories found. Start adding memories to build your store."],
            )

        total_memories = len(memories)
        now = datetime.now(timezone.utc)

        # Calculate component scores
        freshness_score = self._calculate_freshness_score(memories, now)
        diversity_score = self._calculate_diversity_score(memories)
        consistency_score = self._calculate_consistency_score(memories)
        coverage_score = self._calculate_coverage_score(memories)

        # Identify issues
        stale_count = sum(1 for m in memories if self._is_stale(m, now))
        low_quality_count = sum(1 for m in memories if m.confidence < self.min_confidence_threshold)
        duplicate_clusters = len(self.detect_duplicate_clusters(memories, cluster_type="soft"))

        # Calculate overall score (weighted average)
        overall_score = (
            freshness_score * 0.3
            + diversity_score * 0.25
            + consistency_score * 0.25
            + coverage_score * 0.20
        )

        # Determine health status
        if overall_score >= 90:
            status = HealthStatus.EXCELLENT
        elif overall_score >= 75:
            status = HealthStatus.GOOD
        elif overall_score >= 60:
            status = HealthStatus.FAIR
        elif overall_score >= 40:
            status = HealthStatus.POOR
        else:
            status = HealthStatus.CRITICAL

        # Generate recommendations
        recommendations = self._generate_recommendations(
            memories, stale_count, low_quality_count, duplicate_clusters, overall_score
        )

        # Detailed metrics
        metrics = {}
        if detailed:
            metrics = {
                "avg_confidence": np.mean([m.confidence for m in memories]),
                "avg_importance": np.mean([m.importance for m in memories]),
                "avg_age_days": np.mean(
                    [(now - m.created_at).days for m in memories if m.created_at]
                ),
                "avg_access_count": np.mean([m.access_count for m in memories]),
                "memory_types": self._count_by_type(memories),
                "stale_percentage": (stale_count / total_memories) * 100,
                "low_quality_percentage": (low_quality_count / total_memories) * 100,
            }

        return MemoryHealthScore(
            overall_score=overall_score,
            status=status,
            total_memories=total_memories,
            healthy_memories=total_memories - stale_count - low_quality_count,
            stale_memories=stale_count,
            duplicate_clusters=duplicate_clusters,
            low_quality_memories=low_quality_count,
            coverage_score=coverage_score,
            freshness_score=freshness_score,
            diversity_score=diversity_score,
            consistency_score=consistency_score,
            recommendations=recommendations,
            metrics=metrics,
        )

    def detect_duplicate_clusters(
        self,
        memories: list[Memory],
        cluster_type: str = "soft",
        min_cluster_size: int = 2,
    ) -> list[DuplicateCluster]:
        """
        Detect clusters of duplicate or near-duplicate memories.

        Args:
            memories: List of memories to analyze
            cluster_type: Type of clustering ("exact", "soft", "all")
            min_cluster_size: Minimum memories per cluster

        Returns:
            List of duplicate clusters found
        """
        if len(memories) < min_cluster_size:
            return []

        # Generate embeddings for all memories
        texts = [m.text for m in memories]
        embeddings = self.embedder.encode(texts)

        # Calculate pairwise similarities
        similarity_matrix = self._calculate_similarity_matrix(embeddings)

        # Find clusters
        clusters = []
        processed = set()

        for i, memory in enumerate(memories):
            if i in processed:
                continue

            # Find similar memories
            similar_indices = []
            similar_scores = []

            for j in range(i + 1, len(memories)):
                if j in processed:
                    continue

                similarity = similarity_matrix[i][j]

                # Determine if duplicate based on cluster type
                is_duplicate = False
                detected_type = None

                if similarity >= self.exact_duplicate_threshold:
                    is_duplicate = True
                    detected_type = DuplicateClusterType.EXACT
                elif similarity >= self.soft_duplicate_threshold and cluster_type in [
                    "soft",
                    "all",
                ]:
                    is_duplicate = True
                    detected_type = DuplicateClusterType.SOFT
                elif similarity >= 0.75 and cluster_type == "all":
                    is_duplicate = True
                    detected_type = DuplicateClusterType.PARAPHRASE

                if is_duplicate:
                    similar_indices.append(j)
                    similar_scores.append(similarity)

            # Create cluster if enough similar memories found
            if len(similar_indices) >= min_cluster_size - 1:
                cluster_memories = [memory] + [memories[idx] for idx in similar_indices]

                # Find representative (highest confidence or most recent)
                representative = max(
                    cluster_memories,
                    key=lambda m: (m.confidence, m.updated_at or m.created_at),
                )

                # Generate merge suggestion
                merge_suggestion = self._generate_merge_suggestion(cluster_memories)

                cluster = DuplicateCluster(
                    cluster_id=f"cluster_{i}_{len(clusters)}",
                    cluster_type=detected_type or DuplicateClusterType.SOFT,
                    memories=cluster_memories,
                    similarity_scores=[1.0] + similar_scores,
                    representative_memory_id=representative.id,
                    merge_suggestion=merge_suggestion,
                    confidence=min(float(np.mean([1.0] + similar_scores)), 1.0),  # Clamp to 1.0
                )

                clusters.append(cluster)

                # Mark as processed
                processed.add(i)
                for idx in similar_indices:
                    processed.add(idx)

        logger.info(f"Detected {len(clusters)} duplicate clusters")
        return clusters

    def detect_stale_memories(
        self, memories: list[Memory], threshold_days: Optional[int] = None
    ) -> list[StaleMemory]:
        """
        Detect memories that are potentially stale or outdated.

        Args:
            memories: List of memories to analyze
            threshold_days: Custom threshold (uses default if not provided)

        Returns:
            List of stale memories with recommendations
        """
        threshold_days = threshold_days or self.stale_days_threshold
        now = datetime.now(timezone.utc)
        stale_memories = []

        for memory in memories:
            if self._is_stale(memory, now):
                # Determine specific reason
                days_since_creation = (now - memory.created_at).days
                days_since_access = (
                    (now - memory.updated_at).days if memory.updated_at else days_since_creation
                )

                # Determine staleness reason
                if memory.confidence < self.min_confidence_threshold:
                    reason = StaleReason.LOW_CONFIDENCE
                elif (
                    memory.access_count == 0 and days_since_creation > self.no_access_days_threshold
                ):
                    reason = StaleReason.NO_ACTIVITY
                elif days_since_creation > threshold_days * 2:
                    reason = StaleReason.OUTDATED
                else:
                    reason = StaleReason.OUTDATED

                # Calculate staleness score
                staleness_score = self._calculate_staleness_score(memory, now)

                # Generate recommendation
                recommendation, should_archive, should_delete = self._generate_stale_recommendation(
                    memory, staleness_score, reason
                )

                stale_memory = StaleMemory(
                    memory=memory,
                    reason=reason,
                    staleness_score=staleness_score,
                    days_since_access=days_since_access,
                    days_since_creation=days_since_creation,
                    recommendation=recommendation,
                    should_archive=should_archive,
                    should_delete=should_delete,
                )

                stale_memories.append(stale_memory)

        logger.info(f"Detected {len(stale_memories)} stale memories")
        return stale_memories

    def analyze_topic_coverage(
        self, memories: list[Memory], topics: Optional[list[str]] = None
    ) -> list[TopicCoverage]:
        """
        Analyze coverage across different topics/categories.

        Args:
            memories: List of memories to analyze
            topics: Optional list of topics to check (auto-detects if not provided)

        Returns:
            List of topic coverage analyses
        """
        # If no topics provided, extract from tags and types
        if topics is None:
            topics = self._extract_topics(memories)

        coverage_analyses = []

        for topic in topics:
            # Find memories related to this topic
            related_memories = self._find_related_memories(memories, topic)
            memory_count = len(related_memories)

            # Determine coverage level
            if memory_count >= 10:
                level = CoverageLevel.COMPREHENSIVE
            elif memory_count >= 5:
                level = CoverageLevel.ADEQUATE
            elif memory_count >= 2:
                level = CoverageLevel.SPARSE
            elif memory_count == 1:
                level = CoverageLevel.MINIMAL
            else:
                level = CoverageLevel.MISSING

            # Get representative memories
            representative = [m.text[:100] for m in related_memories[:3]]

            # Calculate quality score
            quality_score = (
                self._calculate_topic_quality(related_memories) if related_memories else 0.0
            )

            # Identify gaps
            gaps = self._identify_gaps(topic, related_memories)

            coverage = TopicCoverage(
                topic=topic,
                memory_count=memory_count,
                coverage_level=level,
                representative_memories=representative,
                gaps=gaps,
                quality_score=quality_score,
            )

            coverage_analyses.append(coverage)

        return coverage_analyses

    def generate_quality_report(
        self, memories: list[Memory], user_id: str, include_topics: bool = True
    ) -> MemoryQualityReport:
        """
        Generate comprehensive quality and health report.

        Args:
            memories: List of memories to analyze
            user_id: User identifier
            include_topics: Whether to include topic coverage analysis

        Returns:
            Complete quality report
        """
        logger.info(f"Generating quality report for user {user_id} with {len(memories)} memories")

        # Calculate overall health
        health_score = self.calculate_health_score(memories, detailed=True)

        # Detect duplicates
        duplicate_clusters = self.detect_duplicate_clusters(memories, cluster_type="all")

        # Detect stale memories
        stale_memories = self.detect_stale_memories(memories)

        # Analyze topic coverage
        topic_coverage = []
        if include_topics:
            topic_coverage = self.analyze_topic_coverage(memories)

        report = MemoryQualityReport(
            user_id=user_id,
            health_score=health_score,
            duplicate_clusters=duplicate_clusters,
            stale_memories=stale_memories,
            topic_coverage=topic_coverage,
        )

        logger.info(
            f"Quality report generated: {health_score.status} "
            f"({health_score.overall_score:.1f}/100)"
        )

        return report

    # Helper methods

    def _calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarity matrix."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-10)

        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def _calculate_freshness_score(self, memories: list[Memory], now: datetime) -> float:
        """Calculate freshness score (0-100) based on recency and access."""
        if not memories:
            return 0.0

        freshness_scores = []

        for memory in memories:
            days_old = (now - memory.created_at).days
            days_since_access = (now - memory.updated_at).days if memory.updated_at else days_old

            # Exponential decay based on age and access
            age_score = 100 * np.exp(-days_old / 90)  # Half-life of 90 days
            access_score = 100 * np.exp(-days_since_access / 60)  # Half-life of 60 days

            # Access boost
            access_boost = min(memory.access_count * 2, 20)  # Up to 20% boost

            freshness = (age_score * 0.5 + access_score * 0.5) + access_boost
            freshness_scores.append(min(freshness, 100.0))

        return float(np.mean(freshness_scores))

    def _calculate_diversity_score(self, memories: list[Memory]) -> float:
        """Calculate diversity score (0-100) based on variety."""
        if not memories:
            return 0.0

        # Type diversity
        types = set(m.type for m in memories)
        type_diversity = (len(types) / 6) * 100  # 6 possible types

        # Tag diversity
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)
        tag_diversity = min((len(all_tags) / len(memories)) * 100, 100)

        # Text diversity (average pairwise similarity should be low)
        if len(memories) > 1:
            texts = [m.text for m in memories[:100]]  # Sample for performance
            embeddings = self.embedder.encode(texts)
            similarity_matrix = self._calculate_similarity_matrix(embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            avg_similarity = np.mean(upper_triangle)

            # Lower similarity = higher diversity
            text_diversity = (1 - avg_similarity) * 100
        else:
            text_diversity = 50.0

        # Weighted average
        diversity_score = type_diversity * 0.3 + tag_diversity * 0.3 + text_diversity * 0.4

        return float(diversity_score)

    def _calculate_consistency_score(self, memories: list[Memory]) -> float:
        """Calculate consistency score (0-100) based on quality metrics."""
        if not memories:
            return 0.0

        consistency_scores = []

        for memory in memories:
            # Confidence score
            confidence_score = memory.confidence * 100

            # Importance score (normalized)
            importance_score = (memory.importance / 10) * 100

            # Completeness score (has tags, metadata, etc.)
            completeness = 0
            if memory.tags:
                completeness += 25
            if memory.metadata:
                completeness += 25
            if memory.text and len(memory.text) > 10:
                completeness += 25
            if memory.type:
                completeness += 25

            # Overall consistency
            consistency = confidence_score * 0.4 + importance_score * 0.3 + completeness * 0.3
            consistency_scores.append(consistency)

        return float(np.mean(consistency_scores))

    def _calculate_coverage_score(self, memories: list[Memory]) -> float:
        """Calculate coverage score (0-100) based on breadth."""
        if not memories:
            return 0.0

        # Check coverage across types
        types = set(m.type for m in memories)
        type_coverage = (len(types) / 6) * 100

        # Check temporal coverage (spread across time)
        if len(memories) > 1:
            dates = [m.created_at for m in memories]
            date_range = (max(dates) - min(dates)).days
            # Ideal range is at least 30 days
            temporal_coverage = min((date_range / 30) * 100, 100)
        else:
            temporal_coverage = 50.0

        # Check quantity coverage
        quantity_coverage = min((len(memories) / 100) * 100, 100)  # Ideal: 100+ memories

        coverage_score = type_coverage * 0.3 + temporal_coverage * 0.3 + quantity_coverage * 0.4

        return float(coverage_score)

    def _is_stale(self, memory: Memory, now: datetime) -> bool:
        """Check if memory is stale."""
        days_old = (now - memory.created_at).days
        days_since_access = (now - memory.updated_at).days if memory.updated_at else days_old

        # Stale if:
        # 1. Very old with no recent access
        # 2. Never accessed and old enough
        # 3. Low confidence
        # 4. Expired
        if memory.expires_at and now > memory.expires_at:
            return True

        if memory.confidence < self.min_confidence_threshold:
            return True

        if memory.access_count == 0 and days_since_access > self.no_access_days_threshold:
            return True

        if days_old > self.stale_days_threshold and days_since_access > 30:
            return True

        return False

    def _calculate_staleness_score(self, memory: Memory, now: datetime) -> float:
        """Calculate how stale a memory is (0-1, higher = more stale)."""
        days_old = (now - memory.created_at).days
        days_since_access = (now - memory.updated_at).days if memory.updated_at else days_old

        # Factors contributing to staleness
        age_factor = min(days_old / self.stale_days_threshold, 1.0)
        access_factor = min(days_since_access / self.no_access_days_threshold, 1.0)
        confidence_factor = 1.0 - memory.confidence
        activity_factor = 1.0 if memory.access_count == 0 else (1.0 / (1 + memory.access_count))

        # Weighted average
        staleness = (
            age_factor * 0.3 + access_factor * 0.3 + confidence_factor * 0.2 + activity_factor * 0.2
        )

        return float(min(staleness, 1.0))

    def _generate_stale_recommendation(
        self, memory: Memory, staleness_score: float, reason: StaleReason
    ) -> tuple[str, bool, bool]:
        """Generate recommendation for stale memory."""
        should_archive = False
        should_delete = False

        if staleness_score >= 0.8:
            recommendation = "Consider deleting this memory - very stale and low value"
            should_delete = True
        elif staleness_score >= 0.6:
            recommendation = (
                "Consider archiving this memory - outdated but may have historical value"
            )
            should_archive = True
        elif staleness_score >= 0.4:
            recommendation = "Review and update this memory if still relevant"
        else:
            recommendation = "Monitor this memory - showing early signs of staleness"

        return recommendation, should_archive, should_delete

    def _generate_merge_suggestion(self, memories: list[Memory]) -> str:
        """Generate suggestion for merging duplicate memories."""
        if not memories:
            return ""

        # Find best memory
        best = max(memories, key=lambda m: (m.confidence, m.access_count, m.importance))

        suggestion = f"Merge into: '{best.text[:100]}...' (highest quality)"
        return suggestion

    def _generate_recommendations(
        self,
        memories: list[Memory],
        stale_count: int,
        low_quality_count: int,
        duplicate_clusters: int,
        overall_score: float,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if overall_score >= 90:
            recommendations.append("✅ Memory store is in excellent health!")
        elif overall_score >= 75:
            recommendations.append("✅ Memory store is healthy with minor improvements possible")

        if stale_count > len(memories) * 0.2:
            recommendations.append(
                f"🔴 High stale memory count ({stale_count}). Run cleanup to remove outdated memories."
            )

        if low_quality_count > len(memories) * 0.1:
            recommendations.append(
                f"⚠️ {low_quality_count} low-confidence memories found. Review and update or remove."
            )

        if duplicate_clusters > 0:
            recommendations.append(
                f"🔄 {duplicate_clusters} duplicate clusters detected. Run deduplication."
            )

        if len(memories) < 10:
            recommendations.append("📝 Add more memories to improve coverage and usefulness.")

        if not recommendations:
            recommendations.append("Continue monitoring memory health regularly.")

        return recommendations

    def _count_by_type(self, memories: list[Memory]) -> dict[str, int]:
        """Count memories by type."""
        counts = defaultdict(int)
        for memory in memories:
            counts[memory.type.value] += 1
        return dict(counts)

    def _extract_topics(self, memories: list[Memory]) -> list[str]:
        """Extract topics from memories."""
        topics = set()

        # Add all unique tags
        for memory in memories:
            topics.update(memory.tags)

        # Add memory types
        for memory in memories:
            topics.add(memory.type.value)

        return sorted(list(topics))[:20]  # Limit to top 20

    def _find_related_memories(self, memories: list[Memory], topic: str) -> list[Memory]:
        """Find memories related to a topic."""
        related = []

        for memory in memories:
            # Check if topic in tags or text
            if topic in memory.tags or topic.lower() in memory.text.lower():
                related.append(memory)
            elif topic == memory.type.value:
                related.append(memory)

        return related

    def _calculate_topic_quality(self, memories: list[Memory]) -> float:
        """Calculate quality score for topic coverage."""
        if not memories:
            return 0.0

        # Average confidence
        avg_confidence = np.mean([m.confidence for m in memories])

        # Average importance
        avg_importance = np.mean([m.importance for m in memories])

        # Recency (how fresh are these memories)
        now = datetime.now(timezone.utc)
        avg_age_days = np.mean([(now - m.created_at).days for m in memories])
        recency_score = 100 * np.exp(-avg_age_days / 90)

        # Quality score
        quality = (
            (avg_confidence * 100) * 0.4 + (avg_importance / 10 * 100) * 0.3 + recency_score * 0.3
        )

        return float(quality)

    def _identify_gaps(self, topic: str, memories: list[Memory]) -> list[str]:
        """Identify gaps in topic coverage."""
        gaps = []

        if len(memories) == 0:
            gaps.append(f"No memories found for {topic}")
        elif len(memories) < 3:
            gaps.append(f"Limited coverage for {topic} - consider adding more context")

        # Check if any recent memories (last 30 days)
        now = datetime.now(timezone.utc)
        recent = [m for m in memories if (now - m.created_at).days < 30]
        if not recent:
            gaps.append(f"No recent memories for {topic} - may be outdated")

        return gaps
