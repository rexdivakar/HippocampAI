"""Memory health monitoring and quality scoring."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, MemoryType
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class DuplicateCluster(BaseModel):
    """Cluster of similar/duplicate memories."""

    cluster_id: str
    memory_ids: list[str]
    representative_text: str
    avg_similarity: float
    cluster_size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryHealthScore(BaseModel):
    """Overall health score for memory store."""

    overall_score: float = Field(ge=0.0, le=100.0, description="Overall health score (0-100)")
    quality_score: float = Field(ge=0.0, le=100.0, description="Memory quality score")
    diversity_score: float = Field(ge=0.0, le=100.0, description="Content diversity score")
    freshness_score: float = Field(ge=0.0, le=100.0, description="How up-to-date memories are")
    coverage_score: float = Field(ge=0.0, le=100.0, description="Topic coverage score")
    duplication_score: float = Field(ge=0.0, le=100.0, description="Deduplication health")
    staleness_score: float = Field(ge=0.0, le=100.0, description="Inverse of stale memory ratio")

    total_memories: int = 0
    stale_count: int = 0
    duplicate_clusters: int = 0
    avg_importance: float = 0.0
    avg_confidence: float = 0.0
    avg_age_days: float = 0.0

    breakdown: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StaleMemory(BaseModel):
    """Flagged stale/outdated memory."""

    memory_id: str
    text: str
    age_days: float
    last_accessed: Optional[datetime] = None
    importance: float
    staleness_reason: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in staleness detection")


class CoverageReport(BaseModel):
    """Memory coverage analysis."""

    topic_distribution: dict[str, int] = Field(default_factory=dict)
    well_covered_topics: list[str] = Field(default_factory=list)
    poorly_covered_topics: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    type_distribution: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class NearDuplicateWarning(BaseModel):
    """Warning about near-duplicate memories."""

    memory_id_1: str
    memory_id_2: str
    text_1: str
    text_2: str
    similarity_score: float
    merge_suggestion: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryHealthMonitor:
    """Monitor memory store health and quality."""

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        stale_threshold_days: int = 180,
        near_duplicate_threshold: float = 0.85,
        exact_duplicate_threshold: float = 0.95,
    ):
        """Initialize health monitor.

        Args:
            qdrant_store: Vector store for memories
            embedder: Embedding model
            stale_threshold_days: Days after which memory is considered potentially stale
            near_duplicate_threshold: Similarity threshold for near-duplicates
            exact_duplicate_threshold: Similarity threshold for exact duplicates
        """
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.stale_threshold_days = stale_threshold_days
        self.near_dup_threshold = near_duplicate_threshold
        self.exact_dup_threshold = exact_duplicate_threshold

    def calculate_health_score(
        self,
        user_id: str,
        include_stale_detection: bool = True,
        include_duplicate_detection: bool = True,
    ) -> MemoryHealthScore:
        """Calculate comprehensive health score for user's memory store.

        Args:
            user_id: User identifier
            include_stale_detection: Run stale memory detection
            include_duplicate_detection: Run duplicate clustering

        Returns:
            MemoryHealthScore with overall and component scores
        """
        logger.info(f"Calculating health score for user {user_id}")

        # Fetch all memories for user
        memories = self._fetch_all_user_memories(user_id)

        if not memories:
            return MemoryHealthScore(
                overall_score=0.0,
                quality_score=0.0,
                diversity_score=0.0,
                freshness_score=0.0,
                coverage_score=0.0,
                duplication_score=100.0,
                staleness_score=100.0,
                total_memories=0,
                recommendations=[
                    "No memories found. Start adding memories to build your knowledge base."
                ],
            )

        # Component scores
        quality_score = self._calculate_quality_score(memories)
        diversity_score = self._calculate_diversity_score(memories)
        freshness_score = self._calculate_freshness_score(memories)
        coverage_score = self._calculate_coverage_score(memories)

        # Optional intensive checks
        stale_count = 0
        duplicate_clusters = 0
        duplication_score = 100.0
        staleness_score = 100.0

        if include_stale_detection:
            stale_memories = self.detect_stale_memories(user_id, memories)
            stale_count = len(stale_memories)
            staleness_score = max(0.0, 100.0 - (stale_count / len(memories)) * 100)

        if include_duplicate_detection:
            clusters = self.detect_duplicate_clusters(user_id, memories)
            duplicate_clusters = len(clusters)
            # Penalize based on total duplicates in clusters
            total_dupes = sum(max(0, c.cluster_size - 1) for c in clusters)
            duplication_score = max(0.0, 100.0 - (total_dupes / len(memories)) * 100)

        # Calculate overall score (weighted average)
        overall_score = (
            quality_score * 0.25
            + diversity_score * 0.15
            + freshness_score * 0.15
            + coverage_score * 0.15
            + duplication_score * 0.15
            + staleness_score * 0.15
        )

        # Calculate statistics
        avg_importance = sum(m.importance for m in memories) / len(memories)
        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        avg_age_days = sum(
            (datetime.now(timezone.utc) - m.created_at).days for m in memories
        ) / len(memories)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score,
            quality_score,
            diversity_score,
            freshness_score,
            coverage_score,
            duplication_score,
            staleness_score,
            stale_count,
            duplicate_clusters,
        )

        return MemoryHealthScore(
            overall_score=overall_score,
            quality_score=quality_score,
            diversity_score=diversity_score,
            freshness_score=freshness_score,
            coverage_score=coverage_score,
            duplication_score=duplication_score,
            staleness_score=staleness_score,
            total_memories=len(memories),
            stale_count=stale_count,
            duplicate_clusters=duplicate_clusters,
            avg_importance=avg_importance,
            avg_confidence=avg_confidence,
            avg_age_days=avg_age_days,
            breakdown={
                "quality": quality_score,
                "diversity": diversity_score,
                "freshness": freshness_score,
                "coverage": coverage_score,
                "duplication": duplication_score,
                "staleness": staleness_score,
            },
            recommendations=recommendations,
        )

    def _calculate_quality_score(self, memories: list[Memory]) -> float:
        """Calculate quality score based on importance and confidence."""
        if not memories:
            return 0.0

        # Average importance (normalized to 0-100)
        avg_importance = sum(m.importance for m in memories) / len(memories)
        importance_score = (avg_importance / 10.0) * 100

        # Average confidence
        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        confidence_score = avg_confidence * 100

        # Metadata completeness
        metadata_complete = sum(1 for m in memories if m.tags or m.metadata or m.entities) / len(
            memories
        )
        metadata_score = metadata_complete * 100

        # Weighted average
        result: float = importance_score * 0.4 + confidence_score * 0.4 + metadata_score * 0.2
        return result

    def _calculate_diversity_score(self, memories: list[Memory]) -> float:
        """Calculate diversity based on memory types and content."""
        if not memories:
            return 0.0

        # Type diversity
        type_counts: defaultdict[Any, int] = defaultdict(int)
        for m in memories:
            type_counts[m.type] += 1

        # Shannon entropy for type distribution
        import math

        total = len(memories)
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(MemoryType)) if len(MemoryType) > 0 else 0
        type_diversity = (entropy / max_entropy) * 100 if max_entropy > 0 else 0

        # Text length diversity (not all short or all long)
        lengths = [m.text_length for m in memories if m.text_length > 0]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            variance = sum((length - avg_len) ** 2 for length in lengths) / len(lengths)
            length_diversity = min(100.0, (variance**0.5) / 10)
        else:
            length_diversity = 50.0

        return cast(float, (type_diversity * 0.7 + length_diversity * 0.3))

    def _calculate_freshness_score(self, memories: list[Memory]) -> float:
        """Calculate freshness based on memory ages."""
        if not memories:
            return 0.0

        now = datetime.now(timezone.utc)
        ages_days = [(now - m.created_at).days for m in memories]

        # Recent memories boost score
        recent_count = sum(1 for age in ages_days if age <= 30)
        recent_ratio = recent_count / len(memories)

        # Average age penalty
        avg_age = sum(ages_days) / len(memories)
        age_penalty = min(100.0, avg_age / 3.65)  # 1 year = 100% penalty

        # Access frequency (if tracked)
        active_memories = sum(1 for m in memories if m.access_count > 0)
        activity_ratio = active_memories / len(memories) if len(memories) > 0 else 0

        freshness = recent_ratio * 40 + (100 - age_penalty) * 0.4 + activity_ratio * 20

        result: float = max(0.0, min(100.0, freshness))
        return result

    def _calculate_coverage_score(self, memories: list[Memory]) -> float:
        """Calculate topic coverage score."""
        if not memories:
            return 0.0

        # Tag diversity
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)

        tag_coverage = min(100.0, len(all_tags) * 5)  # 20 unique tags = 100

        # Type coverage (having multiple types is good)
        type_coverage = (len(set(m.type for m in memories)) / len(MemoryType)) * 100

        # Entity coverage (if available)
        entity_coverage = 0.0
        entity_memories = [m for m in memories if m.entities]
        if entity_memories:
            entity_coverage = (len(entity_memories) / len(memories)) * 100

        return tag_coverage * 0.4 + type_coverage * 0.4 + entity_coverage * 0.2

    def detect_stale_memories(
        self,
        user_id: str,
        memories: Optional[list[Memory]] = None,
    ) -> list[StaleMemory]:
        """Detect potentially stale/outdated memories.

        Args:
            user_id: User identifier
            memories: Optional pre-fetched memories

        Returns:
            List of stale memories
        """
        if memories is None:
            memories = self._fetch_all_user_memories(user_id)

        stale_memories = []
        now = datetime.now(timezone.utc)
        threshold_date = now - timedelta(days=self.stale_threshold_days)

        for mem in memories:
            age_days = (now - mem.created_at).days

            # Check various staleness indicators
            staleness_reasons = []
            confidence = 0.0

            # Age-based staleness
            if mem.created_at < threshold_date:
                staleness_reasons.append(f"Memory is {age_days} days old")
                confidence += 0.3

            # Never accessed and old
            if mem.access_count == 0 and age_days > 90:
                staleness_reasons.append(f"Never accessed in {age_days} days")
                confidence += 0.3

            # Low importance and old
            if mem.importance < 3.0 and age_days > 60:
                staleness_reasons.append("Low importance and aging")
                confidence += 0.2

            # Temporal indicators in text (year mentions, dates)
            if self._contains_outdated_temporal_info(mem.text):
                staleness_reasons.append("Contains potentially outdated temporal information")
                confidence += 0.2

            if staleness_reasons:
                stale_memories.append(
                    StaleMemory(
                        memory_id=mem.id,
                        text=mem.text[:200],  # Preview
                        age_days=age_days,
                        last_accessed=mem.updated_at,
                        importance=mem.importance,
                        staleness_reason="; ".join(staleness_reasons),
                        confidence=min(1.0, confidence),
                    )
                )

        # Sort by confidence
        stale_memories.sort(key=lambda x: x.confidence, reverse=True)
        return stale_memories

    def detect_duplicate_clusters(
        self,
        user_id: str,
        memories: Optional[list[Memory]] = None,
        min_cluster_size: int = 2,
    ) -> list[DuplicateCluster]:
        """Detect clusters of duplicate/similar memories using soft clustering.

        Args:
            user_id: User identifier
            memories: Optional pre-fetched memories
            min_cluster_size: Minimum cluster size to report

        Returns:
            List of duplicate clusters
        """
        if memories is None:
            memories = self._fetch_all_user_memories(user_id)

        if len(memories) < 2:
            return []

        clusters: list[DuplicateCluster] = []
        visited = set()

        # Build similarity matrix
        for i, mem_i in enumerate(memories):
            if mem_i.id in visited:
                continue

            cluster_members = [mem_i]
            cluster_ids = [mem_i.id]
            similarities = []

            # Find similar memories
            for j, mem_j in enumerate(memories[i + 1 :], start=i + 1):
                if mem_j.id in visited:
                    continue

                # Calculate similarity
                similarity = self._calculate_text_similarity(mem_i.text, mem_j.text)

                if similarity >= self.near_dup_threshold:
                    cluster_members.append(mem_j)
                    cluster_ids.append(mem_j.id)
                    similarities.append(similarity)

            # Create cluster if size threshold met
            if len(cluster_members) >= min_cluster_size:
                for mem in cluster_members:
                    visited.add(mem.id)

                avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

                # Choose representative (highest importance)
                representative = max(cluster_members, key=lambda m: m.importance)

                clusters.append(
                    DuplicateCluster(
                        cluster_id=f"cluster_{len(clusters) + 1}",
                        memory_ids=cluster_ids,
                        representative_text=representative.text[:200],
                        avg_similarity=avg_similarity,
                        cluster_size=len(cluster_members),
                    )
                )

        return clusters

    def detect_near_duplicates(
        self,
        user_id: str,
        memories: Optional[list[Memory]] = None,
        suggest_merge: bool = True,
    ) -> list[NearDuplicateWarning]:
        """Detect near-duplicate pairs with merge suggestions.

        Args:
            user_id: User identifier
            memories: Optional pre-fetched memories
            suggest_merge: Generate merge suggestions

        Returns:
            List of near-duplicate warnings
        """
        if memories is None:
            memories = self._fetch_all_user_memories(user_id)

        warnings = []

        for i, mem_i in enumerate(memories):
            for mem_j in memories[i + 1 :]:
                similarity = self._calculate_text_similarity(mem_i.text, mem_j.text)

                # Near duplicate but not exact
                if self.near_dup_threshold <= similarity < self.exact_dup_threshold:
                    merge_suggestion = None
                    if suggest_merge:
                        merge_suggestion = self._generate_merge_suggestion(mem_i, mem_j)

                    warnings.append(
                        NearDuplicateWarning(
                            memory_id_1=mem_i.id,
                            memory_id_2=mem_j.id,
                            text_1=mem_i.text[:200],
                            text_2=mem_j.text[:200],
                            similarity_score=similarity,
                            merge_suggestion=merge_suggestion,
                            confidence=min(
                                1.0,
                                (similarity - self.near_dup_threshold)
                                / (1.0 - self.near_dup_threshold),
                            ),
                        )
                    )

        # Sort by similarity
        warnings.sort(key=lambda x: x.similarity_score, reverse=True)
        return warnings

    def analyze_coverage(
        self,
        user_id: str,
        memories: Optional[list[Memory]] = None,
    ) -> CoverageReport:
        """Analyze memory coverage across topics and types.

        Args:
            user_id: User identifier
            memories: Optional pre-fetched memories

        Returns:
            Coverage analysis report
        """
        if memories is None:
            memories = self._fetch_all_user_memories(user_id)

        if not memories:
            return CoverageReport(
                recommendations=["No memories to analyze. Start adding memories."]
            )

        # Topic distribution (based on tags)
        topic_dist: defaultdict[str, int] = defaultdict(int)
        for mem in memories:
            for tag in mem.tags:
                topic_dist[tag] += 1

        # Type distribution
        type_dist: defaultdict[str, int] = defaultdict(int)
        for mem in memories:
            type_dist[mem.type.value] += 1

        # Identify well-covered vs poorly-covered topics
        avg_topic_count = sum(topic_dist.values()) / len(topic_dist) if topic_dist else 0
        well_covered = [
            topic for topic, count in topic_dist.items() if count >= avg_topic_count * 1.5
        ]
        poorly_covered = [
            topic for topic, count in topic_dist.items() if count < avg_topic_count * 0.5
        ]

        # Identify gaps
        gaps = []
        all_types = set(t.value for t in MemoryType)
        existing_types = set(type_dist.keys())
        missing_types = all_types - existing_types

        if missing_types:
            gaps.append(f"Missing memory types: {', '.join(missing_types)}")

        # Generate recommendations
        recommendations = []
        if poorly_covered:
            recommendations.append(
                f"Consider adding more memories about: {', '.join(poorly_covered[:3])}"
            )
        if missing_types:
            recommendations.append(
                f"Add {', '.join(missing_types)} type memories for better coverage"
            )
        if len(topic_dist) < 5:
            recommendations.append("Diversify your memory topics with more tags")

        return CoverageReport(
            topic_distribution=dict(topic_dist),
            well_covered_topics=well_covered,
            poorly_covered_topics=poorly_covered,
            coverage_gaps=gaps,
            type_distribution=dict(type_dist),
            recommendations=recommendations,
        )

    def _fetch_all_user_memories(self, user_id: str) -> list[Memory]:
        """Fetch all memories for a user from both collections."""
        memories = []

        # Fetch from facts collection
        try:
            facts_results = self.qdrant.scroll(
                collection_name=self.qdrant.collection_facts,
                filters={"user_id": user_id},
                limit=10000,
            )
            for result in facts_results:
                if "payload" in result:
                    memories.append(Memory(**result["payload"]))
        except Exception as e:
            logger.warning(f"Error fetching from facts collection: {e}")

        # Fetch from prefs collection
        try:
            prefs_results = self.qdrant.scroll(
                collection_name=self.qdrant.collection_prefs,
                filters={"user_id": user_id},
                limit=10000,
            )
            for result in prefs_results:
                if "payload" in result:
                    memories.append(Memory(**result["payload"]))
        except Exception as e:
            logger.warning(f"Error fetching from prefs collection: {e}")

        return memories

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            vec1 = self.embedder.encode_single(text1)
            vec2 = self.embedder.encode_single(text2)

            # Cosine similarity
            import numpy as np

            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _contains_outdated_temporal_info(self, text: str) -> bool:
        """Check if text contains potentially outdated temporal information."""
        import re

        current_year = datetime.now().year

        # Look for year mentions
        year_pattern = r"\b(19|20)\d{2}\b"
        years = re.findall(year_pattern, text)

        for year_str in years:
            year = int(year_str)
            if current_year - year > 3:  # More than 3 years old
                return True

        # Look for temporal keywords
        outdated_keywords = [
            "currently",
            "now",
            "today",
            "this year",
            "this month",
            "recently",
            "latest",
            "new",
            "upcoming",
        ]

        text_lower = text.lower()
        if any(keyword in text_lower for keyword in outdated_keywords):
            return True

        return False

    def _generate_merge_suggestion(self, mem1: Memory, mem2: Memory) -> str:
        """Generate a suggestion for merging two similar memories."""
        # Combine unique information
        longer = mem1 if len(mem1.text) > len(mem2.text) else mem2

        # Simple heuristic: use longer text as base
        suggestion = f"Keep: '{longer.text[:100]}...' (higher importance/longer)"

        if mem1.importance != mem2.importance:
            higher = mem1 if mem1.importance > mem2.importance else mem2
            suggestion = f"Merge into higher importance memory (importance: {higher.importance})"

        return suggestion

    def _generate_recommendations(
        self,
        overall: float,
        quality: float,
        diversity: float,
        freshness: float,
        coverage: float,
        duplication: float,
        staleness: float,
        stale_count: int,
        dup_clusters: int,
    ) -> list[str]:
        """Generate actionable recommendations based on scores."""
        recommendations = []

        if overall < 50:
            recommendations.append(
                "⚠️ Overall memory health is low. Consider the recommendations below."
            )
        elif overall > 80:
            recommendations.append("✓ Memory health is excellent!")

        if quality < 60:
            recommendations.append(
                "Improve memory quality by adding importance scores and metadata"
            )

        if diversity < 60:
            recommendations.append("Add more diverse memory types and topics")

        if freshness < 60:
            recommendations.append("Update or add recent memories to improve freshness")

        if coverage < 60:
            recommendations.append("Expand topic coverage with more varied tags and entities")

        if duplication < 70:
            recommendations.append(f"Remove or merge {dup_clusters} duplicate clusters")

        if staleness < 70 and stale_count > 0:
            recommendations.append(f"Review and update {stale_count} potentially stale memories")

        return recommendations
