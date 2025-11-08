"""Memory debugging and observability: explainability, visualization, profiling."""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalExplanation(BaseModel):
    """Explanation of why a memory was retrieved."""

    memory_id: str
    rank: int
    final_score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    ranking_factors: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    contributing_factors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimilarityVisualization(BaseModel):
    """Visualization data for similarity scores."""

    query: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    score_distribution: dict[str, int] = Field(default_factory=dict)
    avg_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0


class MemoryAccessHeatmap(BaseModel):
    """Heatmap of memory access patterns."""

    user_id: str
    time_period_days: int
    access_by_hour: dict[int, int] = Field(default_factory=dict)
    access_by_day: dict[str, int] = Field(default_factory=dict)
    access_by_type: dict[str, int] = Field(default_factory=dict)
    hot_memories: list[tuple[str, int]] = Field(default_factory=list)
    cold_memories: list[str] = Field(default_factory=list)
    peak_hours: list[int] = Field(default_factory=list)
    total_accesses: int = 0


class QueryPerformanceProfile(BaseModel):
    """Performance profile for a query."""

    query: str
    total_time_ms: float
    stage_timings: dict[str, float] = Field(default_factory=dict)
    memory_count: int = 0
    vector_search_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    filtering_ms: Optional[float] = None
    bottlenecks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class PerformanceSnapshot(BaseModel):
    """Snapshot of system performance."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_queries: int = 0
    avg_query_time_ms: float = 0.0
    slow_queries: int = 0
    cache_hit_rate: float = 0.0
    total_memories: int = 0
    avg_results_returned: float = 0.0
    performance_score: float = Field(ge=0.0, le=100.0)


class MemoryObservabilityMonitor:
    """Monitor and explain memory retrieval and system performance."""

    def __init__(
        self,
        enable_profiling: bool = True,
        slow_query_threshold_ms: float = 1000.0,
        track_access_patterns: bool = True,
    ):
        """Initialize observability monitor.

        Args:
            enable_profiling: Enable performance profiling
            slow_query_threshold_ms: Threshold for flagging slow queries
            track_access_patterns: Track memory access patterns
        """
        self.enable_profiling = enable_profiling
        self.slow_query_threshold = slow_query_threshold_ms
        self.track_access = track_access_patterns

        # Performance tracking
        self.query_times: list[float] = []
        self.query_history: list[QueryPerformanceProfile] = []
        self.access_log: list[dict[str, Any]] = []

    def explain_retrieval(
        self,
        query: str,
        results: list[RetrievalResult],
        retrieval_metadata: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalExplanation]:
        """Explain why each memory was retrieved and ranked.

        Args:
            query: Original query
            results: Retrieved results
            retrieval_metadata: Additional metadata from retrieval process

        Returns:
            List of explanations for each result
        """
        explanations = []

        for rank, result in enumerate(results, start=1):
            # Extract score components
            score_breakdown = result.breakdown if result.breakdown else {}

            # Generate human-readable explanation
            explanation_parts = []
            contributing_factors = []

            # Analyze score breakdown
            vector_score = score_breakdown.get("vector", 0.0)
            bm25_score = score_breakdown.get("bm25", 0.0)
            recency_score = score_breakdown.get("recency", 0.0)
            importance_score = score_breakdown.get("importance", 0.0)

            if vector_score > 0.7:
                explanation_parts.append("High semantic similarity to query")
                contributing_factors.append("semantic_match")

            if bm25_score > 0.5:
                explanation_parts.append("Strong keyword overlap")
                contributing_factors.append("keyword_match")

            if recency_score > 0.7:
                explanation_parts.append("Recent memory")
                contributing_factors.append("recency")

            if importance_score > 0.7:
                explanation_parts.append("High importance score")
                contributing_factors.append("importance")

            # Check memory properties
            mem = result.memory
            if mem.access_count > 10:
                explanation_parts.append(f"Frequently accessed ({mem.access_count} times)")
                contributing_factors.append("high_access_count")

            if mem.tags:
                matching_tags = self._find_matching_tags(query, mem.tags)
                if matching_tags:
                    explanation_parts.append(f"Tags: {', '.join(matching_tags)}")
                    contributing_factors.append("tag_match")

            # Combine explanation
            if explanation_parts:
                explanation = ". ".join(explanation_parts) + "."
            else:
                explanation = "Retrieved based on overall relevance score."

            explanations.append(
                RetrievalExplanation(
                    memory_id=mem.id,
                    rank=rank,
                    final_score=result.score,
                    score_breakdown=score_breakdown,
                    ranking_factors={
                        "vector": vector_score,
                        "bm25": bm25_score,
                        "recency": recency_score,
                        "importance": importance_score,
                        "access_count": mem.access_count,
                        "tags": mem.tags,
                    },
                    explanation=explanation,
                    contributing_factors=contributing_factors,
                )
            )

        return explanations

    def visualize_similarity_scores(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> SimilarityVisualization:
        """Create visualization data for similarity scores.

        Args:
            query: Query text
            results: Retrieval results
            top_k: Number of top results to visualize

        Returns:
            SimilarityVisualization with data for charts
        """
        top_results = results[:top_k]

        # Prepare visualization data
        viz_results = []
        scores = []

        for i, result in enumerate(top_results, start=1):
            scores.append(result.score)
            viz_results.append({
                "rank": i,
                "memory_id": result.memory.id,
                "score": result.score,
                "text_preview": result.memory.text[:100],
                "type": result.memory.type.value,
                "importance": result.memory.importance,
                "breakdown": result.breakdown,
            })

        # Score distribution (buckets)
        distribution: dict[str, int] = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for score in scores:
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return SimilarityVisualization(
            query=query,
            results=viz_results,
            score_distribution=distribution,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
        )

    def generate_access_heatmap(
        self,
        user_id: str,
        memories: list[Memory],
        time_period_days: int = 30,
    ) -> MemoryAccessHeatmap:
        """Generate heatmap of memory access patterns.

        Args:
            user_id: User identifier
            memories: User's memories
            time_period_days: Time period to analyze

        Returns:
            MemoryAccessHeatmap with access patterns
        """
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=time_period_days)

        # Track accesses by various dimensions
        access_by_hour: dict[int, int] = defaultdict(int)
        access_by_day: dict[str, int] = defaultdict(int)
        access_by_type: dict[str, int] = defaultdict(int)

        total_accesses = 0
        memory_access_counts: dict[str, int] = {}

        for mem in memories:
            # Count accesses
            access_count = mem.access_count
            total_accesses += access_count
            memory_access_counts[mem.id] = access_count

            # By type
            access_by_type[mem.type.value] += access_count

            # Simulate hourly/daily distribution based on updated_at
            # (In real implementation, you'd track actual access timestamps)
            if mem.updated_at and mem.updated_at >= cutoff_date:
                hour = mem.updated_at.hour
                day_name = mem.updated_at.strftime("%A")
                access_by_hour[hour] += 1
                access_by_day[day_name] += 1

        # Identify hot and cold memories
        sorted_memories = sorted(
            memory_access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        hot_memories = sorted_memories[:10]  # Top 10
        cold_memories = [
            mem_id for mem_id, count in sorted_memories
            if count == 0
        ][:20]  # Up to 20 never-accessed

        # Find peak hours
        if access_by_hour:
            max_accesses = max(access_by_hour.values())
            peak_hours = [
                hour for hour, count in access_by_hour.items()
                if count >= max_accesses * 0.8  # Within 80% of peak
            ]
        else:
            peak_hours = []

        return MemoryAccessHeatmap(
            user_id=user_id,
            time_period_days=time_period_days,
            access_by_hour=dict(access_by_hour),
            access_by_day=dict(access_by_day),
            access_by_type=dict(access_by_type),
            hot_memories=hot_memories,
            cold_memories=cold_memories,
            peak_hours=sorted(peak_hours),
            total_accesses=total_accesses,
        )

    def profile_query_performance(
        self,
        query: str,
        stage_timings: dict[str, float],
        result_count: int,
    ) -> QueryPerformanceProfile:
        """Profile query performance and identify bottlenecks.

        Args:
            query: Query text
            stage_timings: Timing for each stage (in ms)
            result_count: Number of results returned

        Returns:
            QueryPerformanceProfile with analysis
        """
        total_time = sum(stage_timings.values())

        # Identify bottlenecks (stages taking >30% of time)
        bottlenecks = []
        for stage, timing in stage_timings.items():
            if timing > total_time * 0.3:
                bottlenecks.append(f"{stage} ({timing:.2f}ms, {timing/total_time*100:.1f}%)")

        # Generate recommendations
        recommendations = []

        if stage_timings.get("vector_search", 0) > 500:
            recommendations.append("Consider increasing HNSW ef_search parameter for faster vector search")

        if stage_timings.get("reranking", 0) > 300:
            recommendations.append("Reranking is slow - consider reducing candidate pool size")

        if stage_timings.get("filtering", 0) > 100:
            recommendations.append("Filtering overhead is high - optimize filter conditions")

        if total_time > self.slow_query_threshold:
            recommendations.append(f"Query exceeded slow threshold ({self.slow_query_threshold}ms)")

        if result_count > 100:
            recommendations.append("Large result set - consider pagination or stricter filtering")

        profile = QueryPerformanceProfile(
            query=query[:100],  # Truncate long queries
            total_time_ms=total_time,
            stage_timings=stage_timings,
            memory_count=result_count,
            vector_search_ms=stage_timings.get("vector_search"),
            reranking_ms=stage_timings.get("reranking"),
            filtering_ms=stage_timings.get("filtering"),
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

        # Track for historical analysis
        if self.enable_profiling:
            self.query_history.append(profile)
            self.query_times.append(total_time)

        return profile

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot.

        Returns:
            PerformanceSnapshot with system metrics
        """
        if not self.query_times:
            return PerformanceSnapshot(
                performance_score=100.0,
            )

        total_queries = len(self.query_times)
        avg_time = sum(self.query_times) / total_queries
        slow_queries = sum(1 for t in self.query_times if t > self.slow_query_threshold)

        # Calculate performance score
        # Lower average time = higher score
        time_score = max(0, 100 - (avg_time / 10))  # 1000ms = 0 score

        # Fewer slow queries = higher score
        slow_ratio = slow_queries / total_queries if total_queries > 0 else 0
        slow_score = (1 - slow_ratio) * 100

        # Combined score
        performance_score = (time_score * 0.6 + slow_score * 0.4)

        # Average results returned
        avg_results = 0.0
        if self.query_history:
            avg_results = sum(q.memory_count for q in self.query_history) / len(self.query_history)

        return PerformanceSnapshot(
            total_queries=total_queries,
            avg_query_time_ms=avg_time,
            slow_queries=slow_queries,
            cache_hit_rate=0.0,  # Placeholder - implement cache tracking
            total_memories=0,  # Placeholder - pass from outside
            avg_results_returned=avg_results,
            performance_score=performance_score,
        )

    def identify_slow_queries(
        self,
        threshold_ms: Optional[float] = None,
    ) -> list[QueryPerformanceProfile]:
        """Identify queries that exceeded performance threshold.

        Args:
            threshold_ms: Custom threshold (uses default if None)

        Returns:
            List of slow query profiles
        """
        threshold = threshold_ms or self.slow_query_threshold

        slow_queries = [
            profile for profile in self.query_history
            if profile.total_time_ms > threshold
        ]

        # Sort by time (slowest first)
        slow_queries.sort(key=lambda q: q.total_time_ms, reverse=True)

        return slow_queries

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Dict with performance analysis
        """
        snapshot = self.get_performance_snapshot()
        slow_queries = self.identify_slow_queries()

        # Stage analysis
        stage_stats: dict[str, list[float]] = defaultdict(list)
        for profile in self.query_history:
            for stage, timing in profile.stage_timings.items():
                stage_stats[stage].append(timing)

        stage_averages = {
            stage: sum(timings) / len(timings)
            for stage, timings in stage_stats.items()
        }

        # Bottleneck analysis
        bottleneck_counter: dict[str, int] = defaultdict(int)
        for profile in self.query_history:
            for bottleneck in profile.bottlenecks:
                stage = bottleneck.split("(")[0].strip()
                bottleneck_counter[stage] += 1

        return {
            "snapshot": snapshot.model_dump(),
            "slow_query_count": len(slow_queries),
            "slowest_queries": [q.model_dump() for q in slow_queries[:5]],
            "stage_averages_ms": stage_averages,
            "common_bottlenecks": dict(
                sorted(bottleneck_counter.items(), key=lambda x: x[1], reverse=True)
            ),
            "recommendations": self._generate_performance_recommendations(
                snapshot, stage_averages, bottleneck_counter
            ),
        }

    def clear_history(self):
        """Clear performance history."""
        self.query_times.clear()
        self.query_history.clear()
        self.access_log.clear()
        logger.info("Performance history cleared")

    # Private helper methods

    def _find_matching_tags(self, query: str, tags: list[str]) -> list[str]:
        """Find tags that match query terms."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matching = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in query_lower or any(word in tag_lower for word in query_words):
                matching.append(tag)

        return matching

    def _generate_performance_recommendations(
        self,
        snapshot: PerformanceSnapshot,
        stage_averages: dict[str, float],
        bottlenecks: dict[str, int],
    ) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if snapshot.avg_query_time_ms > 500:
            recommendations.append("Average query time is high - consider optimizing vector search parameters")

        if snapshot.slow_queries > snapshot.total_queries * 0.1:
            recommendations.append("More than 10% of queries are slow - investigate bottlenecks")

        # Stage-specific recommendations
        if stage_averages.get("vector_search", 0) > 300:
            recommendations.append("Vector search is slow - increase ef_search or reduce vector dimensions")

        if stage_averages.get("reranking", 0) > 200:
            recommendations.append("Reranking is a bottleneck - reduce candidate pool or use faster reranker")

        # Bottleneck recommendations
        if bottlenecks:
            most_common = max(bottlenecks.items(), key=lambda x: x[1])
            recommendations.append(
                f"'{most_common[0]}' is the most common bottleneck ({most_common[1]} occurrences)"
            )

        if not recommendations:
            recommendations.append("Performance is good - no immediate optimizations needed")

        return recommendations


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str):
        """Initialize timer.

        Args:
            operation_name: Name of operation being timed
        """
        self.operation = operation_name
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and calculate elapsed time."""
        if self.start_time:
            elapsed = time.perf_counter() - self.start_time
            self.elapsed_ms = elapsed * 1000  # Convert to ms

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms


def create_timing_context() -> dict[str, PerformanceTimer]:
    """Create a dict to track multiple timers."""
    return {}


def profile_operation(func):
    """Decorator to profile function execution time."""
    def wrapper(*args, **kwargs):
        timer = PerformanceTimer(func.__name__)
        with timer:
            result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} took {timer.elapsed_ms:.2f}ms")
        return result
    return wrapper
