"""Storage integration for monitoring data in Qdrant."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from qdrant_client.models import Distance, VectorParams

from hippocampai.monitoring.memory_health import (
    MemoryQualityReport,
)
from hippocampai.monitoring.metrics import Trace
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class MonitoringStorage:
    """
    Storage layer for monitoring data in Qdrant.

    Stores:
    - Health reports with full metadata
    - Trace data with tags for querying
    - Metrics snapshots for historical analysis
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        collection_health: str = "hippocampai_health_reports",
        collection_traces: str = "hippocampai_traces",
    ):
        """
        Initialize monitoring storage.

        Args:
            qdrant_store: Qdrant vector store instance
            collection_health: Collection name for health reports
            collection_traces: Collection name for traces
        """
        self.qdrant = qdrant_store
        self.collection_health = collection_health
        self.collection_traces = collection_traces

        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Initialize Qdrant collections for monitoring data."""
        # Check if collections exist first
        try:
            existing_collections = [
                col.name for col in self.qdrant.client.get_collections().collections
            ]
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
            existing_collections = []

        # Only create if they don't exist
        if self.collection_health not in existing_collections:
            try:
                # Health reports collection (minimal vectors, metadata-focused)
                # Use client directly to avoid QdrantStore's default vector size
                self.qdrant.client.create_collection(
                    collection_name=self.collection_health,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
                logger.info(f"Initialized collection: {self.collection_health}")
            except Exception as e:
                logger.debug(f"Collection {self.collection_health} creation failed: {e}")
        else:
            logger.debug(f"Collection {self.collection_health} already exists")

        if self.collection_traces not in existing_collections:
            try:
                # Traces collection (minimal vectors, metadata-focused)
                # Use client directly to avoid QdrantStore's default vector size
                self.qdrant.client.create_collection(
                    collection_name=self.collection_traces,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
                logger.info(f"Initialized collection: {self.collection_traces}")
            except Exception as e:
                logger.debug(f"Collection {self.collection_traces} creation failed: {e}")
        else:
            logger.debug(f"Collection {self.collection_traces} already exists")

    def store_health_report(
        self,
        report: MemoryQualityReport,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Store health report in Qdrant with comprehensive metadata.

        Args:
            report: Quality report to store
            tags: Additional tags for filtering

        Returns:
            Report ID (UUID string)
        """
        # Generate UUID for Qdrant (must be UUID or unsigned integer)
        report_uuid = str(uuid.uuid4())

        # Build comprehensive payload with tags
        payload = {
            # Core data
            "report_id": report_uuid,
            "user_id": report.user_id,
            "generated_at": report.generated_at.isoformat(),
            # Health score
            "health_score": report.health_score.overall_score,
            "health_status": report.health_score.status,
            "freshness_score": report.health_score.freshness_score,
            "diversity_score": report.health_score.diversity_score,
            "consistency_score": report.health_score.consistency_score,
            "coverage_score": report.health_score.coverage_score,
            # Counts
            "total_memories": report.health_score.total_memories,
            "healthy_memories": report.health_score.healthy_memories,
            "stale_memories": report.health_score.stale_memories,
            "duplicate_clusters": report.health_score.duplicate_clusters,
            "low_quality_memories": report.health_score.low_quality_memories,
            # Issues
            "duplicate_count": len(report.duplicate_clusters),
            "stale_count": len(report.stale_memories),
            "topic_count": len(report.topic_coverage),
            # Recommendations
            "recommendations": report.health_score.recommendations,
            # Metrics
            "metrics": report.health_score.metrics,
            # Tags for filtering
            "tags": tags or {},
            # Searchable fields
            "timestamp": int(report.generated_at.timestamp()),
        }

        # Store with dummy vector (metadata-only storage)
        self.qdrant.upsert(
            collection_name=self.collection_health,
            id=report_uuid,
            vector=[0.0],  # Dummy vector
            payload=payload,
        )

        logger.info(f"Stored health report: {report_uuid}")
        return report_uuid

    def store_trace(
        self,
        trace: Trace,
        additional_tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Store trace in Qdrant with full metadata and tags.

        Args:
            trace: Trace to store
            additional_tags: Additional tags beyond trace.tags

        Returns:
            Trace ID (UUID string)
        """
        # Convert trace_id to UUID if it's not already
        # trace.trace_id format is "trace_{uuid_without_dashes}"
        # We need to extract the UUID portion or generate a proper UUID
        try:
            # Try to parse as UUID first
            trace_uuid = str(uuid.UUID(trace.trace_id))
        except (ValueError, AttributeError):
            # If trace_id is in format "trace_xxxxxxxx", extract the hex part
            if trace.trace_id.startswith("trace_"):
                hex_part = trace.trace_id[6:]
                try:
                    # Try to reconstruct UUID from hex string
                    trace_uuid = str(uuid.UUID(hex=hex_part))
                except ValueError:
                    # If that fails, generate a new UUID
                    trace_uuid = str(uuid.uuid4())
            else:
                # Generate new UUID if format is unexpected
                trace_uuid = str(uuid.uuid4())

        # Merge tags
        all_tags = {**trace.tags}
        if additional_tags:
            all_tags.update(additional_tags)

        # Build payload
        payload = {
            # Core trace data
            "trace_id": trace_uuid,
            "original_trace_id": trace.trace_id,  # Keep original for reference
            "operation": trace.operation,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "duration_ms": trace.duration_ms,
            "success": trace.success,
            "error": trace.error,
            # Metadata
            "metadata": trace.metadata,
            "tags": all_tags,
            # User/session context
            "user_id": trace.user_id,
            "session_id": trace.session_id,
            "memory_id": trace.memory_id,
            "memory_type": trace.memory_type,
            # Additional fields
            "health_score": trace.health_score,
            "result_count": trace.result_count,
            # Spans
            "span_count": len(trace.spans),
            "spans": [
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "duration_ms": span.duration_ms,
                    "tags": span.tags,
                }
                for span in trace.spans
            ],
            # Searchable fields
            "timestamp": int(trace.start_time.timestamp()),
        }

        # Store with dummy vector
        self.qdrant.upsert(
            collection_name=self.collection_traces,
            id=trace_uuid,
            vector=[0.0],  # Dummy vector
            payload=payload,
        )

        logger.debug(f"Stored trace: {trace_uuid} (original: {trace.trace_id})")
        return trace_uuid

    def query_health_reports(
        self,
        user_id: Optional[str] = None,
        min_health_score: Optional[float] = None,
        max_health_score: Optional[float] = None,
        status: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query health reports with filtering.

        Args:
            user_id: Filter by user ID
            min_health_score: Minimum health score
            max_health_score: Maximum health score
            status: Filter by status
            tags: Filter by tags (all must match)
            start_time: Reports after this time
            end_time: Reports before this time
            limit: Maximum results

        Returns:
            List of health report payloads
        """
        # Build filters
        filters = {}

        if user_id:
            filters["user_id"] = user_id

        if status:
            filters["health_status"] = status

        # Add tag filters
        if tags:
            for key, value in tags.items():
                filters[f"tags.{key}"] = value

        # Query Qdrant
        try:
            results = self.qdrant.scroll(
                collection_name=self.collection_health,
                filters=filters,
                limit=limit * 2,  # Get more for time filtering
            )

            # Post-filter by score and time
            filtered_results = []
            for result in results:
                payload = result["payload"]

                # Health score filter
                if min_health_score is not None:
                    if payload.get("health_score", 0) < min_health_score:
                        continue

                if max_health_score is not None:
                    if payload.get("health_score", 100) > max_health_score:
                        continue

                # Time filters
                if start_time or end_time:
                    generated_at = datetime.fromisoformat(payload["generated_at"])

                    if start_time and generated_at < start_time:
                        continue

                    if end_time and generated_at > end_time:
                        continue

                filtered_results.append(payload)

            # Sort by timestamp descending
            filtered_results.sort(key=lambda x: x["timestamp"], reverse=True)

            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Error querying health reports: {e}")
            return []

    def query_traces(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        success: Optional[bool] = None,
        tags: Optional[dict[str, str]] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query traces with comprehensive filtering.

        Args:
            operation: Filter by operation type
            user_id: Filter by user ID
            session_id: Filter by session ID
            memory_type: Filter by memory type
            success: Filter by success status
            tags: Filter by tags (all must match)
            min_duration_ms: Minimum duration
            max_duration_ms: Maximum duration
            start_time: Traces after this time
            end_time: Traces before this time
            limit: Maximum results

        Returns:
            List of trace payloads
        """
        # Build filters
        filters = {}

        if operation:
            filters["operation"] = operation

        if user_id:
            filters["user_id"] = user_id

        if session_id:
            filters["session_id"] = session_id

        if memory_type:
            filters["memory_type"] = memory_type

        if success is not None:
            filters["success"] = success

        # Add tag filters
        if tags:
            for key, value in tags.items():
                filters[f"tags.{key}"] = value

        # Query Qdrant
        try:
            results = self.qdrant.scroll(
                collection_name=self.collection_traces,
                filters=filters,
                limit=limit * 2,  # Get more for post-filtering
            )

            # Post-filter by duration and time
            filtered_results = []
            for result in results:
                payload = result["payload"]

                # Duration filters
                duration = payload.get("duration_ms")

                if min_duration_ms is not None and (duration is None or duration < min_duration_ms):
                    continue

                if max_duration_ms is not None and (duration is None or duration > max_duration_ms):
                    continue

                # Time filters
                if start_time or end_time:
                    trace_start = datetime.fromisoformat(payload["start_time"])

                    if start_time and trace_start < start_time:
                        continue

                    if end_time and trace_start > end_time:
                        continue

                filtered_results.append(payload)

            # Sort by timestamp descending
            filtered_results.sort(key=lambda x: x["timestamp"], reverse=True)

            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Error querying traces: {e}")
            return []

    def get_health_history(
        self,
        user_id: str,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get health score history for a user.

        Args:
            user_id: User identifier
            days: Number of days of history
            limit: Maximum results

        Returns:
            List of health scores over time
        """
        start_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)

        reports = self.query_health_reports(
            user_id=user_id,
            start_time=start_time,
            limit=limit,
        )

        # Extract time series data
        history = [
            {
                "timestamp": r["generated_at"],
                "health_score": r["health_score"],
                "status": r["health_status"],
                "total_memories": r["total_memories"],
                "stale_memories": r["stale_memories"],
                "duplicate_clusters": r["duplicate_clusters"],
            }
            for r in reports
        ]

        return history

    def get_trace_statistics(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Get trace statistics for a time period.

        Args:
            operation: Filter by operation
            user_id: Filter by user
            tags: Filter by tags
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        start_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)

        traces = self.query_traces(
            operation=operation,
            user_id=user_id,
            tags=tags,
            start_time=start_time,
            limit=10000,
        )

        if not traces:
            return {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
            }

        import numpy as np

        durations = [t["duration_ms"] for t in traces if t.get("duration_ms") is not None]

        stats = {
            "count": len(traces),
            "success_count": sum(1 for t in traces if t["success"]),
            "error_count": sum(1 for t in traces if not t["success"]),
            "success_rate": (sum(1 for t in traces if t["success"]) / len(traces) * 100),
            "time_period_days": days,
        }

        if durations:
            stats["duration_stats"] = {
                "min_ms": float(np.min(durations)),
                "max_ms": float(np.max(durations)),
                "mean_ms": float(np.mean(durations)),
                "median_ms": float(np.median(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "p99_ms": float(np.percentile(durations, 99)),
            }

        return stats

    def cleanup_old_data(
        self,
        health_retention_days: int = 90,
        trace_retention_days: int = 30,
    ) -> dict[str, int]:
        """
        Clean up old monitoring data.

        Args:
            health_retention_days: Days to retain health reports
            trace_retention_days: Days to retain traces

        Returns:
            Dictionary with cleanup counts
        """
        now = datetime.now(timezone.utc)
        health_cutoff = now.replace(day=now.day - health_retention_days)
        trace_cutoff = now.replace(day=now.day - trace_retention_days)

        # Query old health reports
        old_health = self.query_health_reports(
            end_time=health_cutoff,
            limit=10000,
        )

        # Query old traces
        old_traces = self.query_traces(
            end_time=trace_cutoff,
            limit=10000,
        )

        # Delete old health reports
        health_deleted = 0
        if old_health:
            health_ids = [r["report_id"] for r in old_health]
            self.qdrant.delete(
                collection_name=self.collection_health,
                ids=health_ids,
            )
            health_deleted = len(health_ids)

        # Delete old traces
        traces_deleted = 0
        if old_traces:
            trace_ids = [t["trace_id"] for t in old_traces]
            self.qdrant.delete(
                collection_name=self.collection_traces,
                ids=trace_ids,
            )
            traces_deleted = len(trace_ids)

        logger.info(f"Cleaned up {health_deleted} health reports and {traces_deleted} traces")

        return {
            "health_reports_deleted": health_deleted,
            "traces_deleted": traces_deleted,
        }
