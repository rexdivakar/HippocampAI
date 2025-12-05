"""Metrics and tracing for memory operations."""

import functools
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generator, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Type of metric."""

    COUNTER = "counter"  # Incremental count
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurement


class OperationType(str, Enum):
    """Type of memory operation."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    DEDUP = "dedup"
    CONSOLIDATE = "consolidate"
    CONFLICT_RESOLVE = "conflict_resolve"
    HEALTH_CHECK = "health_check"


class Metric(BaseModel):
    """Individual metric measurement."""

    name: str
    type: MetricType
    value: float
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trace(BaseModel):
    """Execution trace for operation."""

    trace_id: str
    operation: OperationType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list["Span"] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)  # For filtering and search

    # Additional metadata for comprehensive tracking
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_id: Optional[str] = None
    memory_type: Optional[str] = None
    health_score: Optional[float] = None
    result_count: Optional[int] = None


class Span(BaseModel):
    """Sub-operation span within a trace."""

    span_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: dict[str, Any] = Field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for memory operations.

    Features:
    - Operation timing and success rates
    - Memory store statistics
    - Performance metrics
    - Distributed tracing support
    """

    def __init__(self, enable_tracing: bool = True):
        """
        Initialize metrics collector.

        Args:
            enable_tracing: Whether to collect detailed traces
        """
        self.enable_tracing = enable_tracing
        self.metrics: dict[str, list[Metric]] = defaultdict(list)
        self.traces: list[Trace] = []
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Record a metric measurement.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for filtering
        """
        metric = Metric(name=name, type=metric_type, value=value, tags=tags or {})

        self.metrics[name].append(metric)

        # Update aggregated values
        if metric_type == MetricType.COUNTER:
            self.counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = value

        logger.debug(f"Recorded metric: {name}={value} ({metric_type})")

    def increment_counter(
        self, name: str, value: float = 1.0, tags: Optional[dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_histogram(
        self, name: str, value: float, tags: Optional[dict[str, str]] = None
    ) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)

    @contextmanager
    def trace_operation(
        self,
        operation: OperationType,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> Generator[Optional[Trace], None, None]:
        """
        Context manager for tracing operations with comprehensive tagging.

        Usage:
            with collector.trace_operation(
                OperationType.CREATE,
                metadata={"key": "value"},
                tags={"environment": "production", "version": "1.0"},
                user_id="user123",
                memory_type="preference"
            ):
                # ... do work ...
                pass
        """
        if not self.enable_tracing:
            yield None
            return

        trace_id = trace_id or self._generate_trace_id()
        start_time = datetime.now(timezone.utc)

        trace = Trace(
            trace_id=trace_id,
            operation=operation,
            start_time=start_time,
            metadata=metadata or {},
            tags=tags or {},
            user_id=user_id,
            session_id=session_id,
            memory_id=memory_id,
            memory_type=memory_type,
        )

        try:
            yield trace
            trace.success = True
        except Exception as e:
            trace.success = False
            trace.error = str(e)
            logger.error(f"Operation {operation} failed: {e}")
            raise
        finally:
            trace.end_time = datetime.now(timezone.utc)
            trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

            self.traces.append(trace)

            # Record metrics
            self.increment_counter(f"operation.{operation.value}.total")
            self.record_histogram(f"operation.{operation.value}.duration_ms", trace.duration_ms)

            if not trace.success:
                self.increment_counter(f"operation.{operation.value}.errors")

            logger.debug(
                f"Trace {trace_id}: {operation} completed in {trace.duration_ms:.2f}ms "
                f"(success={trace.success})"
            )

    @contextmanager
    def span(
        self, trace: Optional[Trace], name: str, tags: Optional[dict[str, Any]] = None
    ) -> Generator[Optional[Span], None, None]:
        """
        Create a span within a trace.

        Usage:
            with collector.trace_operation(OperationType.CREATE) as trace:
                with collector.span(trace, "embed_text"):
                    # ... embedding work ...
                    pass
        """
        if not self.enable_tracing or trace is None:
            yield None
            return

        span_id = f"{trace.trace_id}_span_{len(trace.spans)}"
        start_time = datetime.now(timezone.utc)

        span_obj = Span(span_id=span_id, name=name, start_time=start_time, tags=tags or {})

        try:
            yield span_obj
        finally:
            span_obj.end_time = datetime.now(timezone.utc)
            span_obj.duration_ms = (span_obj.end_time - span_obj.start_time).total_seconds() * 1000

            trace.spans.append(span_obj)

            logger.debug(f"Span {name} completed in {span_obj.duration_ms:.2f}ms")

    def time_function(
        self, operation: OperationType
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to automatically trace function execution.

        Usage:
            @collector.time_function(OperationType.CREATE)
            async def create_memory(...):
                pass
        """
        import inspect

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace_operation(operation, metadata={"function": func.__name__}):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace_operation(operation, metadata={"function": func.__name__}):
                    return func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "total_traces": len(self.traces),
            "successful_operations": sum(1 for t in self.traces if t.success),
            "failed_operations": sum(1 for t in self.traces if not t.success),
        }

        # Calculate operation statistics
        operation_stats: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_duration_ms": 0, "errors": 0}
        )

        for trace in self.traces:
            stats = operation_stats[trace.operation.value]
            stats["count"] += 1
            if trace.duration_ms:
                stats["total_duration_ms"] += trace.duration_ms
            if not trace.success:
                stats["errors"] += 1

        # Calculate averages
        for op, stats in operation_stats.items():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["error_rate"] = stats["errors"] / stats["count"]

        summary["operation_stats"] = dict(operation_stats)

        return summary

    def get_histogram_stats(self, metric_name: str) -> dict[str, float]:
        """Get statistical summary of histogram metric."""
        values = [
            m.value for m in self.metrics.get(metric_name, []) if m.type == MetricType.HISTOGRAM
        ]

        if not values:
            return {}

        import numpy as np

        return {
            "count": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    def get_recent_traces(
        self, limit: int = 10, operation: Optional[OperationType] = None
    ) -> list[Trace]:
        """Get recent traces, optionally filtered by operation."""
        traces = self.traces

        if operation:
            traces = [t for t in traces if t.operation == operation]

        # Sort by start time descending
        traces = sorted(traces, key=lambda t: t.start_time, reverse=True)

        return traces[:limit]

    def query_traces(
        self,
        tags: Optional[dict[str, str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        operation: Optional[OperationType] = None,
        success: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 100,
    ) -> list[Trace]:
        """
        Query traces with advanced filtering.

        Args:
            tags: Filter by tags (all must match)
            user_id: Filter by user ID
            session_id: Filter by session ID
            memory_type: Filter by memory type
            operation: Filter by operation type
            success: Filter by success status
            min_duration_ms: Minimum duration
            max_duration_ms: Maximum duration
            limit: Maximum results

        Returns:
            List of matching traces
        """
        filtered_traces = []

        for trace in self.traces:
            # Apply filters
            if operation and trace.operation != operation:
                continue

            if success is not None and trace.success != success:
                continue

            if user_id and trace.user_id != user_id:
                continue

            if session_id and trace.session_id != session_id:
                continue

            if memory_type and trace.memory_type != memory_type:
                continue

            if min_duration_ms and (
                trace.duration_ms is None or trace.duration_ms < min_duration_ms
            ):
                continue

            if max_duration_ms and (
                trace.duration_ms is None or trace.duration_ms > max_duration_ms
            ):
                continue

            # Check tags (all must match)
            if tags:
                if not all(trace.tags.get(k) == v for k, v in tags.items()):
                    continue

            filtered_traces.append(trace)

        # Sort by start time descending
        filtered_traces = sorted(filtered_traces, key=lambda t: t.start_time, reverse=True)

        return filtered_traces[:limit]

    def get_trace_statistics(
        self,
        operation: Optional[OperationType] = None,
        user_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Get statistics for traces matching filters.

        Args:
            operation: Filter by operation type
            user_id: Filter by user ID
            tags: Filter by tags

        Returns:
            Statistics dictionary
        """
        traces = self.query_traces(operation=operation, user_id=user_id, tags=tags, limit=10000)

        if not traces:
            return {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
            }

        import numpy as np

        durations = [t.duration_ms for t in traces if t.duration_ms is not None]

        stats: dict[str, Any] = {
            "count": len(traces),
            "success_count": sum(1 for t in traces if t.success),
            "error_count": sum(1 for t in traces if not t.success),
            "success_rate": sum(1 for t in traces if t.success) / len(traces) * 100,
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

        # Group by tags
        if tags:
            stats["matching_tags"] = tags

        return stats

    def reset_metrics(self) -> None:
        """Reset all collected metrics and traces."""
        self.metrics.clear()
        self.traces.clear()
        self.counters.clear()
        self.gauges.clear()
        logger.info("Metrics reset")

    def export_metrics(self, format: str = "prometheus") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format ("prometheus", "json")

        Returns:
            Formatted metrics string
        """
        if format == "prometheus":
            return self._export_prometheus()
        elif format == "json":
            import json

            return json.dumps(self.get_metrics_summary(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Histograms
        histogram_metrics = defaultdict(list)
        for name, metrics in self.metrics.items():
            histogram_values = [m.value for m in metrics if m.type == MetricType.HISTOGRAM]
            if histogram_values:
                histogram_metrics[name] = histogram_values

        for name, values in histogram_metrics.items():
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {len(values)}")
            lines.append(f"{name}_sum {sum(values)}")

            # Add quantiles
            import numpy as np

            for quantile in [0.5, 0.9, 0.95, 0.99]:
                value = float(np.percentile(values, quantile * 100))
                lines.append(f'{name}{{quantile="{quantile}"}} {value}')

        return "\n".join(lines)

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        import uuid

        return f"trace_{uuid.uuid4().hex[:16]}"


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def configure_metrics(enable_tracing: bool = True) -> None:
    """Configure global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(enable_tracing=enable_tracing)


# Convenience functions


def record_memory_operation(
    operation: OperationType,
    success: bool = True,
    duration_ms: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Record a memory operation metric with optional metadata tags."""
    collector = get_metrics_collector()

    # Extract tags from metadata for filtering and analysis
    tags = {}
    if metadata:
        # Extract common fields as tags
        if "user_id" in metadata:
            tags["user_id"] = str(metadata["user_id"])
        if "memory_type" in metadata:
            tags["memory_type"] = str(metadata["memory_type"])
        if "session_id" in metadata:
            tags["session_id"] = str(metadata["session_id"])

    collector.increment_counter(f"memory.{operation.value}.total", tags=tags)

    if not success:
        collector.increment_counter(f"memory.{operation.value}.errors", tags=tags)

    if duration_ms is not None:
        collector.record_histogram(f"memory.{operation.value}.duration_ms", duration_ms, tags=tags)


def record_memory_store_stats(
    total_memories: int,
    healthy_memories: int,
    stale_memories: int,
    duplicate_clusters: int,
    health_score: float,
) -> None:
    """Record memory store statistics."""
    collector = get_metrics_collector()

    collector.set_gauge("memory_store.total_memories", total_memories)
    collector.set_gauge("memory_store.healthy_memories", healthy_memories)
    collector.set_gauge("memory_store.stale_memories", stale_memories)
    collector.set_gauge("memory_store.duplicate_clusters", duplicate_clusters)
    collector.set_gauge("memory_store.health_score", health_score)


def record_search_metrics(
    query_time_ms: float,
    results_count: int,
    cache_hit: bool = False,
) -> None:
    """Record search operation metrics."""
    collector = get_metrics_collector()

    collector.record_histogram("search.query_time_ms", query_time_ms)
    collector.set_gauge("search.results_count", results_count)

    if cache_hit:
        collector.increment_counter("search.cache_hits")
    else:
        collector.increment_counter("search.cache_misses")


def record_quality_metrics(
    freshness_score: float,
    diversity_score: float,
    consistency_score: float,
    coverage_score: float,
) -> None:
    """Record memory quality metrics."""
    collector = get_metrics_collector()

    collector.set_gauge("quality.freshness_score", freshness_score)
    collector.set_gauge("quality.diversity_score", diversity_score)
    collector.set_gauge("quality.consistency_score", consistency_score)
    collector.set_gauge("quality.coverage_score", coverage_score)
