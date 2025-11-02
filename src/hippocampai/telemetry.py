"""Telemetry and tracing for HippocampAI operations.

This module provides observability into memory operations, similar to Mem0's platform.
Track memory creation, retrieval, extraction, and performance metrics.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of operations to track."""

    REMEMBER = "remember"
    RECALL = "recall"
    EXTRACT = "extract"
    DEDUPLICATE = "deduplicate"
    CONSOLIDATE = "consolidate"
    DECAY = "decay"
    UPDATE = "update"
    DELETE = "delete"
    GET = "get"
    EXPIRE = "expire"


@dataclass
class TraceEvent:
    """Single trace event in a span."""

    trace_id: str
    span_id: str
    operation: OperationType
    timestamp: datetime
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # success, error, skipped
    error: Optional[str] = None


@dataclass
class MemoryTrace:
    """Complete trace for a memory operation."""

    trace_id: str
    operation: OperationType
    user_id: str
    session_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    status: str = "in_progress"


class MemoryTelemetry:
    """Centralized telemetry collector for memory operations."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.traces: dict[str, MemoryTrace] = {}
        self.metrics: dict[str, list[float]] = {
            "remember_duration": [],
            "recall_duration": [],
            "extract_duration": [],
            "retrieval_count": [],
            "update_duration": [],
            "delete_duration": [],
            "get_duration": [],
            "memory_size_chars": [],
            "memory_size_tokens": [],
        }

    def start_trace(
        self,
        operation: OperationType,
        user_id: str,
        session_id: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        """Start a new trace for an operation."""
        if not self.enabled:
            return ""

        trace_id = str(uuid4())
        trace = MemoryTrace(
            trace_id=trace_id,
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
            metadata=metadata,
        )

        self.traces[trace_id] = trace
        logger.debug(f"Started trace {trace_id} for {operation.value}")
        return trace_id

    def add_event(
        self,
        trace_id: str,
        event_name: str,
        status: str = "success",
        duration_ms: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """Add an event to a trace."""
        if not self.enabled or not trace_id or trace_id not in self.traces:
            return

        trace = self.traces[trace_id]
        event = TraceEvent(
            trace_id=trace_id,
            span_id=str(uuid4()),
            operation=trace.operation,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            metadata=metadata,
            status=status,
        )

        trace.events.append(event)
        logger.debug(f"Added event '{event_name}' to trace {trace_id}")

    def end_trace(
        self, trace_id: str, status: str = "success", result: Optional[dict[str, Any]] = None
    ) -> Optional[MemoryTrace]:
        """End a trace and record metrics."""
        if not self.enabled or not trace_id or trace_id not in self.traces:
            return None

        trace = self.traces[trace_id]
        trace.end_time = datetime.now(timezone.utc)
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        trace.status = status
        trace.result = result

        # Record metrics
        operation_key = f"{trace.operation.value}_duration"
        if operation_key in self.metrics:
            self.metrics[operation_key].append(trace.duration_ms)

        logger.debug(f"Ended trace {trace_id} with status {status} ({trace.duration_ms:.2f}ms)")
        return trace

    def get_trace(self, trace_id: str) -> Optional[MemoryTrace]:
        """Get a specific trace."""
        return self.traces.get(trace_id)

    def get_recent_traces(
        self, limit: int = 10, operation: Optional[OperationType] = None
    ) -> list[MemoryTrace]:
        """Get recent traces, optionally filtered by operation."""
        traces = list(self.traces.values())

        if operation:
            traces = [t for t in traces if t.operation == operation]

        # Sort by start time, most recent first
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    def track_memory_size(self, text_length: int, token_count: int) -> None:
        """Track memory size metrics."""
        if not self.enabled:
            return

        self.metrics["memory_size_chars"].append(float(text_length))
        self.metrics["memory_size_tokens"].append(float(token_count))

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {}

        for key, values in self.metrics.items():
            if not values:
                continue

            summary[key] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }

        return summary

    def clear_traces(self, older_than_minutes: Optional[int] = None) -> int:
        """Clear old traces to prevent memory buildup."""
        if not older_than_minutes:
            count = len(self.traces)
            self.traces.clear()
            return count

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (older_than_minutes * 60)
        old_traces = [
            tid for tid, trace in self.traces.items() if trace.start_time.timestamp() < cutoff
        ]

        for tid in old_traces:
            del self.traces[tid]

        return len(old_traces)

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((p / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_traces(self, trace_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Export traces in a format suitable for external tools (e.g., OpenTelemetry)."""
        if trace_ids:
            traces_to_export = [self.traces[tid] for tid in trace_ids if tid in self.traces]
        else:
            traces_to_export = list(self.traces.values())

        exported = []
        for trace in traces_to_export:
            exported.append(
                {
                    "trace_id": trace.trace_id,
                    "operation": trace.operation.value,
                    "user_id": trace.user_id,
                    "session_id": trace.session_id,
                    "start_time": trace.start_time.isoformat(),
                    "end_time": trace.end_time.isoformat() if trace.end_time else None,
                    "duration_ms": trace.duration_ms,
                    "status": trace.status,
                    "metadata": trace.metadata,
                    "result": trace.result,
                    "events": [
                        {
                            "span_id": event.span_id,
                            "timestamp": event.timestamp.isoformat(),
                            "duration_ms": event.duration_ms,
                            "status": event.status,
                            "metadata": event.metadata,
                        }
                        for event in trace.events
                    ],
                }
            )

        return exported


# Global telemetry instance
_global_telemetry: Optional[MemoryTelemetry] = None


def get_telemetry(enabled: bool = True) -> MemoryTelemetry:
    """Get or create global telemetry instance."""
    global _global_telemetry

    if _global_telemetry is None:
        _global_telemetry = MemoryTelemetry(enabled=enabled)

    return _global_telemetry


class traced:
    """Decorator to automatically trace function calls."""

    def __init__(self, operation: OperationType, capture_result: bool = True):
        self.operation = operation
        self.capture_result = capture_result

    def __call__(self, func: Any) -> Any:
        """Wrap function with tracing."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            telemetry = get_telemetry()

            # Extract user_id and session_id from kwargs if available
            user_id = kwargs.get("user_id", "unknown")
            session_id = kwargs.get("session_id")

            # Start trace
            trace_id = telemetry.start_trace(
                operation=self.operation,
                user_id=user_id,
                session_id=session_id,
                function=func.__name__,
            )

            start_time = time.time()
            status = "success"
            result = None
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                # Prepare result metadata
                result_meta: Optional[dict[str, Any]] = None
                if self.capture_result and result:
                    if isinstance(result, list):
                        result_meta = {"count": len(result)}
                    elif hasattr(result, "__dict__"):
                        result_meta = {"type": type(result).__name__}

                telemetry.end_trace(
                    trace_id,
                    status=status,
                    result=result_meta,
                )

                if error:
                    telemetry.add_event(
                        trace_id,
                        "error",
                        status="error",
                        duration_ms=duration_ms,
                        error=error,
                    )

        return wrapper
