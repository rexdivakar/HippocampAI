"""Telemetry and tracing for HippocampAI operations.

This module provides observability into memory operations, similar to Mem0's platform.
Track memory creation, retrieval, extraction, and performance metrics.
"""

import contextvars
import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator, Optional
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
        self.llm_invocations: dict[str, "LLMInvocation"] = {}
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

    # ------------------------------------------------------------------
    # LLM usage tracing
    # ------------------------------------------------------------------

    def start_llm_invocation(
        self,
        provider: str,
        requested_model: str,
        operation: str,
        streaming: bool = False,
        **explicit_context: Any,
    ) -> str:
        """Start a logical LLM invocation and return its id.

        Attribution (workflow/agent/tool/tenant/...) is pulled from the
        active ``LLMTraceContext`` and merged with any explicit overrides.
        Returns a valid id even when telemetry is disabled, so callers don't
        need to branch on ``self.enabled``.
        """
        invocation_id = str(uuid4())
        if not self.enabled:
            return invocation_id
        ctx = get_current_llm_context().merged(**explicit_context)
        invocation = LLMInvocation(
            invocation_id=invocation_id,
            trace_id=ctx.trace_id or str(uuid4()),
            request_id=str(uuid4()),
            parent_span_id=ctx.parent_span_id,
            provider=provider,
            requested_model=requested_model,
            operation=ctx.feature_name or operation,
            start_time=datetime.now(timezone.utc),
            workflow_id=ctx.workflow_id,
            workflow_name=ctx.workflow_name,
            workflow_step_id=ctx.workflow_step_id,
            workflow_step_name=ctx.workflow_step_name,
            agent_id=ctx.agent_id,
            agent_name=ctx.agent_name,
            tool_name=ctx.tool_name,
            feature_name=ctx.feature_name,
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            conversation_id=ctx.conversation_id,
            billing_bucket=ctx.billing_bucket,
            streaming=streaming,
            metadata=_cap_metadata(dict(ctx.metadata)),
        )
        self.llm_invocations[invocation_id] = invocation
        return invocation_id

    def end_llm_invocation(
        self,
        invocation_id: str,
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ) -> Optional["LLMInvocation"]:
        """Finalize a logical invocation, aggregating its attempts. No-op if
        telemetry is disabled or the invocation is unknown; idempotent
        otherwise (see ``LLMInvocation.finalize``)."""
        if not self.enabled:
            return None
        invocation = self.llm_invocations.get(invocation_id)
        if invocation is None:
            return None
        invocation.finalize(
            status=status,
            error_type=error_type,
            error_message=error_message,
            finish_reason=finish_reason,
        )
        return invocation

    def get_llm_invocation(self, invocation_id: str) -> Optional["LLMInvocation"]:
        """Get a specific logical LLM invocation, including all its attempts."""
        return self.llm_invocations.get(invocation_id)

    def get_recent_llm_invocations(self, limit: int = 10, **filters: Any) -> list["LLMInvocation"]:
        """Get recent LLM invocations, optionally filtered by any attribute
        (e.g. ``workflow_id="wf-1"``, ``provider="openai"``, ``status="error"``)."""
        items = list(self.llm_invocations.values())
        for key, value in filters.items():
            if value is not None:
                items = [i for i in items if getattr(i, key, None) == value]
        items.sort(key=lambda i: i.start_time, reverse=True)
        return items[:limit]

    def get_llm_usage_summary(self, group_by: str = "provider") -> dict[str, dict[str, Any]]:
        """Aggregate tokens/cost/retries/fallbacks by an attribution
        dimension (``workflow_id``, ``workflow_step_id``, ``agent_id``,
        ``tool_name``, ``provider``, ``user_id``, ``workspace_id``,
        ``tenant_id``, ...). Answers questions like "which workflow used the
        most tokens" or "which agent generated the most retries".

        ``estimated_cost`` sums known costs and remains ``None`` when none
        are known. ``unknown_cost_invocations`` reports how many invocations
        were excluded from that sum because their cost was unknown.
        """
        summary: dict[str, dict[str, Any]] = {}
        for invocation in self.llm_invocations.values():
            if invocation.status == "in_progress":
                continue
            key = getattr(invocation, group_by, None)
            key = str(key) if key is not None else "unknown"
            bucket = summary.setdefault(
                key,
                {
                    "invocation_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": None,
                    "unknown_cost_invocations": 0,
                    "total_attempts": 0,
                    "total_retries": 0,
                    "total_fallbacks": 0,
                    "errors": 0,
                    "total_latency_ms": 0.0,
                },
            )
            bucket["invocation_count"] += 1
            bucket["input_tokens"] += invocation.input_tokens
            bucket["output_tokens"] += invocation.output_tokens
            bucket["total_tokens"] += invocation.total_tokens
            if invocation.estimated_cost is None:
                bucket["unknown_cost_invocations"] += 1
            else:
                known_cost = bucket["estimated_cost"]
                bucket["estimated_cost"] = (
                    invocation.estimated_cost
                    if known_cost is None
                    else known_cost + invocation.estimated_cost
                )
            bucket["total_attempts"] += invocation.total_attempts
            bucket["total_retries"] += invocation.total_retries
            bucket["total_fallbacks"] += invocation.total_fallbacks
            bucket["total_latency_ms"] += invocation.latency_ms or 0.0
            if invocation.status == "error":
                bucket["errors"] += 1
        for bucket in summary.values():
            bucket["avg_latency_ms"] = (
                bucket["total_latency_ms"] / bucket["invocation_count"]
                if bucket["invocation_count"]
                else 0.0
            )
        return summary

    def get_costliest_llm_attempts(self, limit: int = 10) -> list["UpstreamAttempt"]:
        """Upstream attempts sorted by estimated cost, highest first -
        surfaces which attempt caused a cost spike."""
        attempts = [a for inv in self.llm_invocations.values() for a in inv.attempts]
        attempts.sort(key=lambda a: a.estimated_cost or 0.0, reverse=True)
        return attempts[:limit]

    def get_slowest_llm_attempts(self, limit: int = 10) -> list["UpstreamAttempt"]:
        """Upstream attempts sorted by latency, highest first - surfaces
        which attempt caused a latency spike."""
        attempts = [a for inv in self.llm_invocations.values() for a in inv.attempts]
        attempts.sort(key=lambda a: a.latency_ms or 0.0, reverse=True)
        return attempts[:limit]

    def export_llm_invocations(
        self, invocation_ids: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """Export LLM invocations (with nested attempts) as JSON-safe dicts,
        suitable for shipping to an external store."""
        if invocation_ids:
            invocations = [
                self.llm_invocations[i] for i in invocation_ids if i in self.llm_invocations
            ]
        else:
            invocations = list(self.llm_invocations.values())
        return [_invocation_to_dict(inv) for inv in invocations]


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


# ======================================================================
# LLM usage tracing: logical invocations vs. upstream attempts
#
# A "logical invocation" is one model call as requested by application code
# (e.g. one client.chat(...) call). It may involve several "upstream
# attempts" against the real provider due to retries, rate limits, timeouts,
# or model/provider fallback. Both levels are recorded so that cost and
# latency of a logical call reflect every attempt, not just the last one.
# ======================================================================

_MAX_ERROR_MESSAGE_LEN = 500

_SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"sk-ant-[a-z0-9\-_]{8,}",
        r"sk-[a-z0-9]{8,}",
        r"gsk_[a-z0-9]{8,}",
        r"bearer\s+[a-z0-9._\-]{8,}",
        r"api[_-]?key[\"']?\s*[:=]\s*[\"']?[a-z0-9._\-]{8,}",
        r"authorization[\"']?\s*[:=]\s*[\"']?[a-z0-9._\-]{8,}",
    )
]


def sanitize_error_message(message: Optional[str]) -> str:
    """Redact credentials/API keys/auth headers from a provider error message.

    Applied to every error stored in telemetry so secrets in exception text
    (e.g. an SDK echoing request headers) never reach the trace store.
    """
    if not message:
        return ""
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    if len(sanitized) > _MAX_ERROR_MESSAGE_LEN:
        sanitized = sanitized[:_MAX_ERROR_MESSAGE_LEN] + "...[truncated]"
    return sanitized


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Classify a provider exception into (error_type, retry_reason).

    Uses exception class name matching instead of importing every provider
    SDK's exception hierarchy, so this works whether or not a given provider
    package is installed.
    """
    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()

    if "ratelimit" in name:
        return "rate_limit", "rate_limit"
    if "timeout" in name:
        return "timeout", "timeout"
    if "connect" in name:
        return "connection_error", "connection_error"
    if "authenticat" in name or "permissiondenied" in name:
        return "auth_error", "non_retryable"
    if "notfound" in name:
        return "not_found", "non_retryable"
    if "apistatus" in name or "httpstatus" in name:
        status = getattr(exc, "status_code", None)
        if status is None:
            response_obj = getattr(exc, "response", None)
            status = getattr(response_obj, "status_code", None)
        if status == 429:
            return "rate_limit", "rate_limit"
        if isinstance(status, int) and 500 <= status < 600:
            return "server_error", "server_error"
        return "api_error", "non_retryable"
    if "badrequest" in name or "invalidrequest" in name:
        return "invalid_request", "non_retryable"
    if isinstance(exc, (ConnectionError, OSError)):
        return "connection_error", "connection_error"
    if isinstance(exc, TimeoutError):
        return "timeout", "timeout"
    return f"{module}.{type(exc).__name__}", "unknown"


@dataclass(frozen=True)
class LLMTraceContext:
    """Immutable attribution context propagated to LLM telemetry.

    Callers attach this via the ``llm_trace_context`` context manager around
    any code that invokes a ``BaseLLM`` provider. It never needs to be passed
    as an explicit argument through intermediate function signatures.
    """

    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    workflow_step_id: Optional[str] = None
    workflow_step_name: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    feature_name: Optional[str] = None
    billing_bucket: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def merged(self, **overrides: Any) -> "LLMTraceContext":
        """Return a new context with non-None overrides applied.

        ``metadata`` is merged (union) rather than replaced, so nested
        ``llm_trace_context`` blocks accumulate metadata instead of losing
        the outer block's keys.
        """
        extra_metadata = overrides.pop("metadata", None)
        current = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "metadata"}
        for key, value in overrides.items():
            if value is not None:
                current[key] = value
        merged_metadata = {**self.metadata, **(extra_metadata or {})}
        return LLMTraceContext(metadata=merged_metadata, **current)


_llm_context_var: "contextvars.ContextVar[LLMTraceContext]" = contextvars.ContextVar(
    "hippocampai_llm_trace_context", default=LLMTraceContext()
)
_current_llm_invocation_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "hippocampai_current_llm_invocation_id", default=None
)


def get_current_llm_context() -> LLMTraceContext:
    """Return the attribution context active in the current thread/task."""
    return _llm_context_var.get()


@contextmanager
def llm_trace_context(**overrides: Any) -> Iterator[LLMTraceContext]:
    """Attach attribution (workflow/agent/tool/tenant/...) to LLM calls made
    within this block.

    Backed by a ``contextvars.ContextVar``, so it nests safely and cannot
    leak between concurrent threads or asyncio tasks.

    Example:
        with llm_trace_context(workflow_id="wf-1", agent_id="researcher"):
            with llm_trace_context(workflow_step_id="draft"):
                llm.chat(messages)  # tagged with both workflow_id and step
    """
    new_ctx = get_current_llm_context().merged(**overrides)
    token = _llm_context_var.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _llm_context_var.reset(token)


def _cap_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata whose JSON representation fits the configured budget."""
    if not metadata:
        return metadata
    try:
        from hippocampai.config import get_config

        max_bytes: int = get_config().llm_telemetry_max_metadata_bytes
    except Exception:  # noqa: BLE001 - telemetry must never break the caller
        max_bytes = 8192

    def fits(candidate: dict[str, Any]) -> bool:
        return len(json.dumps(candidate, default=str).encode("utf-8")) <= max_bytes

    try:
        encoded = json.dumps(metadata, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001
        replacements: tuple[dict[str, Any], ...] = ({"_unserializable": True}, {})
        return next(candidate for candidate in replacements if fits(candidate))
    if len(encoded) <= max_bytes:
        return metadata
    replacements = (
        {"_truncated": True, "_key_count": len(metadata)},
        {"_truncated": True},
        {},
    )
    return next(candidate for candidate in replacements if fits(candidate))


@dataclass
class UpstreamMetadata:
    """Provider-neutral optional metadata about one upstream request.

    Populated by provider adapters from raw SDK responses, or supplied
    explicitly by an OpenAI-compatible routing layer (e.g. a self-hosted
    router) without HippocampAI depending on that layer. Every field is
    optional; unknown/missing fields never raise.

    Identity fields are deliberately kept separate and are never merged
    into one another:
      - HippocampAI's own ``LLMInvocation.request_id`` (the logical
        invocation identity) is authoritative and is never overwritten by
        anything in this class.
      - ``router_request_id`` is an *external* gateway/router's own
        identifier for the logical request as it crosses that boundary
        (e.g. a client-request-ID header an OpenAI-compatible proxy
        assigns). It is provider-neutral - no specific router product is
        named or required here.
      - ``provider_request_id`` identifies exactly one upstream attempt at
        exactly one provider (e.g. OpenAI's ``response.id``). It must never
        be populated from a router/gateway-level ID, since that ID does not
        reliably identify a single attempt.
    """

    router_request_id: Optional[str] = None
    request_id: Optional[str] = None
    """Deprecated alias for ``router_request_id``, kept for backwards
    compatibility with callers written against the pre-router-identity
    contract. ``apply_metadata()`` falls back to this only when
    ``router_request_id`` is unset. Never mapped into ``provider_request_id``.
    New code should set ``router_request_id`` directly."""
    provider_request_id: Optional[str] = None
    selected_model: Optional[str] = None
    routing_reason: Optional[str] = None
    retry_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    upstream_attempt: Optional[int] = None
    upstream_latency_ms: Optional[float] = None
    upstream_token_usage: dict[str, int] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai_response(
        cls, response: Any, headers: Optional[dict[str, str]] = None
    ) -> "UpstreamMetadata":
        """Extract usage/ids from an OpenAI (or OpenAI-compatible, e.g. Groq)
        chat completion response."""
        token_usage: dict[str, int] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if prompt_tokens is not None:
                token_usage["input_tokens"] = prompt_tokens
            if completion_tokens is not None:
                token_usage["output_tokens"] = completion_tokens
            if total_tokens is not None:
                token_usage["total_tokens"] = total_tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
            if cached is not None:
                token_usage["cached_input_tokens"] = cached
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", None)
                if completion_details
                else None
            )
            if reasoning is not None:
                token_usage["reasoning_tokens"] = reasoning

        provider_metadata: dict[str, Any] = {}
        if headers:
            for key in (
                "x-request-id",
                "openai-processing-ms",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-remaining-tokens",
            ):
                if key in headers:
                    provider_metadata[key] = headers[key]

        return cls(
            provider_request_id=getattr(response, "id", None),
            selected_model=getattr(response, "model", None),
            upstream_token_usage=token_usage,
            provider_metadata=provider_metadata,
        )

    @classmethod
    def from_anthropic_response(cls, response: Any) -> "UpstreamMetadata":
        """Extract usage/ids from an Anthropic Messages API response."""
        token_usage: dict[str, int] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            cache_read = getattr(usage, "cache_read_input_tokens", None)
            if input_tokens is not None:
                token_usage["input_tokens"] = input_tokens
            if output_tokens is not None:
                token_usage["output_tokens"] = output_tokens
            if cache_read is not None:
                token_usage["cached_input_tokens"] = cache_read
        return cls(
            provider_request_id=getattr(response, "id", None),
            selected_model=getattr(response, "model", None),
            upstream_token_usage=token_usage,
        )

    @classmethod
    def from_ollama_response(cls, payload: dict[str, Any]) -> "UpstreamMetadata":
        """Extract usage from an Ollama ``/api/generate`` or ``/api/chat``
        JSON payload. Ollama is local, so there is no provider request id."""
        token_usage: dict[str, int] = {}
        if payload.get("prompt_eval_count") is not None:
            token_usage["input_tokens"] = payload["prompt_eval_count"]
        if payload.get("eval_count") is not None:
            token_usage["output_tokens"] = payload["eval_count"]
        return cls(
            selected_model=payload.get("model"),
            upstream_token_usage=token_usage,
        )


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Optional[float]:
    """Estimate cost in USD from configured per-token pricing.

    Pricing is supplied entirely through ``Config.llm_pricing`` (dollars per
    1M tokens, keyed by ``"<provider>:<model>"`` or bare ``"<model>"``).
    HippocampAI ships no built-in price catalog, so this returns ``None``
    (unknown, never a fabricated number) when no rate is configured or cost
    estimation is disabled.
    """
    try:
        from hippocampai.config import get_config

        config = get_config()
    except Exception:  # noqa: BLE001
        return None
    if not config.llm_cost_estimation_enabled:
        return None
    # Pydantic's Field() defaults are seen as Any by mypy without the pydantic
    # plugin; annotate explicitly so the arithmetic below stays float, not Any.
    pricing: dict[str, dict[str, float]] = config.llm_pricing
    rates: Optional[dict[str, float]] = pricing.get(f"{provider}:{model}") or pricing.get(model)
    if not rates:
        return None

    billable_input = max(0, input_tokens - cached_input_tokens)
    cost = billable_input / 1_000_000 * rates.get("input", 0.0)
    cost += cached_input_tokens / 1_000_000 * rates.get("cached_input", rates.get("input", 0.0))
    cost += output_tokens / 1_000_000 * rates.get("output", 0.0)
    cost += reasoning_tokens / 1_000_000 * rates.get("reasoning", rates.get("output", 0.0))
    return round(cost, 8)


@dataclass
class UpstreamAttempt:
    """One real request made to an LLM provider.

    A logical invocation may contain several of these (retries, rate-limit
    backoff, timeouts, or model/provider fallback all produce a new attempt).
    """

    attempt_number: int
    provider: str
    requested_model: str
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    selected_model: Optional[str] = None
    provider_request_id: Optional[str] = None
    router_request_id: Optional[str] = None
    """External gateway/router's own identifier for the logical request
    this attempt belongs to, if supplied via ``UpstreamMetadata``. Distinct
    from ``provider_request_id`` (this specific attempt's upstream ID) and
    from the owning ``LLMInvocation.request_id`` (HippocampAI's own
    authoritative logical invocation ID, never overwritten by this field)."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    status: str = "in_progress"  # in_progress, success, error
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    routing_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_metadata(self, meta: UpstreamMetadata) -> None:
        """Merge an ``UpstreamMetadata`` payload onto this attempt.

        Only fields actually present in ``meta`` overwrite existing values,
        so a provider that exposes partial metadata (e.g. no request id)
        never clobbers fields set elsewhere.
        """
        if meta.provider_request_id:
            self.provider_request_id = meta.provider_request_id
        router_request_id = meta.router_request_id or meta.request_id
        if router_request_id:
            self.router_request_id = router_request_id
        if meta.selected_model:
            self.selected_model = meta.selected_model
        if meta.routing_reason:
            self.routing_reason = meta.routing_reason
        if meta.fallback_reason:
            self.fallback_reason = meta.fallback_reason
        if meta.retry_reason:
            self.retry_reason = meta.retry_reason
        usage = meta.upstream_token_usage or {}
        if "input_tokens" in usage:
            self.input_tokens = usage["input_tokens"]
        if "output_tokens" in usage:
            self.output_tokens = usage["output_tokens"]
        if "cached_input_tokens" in usage:
            self.cached_input_tokens = usage["cached_input_tokens"]
        if "reasoning_tokens" in usage:
            self.reasoning_tokens = usage["reasoning_tokens"]
        if "total_tokens" in usage:
            self.total_tokens = usage["total_tokens"]
        if meta.provider_metadata:
            self.metadata.update(meta.provider_metadata)


@dataclass
class LLMInvocation:
    """One logical model call as requested by application code.

    Aggregates every ``UpstreamAttempt`` it took to complete (or fail), so
    cost/latency reflect retries and fallbacks, not just the final try.
    """

    invocation_id: str
    trace_id: str
    request_id: str
    """HippocampAI's own authoritative logical invocation identity. Never
    overwritten by any router/provider-supplied identifier - see
    ``router_request_id`` for an external gateway's own ID for this same
    logical request."""
    provider: str
    requested_model: str
    operation: str
    start_time: datetime
    parent_span_id: Optional[str] = None
    # Attribution
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    workflow_step_id: Optional[str] = None
    workflow_step_name: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    feature_name: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    billing_bucket: Optional[str] = None
    # Streaming
    streaming: bool = False
    time_to_first_token_ms: Optional[float] = None
    # Outcome (populated by finalize())
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    selected_provider: Optional[str] = None
    selected_model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: Optional[float] = None
    status: str = "in_progress"  # in_progress, success, error
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None
    total_attempts: int = 0
    total_retries: int = 0
    total_fallbacks: int = 0
    router_request_id: Optional[str] = None
    """External gateway/router's own identifier for this logical request,
    if any attempt reported one via ``UpstreamMetadata``. Distinct from
    ``request_id`` (HippocampAI's own ID, always authoritative) and from
    each attempt's own ``provider_request_id``."""
    attempts: list[UpstreamAttempt] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(
        self,
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        """Compute aggregates across all attempts. Idempotent: only the
        first call takes effect, so a manually-recorded failure inside a
        provider adapter is never silently overwritten by an automatic
        success finalize on the way back out."""
        if self.status != "in_progress":
            return

        self.end_time = datetime.now(timezone.utc)
        self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        self.error_type = error_type
        self.error_message = error_message

        self.total_attempts = len(self.attempts)
        # A fallback attempt is not also a retry: total_retries only counts
        # additional attempts along the *same* route (no fallback_reason).
        # See docs/provider_tracing.md#retry-behavior for the worked examples.
        self.total_fallbacks = sum(1 for a in self.attempts if a.fallback_reason)
        self.total_retries = max(0, self.total_attempts - 1) - self.total_fallbacks
        self.router_request_id = next(
            (a.router_request_id for a in self.attempts if a.router_request_id), None
        )
        self.input_tokens = sum(a.input_tokens or 0 for a in self.attempts)
        self.output_tokens = sum(a.output_tokens or 0 for a in self.attempts)
        self.cached_input_tokens = sum(a.cached_input_tokens or 0 for a in self.attempts)
        self.reasoning_tokens = sum(a.reasoning_tokens or 0 for a in self.attempts)
        self.total_tokens = sum(
            (attempt.input_tokens or 0) + (attempt.output_tokens or 0)
            if attempt.input_tokens is not None or attempt.output_tokens is not None
            else (attempt.total_tokens or 0)
            for attempt in self.attempts
        )

        costs = [a.estimated_cost for a in self.attempts if a.estimated_cost is not None]
        self.estimated_cost = sum(costs) if costs else None

        final_attempt = next(
            (a for a in reversed(self.attempts) if a.is_final),
            self.attempts[-1] if self.attempts else None,
        )
        if final_attempt is not None:
            self.selected_provider = final_attempt.provider
            self.selected_model = final_attempt.selected_model or final_attempt.requested_model
            self.finish_reason = finish_reason or final_attempt.finish_reason
        else:
            self.finish_reason = finish_reason


def _invocation_to_dict(invocation: LLMInvocation) -> dict[str, Any]:
    """Export an ``LLMInvocation`` (with nested attempts) as JSON-safe dict."""

    def _attempt_dict(attempt: UpstreamAttempt) -> dict[str, Any]:
        return {
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "provider": attempt.provider,
            "requested_model": attempt.requested_model,
            "selected_model": attempt.selected_model,
            "provider_request_id": attempt.provider_request_id,
            "router_request_id": attempt.router_request_id,
            "start_time": attempt.start_time.isoformat(),
            "end_time": attempt.end_time.isoformat() if attempt.end_time else None,
            "latency_ms": attempt.latency_ms,
            "input_tokens": attempt.input_tokens,
            "output_tokens": attempt.output_tokens,
            "cached_input_tokens": attempt.cached_input_tokens,
            "reasoning_tokens": attempt.reasoning_tokens,
            "total_tokens": attempt.total_tokens,
            "estimated_cost": attempt.estimated_cost,
            "status": attempt.status,
            "error_type": attempt.error_type,
            "error_message": attempt.error_message,
            "retry_reason": attempt.retry_reason,
            "fallback_reason": attempt.fallback_reason,
            "routing_reason": attempt.routing_reason,
            "finish_reason": attempt.finish_reason,
            "is_final": attempt.is_final,
            "metadata": attempt.metadata,
        }

    return {
        "invocation_id": invocation.invocation_id,
        "trace_id": invocation.trace_id,
        "request_id": invocation.request_id,
        "parent_span_id": invocation.parent_span_id,
        "provider": invocation.provider,
        "requested_model": invocation.requested_model,
        "operation": invocation.operation,
        "workflow_id": invocation.workflow_id,
        "workflow_name": invocation.workflow_name,
        "workflow_step_id": invocation.workflow_step_id,
        "workflow_step_name": invocation.workflow_step_name,
        "agent_id": invocation.agent_id,
        "agent_name": invocation.agent_name,
        "tool_name": invocation.tool_name,
        "feature_name": invocation.feature_name,
        "tenant_id": invocation.tenant_id,
        "workspace_id": invocation.workspace_id,
        "user_id": invocation.user_id,
        "session_id": invocation.session_id,
        "conversation_id": invocation.conversation_id,
        "billing_bucket": invocation.billing_bucket,
        "streaming": invocation.streaming,
        "time_to_first_token_ms": invocation.time_to_first_token_ms,
        "start_time": invocation.start_time.isoformat(),
        "end_time": invocation.end_time.isoformat() if invocation.end_time else None,
        "latency_ms": invocation.latency_ms,
        "selected_provider": invocation.selected_provider,
        "selected_model": invocation.selected_model,
        "input_tokens": invocation.input_tokens,
        "output_tokens": invocation.output_tokens,
        "cached_input_tokens": invocation.cached_input_tokens,
        "reasoning_tokens": invocation.reasoning_tokens,
        "total_tokens": invocation.total_tokens,
        "estimated_cost": invocation.estimated_cost,
        "status": invocation.status,
        "error_type": invocation.error_type,
        "error_message": invocation.error_message,
        "finish_reason": invocation.finish_reason,
        "total_attempts": invocation.total_attempts,
        "total_retries": invocation.total_retries,
        "total_fallbacks": invocation.total_fallbacks,
        "router_request_id": invocation.router_request_id,
        "attempts": [_attempt_dict(a) for a in invocation.attempts],
        "metadata": invocation.metadata,
    }


@contextmanager
def llm_invocation(
    provider: str,
    requested_model: str,
    operation: str,
    streaming: bool = False,
    **explicit_context: Any,
) -> Iterator[str]:
    """Start one logical LLM invocation. Use at the public entrypoint of a
    provider adapter (``chat``/``generate``), outside of any retry decorator.

    Nested ``llm_attempt`` calls made while this block is active are
    automatically attached to this invocation via a context variable.
    Telemetry bookkeeping failures are logged and swallowed - they never
    break the underlying LLM call.
    """
    telemetry = get_telemetry()
    try:
        invocation_id = telemetry.start_llm_invocation(
            provider=provider,
            requested_model=requested_model,
            operation=operation,
            streaming=streaming,
            **explicit_context,
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to start LLM invocation telemetry", exc_info=True)
        invocation_id = str(uuid4())

    token = _current_llm_invocation_id.set(invocation_id)
    try:
        yield invocation_id
    except Exception as exc:
        error_type, _ = classify_llm_error(exc)
        try:
            telemetry.end_llm_invocation(
                invocation_id,
                status="error",
                error_type=error_type,
                error_message=sanitize_error_message(str(exc)),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to finalize LLM invocation telemetry", exc_info=True)
        raise
    else:
        try:
            telemetry.end_llm_invocation(invocation_id, status="success")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to finalize LLM invocation telemetry", exc_info=True)
    finally:
        _current_llm_invocation_id.reset(token)


@contextmanager
def llm_attempt(
    provider: str,
    requested_model: str,
    selected_model: Optional[str] = None,
    retry_reason: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    routing_reason: Optional[str] = None,
) -> Iterator[UpstreamAttempt]:
    """Record one real upstream request. Use inside the retried call of a
    provider adapter, i.e. the function tenacity re-invokes on every retry.

    Attaches itself to the invocation started by the enclosing
    ``llm_invocation`` block (found via a context variable), if any. Yields
    the ``UpstreamAttempt`` so the caller can fill in usage/ids via
    ``attempt.apply_metadata(...)`` before returning.

    ``retry_reason`` semantics: it belongs to the attempt that was started
    *because* a previous attempt failed, not to the failing attempt itself
    (a failed attempt's ``error_type`` already says why it failed; its
    ``retry_reason`` stays ``None``). If the previous attempt in this
    invocation errored and this attempt isn't an explicit fallback, its
    ``retry_reason`` is inherited from that previous attempt's
    ``error_type`` unless the caller passes one explicitly.
    """
    telemetry = get_telemetry()
    invocation_id = _current_llm_invocation_id.get()
    invocation = (
        telemetry.llm_invocations.get(invocation_id)
        if (invocation_id and telemetry.enabled)
        else None
    )
    previous_attempt = invocation.attempts[-1] if invocation and invocation.attempts else None
    if retry_reason is None and fallback_reason is None and previous_attempt is not None:
        if previous_attempt.status == "error":
            retry_reason = previous_attempt.error_type
    attempt_number = (len(invocation.attempts) + 1) if invocation is not None else 1
    attempt = UpstreamAttempt(
        attempt_number=attempt_number,
        provider=provider,
        requested_model=requested_model,
        selected_model=selected_model,
        retry_reason=retry_reason,
        fallback_reason=fallback_reason,
        routing_reason=routing_reason,
    )
    start = time.monotonic()
    try:
        yield attempt
        attempt.status = "success"
        attempt.is_final = True
    except Exception as exc:
        error_type, _ = classify_llm_error(exc)
        attempt.status = "error"
        attempt.error_type = error_type
        attempt.error_message = sanitize_error_message(str(exc))
        raise
    finally:
        attempt.end_time = datetime.now(timezone.utc)
        attempt.latency_ms = (time.monotonic() - start) * 1000
        if attempt.total_tokens is None and (
            attempt.input_tokens is not None or attempt.output_tokens is not None
        ):
            attempt.total_tokens = (attempt.input_tokens or 0) + (attempt.output_tokens or 0)
        attempt.estimated_cost = estimate_cost(
            provider=attempt.provider,
            model=attempt.selected_model or attempt.requested_model,
            input_tokens=attempt.input_tokens or 0,
            output_tokens=attempt.output_tokens or 0,
            cached_input_tokens=attempt.cached_input_tokens or 0,
            reasoning_tokens=attempt.reasoning_tokens or 0,
        )
        if invocation is not None:
            invocation.attempts.append(attempt)
