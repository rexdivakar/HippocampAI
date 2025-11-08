"""Prometheus metrics integration for HippocampAI."""

import time
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ==============================================================================
# Application Metrics
# ==============================================================================

# API Request Metrics
http_requests_total = Counter(
    "hippocampai_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "hippocampai_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

http_requests_in_progress = Gauge(
    "hippocampai_http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
)

# ==============================================================================
# Memory Operation Metrics
# ==============================================================================

# Memory CRUD Operations
memory_operations_total = Counter(
    "hippocampai_memory_operations_total",
    "Total memory operations",
    ["operation", "status"],
)

memory_operation_duration_seconds = Histogram(
    "hippocampai_memory_operation_duration_seconds",
    "Memory operation duration in seconds",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Memory Store Metrics
total_memories = Gauge(
    "hippocampai_total_memories",
    "Total number of memories in the system",
    ["user_id", "memory_type"],
)

memory_size_bytes = Histogram(
    "hippocampai_memory_size_bytes",
    "Memory text size in bytes",
    ["memory_type"],
    buckets=(100, 500, 1000, 2000, 5000, 10000, 50000, 100000),
)

memories_created_total = Counter(
    "hippocampai_memories_created_total",
    "Total memories created",
    ["memory_type"],
)

memories_deleted_total = Counter(
    "hippocampai_memories_deleted_total",
    "Total memories deleted",
    ["memory_type"],
)

# ==============================================================================
# Search & Retrieval Metrics
# ==============================================================================

search_requests_total = Counter(
    "hippocampai_search_requests_total",
    "Total search requests",
    ["search_type", "status"],
)

search_duration_seconds = Histogram(
    "hippocampai_search_duration_seconds",
    "Search operation duration in seconds",
    ["search_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

search_results_count = Histogram(
    "hippocampai_search_results_count",
    "Number of results returned from search",
    buckets=(0, 1, 5, 10, 20, 50, 100, 500),
)

# Cache Metrics
cache_hits_total = Counter(
    "hippocampai_cache_hits_total",
    "Total cache hits",
    ["cache_type"],
)

cache_misses_total = Counter(
    "hippocampai_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)

cache_hit_rate = Gauge(
    "hippocampai_cache_hit_rate",
    "Cache hit rate (0-1)",
    ["cache_type"],
)

# ==============================================================================
# Advanced Feature Metrics
# ==============================================================================

# Conflict Resolution
conflicts_detected_total = Counter(
    "hippocampai_conflicts_detected_total",
    "Total memory conflicts detected",
    ["conflict_type"],
)

conflicts_resolved_total = Counter(
    "hippocampai_conflicts_resolved_total",
    "Total conflicts resolved",
    ["resolution_strategy"],
)

# Memory Health
memory_health_score = Gauge(
    "hippocampai_memory_health_score",
    "Memory health score (0-100)",
    ["user_id"],
)

stale_memories_count = Gauge(
    "hippocampai_stale_memories_count",
    "Number of stale memories detected",
    ["user_id"],
)

duplicate_memories_count = Gauge(
    "hippocampai_duplicate_memories_count",
    "Number of duplicate memories detected",
    ["user_id"],
)

# Temporal Features
memory_freshness_score = Gauge(
    "hippocampai_memory_freshness_score",
    "Memory freshness score (0-1)",
    ["memory_id"],
)

memory_age_days = Histogram(
    "hippocampai_memory_age_days",
    "Memory age in days",
    buckets=(1, 7, 14, 30, 60, 90, 180, 365),
)

# ==============================================================================
# Background Task Metrics
# ==============================================================================

background_tasks_total = Counter(
    "hippocampai_background_tasks_total",
    "Total background tasks executed",
    ["task_name", "status"],
)

background_task_duration_seconds = Histogram(
    "hippocampai_background_task_duration_seconds",
    "Background task duration in seconds",
    ["task_name"],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600),
)

deduplication_runs_total = Counter(
    "hippocampai_deduplication_runs_total",
    "Total deduplication runs",
    ["status"],
)

consolidation_runs_total = Counter(
    "hippocampai_consolidation_runs_total",
    "Total consolidation runs",
    ["status"],
)

# ==============================================================================
# Vector Database Metrics
# ==============================================================================

qdrant_operations_total = Counter(
    "hippocampai_qdrant_operations_total",
    "Total Qdrant operations",
    ["operation", "collection", "status"],
)

qdrant_operation_duration_seconds = Histogram(
    "hippocampai_qdrant_operation_duration_seconds",
    "Qdrant operation duration in seconds",
    ["operation", "collection"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

vector_search_latency_seconds = Histogram(
    "hippocampai_vector_search_latency_seconds",
    "Vector search latency in seconds",
    ["collection"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ==============================================================================
# System Metrics
# ==============================================================================

# Application Info
app_info = Info("hippocampai_app", "HippocampAI application information")
app_info.info({
    "version": "0.2.5",
    "environment": "production",
})

# Active connections
active_connections = Gauge(
    "hippocampai_active_connections",
    "Number of active connections",
)

# Error Metrics
errors_total = Counter(
    "hippocampai_errors_total",
    "Total errors encountered",
    ["error_type", "operation"],
)

# ==============================================================================
# Middleware
# ==============================================================================


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        # Time the request
        start_time = time.time()

        try:
            response = await call_next(request)
            status = response.status_code

            # Record metrics
            duration = time.time() - start_time
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status,
            ).inc()
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            return response

        except Exception as e:
            # Record error
            errors_total.labels(
                error_type=type(e).__name__,
                operation="http_request",
            ).inc()
            raise

        finally:
            # Always decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# ==============================================================================
# Helper Functions
# ==============================================================================


def record_memory_operation(
    operation: str,
    duration: float,
    status: str = "success",
    memory_type: str = "unknown",
):
    """Record a memory operation metric."""
    memory_operations_total.labels(
        operation=operation,
        status=status,
    ).inc()
    memory_operation_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def record_search_operation(
    search_type: str,
    duration: float,
    result_count: int,
    status: str = "success",
):
    """Record a search operation metric."""
    search_requests_total.labels(
        search_type=search_type,
        status=status,
    ).inc()
    search_duration_seconds.labels(
        search_type=search_type,
    ).observe(duration)
    search_results_count.observe(result_count)


def record_cache_access(cache_type: str, hit: bool):
    """Record a cache access metric."""
    if hit:
        cache_hits_total.labels(cache_type=cache_type).inc()
    else:
        cache_misses_total.labels(cache_type=cache_type).inc()


def update_memory_health_metrics(user_id: str, health_data: dict):
    """Update memory health metrics."""
    memory_health_score.labels(user_id=user_id).set(
        health_data.get("overall_score", 0)
    )
    stale_memories_count.labels(user_id=user_id).set(
        len(health_data.get("stale_memories", []))
    )
    duplicate_memories_count.labels(user_id=user_id).set(
        len(health_data.get("duplicates", []))
    )


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    return generate_latest()
