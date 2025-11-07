"""Tests for metrics and tracing."""

import asyncio
import time

import pytest

from hippocampai.monitoring.metrics import (
    MetricsCollector,
    OperationType,
    configure_metrics,
    get_metrics_collector,
    record_memory_operation,
    record_memory_store_stats,
    record_quality_metrics,
    record_search_metrics,
)


@pytest.fixture
def collector():
    """Create fresh metrics collector."""
    return MetricsCollector(enable_tracing=True)


class TestBasicMetrics:
    """Test basic metric recording."""

    def test_counter_metric(self, collector):
        """Test counter metric recording."""
        collector.increment_counter("test.counter", value=1.0)
        collector.increment_counter("test.counter", value=2.0)

        assert collector.counters["test.counter"] == 3.0

    def test_gauge_metric(self, collector):
        """Test gauge metric recording."""
        collector.set_gauge("test.gauge", 42.0)
        collector.set_gauge("test.gauge", 100.0)

        assert collector.gauges["test.gauge"] == 100.0  # Latest value

    def test_histogram_metric(self, collector):
        """Test histogram metric recording."""
        for value in [10.0, 20.0, 30.0, 40.0, 50.0]:
            collector.record_histogram("test.histogram", value)

        metrics = collector.metrics["test.histogram"]
        assert len(metrics) == 5

        stats = collector.get_histogram_stats("test.histogram")
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0

    def test_metric_with_tags(self, collector):
        """Test metrics with tags."""
        collector.increment_counter(
            "test.tagged",
            value=1.0,
            tags={"user": "user1", "operation": "create"},
        )

        metrics = collector.metrics["test.tagged"]
        assert len(metrics) == 1
        assert metrics[0].tags["user"] == "user1"


class TestTracing:
    """Test distributed tracing."""

    def test_trace_operation(self, collector):
        """Test basic operation tracing."""
        with collector.trace_operation(OperationType.CREATE) as trace:
            time.sleep(0.01)  # Simulate work

        assert trace is not None
        assert trace.success is True
        assert trace.duration_ms > 0
        assert len(collector.traces) == 1

    def test_trace_with_error(self, collector):
        """Test tracing with error handling."""
        with pytest.raises(ValueError):
            with collector.trace_operation(OperationType.CREATE) as trace:
                raise ValueError("Test error")

        assert len(collector.traces) == 1
        trace = collector.traces[0]
        assert trace.success is False
        assert trace.error == "Test error"

    def test_trace_with_metadata(self, collector):
        """Test tracing with metadata."""
        with collector.trace_operation(
            OperationType.SEARCH, metadata={"user_id": "user1", "query": "test"}
        ) as trace:
            pass

        assert trace.metadata["user_id"] == "user1"
        assert trace.metadata["query"] == "test"

    def test_span_within_trace(self, collector):
        """Test creating spans within a trace."""
        with collector.trace_operation(OperationType.CREATE) as trace:
            with collector.span(trace, "embed_text"):
                time.sleep(0.005)

            with collector.span(trace, "store_vector"):
                time.sleep(0.005)

        assert len(trace.spans) == 2
        assert trace.spans[0].name == "embed_text"
        assert trace.spans[1].name == "store_vector"
        assert all(span.duration_ms > 0 for span in trace.spans)

    def test_span_with_tags(self, collector):
        """Test spans with tags."""
        with collector.trace_operation(OperationType.CREATE) as trace:
            with collector.span(trace, "process", tags={"step": 1, "batch_size": 10}):
                pass

        span = trace.spans[0]
        assert span.tags["step"] == 1
        assert span.tags["batch_size"] == 10

    @pytest.mark.asyncio
    async def test_async_tracing(self, collector):
        """Test tracing with async operations."""

        @collector.time_function(OperationType.CREATE)
        async def async_operation():
            await asyncio.sleep(0.01)
            return "result"

        result = await async_operation()

        assert result == "result"
        assert len(collector.traces) == 1
        assert collector.traces[0].success is True

    def test_sync_tracing(self, collector):
        """Test tracing with sync operations."""

        @collector.time_function(OperationType.UPDATE)
        def sync_operation():
            time.sleep(0.01)
            return "result"

        result = sync_operation()

        assert result == "result"
        assert len(collector.traces) == 1
        assert collector.traces[0].success is True


class TestMetricsSummary:
    """Test metrics summary and aggregation."""

    def test_empty_summary(self, collector):
        """Test summary with no metrics."""
        summary = collector.get_metrics_summary()

        assert summary["total_traces"] == 0
        assert summary["successful_operations"] == 0
        assert summary["failed_operations"] == 0

    def test_summary_with_traces(self, collector):
        """Test summary with multiple traces."""
        # Successful operations
        for _ in range(5):
            with collector.trace_operation(OperationType.CREATE):
                pass

        # Failed operations
        for _ in range(2):
            try:
                with collector.trace_operation(OperationType.DELETE):
                    raise ValueError("Error")
            except ValueError:
                pass

        summary = collector.get_metrics_summary()

        assert summary["total_traces"] == 7
        assert summary["successful_operations"] == 5
        assert summary["failed_operations"] == 2

    def test_operation_stats(self, collector):
        """Test per-operation statistics."""
        with collector.trace_operation(OperationType.CREATE):
            time.sleep(0.01)

        with collector.trace_operation(OperationType.CREATE):
            time.sleep(0.02)

        summary = collector.get_metrics_summary()
        create_stats = summary["operation_stats"]["create"]

        assert create_stats["count"] == 2
        assert create_stats["avg_duration_ms"] > 0
        assert create_stats["error_rate"] == 0.0

    def test_histogram_stats(self, collector):
        """Test histogram statistics."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

        for value in values:
            collector.record_histogram("latency", value)

        stats = collector.get_histogram_stats("latency")

        assert stats["min"] == 10.0
        assert stats["max"] == 100.0
        assert stats["mean"] == 55.0
        assert stats["median"] == 55.0
        assert stats["p95"] > 90.0
        assert stats["p99"] > 95.0


class TestRecentTraces:
    """Test querying recent traces."""

    def test_get_recent_traces(self, collector):
        """Test getting recent traces."""
        for i in range(15):
            with collector.trace_operation(OperationType.CREATE):
                time.sleep(0.001)

        recent = collector.get_recent_traces(limit=10)

        assert len(recent) == 10
        # Should be sorted by start time descending
        assert all(recent[i].start_time >= recent[i + 1].start_time for i in range(len(recent) - 1))

    def test_filter_by_operation(self, collector):
        """Test filtering traces by operation type."""
        with collector.trace_operation(OperationType.CREATE):
            pass

        with collector.trace_operation(OperationType.UPDATE):
            pass

        with collector.trace_operation(OperationType.DELETE):
            pass

        create_traces = collector.get_recent_traces(operation=OperationType.CREATE)
        assert len(create_traces) == 1
        assert create_traces[0].operation == OperationType.CREATE


class TestMetricsExport:
    """Test metrics export functionality."""

    def test_prometheus_export(self, collector):
        """Test Prometheus format export."""
        collector.increment_counter("requests_total", 100.0)
        collector.set_gauge("active_users", 42.0)

        for value in [10.0, 20.0, 30.0]:
            collector.record_histogram("response_time", value)

        output = collector.export_metrics(format="prometheus")

        assert "requests_total 100" in output
        assert "active_users 42" in output
        assert "response_time_count" in output
        assert "response_time_sum" in output

    def test_json_export(self, collector):
        """Test JSON format export."""
        collector.increment_counter("test.counter", 10.0)

        output = collector.export_metrics(format="json")

        assert "counters" in output
        assert "test.counter" in output

    def test_invalid_format(self, collector):
        """Test export with invalid format."""
        with pytest.raises(ValueError):
            collector.export_metrics(format="invalid")


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_reset_all_metrics(self, collector):
        """Test resetting all metrics."""
        collector.increment_counter("test.counter", 10.0)
        collector.set_gauge("test.gauge", 42.0)

        with collector.trace_operation(OperationType.CREATE):
            pass

        assert len(collector.metrics) > 0
        assert len(collector.traces) > 0

        collector.reset_metrics()

        assert len(collector.metrics) == 0
        assert len(collector.traces) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0


class TestGlobalCollector:
    """Test global metrics collector."""

    def test_get_global_collector(self):
        """Test getting global collector instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2  # Should be same instance

    def test_configure_metrics(self):
        """Test configuring global collector."""
        configure_metrics(enable_tracing=False)

        collector = get_metrics_collector()
        assert collector.enable_tracing is False

        # Reset for other tests
        configure_metrics(enable_tracing=True)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_record_memory_operation(self):
        """Test recording memory operation."""
        configure_metrics(enable_tracing=True)
        collector = get_metrics_collector()
        collector.reset_metrics()

        record_memory_operation(OperationType.CREATE, success=True, duration_ms=25.5)

        assert collector.counters["memory.create.total"] >= 1.0

    def test_record_memory_operation_error(self):
        """Test recording failed operation."""
        configure_metrics(enable_tracing=True)
        collector = get_metrics_collector()
        collector.reset_metrics()

        record_memory_operation(OperationType.DELETE, success=False)

        assert collector.counters["memory.delete.errors"] >= 1.0

    def test_record_memory_store_stats(self):
        """Test recording store statistics."""
        configure_metrics(enable_tracing=True)
        collector = get_metrics_collector()
        collector.reset_metrics()

        record_memory_store_stats(
            total_memories=100,
            healthy_memories=85,
            stale_memories=15,
            duplicate_clusters=3,
            health_score=82.5,
        )

        assert collector.gauges["memory_store.total_memories"] == 100
        assert collector.gauges["memory_store.health_score"] == 82.5

    def test_record_search_metrics(self):
        """Test recording search metrics."""
        configure_metrics(enable_tracing=True)
        collector = get_metrics_collector()
        collector.reset_metrics()

        record_search_metrics(query_time_ms=15.5, results_count=10, cache_hit=True)

        assert collector.gauges["search.results_count"] == 10
        assert collector.counters["search.cache_hits"] >= 1.0

    def test_record_quality_metrics(self):
        """Test recording quality metrics."""
        configure_metrics(enable_tracing=True)
        collector = get_metrics_collector()
        collector.reset_metrics()

        record_quality_metrics(
            freshness_score=85.0,
            diversity_score=75.0,
            consistency_score=90.0,
            coverage_score=80.0,
        )

        assert collector.gauges["quality.freshness_score"] == 85.0
        assert collector.gauges["quality.diversity_score"] == 75.0


class TestTracingDisabled:
    """Test behavior when tracing is disabled."""

    def test_trace_with_tracing_disabled(self):
        """Test that tracing can be disabled."""
        collector = MetricsCollector(enable_tracing=False)

        with collector.trace_operation(OperationType.CREATE) as trace:
            pass

        assert trace is None
        assert len(collector.traces) == 0

    def test_span_with_tracing_disabled(self):
        """Test that spans are skipped when tracing disabled."""
        collector = MetricsCollector(enable_tracing=False)

        with collector.trace_operation(OperationType.CREATE) as trace:
            with collector.span(trace, "test_span") as span:
                pass

        assert span is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
