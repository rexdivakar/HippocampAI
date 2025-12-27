"""Tests for benchmark suite."""


from bench.data_generator import (
    SyntheticMemory,
    generate_bitemporal_facts,
    generate_fact_content,
    generate_memories,
    generate_memory,
    generate_queries,
)
from bench.runner import (
    BenchmarkResult,
    BenchmarkSuite,
    percentile,
    run_benchmark,
)


class TestDataGenerator:
    """Tests for synthetic data generator."""

    def test_generate_memory(self) -> None:
        """Test generating a single memory."""
        memory = generate_memory("user_1")

        assert isinstance(memory, SyntheticMemory)
        assert memory.user_id == "user_1"
        assert memory.content
        assert memory.memory_type in ["fact", "preference", "event"]
        assert 1 <= memory.importance <= 10
        assert memory.created_at is not None

    def test_generate_memory_with_type(self) -> None:
        """Test generating memory with specific type."""
        memory = generate_memory("user_1", memory_type="preference")
        assert memory.memory_type == "preference"

    def test_generate_memories(self) -> None:
        """Test generating multiple memories."""
        memories = list(generate_memories(10, user_id="user_1"))

        assert len(memories) == 10
        assert all(m.user_id == "user_1" for m in memories)

    def test_generate_memories_multiple_users(self) -> None:
        """Test generating memories for multiple users."""
        memories = list(generate_memories(10, num_users=3))

        assert len(memories) == 10
        user_ids = set(m.user_id for m in memories)
        assert len(user_ids) == 3

    def test_generate_queries(self) -> None:
        """Test generating search queries."""
        queries = generate_queries(5)

        assert len(queries) == 5
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 0 for q in queries)

    def test_generate_bitemporal_facts(self) -> None:
        """Test generating bi-temporal facts."""
        facts = generate_bitemporal_facts(5, "user_1")

        assert len(facts) == 5
        for fact in facts:
            assert fact["user_id"] == "user_1"
            assert fact["subject"]
            assert fact["predicate"]
            assert fact["object_value"]
            assert fact["valid_from"] is not None

    def test_generate_fact_content(self) -> None:
        """Test fact content generation."""
        content = generate_fact_content()
        assert isinstance(content, str)
        assert len(content) > 10


class TestBenchmarkRunner:
    """Tests for benchmark runner."""

    def test_percentile(self) -> None:
        """Test percentile calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert percentile(data, 50) == 5.5
        assert percentile(data, 0) == 1.0
        assert percentile(data, 100) == 10.0

    def test_percentile_empty(self) -> None:
        """Test percentile with empty data."""
        assert percentile([], 50) == 0.0

    def test_run_benchmark(self) -> None:
        """Test running a simple benchmark."""
        counter = [0]

        def simple_op() -> None:
            counter[0] += 1

        result = run_benchmark(
            name="Test",
            operation=simple_op,
            iterations=10,
            warmup=2,
        )

        assert result.name == "Test"
        assert result.operations == 10
        assert result.errors == 0
        assert result.ops_per_second > 0
        assert counter[0] == 12  # 10 iterations + 2 warmup

    def test_run_benchmark_with_errors(self) -> None:
        """Test benchmark with failing operations."""
        call_count = [0]

        def failing_op() -> None:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("Test error")

        result = run_benchmark(
            name="Failing",
            operation=failing_op,
            iterations=10,
            warmup=0,
        )

        assert result.errors == 5  # Half should fail


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            name="Test",
            operations=100,
            total_time_ms=1000.0,
            ops_per_second=100.0,
            latency_p50_ms=8.0,
            latency_p95_ms=15.0,
            latency_p99_ms=20.0,
            latency_min_ms=5.0,
            latency_max_ms=25.0,
            errors=2,
        )

        d = result.to_dict()

        assert d["name"] == "Test"
        assert d["operations"] == 100
        assert d["ops_per_second"] == 100.0
        assert d["errors"] == 2


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_to_dict(self) -> None:
        """Test converting suite to dictionary."""
        suite = BenchmarkSuite(
            name="Test Suite",
            timestamp="2024-01-01T00:00:00",
            system_info={"python": "3.12"},
        )
        suite.results.append(
            BenchmarkResult(
                name="Test",
                operations=10,
                total_time_ms=100.0,
                ops_per_second=100.0,
                latency_p50_ms=8.0,
                latency_p95_ms=15.0,
                latency_p99_ms=20.0,
                latency_min_ms=5.0,
                latency_max_ms=25.0,
            )
        )

        d = suite.to_dict()

        assert d["name"] == "Test Suite"
        assert len(d["results"]) == 1

    def test_to_markdown(self) -> None:
        """Test generating markdown summary."""
        suite = BenchmarkSuite(
            name="Test Suite",
            timestamp="2024-01-01T00:00:00",
            system_info={"python": "3.12"},
        )
        suite.results.append(
            BenchmarkResult(
                name="Test",
                operations=10,
                total_time_ms=100.0,
                ops_per_second=100.0,
                latency_p50_ms=8.0,
                latency_p95_ms=15.0,
                latency_p99_ms=20.0,
                latency_min_ms=5.0,
                latency_max_ms=25.0,
            )
        )

        md = suite.to_markdown()

        assert "# Benchmark Results" in md
        assert "Test Suite" in md
        assert "| Test |" in md
