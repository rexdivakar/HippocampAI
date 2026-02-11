"""Benchmark runner for HippocampAI.

Runs benchmarks and collects metrics for:
- Ingestion throughput
- Retrieval latency
- Reranking overhead
- Context assembly performance
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from bench.data_generator import (
    generate_memories,
    generate_queries,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    operations: int
    total_time_ms: float
    ops_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    errors: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "operations": self.operations,
            "total_time_ms": round(self.total_time_ms, 2),
            "ops_per_second": round(self.ops_per_second, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2),
            "latency_min_ms": round(self.latency_min_ms, 2),
            "latency_max_ms": round(self.latency_max_ms, 2),
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Generate markdown summary."""
        lines = [
            f"# Benchmark Results: {self.name}",
            "",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## System Info",
            "",
        ]

        for key, value in self.system_info.items():
            lines.append(f"- **{key}:** {value}")

        lines.extend(
            [
                "",
                "## Results",
                "",
                "| Benchmark | Ops | Ops/sec | P50 (ms) | P95 (ms) | P99 (ms) | Errors |",
                "|-----------|-----|---------|----------|----------|----------|--------|",
            ]
        )

        for r in self.results:
            lines.append(
                f"| {r.name} | {r.operations} | {r.ops_per_second:.1f} | "
                f"{r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | "
                f"{r.latency_p99_ms:.2f} | {r.errors} |"
            )

        lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for r in self.results:
            lines.extend(
                [
                    f"### {r.name}",
                    "",
                    f"- **Total Operations:** {r.operations}",
                    f"- **Total Time:** {r.total_time_ms:.2f} ms",
                    f"- **Throughput:** {r.ops_per_second:.2f} ops/sec",
                    f"- **Latency (min/p50/p95/p99/max):** "
                    f"{r.latency_min_ms:.2f} / {r.latency_p50_ms:.2f} / "
                    f"{r.latency_p95_ms:.2f} / {r.latency_p99_ms:.2f} / "
                    f"{r.latency_max_ms:.2f} ms",
                    f"- **Errors:** {r.errors}",
                    "",
                ]
            )

        return "\n".join(lines)


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_benchmark(
    name: str,
    operation: Callable[[], Any],
    iterations: int,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run a benchmark.

    Args:
        name: Benchmark name
        operation: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult
    """
    logger.info(f"Running benchmark: {name} ({iterations} iterations)")

    # Warmup
    for _ in range(warmup):
        try:
            operation()
        except Exception:
            pass

    # Actual benchmark
    latencies: list[float] = []
    errors = 0

    start_total = time.perf_counter()

    for i in range(iterations):
        start = time.perf_counter()
        try:
            operation()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)
        except Exception as e:
            errors += 1
            logger.warning(f"Benchmark error: {e}")

        if (i + 1) % 100 == 0:
            logger.debug(f"  Progress: {i + 1}/{iterations}")

    total_time = (time.perf_counter() - start_total) * 1000  # ms

    if not latencies:
        latencies = [0.0]

    successful = max(0, iterations - errors)
    duration_sec = total_time / 1000 if total_time > 0 else 0
    ops_sec = (successful / duration_sec) if duration_sec > 0 and successful > 0 else 0.0

    return BenchmarkResult(
        name=name,
        operations=iterations,
        total_time_ms=total_time,
        ops_per_second=ops_sec,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        errors=errors,
    )


class HippocampBenchmarks:
    """Benchmark suite for HippocampAI."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize benchmarks.

        Args:
            api_url: HippocampAI API URL
            api_key: API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
        self._client: Any = None
        self._test_user_id = f"bench_user_{int(time.time())}"

    @property
    def client(self) -> Any:
        """Get HippocampAI client."""
        if self._client is None:
            try:
                from hippocampai import HippocampAI
            except Exception as e:
                raise RuntimeError(f"Failed to import HippocampAI client: {e}")
            try:
                self._client = HippocampAI(
                    api_key=self.api_key,
                    base_url=self.api_url,
                    timeout=30,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize HippocampAI client (url={self.api_url}): {e}")
        return self._client

    def run_all(
        self,
        ingestion_count: int = 100,
        retrieval_count: int = 50,
        context_count: int = 20,
    ) -> BenchmarkSuite:
        """Run all benchmarks.

        Args:
            ingestion_count: Number of memories to ingest
            retrieval_count: Number of retrieval queries
            context_count: Number of context assembly calls

        Returns:
            BenchmarkSuite with all results
        """
        import platform

        suite = BenchmarkSuite(
            name="HippocampAI Benchmarks",
            timestamp=datetime.utcnow().isoformat(),
            system_info={
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "api_url": self.api_url,
            },
        )

        # Run benchmarks
        try:
            suite.results.append(self.benchmark_ingestion(ingestion_count))
        except Exception as e:
            logger.error(f"Ingestion benchmark failed: {e}")

        try:
            suite.results.append(self.benchmark_retrieval(retrieval_count))
        except Exception as e:
            logger.error(f"Retrieval benchmark failed: {e}")

        try:
            suite.results.append(self.benchmark_context_assembly(context_count))
        except Exception as e:
            logger.error(f"Context assembly benchmark failed: {e}")

        return suite

    def benchmark_ingestion(self, count: int = 100) -> BenchmarkResult:
        """Benchmark memory ingestion throughput.

        Args:
            count: Number of memories to ingest

        Returns:
            BenchmarkResult
        """
        memories = list(generate_memories(count, user_id=self._test_user_id))
        memory_iter = iter(memories)

        def ingest_one() -> None:
            memory = next(memory_iter, None)
            if memory is None:
                return
            for attempt in range(3):
                try:
                    self.client.add(
                        user_id=memory.user_id,
                        content=memory.content,
                        memory_type=memory.memory_type,
                        importance=memory.importance,
                        timeout=15,
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise

        # Reset iterator for actual benchmark
        memory_iter = iter(memories)

        return run_benchmark(
            name="Ingestion",
            operation=ingest_one,
            iterations=count,
            warmup=0,  # No warmup for ingestion
        )

    def benchmark_retrieval(self, count: int = 50) -> BenchmarkResult:
        """Benchmark retrieval latency.

        Args:
            count: Number of queries to run

        Returns:
            BenchmarkResult
        """
        queries = generate_queries(count + 10)  # Extra for warmup
        query_iter = iter(queries)

        def retrieve_one() -> None:
            query = next(query_iter, None)
            if query is None:
                return
            self.client.search(
                user_id=self._test_user_id,
                query=query,
                top_k=10,
                timeout=15,
            )

        return run_benchmark(
            name="Retrieval",
            operation=retrieve_one,
            iterations=count,
            warmup=5,
        )

    def benchmark_context_assembly(self, count: int = 20) -> BenchmarkResult:
        """Benchmark context assembly performance.

        Args:
            count: Number of context assembly calls

        Returns:
            BenchmarkResult
        """
        queries = generate_queries(count + 5)
        query_iter = iter(queries)

        def assemble_one() -> None:
            query = next(query_iter, None)
            if query is None:
                return
            from hippocampai.context.models import ContextConstraints

            constraints = ContextConstraints(token_budget=4000)
            self.client.assemble_context(
                user_id=self._test_user_id,
                query=query,
                constraints=constraints,
                timeout=30,
            )

        return run_benchmark(
            name="Context Assembly",
            operation=assemble_one,
            iterations=count,
            warmup=2,
        )

    def cleanup(self) -> None:
        """Clean up benchmark data.

        Note: This will only run if the client exposes a safe delete API.
        In environments without delete support, this is a no-op and users
        should manually purge test data for the bench user.
        """
        logger.info(f"Cleaning up benchmark user: {self._test_user_id}")
        try:
            if hasattr(self.client, "delete_user_data"):
                self.client.delete_user_data(user_id=self._test_user_id)
        except Exception as e:
            logger.warning(f"Cleanup skipped/failed: {e}")


def save_results(
    suite: BenchmarkSuite,
    output_dir: str = "bench/results",
) -> tuple[str, str]:
    """Save benchmark results.

    Args:
        suite: Benchmark suite to save
        output_dir: Output directory

    Returns:
        Tuple of (json_path, markdown_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_path / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)

    # Save Markdown
    md_path = output_path / f"benchmark_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(suite.to_markdown())

    logger.info(f"Results saved to {json_path} and {md_path}")

    return str(json_path), str(md_path)
