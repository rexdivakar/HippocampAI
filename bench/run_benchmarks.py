#!/usr/bin/env python
"""Run HippocampAI benchmarks.

Usage:
    python -m bench.run_benchmarks [options]

Options:
    --api-url URL       HippocampAI API URL (default: http://localhost:8000)
    --api-key KEY       API key for authentication
    --ingestion N       Number of memories to ingest (default: 100)
    --retrieval N       Number of retrieval queries (default: 50)
    --context N         Number of context assembly calls (default: 20)
    --output DIR        Output directory (default: bench/results)
    --quick             Run quick benchmark (reduced iterations)
    --verbose           Enable verbose logging
"""

import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.runner import HippocampBenchmarks, save_results


def main() -> int:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run HippocampAI benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-url",
        default=os.getenv("HIPPOCAMPAI_API_URL", "http://localhost:8000"),
        help="HippocampAI API URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("HIPPOCAMPAI_API_KEY"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--ingestion",
        type=int,
        default=100,
        help="Number of memories to ingest",
    )
    parser.add_argument(
        "--retrieval",
        type=int,
        default=50,
        help="Number of retrieval queries",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=20,
        help="Number of context assembly calls",
    )
    parser.add_argument(
        "--output",
        default="bench/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced iterations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Adjust counts for quick mode
    if args.quick:
        args.ingestion = min(args.ingestion, 20)
        args.retrieval = min(args.retrieval, 10)
        args.context = min(args.context, 5)
        logger.info("Running in quick mode with reduced iterations")

    logger.info("=" * 60)
    logger.info("HippocampAI Benchmark Suite")
    logger.info("=" * 60)
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Ingestion count: {args.ingestion}")
    logger.info(f"Retrieval count: {args.retrieval}")
    logger.info(f"Context assembly count: {args.context}")
    logger.info("=" * 60)

    try:
        # Initialize benchmarks
        benchmarks = HippocampBenchmarks(
            api_url=args.api_url,
            api_key=args.api_key,
        )

        # Run all benchmarks
        suite = benchmarks.run_all(
            ingestion_count=args.ingestion,
            retrieval_count=args.retrieval,
            context_count=args.context,
        )

        # Save results
        json_path, md_path = save_results(suite, args.output)

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)

        for result in suite.results:
            logger.info(
                f"{result.name}: {result.ops_per_second:.1f} ops/sec, "
                f"p50={result.latency_p50_ms:.2f}ms, "
                f"p95={result.latency_p95_ms:.2f}ms"
            )

        logger.info("")
        logger.info("Results saved to:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")

        # Cleanup
        try:
            benchmarks.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
