#!/usr/bin/env python3
"""
Unified Test Runner for HippocampAI

This script provides a comprehensive test suite runner that organizes and executes
all tests in a structured manner, similar to mem0 and zep test suites.

Usage:
    python tests/run_all_tests.py                    # Run all unit tests
    python tests/run_all_tests.py --integration      # Run integration tests
    python tests/run_all_tests.py --all              # Run everything
    python tests/run_all_tests.py --quick            # Run quick smoke tests
    python tests/run_all_tests.py --category core    # Run specific category
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


# Test categories organized by functionality
TEST_CATEGORIES: Dict[str, List[str]] = {
    "core": [
        "test_functional.py",
        "test_async.py",
        "test_retrieval.py",
        "test_comprehensive_validation.py",
    ],
    "scheduler": [
        "test_scheduler.py",
        "test_auto_consolidation.py",
        "test_auto_summarization.py",
        "test_importance_decay.py",
    ],
    "intelligence": [
        "test_intelligence_integration.py",
        "test_advanced_features.py",
    ],
    "memory_management": [
        "test_memory_management_api.py",
        "test_smart_memory.py",
        "test_memory_health.py",
        "test_advanced_compression.py",
    ],
    "multiagent": [
        "test_multiagent.py",
        "test_conflict_resolution.py",
    ],
    "monitoring": [
        "test_metrics.py",
        "test_monitoring_tags_storage.py",
    ],
    "integration": [
        "test_all_features_integration.py",
        "test_library_saas_integration.py",
    ],
}

# Quick smoke tests for rapid validation
QUICK_TESTS = [
    "tests/test_functional.py::TestMemoryClient::test_remember",
    "tests/test_functional.py::TestMemoryClient::test_recall",
    "tests/test_async.py::TestAsyncOperations::test_remember",
]


def run_pytest(
    test_files: List[str], verbose: bool = True, capture: str = "no", markers: Optional[str] = None
) -> int:
    """Run pytest with specified test files."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if capture:
        cmd.extend(["-s" if capture == "no" else f"--capture={capture}"])

    if markers:
        cmd.extend(["-m", markers])

    cmd.extend(test_files)

    print(f"{Colors.BLUE}Running: {' '.join(cmd)}{Colors.END}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_category(category: str, verbose: bool = True) -> int:
    """Run tests for a specific category."""
    if category not in TEST_CATEGORIES:
        print_error(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return 1

    print_header(f"Running {category.upper()} Tests")

    test_files = [f"tests/{f}" for f in TEST_CATEGORIES[category]]
    return run_pytest(test_files, verbose=verbose)


def run_quick_tests() -> int:
    """Run quick smoke tests."""
    print_header("Running Quick Smoke Tests")
    return run_pytest(QUICK_TESTS, verbose=True)


def run_unit_tests(verbose: bool = True) -> int:
    """Run all unit tests (excluding integration)."""
    print_header("Running Unit Tests")

    all_unit_tests = []
    for category, files in TEST_CATEGORIES.items():
        if category != "integration":
            all_unit_tests.extend([f"tests/{f}" for f in files])

    return run_pytest(all_unit_tests, verbose=verbose)


def run_integration_tests() -> int:
    """Run integration tests as standalone scripts."""
    print_header("Running Integration Tests")

    # Note: Integration tests are run as standalone scripts, not via pytest
    print_warning("Integration tests should be run as standalone scripts:")
    print("\n  python tests/test_all_features_integration.py")
    print("  python tests/test_library_saas_integration.py\n")

    print("These tests require:")
    print("  - Qdrant running on localhost:6333")
    print("  - Redis running on localhost:6379")
    print("  - Optional: LLM provider (Groq/OpenAI/Ollama)")
    print("  - For SaaS test: API server at localhost:8000\n")

    return 0


def run_all_tests(verbose: bool = True) -> int:
    """Run all tests."""
    print_header("Running Complete Test Suite")

    results = {}

    # Run unit tests by category
    for category in TEST_CATEGORIES.keys():
        if category != "integration":
            result = run_category(category, verbose=verbose)
            results[category] = result

    # Print summary
    print_header("Test Results Summary")

    total = len(results)
    passed = sum(1 for r in results.values() if r == 0)
    failed = total - passed

    for category, result in results.items():
        if result == 0:
            print_success(f"{category}: PASSED")
        else:
            print_error(f"{category}: FAILED")

    print(
        f"\n{Colors.BOLD}Total: {total} categories, {passed} passed, {failed} failed{Colors.END}\n"
    )

    if failed == 0:
        print_success("All tests passed! 🎉")
        print_warning("Don't forget to run integration tests separately if needed.")
        return 0
    else:
        print_error(f"{failed} test category(ies) failed")
        return 1


def list_categories() -> None:
    """List all test categories."""
    print_header("Available Test Categories")

    for category, files in TEST_CATEGORIES.items():
        print(f"{Colors.BOLD}{category}{Colors.END}")
        for file in files:
            print(f"  - {file}")
        print()


def check_services() -> None:
    """Check if required services are running."""
    print_header("Service Status Check")

    # Check Qdrant
    try:
        import httpx

        response = httpx.get("http://localhost:6333/health", timeout=2.0)
        if response.status_code == 200:
            print_success("Qdrant is running on localhost:6333")
        else:
            print_warning(f"Qdrant returned status {response.status_code}")
    except Exception as e:
        print_error(f"Qdrant not accessible: {e}")
        print("  Start with: docker run -p 6333:6333 qdrant/qdrant")

    # Check Redis
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
        r.ping()
        print_success("Redis is running on localhost:6379")
    except Exception as e:
        print_warning(f"Redis not accessible: {e}")
        print("  Optional for most tests")
        print("  Start with: docker run -p 6379:6379 redis")

    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Test Runner for HippocampAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run all unit tests
  %(prog)s --quick              # Run quick smoke tests
  %(prog)s --category core      # Run core tests only
  %(prog)s --all                # Run everything
  %(prog)s --integration        # Show integration test info
  %(prog)s --list               # List all test categories
  %(prog)s --check-services     # Check service availability
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")

    parser.add_argument(
        "--category",
        type=str,
        choices=list(TEST_CATEGORIES.keys()),
        help="Run specific test category",
    )

    parser.add_argument("--all", action="store_true", help="Run all unit tests")

    parser.add_argument(
        "--integration", action="store_true", help="Show integration test information"
    )

    parser.add_argument("--list", action="store_true", help="List all test categories")

    parser.add_argument(
        "--check-services", action="store_true", help="Check if required services are running"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True, help="Verbose output (default)"
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Handle different modes
    if args.list:
        list_categories()
        return 0

    if args.check_services:
        check_services()
        return 0

    if args.quick:
        return run_quick_tests()

    if args.category:
        return run_category(args.category, verbose=verbose)

    if args.integration:
        return run_integration_tests()

    if args.all:
        return run_all_tests(verbose=verbose)

    # Default: run unit tests
    return run_unit_tests(verbose=verbose)


if __name__ == "__main__":
    sys.exit(main())
