#!/usr/bin/env python3
"""
Test script to validate HippocampAI installation and basic functionality.

Usage:
    # After installing from TestPyPI:
    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai

    # Run this script:
    python test_install.py
"""

import sys
import traceback
from typing import List


class TestResult:
    """Store test results."""
    def __init__(self, name: str, passed: bool, error: str = ""):
        self.name = name
        self.passed = passed
        self.error = error

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        error_msg = f"\n  Error: {self.error}" if self.error else ""
        return f"{status}: {self.name}{error_msg}"


def run_test(test_name: str, test_func) -> TestResult:
    """Run a single test and return result."""
    try:
        test_func()
        return TestResult(test_name, True)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return TestResult(test_name, False, error_msg)


def test_basic_imports():
    """Test basic package imports."""
    from hippocampai import Memory, MemoryClient, MemoryType

    assert MemoryClient is not None
    assert Memory is not None
    assert MemoryType is not None


def test_version():
    """Test package version."""
    import hippocampai
    assert hasattr(hippocampai, '__version__')
    print(f"  Package version: {hippocampai.__version__}")


def test_memory_type_enum():
    """Test MemoryType enum."""
    from hippocampai.models.memory import MemoryType

    # Check all expected memory types exist
    expected_types = ['preference', 'fact', 'goal', 'event', 'habit', 'context']
    for mem_type in expected_types:
        assert hasattr(MemoryType, mem_type.upper())

    print(f"  MemoryType enum has {len(MemoryType)} types")


def test_memory_model():
    """Test Memory model creation."""
    from datetime import datetime

    from hippocampai.models.memory import Memory, MemoryType

    memory = Memory(
        id="test_123",
        user_id="user_456",
        session_id="session_789",
        text="Test memory content",
        type=MemoryType.FACT,
        timestamp=datetime.now(),
        importance=0.8,
        embedding=[0.1] * 384  # Typical embedding dimension
    )

    assert memory.id == "test_123"
    assert memory.user_id == "user_456"
    assert memory.type == MemoryType.FACT
    assert memory.importance == 0.8
    print(f"  Created Memory with id: {memory.id}")


def test_config_import():
    """Test configuration module."""
    from hippocampai.config import Config  # noqa: F401

    # Just test that Config class exists and can be imported
    print("  Config class imported successfully")


def test_cli_imports():
    """Test CLI module imports."""
    from hippocampai.cli import main  # noqa: F401

    print("  CLI module imported successfully")


def test_api_imports():
    """Test API module imports."""
    from hippocampai.api import app
    assert app is not None
    print("  API module imported successfully")


def test_pipeline_imports():
    """Test pipeline modules."""
    from hippocampai.pipeline import consolidate, dedup, extractor, importance

    assert extractor is not None
    assert dedup is not None
    assert consolidate is not None
    assert importance is not None

    print("  All pipeline modules imported successfully")


def test_retrieval_imports():
    """Test retrieval modules."""
    from hippocampai.retrieval import bm25, rerank, retriever, router, rrf

    assert bm25 is not None
    assert rerank is not None
    assert retriever is not None
    assert router is not None
    assert rrf is not None

    print("  All retrieval modules imported successfully")


def test_adapters_imports():
    """Test LLM adapter modules."""
    from hippocampai.adapters import llm_base, provider_ollama, provider_openai

    assert llm_base is not None
    assert provider_ollama is not None
    assert provider_openai is not None

    print("  All adapter modules imported successfully")


def test_embedder_import():
    """Test embedder module."""
    from hippocampai.embed.embedder import Embedder

    assert Embedder is not None
    print("  Embedder class imported successfully")


def test_vector_store_import():
    """Test vector store module."""
    from hippocampai.vector.qdrant_store import QdrantStore

    assert QdrantStore is not None
    print("  QdrantStore class imported successfully")


def test_utils_imports():
    """Test utility modules."""
    from hippocampai.utils import cache, scoring, time

    assert cache is not None
    assert scoring is not None
    assert time is not None

    print("  All utility modules imported successfully")


def test_memory_client_creation():
    """Test MemoryClient class availability."""
    from hippocampai import MemoryClient  # noqa: F401

    # Just test that the class exists
    print("  MemoryClient class imported successfully")
    print("  (Note: Full instantiation requires Qdrant connection)")


def test_package_metadata():
    """Test package metadata."""
    import hippocampai

    # Check for standard attributes
    assert hasattr(hippocampai, '__version__')
    assert hasattr(hippocampai, '__all__')

    # Check exported symbols
    assert 'MemoryClient' in hippocampai.__all__
    assert 'Memory' in hippocampai.__all__
    assert 'MemoryType' in hippocampai.__all__

    print(f"  Package exports: {', '.join(hippocampai.__all__)}")


def test_dependencies():
    """Test that key dependencies are available."""
    dependencies = [
        ('qdrant_client', 'Qdrant Client'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('rank_bm25', 'BM25'),
        ('pydantic', 'Pydantic'),
        ('fastapi', 'FastAPI'),
        ('typer', 'Typer'),
    ]

    missing = []
    for module_name, _display_name in dependencies:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(_display_name)

    if missing:
        raise ImportError(f"Missing dependencies: {', '.join(missing)}")

    print(f"  All {len(dependencies)} key dependencies are installed")


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_summary(results: List[TestResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print_header("TEST SUMMARY")
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/total*100):.1f}%\n")

    if failed > 0:
        print("Failed Tests:")
        for result in results:
            if not result.passed:
                print(f"  - {result.name}")
                print(f"    {result.error}\n")


def main():
    """Run all tests."""
    print_header("HippocampAI Installation Test Suite")
    print("\nTesting package installation and basic functionality...")

    # Define all tests
    tests = [
        ("Basic imports", test_basic_imports),
        ("Package version", test_version),
        ("Package metadata", test_package_metadata),
        ("Dependencies", test_dependencies),
        ("MemoryType enum", test_memory_type_enum),
        ("Memory model", test_memory_model),
        ("Config module", test_config_import),
        ("CLI module", test_cli_imports),
        ("API module", test_api_imports),
        ("Pipeline modules", test_pipeline_imports),
        ("Retrieval modules", test_retrieval_imports),
        ("Adapter modules", test_adapters_imports),
        ("Embedder module", test_embedder_import),
        ("Vector store module", test_vector_store_import),
        ("Utility modules", test_utils_imports),
        ("MemoryClient creation", test_memory_client_creation),
    ]

    # Run all tests
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}...", end=" ")
        result = run_test(test_name, test_func)
        results.append(result)
        print(result)

    # Print summary
    print_summary(results)

    # Return exit code
    failed_count = sum(1 for r in results if not r.passed)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        print_header("TEST COMPLETE")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception:
        print("\n\nFatal error during testing:")
        traceback.print_exc()
        sys.exit(1)
