"""
Comprehensive integration test for Library <-> SaaS connectivity.

Tests the Python client library (RemoteBackend) connecting to the
running SaaS API server to ensure all features work end-to-end.
"""

# ruff: noqa: S101  # Use of assert detected - expected in test files

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

# Must add src to path before importing hippocampai modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hippocampai.backends.remote import RemoteBackend
from hippocampai.models.memory import Memory, MemoryType

# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_test(name: str) -> None:
    """Print test name."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing: {name}{Colors.END}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_section(name: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}{Colors.END}")


def test_api_health():
    """Test 0: API Health Check."""
    print_test("API Health Check")

    import httpx
    try:
        response = httpx.get("http://localhost:8000/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        print_success(f"API is healthy: {data}")
        return True
    except Exception as e:
        print_error(f"API health check failed: {e}")
        return False


def test_remote_backend_initialization():
    """Test 1: Remote backend initialization."""
    print_test("Remote Backend Initialization")

    try:
        backend = RemoteBackend(api_url="http://localhost:8000")
        print_success("RemoteBackend initialized successfully")
        return backend
    except Exception as e:
        print_error(f"Failed to initialize RemoteBackend: {e}")
        return None


def test_memory_creation(backend: RemoteBackend, user_id: str):
    """Test 2: Memory creation via SaaS API."""
    print_test("Memory Creation")

    try:
        memory = backend.remember(
            text="The capital of France is Paris",
            user_id=user_id,
            memory_type=MemoryType.FACT,
            metadata={"test": "library_saas_integration", "timestamp": datetime.now(timezone.utc).isoformat()}
        )

        assert memory is not None, "Memory creation returned None"
        assert memory.text == "The capital of France is Paris"
        assert memory.user_id == user_id
        assert memory.memory_type == MemoryType.FACT

        print_success(f"Memory created: ID={memory.id}")
        return memory
    except Exception as e:
        print_error(f"Memory creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_retrieval(backend: RemoteBackend, memory_id: str):
    """Test 3: Memory retrieval by ID."""
    print_test("Memory Retrieval by ID")

    try:
        memory = backend.get_memory(memory_id)

        assert memory is not None, f"Failed to retrieve memory {memory_id}"
        assert memory.id == memory_id

        print_success(f"Memory retrieved: {memory.text[:50]}...")
        return True
    except Exception as e:
        print_error(f"Memory retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_search(backend: RemoteBackend, user_id: str):
    """Test 4: Memory search (recall)."""
    print_test("Memory Search (Recall)")

    try:
        results = backend.recall(
            query="What is the capital of France?",
            user_id=user_id,
            limit=5
        )

        assert results is not None, "Search returned None"
        assert len(results) > 0, "Search returned no results"

        print_success(f"Found {len(results)} memories")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. Score: {result.score:.3f} - {result.memory.text[:50]}...")

        return True
    except Exception as e:
        print_error(f"Memory search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_update(backend: RemoteBackend, memory_id: str, user_id: str):
    """Test 5: Memory update."""
    print_test("Memory Update")

    try:
        updated_memory = backend.update_memory(
            memory_id=memory_id,
            user_id=user_id,
            text="The capital of France is Paris, known as the City of Light",
            metadata={"updated": True, "update_time": datetime.now(timezone.utc).isoformat()}
        )

        assert updated_memory is not None, "Update returned None"
        assert "City of Light" in updated_memory.text

        print_success(f"Memory updated: {updated_memory.text[:60]}...")
        return True
    except Exception as e:
        print_error(f"Memory update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_operations(backend: RemoteBackend, user_id: str):
    """Test 6: Batch memory operations."""
    print_test("Batch Memory Operations")

    try:
        # Create multiple memories in batch
        memories_data = [
            {
                "text": f"Test memory {i}: The color of the sky is blue",
                "user_id": user_id,
                "type": "fact",
                "metadata": {"batch_test": True, "index": i}
            }
            for i in range(3)
        ]

        memories = backend.batch_create_memories(memories_data)

        assert memories is not None, "Batch create returned None"
        assert len(memories) == 3, f"Expected 3 memories, got {len(memories)}"

        print_success(f"Batch created {len(memories)} memories")
        return [m.id for m in memories]
    except Exception as e:
        print_error(f"Batch operations failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_memory_deletion(backend: RemoteBackend, memory_ids: list[str]):
    """Test 7: Memory deletion."""
    print_test("Memory Deletion")

    try:
        deleted_count = 0
        for memory_id in memory_ids:
            result = backend.delete_memory(memory_id)
            if result:
                deleted_count += 1

        print_success(f"Deleted {deleted_count}/{len(memory_ids)} memories")
        return True
    except Exception as e:
        print_error(f"Memory deletion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_features(backend: RemoteBackend, user_id: str):
    """Test 8: Advanced features (extraction, deduplication, etc.)."""
    print_test("Advanced Features")

    successes = []

    # Test extraction from conversation
    try:
        conversation = [
            {"role": "user", "content": "I love pizza"},
            {"role": "assistant", "content": "That's great! Pizza is delicious."},
            {"role": "user", "content": "My favorite is Margherita"}
        ]

        extracted = backend.extract_from_conversation(
            conversation=conversation,
            user_id=user_id,
            session_id=f"test_session_{int(time.time())}"
        )

        print_success(f"Extracted {len(extracted)} memories from conversation")
        successes.append(True)
    except Exception as e:
        print_warning(f"Extraction test skipped or failed: {e}")
        successes.append(False)

    # Test deduplication (dry run)
    try:
        result = backend.deduplicate_memories(user_id=user_id, dry_run=True)
        print_success(f"Deduplication check complete: {result}")
        successes.append(True)
    except Exception as e:
        print_warning(f"Deduplication test skipped or failed: {e}")
        successes.append(False)

    return any(successes)


def test_query_with_filters(backend: RemoteBackend, user_id: str):
    """Test 9: Query with filters."""
    print_test("Query with Filters")

    try:
        memories = backend.get_memories(
            user_id=user_id,
            filters={"memory_type": "fact"},
            limit=10
        )

        assert memories is not None
        print_success(f"Filtered query returned {len(memories)} memories")
        return True
    except Exception as e:
        print_error(f"Filtered query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print_section("LIBRARY-SAAS INTEGRATION TEST SUITE")
    print(f"Testing library connection to SaaS API at http://localhost:8000")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")

    # Generate unique user ID for this test run
    user_id = f"test_user_{int(time.time())}"
    print(f"Test user ID: {user_id}")

    results = {}

    # Test 0: Health check
    results["health_check"] = test_api_health()
    if not results["health_check"]:
        print_error("API is not healthy, aborting tests")
        return

    # Test 1: Backend initialization
    backend = test_remote_backend_initialization()
    if not backend:
        print_error("Failed to initialize backend, aborting tests")
        return

    results["backend_init"] = True

    # Test 2: Memory creation
    memory = test_memory_creation(backend, user_id)
    results["memory_creation"] = memory is not None

    if memory:
        # Test 3: Memory retrieval
        results["memory_retrieval"] = test_memory_retrieval(backend, memory.id)

        # Test 4: Memory search
        results["memory_search"] = test_memory_search(backend, user_id)

        # Test 5: Memory update
        results["memory_update"] = test_memory_update(backend, memory.id, user_id)

    # Test 6: Batch operations
    batch_memory_ids = test_batch_operations(backend, user_id)
    results["batch_operations"] = len(batch_memory_ids) > 0

    # Test 7: Memory deletion
    if batch_memory_ids:
        results["memory_deletion"] = test_memory_deletion(backend, batch_memory_ids)

    # Test 8: Advanced features
    results["advanced_features"] = test_advanced_features(backend, user_id)

    # Test 9: Query with filters
    results["query_with_filters"] = test_query_with_filters(backend, user_id)

    # Cleanup: Delete test memory if it exists
    if memory:
        try:
            backend.delete_memory(memory.id)
        except Exception:
            pass

    # Print summary
    print_section("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"Total tests: {total}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_error(f"Failed: {failed}")

    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END} - {test_name}")

    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n{Colors.BOLD}Success rate: {success_rate:.1f}%{Colors.END}")

    if success_rate == 100:
        print_success("\n🎉 All tests passed! Library-SaaS integration is working perfectly.")
    elif success_rate >= 80:
        print_warning(f"\n⚠️  Most tests passed ({success_rate:.1f}%), but some issues need attention.")
    else:
        print_error(f"\n❌ Multiple tests failed ({success_rate:.1f}%). Integration needs fixes.")

    return success_rate == 100


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
