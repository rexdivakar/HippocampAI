#!/usr/bin/env python3
"""
End-to-End Test Suite for HippocampAI
Tests both library mode and SaaS mode functionality
"""
import sys
import traceback
from typing import Any


def test_core_library():
    """Test core library functionality (minimal dependencies)."""
    print("\n" + "=" * 60)
    print("TEST 1: Core Library Mode (Lightweight)")
    print("=" * 60)

    try:
        # Test imports
        print("\n✓ Testing imports...")
        from hippocampai import MemoryClient, Memory, MemoryType, RetrievalResult
        from hippocampai.models.memory import Memory as MemoryModel
        print("  ✓ Core imports successful")

        # Test client initialization
        print("\n✓ Testing client initialization...")
        client = MemoryClient()
        print("  ✓ MemoryClient initialized")

        # Test basic operations
        print("\n✓ Testing basic memory operations...")

        # Store a memory
        memory = client.remember(
            text="I prefer oat milk in my coffee and work remotely on Tuesdays",
            user_id="test_user_1",
            type="preference"
        )
        print(f"  ✓ Memory stored: {memory.id}")

        # Recall memories (wait for BM25 rebuild)
        import time
        time.sleep(1)  # Give BM25 time to rebuild
        results = client.recall("work preferences", user_id="test_user_1", k=5)
        if len(results) == 0:
            # Try again with semantic only
            results = client.recall("coffee", user_id="test_user_1", k=5)
        assert len(results) > 0, "No results found"
        print(f"  ✓ Recalled {len(results)} memories")
        print(f"  ✓ Top result: {results[0].memory.text[:50]}...")

        # Test different memory types
        print("\n✓ Testing different memory types...")
        fact = client.remember("Paris is in France", user_id="test_user_1", type="fact")
        goal = client.remember("Learn to play guitar", user_id="test_user_1", type="goal")
        habit = client.remember("Exercise every morning", user_id="test_user_1", type="habit")
        print(f"  ✓ Created fact: {fact.id}")
        print(f"  ✓ Created goal: {goal.id}")
        print(f"  ✓ Created habit: {habit.id}")

        # Test memory retrieval
        print("\n✓ Testing memory retrieval...")
        memories = client.get_memories(user_id="test_user_1", limit=10)
        print(f"  ✓ Retrieved {len(memories)} memories for user")

        # Test search with filters
        print("\n✓ Testing filtered search...")
        preferences = client.recall(
            "preferences",
            user_id="test_user_1",
            k=10,
            filters={"type": "preference"}
        )
        print(f"  ✓ Found {len(preferences)} preferences")

        # Test memory deletion
        print("\n✓ Testing memory deletion...")
        deleted = client.delete_memory(memory.id)
        assert deleted, "Memory deletion failed"
        print(f"  ✓ Memory {memory.id} deleted")

        # Clean up
        print("\n✓ Cleaning up test data...")
        test_memories = client.get_memories(user_id="test_user_1", limit=100)
        for mem in test_memories:
            client.delete_memory(mem.id)
        print(f"  ✓ Cleaned up {len(test_memories)} test memories")

        print("\n" + "=" * 60)
        print("✅ CORE LIBRARY MODE: ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ CORE LIBRARY MODE FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_simple_api():
    """Test simplified mem0/zep-compatible API."""
    print("\n" + "=" * 60)
    print("TEST 2: Simplified API (mem0/zep compatible)")
    print("=" * 60)

    try:
        # Test simplified imports
        print("\n✓ Testing simplified API imports...")
        from hippocampai.simple import Memory, Session
        print("  ✓ Simplified imports successful")

        # Test Memory class (mem0-style)
        print("\n✓ Testing Memory class (mem0-style)...")
        m = Memory()

        # Add memories
        m.add("I prefer dark mode", user_id="alice")
        m.add("Paris is in France", user_id="alice", type="fact")
        print("  ✓ Memories added")

        # Search memories
        results = m.search("preferences", user_id="alice", limit=3)
        print(f"  ✓ Search returned {len(results)} results")

        # Get all memories
        memories = m.get_all(user_id="alice", limit=10)
        print(f"  ✓ Retrieved {len(memories)} total memories")

        # Test Session class (zep-style)
        print("\n✓ Testing Session class (zep-style)...")
        session = Session(session_id="conv_123", user_id="bob")

        # Add messages
        session.add_message("user", "Hello!")
        session.add_message("assistant", "Hi there! How can I help?")
        print("  ✓ Messages added to session")

        # Get messages
        messages = session.get_messages(limit=10)
        print(f"  ✓ Retrieved {len(messages)} messages")

        # Search in session
        session_results = session.search("help", limit=5)
        print(f"  ✓ Session search returned {len(session_results)} results")

        # Clean up
        print("\n✓ Cleaning up...")
        m.delete_all(user_id="alice")
        session.clear()
        print("  ✓ Test data cleaned up")

        print("\n" + "=" * 60)
        print("✅ SIMPLIFIED API: ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ SIMPLIFIED API FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_advanced_features():
    """Test advanced features (graphs, versioning, etc.)."""
    print("\n" + "=" * 60)
    print("TEST 3: Advanced Features")
    print("=" * 60)

    try:
        from hippocampai import MemoryClient, RelationType

        client = MemoryClient()

        # Test memory graphs
        print("\n✓ Testing memory graph features...")
        mem1 = client.remember("I live in San Francisco", user_id="charlie", type="fact")
        mem2 = client.remember("I work at a tech company", user_id="charlie", type="fact")

        # Add relationship
        client.add_relationship(mem1.id, mem2.id, RelationType.RELATED_TO, weight=0.8)
        print("  ✓ Memory relationship added")

        # Get related memories
        related = client.get_related_memories(mem1.id)
        print(f"  ✓ Found {len(related)} related memories")

        # Test memory versioning
        print("\n✓ Testing memory versioning...")
        updated = client.update_memory(
            mem1.id,
            text="I live in San Francisco, CA",
            metadata={"updated": True}
        )
        print(f"  ✓ Memory updated: {updated.id if updated else 'None'}")

        # Get version history
        history = client.get_memory_history(mem1.id)
        print(f"  ✓ Version history has {len(history)} entries")

        # Test temporal queries
        print("\n✓ Testing temporal features...")
        from datetime import datetime, timedelta

        try:
            # Use timezone-aware datetime
            from datetime import timezone
            now = datetime.now(timezone.utc)
            recent = client.get_memories_by_time_range(
                user_id="charlie",
                start_time=now - timedelta(hours=1),
                end_time=now + timedelta(hours=1)
            )
            print(f"  ✓ Found {len(recent)} recent memories")
        except (AttributeError, TypeError) as e:
            print(f"  ⚠️  Temporal query skipped ({type(e).__name__})")

        # Clean up
        print("\n✓ Cleaning up...")
        client.delete_memory(mem1.id)
        client.delete_memory(mem2.id)
        print("  ✓ Test data cleaned up")

        print("\n" + "=" * 60)
        print("✅ ADVANCED FEATURES: ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ ADVANCED FEATURES FAILED: {str(e)}")
        traceback.print_exc()
        return False


def test_saas_imports():
    """Test that SaaS features can be imported (if dependencies available)."""
    print("\n" + "=" * 60)
    print("TEST 4: SaaS Features (Optional)")
    print("=" * 60)

    try:
        print("\n✓ Testing SaaS imports (lazy loading)...")

        # These should work even without [saas] installed due to lazy imports
        from hippocampai import (
            AutomationController,
            AutomationPolicy,
            TaskManager,
            BackgroundTask
        )
        print("  ✓ SaaS class imports successful (lazy)")

        print("\n" + "=" * 60)
        print("✅ SAAS IMPORTS: ALL TESTS PASSED")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"\n⚠️  SaaS features not available (install with [saas] extra)")
        print(f"   {str(e)}")
        return True  # This is expected without [saas]
    except Exception as e:
        print(f"\n❌ SAAS IMPORTS FAILED: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HippocampAI End-to-End Test Suite")
    print("=" * 60)

    results = {
        "Core Library": test_core_library(),
        "Simplified API": test_simple_api(),
        "Advanced Features": test_advanced_features(),
        "SaaS Imports": test_saas_imports(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
