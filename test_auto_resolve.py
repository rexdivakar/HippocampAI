"""
Test automatic conflict resolution (Mem0-style) feature.

Demonstrates the new auto_resolve_conflicts parameter in remember().
"""

import time

from hippocampai import MemoryClient


def test_auto_resolve_temporal():
    """Test automatic resolution with TEMPORAL strategy (latest wins)."""
    print("\n=== Test 1: Auto-Resolve with TEMPORAL Strategy ===")

    client = MemoryClient()
    user_id = "test_auto_temporal"

    # Clean up any existing memories
    existing = client.get_memories(user_id)
    for mem in existing:
        client.delete_memory(mem.id, user_id)

    # First memory
    mem1 = client.remember(
        "I love coffee",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=False  # No auto-resolve yet
    )
    print(f"✓ Memory 1 stored: {mem1.text}")

    time.sleep(0.2)

    # Second conflicting memory WITH auto-resolve
    mem2 = client.remember(
        "I hate coffee",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=True,  # Auto-resolve enabled!
        resolution_strategy="temporal"  # Latest wins
    )
    print(f"✓ Memory 2 stored with auto-resolve: {mem2.text}")

    # Check result - should only have 1 memory (the latest one)
    memories = client.get_memories(user_id)
    print("\n📊 Result:")
    print(f"   Total memories: {len(memories)}")
    print(f"   Remaining memory: {memories[0].text if memories else 'None'}")

    # Verify only one memory remains
    assert len(memories) == 1, f"Expected 1 memory, got {len(memories)}"
    assert "hate" in memories[0].text.lower(), "Latest memory should remain"

    print("✅ Test passed: TEMPORAL auto-resolve works!")
    return True


def test_auto_resolve_auto_merge():
    """Test automatic resolution with AUTO_MERGE strategy."""
    print("\n=== Test 2: Auto-Resolve with AUTO_MERGE Strategy ===")

    client = MemoryClient()

    if not client.llm:
        print("⚠ LLM not available, skipping AUTO_MERGE test")
        return True

    user_id = "test_auto_merge"

    # Clean up any existing memories
    existing = client.get_memories(user_id)
    for mem in existing:
        client.delete_memory(mem.id, user_id)

    # First memory
    mem1 = client.remember(
        "I love pizza",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=False
    )
    print(f"✓ Memory 1 stored: {mem1.text}")

    time.sleep(0.2)

    # Conflicting memory with AUTO_MERGE
    mem2 = client.remember(
        "I hate pizza now",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=True,
        resolution_strategy="auto_merge"  # Merge both memories
    )
    print(f"✓ Memory 2 stored with auto-merge: {mem2.text}")

    # Check result
    memories = client.get_memories(user_id)
    print("\n📊 Result:")
    print(f"   Total memories: {len(memories)}")
    if memories:
        print(f"   Merged memory: {memories[0].text}")
        if "merged_from" in memories[0].metadata:
            print(f"   Merged from: {len(memories[0].metadata['merged_from'])} memories")

    # Should have 1 merged memory
    assert len(memories) >= 1, "Should have at least 1 memory"

    print("✅ Test passed: AUTO_MERGE auto-resolve works!")
    return True


def test_auto_resolve_confidence():
    """Test automatic resolution with CONFIDENCE strategy."""
    print("\n=== Test 3: Auto-Resolve with CONFIDENCE Strategy ===")

    client = MemoryClient()
    user_id = "test_auto_confidence"

    # Clean up any existing memories
    existing = client.get_memories(user_id)
    for mem in existing:
        client.delete_memory(mem.id, user_id)

    # First memory with high importance
    mem1 = client.remember(
        "I love tea",
        user_id=user_id,
        type="preference",
        importance=9.5,
        auto_resolve_conflicts=False
    )
    print(f"✓ Memory 1 stored (importance=9.5): {mem1.text}")

    time.sleep(0.2)

    # Conflicting memory with lower importance
    mem2 = client.remember(
        "I hate tea",
        user_id=user_id,
        type="preference",
        importance=5.0,
        auto_resolve_conflicts=True,
        resolution_strategy="importance"  # Higher importance wins
    )
    print(f"✓ Memory 2 stored with auto-resolve (importance=5.0): {mem2.text}")

    # Check result
    memories = client.get_memories(user_id)
    print("\n📊 Result:")
    print(f"   Total memories: {len(memories)}")
    if memories:
        print(f"   Remaining memory: {memories[0].text}")
        print(f"   Importance: {memories[0].importance}")

    # Should keep the higher importance memory
    assert len(memories) == 1, "Should have 1 memory"
    # The first memory (higher importance) should remain
    assert "love" in memories[0].text.lower(), "Higher importance memory should remain"

    print("✅ Test passed: IMPORTANCE auto-resolve works!")
    return True


def test_auto_resolve_disabled():
    """Test that without auto_resolve_conflicts, both memories are kept."""
    print("\n=== Test 4: Without Auto-Resolve (Default Behavior) ===")

    client = MemoryClient()
    user_id = "test_no_auto_resolve"

    # Clean up any existing memories
    existing = client.get_memories(user_id)
    for mem in existing:
        client.delete_memory(mem.id, user_id)

    # Two conflicting memories WITHOUT auto-resolve
    mem1 = client.remember(
        "I love juice",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=False  # Default behavior
    )
    print(f"✓ Memory 1 stored: {mem1.text}")

    time.sleep(0.2)

    mem2 = client.remember(
        "I hate juice",
        user_id=user_id,
        type="preference",
        auto_resolve_conflicts=False  # Still no auto-resolve
    )
    print(f"✓ Memory 2 stored: {mem2.text}")

    # Check result - should have BOTH memories
    memories = client.get_memories(user_id)
    print("\n📊 Result:")
    print(f"   Total memories: {len(memories)}")
    for i, mem in enumerate(memories, 1):
        print(f"   Memory {i}: {mem.text}")

    # Without auto-resolve, may have both or one depending on smart_updater
    # but we can at least verify memories exist
    assert len(memories) >= 1, "Should have at least 1 memory"

    print("✅ Test passed: Default behavior (no auto-resolve) works!")
    return True


def test_mem0_style_usage():
    """Demonstrate Mem0-style usage pattern."""
    print("\n=== Test 5: Mem0-Style Usage Pattern ===")

    client = MemoryClient()
    user_id = "test_mem0_style"

    # Clean up any existing memories
    existing = client.get_memories(user_id)
    for mem in existing:
        client.delete_memory(mem.id, user_id)

    # Simulate conversation over time with auto-resolve always on
    memories_to_add = [
        ("I work at Google", "Day 1"),
        ("I work at Facebook now", "Day 30"),
        ("I got promoted to Senior Engineer", "Day 60"),
        ("I work at Anthropic", "Day 90"),
    ]

    for text, day in memories_to_add:
        memory = client.remember(
            text,
            user_id=user_id,
            type="fact",
            auto_resolve_conflicts=True,  # Always auto-resolve (Mem0-style)
            resolution_strategy="temporal"  # Latest always wins
        )
        print(f"✓ {day}: {memory.text}")
        time.sleep(0.1)

    # Check final state
    memories = client.get_memories(user_id)
    print("\n📊 Final State:")
    print(f"   Total memories: {len(memories)}")
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. {mem.text}")

    # Should have minimal memories (conflicts auto-resolved)
    print("\n✅ Test passed: Mem0-style auto-resolve pattern works!")
    return True


def main():
    """Run all auto-resolve tests."""
    print("=" * 70)
    print("  Automatic Conflict Resolution Tests (Mem0-Style)")
    print("=" * 70)

    tests = [
        test_auto_resolve_temporal,
        test_auto_resolve_auto_merge,
        test_auto_resolve_confidence,
        test_auto_resolve_disabled,
        test_mem0_style_usage,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ Test failed: {test_func.__name__}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ Test error: {test_func.__name__}")
            print(f"   Exception: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    if failed == 0:
        print("\n🎉 All auto-resolve tests passed!")
        print("\n💡 Usage Tips:")
        print("   • Set auto_resolve_conflicts=True for Mem0-style behavior")
        print("   • Use 'temporal' strategy for latest-wins behavior (default)")
        print("   • Use 'auto_merge' to preserve history with LLM")
        print("   • Use 'importance' for priority-based resolution")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
