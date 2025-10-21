"""Example 06: Advanced Memory Management

This example demonstrates all the new high-priority features:
1. update_memory() - Modify existing memories
2. delete_memory() - Remove memories
3. get_memories() - Retrieve with advanced filters
4. Tag-based filtering in recall
5. Memory TTL and automatic expiration
"""

from datetime import datetime, timedelta

from hippocampai import MemoryClient


def main():
    # Initialize client
    client = MemoryClient()
    user_id = "demo_user"

    print("=" * 60)
    print("Advanced Memory Management Demo")
    print("=" * 60)

    # 1. Create memories with tags and TTL
    print("\n1. Creating memories with tags and TTL...")

    coffee_memory = client.remember(
        text="I prefer dark roast coffee with oat milk",
        user_id=user_id,
        type="preference",
        tags=["beverages", "coffee", "morning"],
        importance=8.0,
        ttl_days=90,  # Expires in 90 days
    )
    print(f"✓ Created coffee preference: {coffee_memory.id[:8]}...")

    tea_memory = client.remember(
        text="I drink green tea in the afternoon",
        user_id=user_id,
        type="habit",
        tags=["beverages", "tea", "afternoon"],
        importance=6.0,
        ttl_days=60,
    )
    print(f"✓ Created tea habit: {tea_memory.id[:8]}...")

    workout_memory = client.remember(
        text="I go to the gym at 6am every Monday, Wednesday, Friday",
        user_id=user_id,
        type="habit",
        tags=["fitness", "routine", "morning"],
        importance=9.0,
    )
    print(f"✓ Created workout habit: {workout_memory.id[:8]}...")

    temp_note = client.remember(
        text="Remember to buy groceries today",
        user_id=user_id,
        type="context",
        tags=["todo", "temporary"],
        importance=3.0,
        ttl_days=1,  # Expires tomorrow
    )
    print(f"✓ Created temporary note: {temp_note.id[:8]}...")

    # 2. Update existing memories
    print("\n2. Updating memories...")

    # Update coffee preference with more details
    updated_coffee = client.update_memory(
        memory_id=coffee_memory.id,
        text="I prefer dark roast coffee with oat milk, no sugar",
        tags=["beverages", "coffee", "morning", "dietary"],
        importance=9.0,  # Increased importance
    )
    print("✓ Updated coffee preference with new details")
    print(f"  New text: {updated_coffee.text}")
    print(f"  New tags: {updated_coffee.tags}")
    print(f"  New importance: {updated_coffee.importance}")

    # 3. Get memories with advanced filtering
    print("\n3. Retrieving memories with filters...")

    # Get all beverage-related memories
    beverage_memories = client.get_memories(
        user_id=user_id,
        filters={"tags": "beverages"},
    )
    print(f"\n✓ Found {len(beverage_memories)} beverage-related memories:")
    for mem in beverage_memories:
        print(f"  - {mem.text[:50]}... (tags: {mem.tags})")

    # Get high-importance memories only
    important_memories = client.get_memories(
        user_id=user_id,
        filters={"min_importance": 8.0},
    )
    print(f"\n✓ Found {len(important_memories)} high-importance memories:")
    for mem in important_memories:
        print(f"  - {mem.text[:50]}... (importance: {mem.importance})")

    # Get memories by type and tags
    morning_habits = client.get_memories(
        user_id=user_id,
        filters={"type": "habit", "tags": "morning"},
    )
    print(f"\n✓ Found {len(morning_habits)} morning habits:")
    for mem in morning_habits:
        print(f"  - {mem.text[:50]}...")

    # 4. Semantic recall with tag filtering
    print("\n4. Semantic recall with tag filtering...")

    # Recall memories about routines, filtered by tag
    routine_results = client.recall(
        query="What are my morning routines?",
        user_id=user_id,
        k=5,
        filters={"tags": "morning"},
    )
    print(f"\n✓ Found {len(routine_results)} relevant morning routines:")
    for result in routine_results:
        print(f"  - {result.memory.text[:50]}...")
        print(f"    Score: {result.score:.3f}, Tags: {result.memory.tags}")

    # Recall fitness-related memories
    fitness_results = client.recall(
        query="fitness and exercise",
        user_id=user_id,
        filters={"tags": "fitness"},
    )
    print(f"\n✓ Found {len(fitness_results)} fitness-related memories:")
    for result in fitness_results:
        print(f"  - {result.memory.text[:50]}...")

    # 5. Memory TTL and expiration
    print("\n5. Demonstrating TTL and expiration...")

    # Check which memories have TTL set
    all_memories = client.get_memories(
        user_id=user_id,
        filters={"include_expired": True},
    )
    print("\n✓ Memory TTL status:")
    for mem in all_memories:
        if mem.expires_at:
            days_until_expiry = (mem.expires_at - datetime.utcnow()).days
            print(f"  - {mem.text[:40]}...")
            print(f"    Expires in {days_until_expiry} days ({mem.expires_at.date()})")
        else:
            print(f"  - {mem.text[:40]}... (no expiration)")

    # Manually expire the temporary note for demonstration
    print("\n✓ Marking temporary note as expired...")
    client.update_memory(
        memory_id=temp_note.id,
        expires_at=datetime.utcnow() - timedelta(hours=1),  # Already expired
    )

    # Get memories without expired ones (default)
    active_memories = client.get_memories(user_id=user_id)
    print(f"\n✓ Active memories (excluding expired): {len(active_memories)}")

    # Get all memories including expired
    all_with_expired = client.get_memories(
        user_id=user_id,
        filters={"include_expired": True},
    )
    print(f"✓ Total memories (including expired): {len(all_with_expired)}")

    # Run expiration cleanup
    expired_count = client.expire_memories(user_id=user_id)
    print(f"✓ Cleaned up {expired_count} expired memory(ies)")

    # 6. Delete specific memory
    print("\n6. Deleting a memory...")

    # Delete the tea memory
    deleted = client.delete_memory(memory_id=tea_memory.id, user_id=user_id)
    if deleted:
        print("✓ Successfully deleted tea habit memory")

    # Verify deletion
    remaining = client.get_memories(user_id=user_id)
    print(f"✓ Remaining memories: {len(remaining)}")

    # 7. View telemetry
    print("\n7. Telemetry tracking...")

    # Get metrics summary
    metrics = client.get_telemetry_metrics()
    print("\n✓ Operation metrics:")
    for operation, stats in metrics.items():
        if stats and "count" in stats:
            print(f"  {operation}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg duration: {stats['avg']:.2f}ms")
            print(f"    P95: {stats['p95']:.2f}ms")

    # Get recent operations
    print("\n✓ Recent operations:")
    recent_ops = client.get_recent_operations(limit=5)
    for op in recent_ops:
        print(f"  - {op.operation.value}: {op.status} ({op.duration_ms:.2f}ms)")

    # 8. Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    final_memories = client.get_memories(user_id=user_id)
    print(f"\nFinal memory count: {len(final_memories)}")
    print("\nAll memories:")
    for mem in final_memories:
        print(f"\n  [{mem.type.value}] {mem.text}")
        print(f"    ID: {mem.id}")
        print(f"    Tags: {', '.join(mem.tags)}")
        print(f"    Importance: {mem.importance}/10")
        if mem.expires_at:
            days_left = (mem.expires_at - datetime.utcnow()).days
            print(f"    Expires in: {days_left} days")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
