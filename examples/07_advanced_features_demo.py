"""Demonstration of all advanced memory features in HippocampAI.

This example showcases:
1. Batch operations (add_memories, delete_memories)
2. Graph indexing and relationships
3. Version control and rollback
4. Context injection for LLM prompts
5. Memory access tracking
6. Advanced filtering and sorting
7. Snapshots
8. Audit trail
9. KV store operations

Run this with:
    python examples/07_advanced_features_demo.py
"""

import time

from hippocampai import ChangeType, MemoryClient, RelationType


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Run the advanced features demonstration."""
    # Initialize client
    client = MemoryClient(
        qdrant_url="http://localhost:6333",
        collection_facts="demo_facts",
        collection_prefs="demo_prefs",
    )

    user_id = "demo_user"

    # === 1. BATCH OPERATIONS ===
    print_section("1. Batch Operations")

    # Batch add memories
    memories_data = [
        {
            "text": "Python is great for machine learning",
            "tags": ["programming", "ml"],
            "importance": 8.5,
        },
        {
            "text": "I prefer working in the mornings",
            "tags": ["productivity", "lifestyle"],
            "type": "preference",
            "importance": 7.0,
        },
        {
            "text": "Neural networks are inspired by the brain",
            "tags": ["ml", "ai", "neuroscience"],
            "importance": 9.0,
        },
        {
            "text": "I need to finish the project by Friday",
            "type": "goal",
            "importance": 9.5,
            "ttl_days": 7,  # Expires in 7 days
        },
    ]

    print("Adding 4 memories in batch...")
    created_memories = client.add_memories(memories_data, user_id=user_id)
    print(f"✓ Created {len(created_memories)} memories")
    for i, mem in enumerate(created_memories, 1):
        print(f"  {i}. [{mem.type.value}] {mem.text[:50]}... (ID: {mem.id[:8]})")

    # === 2. GRAPH INDEXING ===
    print_section("2. Graph Indexing & Relationships")

    # Add memories to graph
    print("Adding memories to graph index...")
    for mem in created_memories:
        client.graph.add_memory(mem.id, user_id, {"type": mem.type.value})

    # Create relationships
    print("\nCreating relationships:")
    ml_memories = [m for m in created_memories if "ml" in m.tags]
    if len(ml_memories) >= 2:
        client.add_relationship(
            ml_memories[0].id, ml_memories[1].id, RelationType.RELATED_TO, weight=0.9
        )
        print(f"  ✓ Linked '{ml_memories[0].text[:40]}...'")
        print(f"    → '{ml_memories[1].text[:40]}...'")

    # Get related memories
    if ml_memories:
        related = client.get_related_memories(ml_memories[0].id, max_depth=1)
        print(f"\n✓ Found {len(related)} related memories")
        for rel_id, rel_type, weight in related:
            print(f"  - {rel_id[:8]} ({rel_type}, strength: {weight})")

    # === 3. VERSION CONTROL ===
    print_section("3. Version Control & History")

    # Create a version before updating
    test_memory = created_memories[0]
    print(f"Creating version for: {test_memory.text[:50]}...")

    # Small delay to ensure memory is fully written
    time.sleep(0.2)

    version = client.version_control.create_version(
        test_memory.id,
        test_memory.model_dump(mode="json"),
        created_by=user_id,
        change_summary="Initial version",
    )
    print(f"✓ Version {version.version_number} created")

    # Update the memory
    print("\nUpdating memory importance...")
    time.sleep(0.1)  # Ensure previous operation completed
    updated = client.update_memory(test_memory.id, importance=9.8)
    if updated:
        print(f"✓ Updated importance: {test_memory.importance} → {updated.importance}")
        test_memory = updated  # Use updated memory for next steps
    else:
        print("⚠ Skipping version control demo (memory update failed)")
        print("  This can happen due to timing - continuing with other demos...\n")
        # Continue with other demonstrations

    # Create another version if update succeeded
    if updated:
        version2 = client.version_control.create_version(
            test_memory.id,
            updated.model_dump(mode="json"),
            created_by=user_id,
            change_summary="Increased importance",
        )
        print(f"✓ Version {version2.version_number} created")

        # View history
        history = client.get_memory_history(test_memory.id)
        print(f"\n✓ Version history ({len(history)} versions):")
        for v in history:
            print(
                f"  - v{v.version_number}: {v.change_summary} ({v.created_at.strftime('%H:%M:%S')})"
            )

        # Compare versions
        diff = client.version_control.compare_versions(test_memory.id, 1, 2)
        print("\n✓ Changes between v1 and v2:")
        if diff and diff.get("changed"):
            for key, change in diff["changed"].items():
                print(f"  - {key}: {change['old']} → {change['new']}")

    # === 4. CONTEXT INJECTION ===
    print_section("4. Context Injection for LLMs")

    user_query = "What do I know about machine learning?"
    print(f"User query: '{user_query}'\n")

    # Inject with default template
    prompt_default = client.inject_context(
        prompt=user_query,
        query="machine learning AI",
        user_id=user_id,
        k=3,
        template="default",
    )
    print("✓ Default template:")
    print(prompt_default[:200] + "...\n")

    # Inject with minimal template
    prompt_minimal = client.inject_context(
        prompt=user_query,
        query="machine learning",
        user_id=user_id,
        k=2,
        template="minimal",
    )
    print("✓ Minimal template:")
    print(prompt_minimal[:150] + "...\n")

    # === 5. ACCESS TRACKING ===
    print_section("5. Memory Access Tracking")

    track_memory = created_memories[1]
    print(f"Tracking access for: {track_memory.text[:50]}...")

    # Track multiple accesses
    for i in range(3):
        client.track_memory_access(track_memory.id, user_id)
        time.sleep(0.1)  # Small delay

    # Retrieve and check access count
    memories = client.get_memories(user_id=user_id)
    tracked = next(m for m in memories if m.id == track_memory.id)
    print(f"✓ Access count: {tracked.access_count}")
    print("✓ Access tracking enabled (access count stored in memory)")

    # === 6. ADVANCED FILTERING ===
    print_section("6. Advanced Filtering & Sorting")

    # Filter by importance
    print("High importance memories (>= 8.0):")
    high_importance = client.get_memories_advanced(
        user_id=user_id,
        filters={"min_importance": 8.0},
        sort_by="importance",
        sort_order="desc",
    )
    for mem in high_importance:
        print(f"  - [{mem.importance:.1f}] {mem.text[:50]}...")

    # Filter by tags
    print("\nMemories tagged with 'ml':")
    ml_tagged = client.get_memories_advanced(
        user_id=user_id,
        filters={"tags": "ml"},
        sort_by="created_at",
        sort_order="desc",
    )
    for mem in ml_tagged:
        print(f"  - {mem.text[:50]}... (tags: {', '.join(mem.tags)})")

    # Sort by access count
    print("\nMost accessed memories:")
    most_accessed = client.get_memories_advanced(
        user_id=user_id, sort_by="access_count", sort_order="desc", limit=3
    )
    for mem in most_accessed:
        print(f"  - [{mem.access_count} accesses] {mem.text[:40]}...")

    # === 7. SNAPSHOTS ===
    print_section("7. Memory Snapshots")

    print("Creating snapshot of facts collection...")
    snapshot_name = client.create_snapshot("facts")
    print(f"✓ Snapshot created: {snapshot_name}")
    print("  (Can be restored using Qdrant CLI or API)")

    # === 8. AUDIT TRAIL ===
    print_section("8. Audit Trail")

    # Get audit trail for a specific memory
    audit_entries = client.get_audit_trail(memory_id=test_memory.id, limit=10)
    print(f"Audit trail for memory {test_memory.id[:8]}:")
    for entry in audit_entries[:5]:  # Show first 5
        print(f"  - {entry.change_type.value}: {entry.timestamp.strftime('%H:%M:%S')}")
        if entry.metadata:
            print(f"    Metadata: {entry.metadata}")

    # Get audit trail by change type
    relationship_audits = client.get_audit_trail(change_type=ChangeType.RELATIONSHIP_ADDED, limit=5)
    print(f"\n✓ Relationship changes: {len(relationship_audits)}")

    # Get all user activity
    user_activity = client.get_audit_trail(user_id=user_id, limit=10)
    print(f"✓ Total user activity entries: {len(user_activity)}")

    # === 9. KV STORE ===
    print_section("9. Key-Value Store Operations")

    # KV store is used internally, but we can demonstrate direct usage
    print("KV Store is used internally for fast lookups")
    print("Demonstrating direct KV operations:")

    memory_data = {
        "id": "kv-test-1",
        "text": "KV store test memory",
        "user_id": user_id,
        "tags": ["test", "kv"],
    }

    client.kv_store.set_memory("kv-test-1", memory_data)
    retrieved = client.kv_store.get_memory("kv-test-1")
    print(f"✓ Stored and retrieved: {retrieved['text']}")

    # Get by user
    user_mem_ids = client.kv_store.get_user_memories(user_id)
    print(f"✓ User has {len(user_mem_ids)} memories in KV store")

    # Get by tag
    ml_ids = client.kv_store.get_memories_by_tag("ml")
    print(f"✓ Found {len(ml_ids)} memories tagged with 'ml'")

    # === 10. INTEGRATION EXAMPLE ===
    print_section("10. Full Integration Example")

    print("Demonstrating a complete workflow:\n")

    # 1. Batch add new memories
    new_memories = [
        {"text": "Deep learning uses neural networks", "tags": ["ml", "dl"], "importance": 8.0},
        {
            "text": "TensorFlow is a popular ML framework",
            "tags": ["ml", "tools"],
            "importance": 7.5,
        },
    ]
    created = client.add_memories(new_memories, user_id=user_id)
    print(f"1. Added {len(created)} new memories")

    # 2. Add to graph and create relationships
    for mem in created:
        client.graph.add_memory(mem.id, user_id, {})
    client.add_relationship(created[0].id, created[1].id, RelationType.SUPPORTS)
    print(f"2. Created relationship: {created[0].text[:30]}... → {created[1].text[:30]}...")

    # 3. Track access
    client.track_memory_access(created[0].id, user_id)
    print("3. Tracked memory access")

    # 4. Create context for LLM
    context_prompt = client.inject_context(
        "Explain deep learning to me", "deep learning neural networks", user_id, k=3
    )
    print(f"4. Created LLM prompt with context ({len(context_prompt)} chars)")

    # 5. Check audit trail
    final_audit = client.get_audit_trail(user_id=user_id, limit=5)
    print(f"5. Retrieved audit trail ({len(final_audit)} recent entries)")

    print("\n✓ Full workflow completed successfully!")

    # === CLEANUP ===
    print_section("Cleanup")

    # Get all memory IDs
    all_memories = client.get_memories(user_id=user_id, limit=1000)
    memory_ids = [m.id for m in all_memories]

    if memory_ids:
        print(f"Cleaning up {len(memory_ids)} memories...")
        deleted = client.delete_memories(memory_ids, user_id=user_id)
        print(f"✓ Deleted {deleted} memories")
    else:
        print("No memories to clean up")

    # === STATISTICS ===
    print_section("Final Statistics")

    # Version control stats
    vc_stats = client.version_control.get_statistics()
    print("Version Control:")
    for key, value in vc_stats.items():
        print(f"  - {key}: {value}")

    # KV store stats
    kv_stats = client.kv_store.get_stats()
    print("\nKV Store:")
    for key, value in kv_stats.items():
        print(f"  - {key}: {value}")

    # Graph stats
    print("\nGraph Index:")
    print(f"  - Total nodes: {client.graph.graph.number_of_nodes()}")
    print(f"  - Total edges: {client.graph.graph.number_of_edges()}")

    print("\n" + "=" * 60)
    print("  All advanced features demonstrated successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
