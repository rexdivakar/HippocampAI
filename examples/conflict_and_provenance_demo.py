"""
Conflict Resolution & Provenance Tracking Demo

Demonstrates:
1. Automatic conflict detection
2. Multiple resolution strategies
3. Provenance tracking
4. Quality assessment
5. Citation management
6. Provenance chains
"""

import time

from hippocampai import MemoryClient


def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def demo_basic_conflict_resolution():
    """Demonstrate basic conflict detection and resolution."""
    print_section("1. Basic Conflict Resolution")

    client = MemoryClient()
    user_id = "demo_user_1"

    # Store conflicting memories
    print("Storing conflicting memories...")
    mem1 = client.remember(
        "I love coffee and drink it every day",
        user_id=user_id,
        type="preference",
        importance=7,
        confidence=0.9,
    )
    print(f"✓ Memory 1: {mem1.text}")

    time.sleep(0.1)

    mem2 = client.remember(
        "I hate coffee now, switched to tea",
        user_id=user_id,
        type="preference",
        importance=8,
        confidence=0.95,
    )
    print(f"✓ Memory 2: {mem2.text}")

    # Detect conflicts
    print("\nDetecting conflicts...")
    conflicts = client.detect_memory_conflicts(user_id, check_llm=True)

    if conflicts:
        conflict = conflicts[0]
        print("\n⚠️  Conflict Detected!")
        print(f"   Type: {conflict.conflict_type}")
        print(f"   Confidence: {conflict.confidence_score:.2f}")
        print(f"   Similarity: {conflict.similarity_score:.2f}")
        print(f"\n   Memory 1: {conflict.memory_1.text}")
        print(f"   Created: {conflict.memory_1.created_at}")
        print(f"\n   Memory 2: {conflict.memory_2.text}")
        print(f"   Created: {conflict.memory_2.created_at}")
    else:
        print("No conflicts detected")

    # Resolve using TEMPORAL strategy (latest wins)
    print("\nResolving with TEMPORAL strategy (latest wins)...")
    result = client.auto_resolve_conflicts(user_id, strategy="temporal")

    print("\n✓ Resolution Complete:")
    print(f"   Conflicts found: {result['conflicts_found']}")
    print(f"   Resolved: {result['resolved_count']}")
    print(f"   Deleted: {result['deleted_count']} memories")

    # Show remaining memory
    memories = client.get_memories(user_id, limit=1)
    if memories:
        print(f"\n   Remaining memory: {memories[0].text}")


def demo_resolution_strategies():
    """Demonstrate different resolution strategies."""
    print_section("2. Resolution Strategies")

    client = MemoryClient()

    strategies = [
        ("temporal", "demo_user_temporal"),
        ("confidence", "demo_user_confidence"),
        ("auto_merge", "demo_user_merge"),
        ("user_review", "demo_user_review"),
    ]

    for strategy, user_id in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")

        # Create conflicting memories
        client.remember(
            "I work at Google",
            user_id=user_id,
            type="fact",
            importance=8,
            confidence=0.95,
        )

        time.sleep(0.1)

        client.remember(
            "I work at Facebook",
            user_id=user_id,
            type="fact",
            importance=8,
            confidence=0.80,
        )

        # Resolve
        result = client.auto_resolve_conflicts(user_id, strategy=strategy)

        print(f"Resolved: {result['resolved_count']}")
        print(f"Deleted: {result['deleted_count']}")
        if strategy == "auto_merge":
            print(f"Merged: {result['merged_count']}")

        # Show result
        memories = client.get_memories(user_id, limit=2)
        for mem in memories:
            print(f"  Memory: {mem.text}")
            if mem.metadata.get("has_conflict"):
                print("    ⚠️ Flagged for review")
            if mem.metadata.get("merged_from"):
                print(f"    📎 Merged from {len(mem.metadata['merged_from'])} memories")


def demo_provenance_tracking():
    """Demonstrate provenance tracking."""
    print_section("3. Provenance Tracking")

    client = MemoryClient()
    user_id = "demo_user_provenance"

    # Create memory with provenance
    print("Creating memory with provenance...")
    memory = client.remember(
        "John Smith works at Google as a Senior Software Engineer",
        user_id=user_id,
        type="fact",
        importance=8,
    )

    # Track provenance
    lineage = client.track_memory_provenance(
        memory,
        source="conversation",
        citations=[
            {
                "source_type": "message",
                "source_text": "User mentioned: My colleague John just started at Google",
            }
        ],
    )

    print(f"\n✓ Memory created: {memory.text}")
    print("\nProvenance:")
    print(f"  Source: {lineage['source']}")
    print(f"  Created at: {lineage['created_at']}")
    print(f"  Created by: {lineage['created_by']}")
    print(f"  Citations: {len(lineage['citations'])}")

    # Add more citations
    print("\nAdding additional citation...")
    client.add_memory_citation(
        memory.id,
        source_type="url",
        source_url="https://linkedin.com/in/johnsmith",
        source_text="LinkedIn profile confirms Google employment",
        confidence=0.95,
    )

    # Get updated lineage
    lineage = client.get_memory_lineage(memory.id)
    print(f"✓ Total citations: {len(lineage['citations'])}")

    for i, citation in enumerate(lineage["citations"], 1):
        print(f"\n  Citation {i}:")
        print(f"    Type: {citation['source_type']}")
        print(f"    Text: {citation.get('source_text', 'N/A')[:60]}...")
        print(f"    Confidence: {citation['confidence']:.2f}")


def demo_quality_assessment():
    """Demonstrate quality assessment."""
    print_section("4. Quality Assessment")

    client = MemoryClient()
    user_id = "demo_user_quality"

    # Create memories with different quality levels
    memories_data = [
        (
            "John works somewhere in tech",
            "vague, low specificity",
        ),
        (
            "John Smith works at Google in Mountain View, CA as a Senior Software Engineer in the Search team",
            "specific, high quality",
        ),
        (
            "Person does stuff",
            "extremely vague",
        ),
    ]

    print("Assessing quality of different memories...\n")

    for text, description in memories_data:
        mem = client.remember(text, user_id=user_id, type="fact")

        # Assess quality (using heuristic for speed)
        quality = client.assess_memory_quality(mem.id, use_llm=False)

        print(f"Memory: {text}")
        print(f"Description: {description}")
        print("\nQuality Scores:")
        print(f"  Specificity:   {quality['specificity']:.2f}")
        print(f"  Verifiability: {quality['verifiability']:.2f}")
        print(f"  Completeness:  {quality['completeness']:.2f}")
        print(f"  Clarity:       {quality['clarity']:.2f}")
        print(f"  Relevance:     {quality['relevance']:.2f}")
        print(f"  Overall:       {quality['overall_score']:.2f}")
        print()


def demo_provenance_chains():
    """Demonstrate provenance chain tracking."""
    print_section("5. Provenance Chains")

    client = MemoryClient()
    user_id = "demo_user_chain"

    # Create original memory
    print("Creating provenance chain...\n")

    mem1 = client.remember(
        "Alice works at Anthropic",
        user_id=user_id,
        type="fact",
    )
    client.track_memory_provenance(mem1, source="conversation")
    print(f"1. Original: {mem1.text}")

    # Create derived memory (inference)
    mem2 = client.remember(
        "Alice likely works with AI and language models",
        user_id=user_id,
        type="fact",
    )

    client.provenance_tracker.track_inference(
        mem2, source_memories=[mem1], inference_method="llm", confidence=0.85
    )

    client.update_memory(mem2.id, metadata=mem2.metadata)
    print(f"2. Inferred: {mem2.text}")
    print("   (derived from memory 1)")

    # Create another derived memory
    mem3 = client.remember(
        "Alice probably lives in San Francisco area",
        user_id=user_id,
        type="fact",
    )

    client.provenance_tracker.track_inference(
        mem3, source_memories=[mem1], inference_method="llm", confidence=0.75
    )

    client.update_memory(mem3.id, metadata=mem3.metadata)
    print(f"3. Inferred: {mem3.text}")
    print("   (derived from memory 1)")

    # Get provenance chain for mem2
    print("\nProvenance chain for memory 2:")
    lineage = client.get_memory_lineage(mem2.id)

    if lineage:
        print(f"  Source: {lineage['source']}")
        print(f"  Parent memories: {len(lineage['parent_memory_ids'])}")
        print(f"  Transformations: {len(lineage['transformations'])}")

        for transform in lineage["transformations"]:
            print(f"    - {transform['transformation_type']}: {transform['description']}")


def demo_complete_workflow():
    """Demonstrate complete workflow combining all features."""
    print_section("6. Complete Workflow")

    client = MemoryClient()
    user_id = "demo_user_complete"

    print("Scenario: Tracking evolving preference with full provenance\n")

    # Step 1: Initial preference
    print("Step 1: User mentions preference (Jan 2024)")
    mem1 = client.remember(
        "I love drinking coffee every morning",
        user_id=user_id,
        type="preference",
        importance=7,
        confidence=0.9,
    )

    client.track_memory_provenance(
        mem1,
        source="conversation",
        citations=[
            {
                "source_type": "message",
                "source_text": "User: I can't start my day without coffee!",
            }
        ],
    )

    quality1 = client.assess_memory_quality(mem1.id, use_llm=False)
    print(f"✓ Memory stored (quality: {quality1['overall_score']:.2f})")

    time.sleep(0.2)

    # Step 2: Conflicting preference
    print("\nStep 2: User mentions change (Mar 2024)")
    mem2 = client.remember(
        "I quit coffee and switched to green tea for health",
        user_id=user_id,
        type="preference",
        importance=8,
        confidence=0.95,
    )

    client.track_memory_provenance(
        mem2,
        source="conversation",
        citations=[
            {
                "source_type": "message",
                "source_text": "User: I gave up coffee for Lent and feel much better",
            }
        ],
    )

    print("✓ Memory stored")

    # Step 3: Detect conflict
    print("\nStep 3: Detecting conflicts...")
    conflicts = client.detect_memory_conflicts(user_id, check_llm=True)

    if conflicts:
        print(f"⚠️  Found {len(conflicts)} conflict(s)")
        conflict = conflicts[0]
        print(f"   Type: {conflict.conflict_type}")
        print(f"   Confidence: {conflict.confidence_score:.2f}")

    # Step 4: Resolve with AUTO_MERGE
    print("\nStep 4: Resolving with AUTO_MERGE strategy...")
    result = client.auto_resolve_conflicts(user_id, strategy="auto_merge")

    print(f"✓ Merged {result['merged_count']} memory/memories")

    # Step 5: Examine result
    print("\nStep 5: Final result:")
    memories = client.get_memories(user_id, limit=1)

    if memories:
        merged = memories[0]
        print(f"   Text: {merged.text}")
        print(f"   Importance: {merged.importance}")
        print(f"   Confidence: {merged.confidence:.2f}")

        # Get provenance
        lineage = client.get_memory_lineage(merged.id)
        if lineage:
            print("\n   Provenance:")
            print(f"     Source: {lineage['source']}")
            print(f"     Parent memories: {len(lineage['parent_memory_ids'])}")
            print(f"     Total citations: {len(lineage['citations'])}")
            print(f"     Transformations: {len(lineage['transformations'])}")

            for citation in lineage["citations"]:
                print("\n     Citation:")
                print(f"       Type: {citation['source_type']}")
                print(f"       Text: {citation.get('source_text', 'N/A')[:50]}...")

        # Quality
        quality = client.assess_memory_quality(merged.id, use_llm=False)
        print(f"\n   Quality score: {quality['overall_score']:.2f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print(" HippocampAI - Conflict Resolution & Provenance Demo")
    print("=" * 60)

    try:
        demo_basic_conflict_resolution()
        demo_resolution_strategies()
        demo_provenance_tracking()
        demo_quality_assessment()
        demo_provenance_chains()
        demo_complete_workflow()

        print_section("Demo Complete!")
        print("✅ All features demonstrated successfully")
        print("\nKey features shown:")
        print("  • Automatic conflict detection")
        print("  • Multiple resolution strategies")
        print("  • Provenance tracking")
        print("  • Quality assessment")
        print("  • Citation management")
        print("  • Provenance chains")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
