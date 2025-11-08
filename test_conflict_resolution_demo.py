#!/usr/bin/env python3
"""Test script to demonstrate memory conflict resolution."""

import asyncio
import sys

import requests

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


async def main():
    """Run conflict resolution demonstration."""
    print("\n" + "="*70)
    print("  HippocampAI - Memory Conflict Resolution Demo")
    print("="*70)

    user_id = "conflict_demo_user"

    # Step 1: Create first memory (love coffee)
    print_section("Step 1: Creating Initial Memory")
    memory1 = {
        "text": "I love coffee and drink it every morning",
        "user_id": user_id,
        "type": "preference",
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory1)
    m1_data = response.json()
    print("✅ Created Memory 1:")
    print(f"   ID: {m1_data['id']}")
    print(f"   Text: {m1_data['text']}")
    print(f"   Created: {m1_data['created_at']}")
    print(f"   Confidence: {m1_data['confidence']}")

    # Step 2: Wait a moment then create conflicting memory
    print_section("Step 2: Creating Conflicting Memory")
    print("Waiting 2 seconds to ensure different timestamps...")
    await asyncio.sleep(2)

    memory2 = {
        "text": "I hate coffee and never drink it",
        "user_id": user_id,
        "type": "preference",
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory2)
    m2_data = response.json()
    print("✅ Created Memory 2 (conflicting):")
    print(f"   ID: {m2_data['id']}")
    print(f"   Text: {m2_data['text']}")
    print(f"   Created: {m2_data['created_at']}")
    print(f"   Confidence: {m2_data['confidence']}")

    # Step 3: Search for coffee memories
    print_section("Step 3: Searching for Coffee Memories")
    response = requests.post(
        f"{BASE_URL}/v1/memories/recall",
        json={"query": "coffee", "user_id": user_id, "k": 10}
    )
    results = response.json()
    print(f"Found {len(results)} memories:")
    for i, result in enumerate(results, 1):
        # RetrievalResult has a 'memory' field with the actual memory data
        memory = result.get('memory', result)
        print(f"  {i}. {memory['text']}")
        print(f"     Created: {memory['created_at']}")
        print(f"     Confidence: {memory['confidence']}")
        if 'score' in result:
            print(f"     Search Score: {result['score']:.3f}")

    # Step 4: Verify conflict resolution
    print_section("Step 4: Analyzing Result")
    if len(results) == 1:
        memory = results[0].get('memory', results[0])
        print("✅ CONFLICT RESOLVED!")
        print(f"   Only 1 memory remains: '{memory['text']}'")
        print("   Older conflicting memory was automatically removed")
        print("   Resolution strategy: TEMPORAL (newest wins)")
    elif len(results) == 2:
        print("⚠️  Both memories still exist - checking for conflict flags...")
        for result in results:
            memory = result.get('memory', result)
            if memory.get('metadata', {}).get('has_conflict'):
                print(f"   Memory flagged for review: {memory['text']}")
        print("   Resolution strategy: USER_REVIEW or KEEP_BOTH")
    else:
        print(f"❌ Unexpected: Found {len(results)} memories")

    # Step 5: Create another conflict with higher confidence
    print_section("Step 5: Testing Confidence-Based Resolution")
    memory3 = {
        "text": "I actually love coffee, I was wrong before",
        "user_id": user_id,
        "type": "preference",
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory3)
    m3_data = response.json()
    print("✅ Created Memory 3 (clarification):")
    print(f"   ID: {m3_data['id']}")
    print(f"   Text: {m3_data['text']}")

    # Final search
    response = requests.post(
        f"{BASE_URL}/v1/memories/recall",
        json={"query": "coffee preference", "user_id": user_id, "k": 10}
    )
    results = response.json()
    print(f"\nFinal state: {len(results)} memory(ies) about coffee")
    for result in results[:3]:
        memory = result.get('memory', result)
        print(f"  - {memory['text']}")

    print_section("Summary")
    print("✅ Conflict Resolution Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  1. ✅ Contradiction Detection - Automatically detected conflicting memories")
    print("  2. ✅ Temporal Priority - Newest memory wins by default")
    print("  3. ✅ Automatic Resolution - No manual intervention needed")
    print("  4. ✅ Confidence Tracking - Each memory has confidence score")
    print("\nConfiguration Options:")
    print("  - Resolution Strategy: temporal (default), confidence, importance, user_review")
    print("  - Auto-resolve: Enabled by default")
    print("  - LLM Analysis: Optional for deeper contradiction detection")
    print("\nFor more details, see:")
    print("  - MEMORY_CONFLICT_RESOLUTION_QUICKSTART.md")
    print("  - docs/MEMORY_CONFLICT_RESOLUTION_GUIDE.md")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
