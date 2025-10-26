"""Demo: Smart Memory Updates and Semantic Clustering.

This example demonstrates the advanced memory management features:
- Smart memory updates (merge, update, skip decisions)
- Conflict resolution
- Auto-categorization and tagging
- Semantic clustering by topics
- Topic shift detection
"""

import os

from hippocampai import EnhancedMemoryClient

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: Please set GROQ_API_KEY environment variable")
    print("export GROQ_API_KEY='your-api-key-here'")
    exit(1)

# Initialize client
print("=" * 60)
print("Smart Memory & Clustering Demo")
print("=" * 60)

client = EnhancedMemoryClient(provider="groq")
user_id = "smart_demo_user"

print("\n1. SMART MEMORY UPDATES")
print("-" * 60)

# Store initial memory
print("\n→ Storing: 'I love coffee'")
mem1 = client.remember(text="I love coffee", user_id=user_id, type="preference")
print(f"  ✓ Stored with tags: {mem1.tags}")
print(f"  ✓ Category: {mem1.type}")
print(f"  ✓ Confidence: {mem1.confidence}")

# Try to store similar memory - should intelligently merge/update
print("\n→ Storing similar: 'I really enjoy drinking coffee in the morning'")
mem2 = client.remember(
    text="I really enjoy drinking coffee in the morning", user_id=user_id, type="preference"
)
print(f"  ✓ Action taken: {mem2.text}")
print(f"  ✓ Tags: {mem2.tags}")

# Store conflicting information - should detect and resolve
print("\n→ Storing conflicting: 'I don't like coffee anymore'")
mem3 = client.remember(text="I don't like coffee anymore", user_id=user_id, type="preference")
print(f"  ✓ Conflict resolved: {mem3.text}")
print(f"  ✓ Confidence: {mem3.confidence}")

print("\n2. AUTO-CATEGORIZATION & TAGGING")
print("-" * 60)

# Store various memories that will be auto-categorized
memories_to_store = [
    "I work at Google as a software engineer",
    "I want to learn Python and machine learning",
    "I usually exercise in the morning",
    "I visited Paris last summer",
    "I need to finish the project by Friday",
]

print("\nStoring memories with auto-categorization:")
for text in memories_to_store:
    mem = client.remember(text=text, user_id=user_id)
    print(f"  '{text[:40]}...'")
    print(f"    → Category: {mem.type.value}, Tags: {mem.tags}")

print("\n3. SEMANTIC CLUSTERING")
print("-" * 60)

# Cluster all memories by topic
clusters = client.cluster_user_memories(user_id, max_clusters=5)
print(f"\nFound {len(clusters)} topic clusters:")
for i, cluster in enumerate(clusters, 1):
    print(f"\n  Cluster {i}: {cluster.topic}")
    print(f"  Common tags: {cluster.tags}")
    print(f"  Memories: {len(cluster.memories)}")
    for mem in cluster.memories[:3]:  # Show first 3
        print(f"    - {mem.text[:50]}...")

print("\n4. TOPIC SHIFT DETECTION")
print("-" * 60)

# Add memories to create a topic shift
print("\nAdding work-related memories...")
for text in [
    "I need to review the code before the meeting",
    "The team meeting is at 2 PM today",
    "I should prepare the presentation slides",
]:
    client.remember(text=text, user_id=user_id)

# Detect topic shift
topic = client.detect_topic_shift(user_id, window_size=5)
if topic:
    print(f"  ✓ Topic shift detected! New topic: {topic}")
else:
    print("  No significant topic shift detected")

print("\n5. MEMORY RECONCILIATION")
print("-" * 60)

# Reconcile all memories (resolve conflicts)
print("\nReconciling memories to resolve conflicts...")
reconciled = client.reconcile_user_memories(user_id)
print(f"  ✓ Reconciled {len(reconciled)} memories")

# Show remaining memories by type
memories = client.get_memories(user_id, limit=100)
by_type = {}
for mem in memories:
    t = mem.type.value
    by_type[t] = by_type.get(t, 0) + 1

print("\nFinal memory distribution:")
for mem_type, count in sorted(by_type.items()):
    print(f"  {mem_type}: {count} memories")

print("\n6. MEMORY STATISTICS")
print("-" * 60)

stats = client.get_memory_statistics(user_id)
print(f"\nTotal memories: {stats['total_memories']}")
print(f"Total characters: {stats['total_characters']}")
print(f"Average memory size: {stats['avg_memory_size_chars']:.1f} chars")

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
