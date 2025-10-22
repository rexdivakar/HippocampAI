"""Demo: Semantic Clustering & Auto-Categorization.

This example demonstrates:
- Automatic tag suggestion
- Auto category assignment
- Memory clustering by topics
- Similar memory detection
- Topic shift detection
- Automatic memory enrichment
"""

import os
from hippocampai import EnhancedMemoryClient, MemoryType

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: Please set GROQ_API_KEY environment variable")
    print("export GROQ_API_KEY='your-api-key-here'")
    exit(1)

# Initialize client
print("=" * 70)
print("Semantic Clustering & Auto-Categorization Demo")
print("=" * 70)

client = EnhancedMemoryClient(provider="groq")
user_id = "alice"

print("\n1. AUTO TAG SUGGESTION")
print("-" * 70)

# Store memories with automatic tag suggestion
memories_data = [
    "I love eating pizza and pasta at Italian restaurants",
    "I work as a software engineer at Google",
    "I want to learn machine learning and AI",
    "I exercise at the gym every Monday and Wednesday",
    "I'm planning a vacation to Japan next summer",
]

print("\nStoring memories with auto-suggested tags:\n")
stored_memories = []
for text in memories_data:
    memory = client.remember(text, user_id)
    stored_memories.append(memory)
    print(f"• \"{text[:50]}...\"")
    print(f"  Auto-tags: {memory.tags}")
    print(f"  Category: {memory.type.value}")
    print()

print("\n2. MEMORY CLUSTERING BY TOPICS")
print("-" * 70)

# Add more memories for better clustering
additional_memories = [
    "My project deadline is next Friday",
    "I enjoy hiking in the mountains",
    "I need to buy groceries from the store",
    "I'm reading a book about Python programming",
    "I had dinner at a great sushi restaurant",
    "I need to finish the quarterly report for work",
    "I love trying different cuisines",
    "I want to improve my coding skills",
]

print("\nAdding more memories for clustering...")
for text in additional_memories:
    client.remember(text, user_id)
print(f"✓ Added {len(additional_memories)} more memories")

# Cluster memories
print("\nClustering memories by topics...")
clusters = client.cluster_user_memories(user_id, max_clusters=8)

print(f"\nFound {len(clusters)} topic clusters:\n")
for i, cluster in enumerate(clusters, 1):
    print(f"{i}. Topic: '{cluster.topic.upper()}'")
    print(f"   Memories: {len(cluster.memories)}")
    print(f"   Common tags: {cluster.tags[:5]}")
    print(f"   Examples:")
    for mem in cluster.memories[:2]:  # Show first 2 memories
        print(f"     - {mem.text[:60]}...")
    print()

print("\n3. AUTO CATEGORY ASSIGNMENT")
print("-" * 70)

# Test category detection with different patterns
test_cases = [
    ("I love chocolate ice cream", "Expected: preference"),
    ("I want to learn Spanish", "Expected: goal"),
    ("I exercise every morning", "Expected: habit"),
    ("I went to the concert yesterday", "Expected: event"),
    ("My name is Alice", "Expected: fact"),
]

print("\nTesting auto-category assignment:\n")
from hippocampai.models.memory import Memory
from hippocampai.pipeline.semantic_clustering import SemanticCategorizer
from datetime import datetime, timezone

categorizer = SemanticCategorizer()

for text, expected in test_cases:
    # Create temporary memory for testing
    temp_memory = Memory(
        id="temp",
        text=text,
        user_id=user_id,
        type=MemoryType.FACT,  # Start with wrong type
        created_at=datetime.now(timezone.utc),
        importance=0.5,
        tags=[]
    )

    assigned_type = categorizer.assign_category(temp_memory)
    print(f"Text: \"{text}\"")
    print(f"  {expected}")
    print(f"  Assigned: {assigned_type.value}")
    print()

print("\n4. SIMILAR MEMORY DETECTION")
print("-" * 70)

# Store a new memory
new_memory = client.remember(
    "I enjoy eating pizza at Italian places",
    user_id
)

print(f"\nNew memory: \"{new_memory.text}\"\n")
print("Finding similar existing memories...")

# Get all memories and find similar ones
all_memories = client.get_memories(user_id, limit=100)
similar = categorizer.find_similar_memories(
    new_memory,
    all_memories,
    similarity_threshold=0.3  # Lower threshold to see more results
)

if similar:
    print(f"\nFound {len(similar)} similar memories:\n")
    for mem, score in similar[:3]:  # Top 3
        print(f"• Similarity: {score:.2f}")
        print(f"  \"{mem.text[:60]}...\"")
        print()
else:
    print("No similar memories found")

print("\n5. TOPIC SHIFT DETECTION")
print("-" * 70)

print("\nSimulating a conversation with topic changes...\n")

# Simulate conversation about work
print("→ Talking about work...")
for text in [
    "I have a meeting at 3pm",
    "My project is going well",
    "I need to review the code",
]:
    client.remember(text, user_id)
    print(f"   {text}")

# Check for topic shift
shift = client.detect_topic_shift(user_id, window_size=5)
print(f"\nTopic shift detected: {shift or 'None (still on same topic)'}")

# Now switch to food topic
print("\n→ Switching to food...")
for text in [
    "I'm hungry for lunch",
    "Let's order some pizza",
    "I love Italian food",
]:
    client.remember(text, user_id)
    print(f"   {text}")

# Check again
shift = client.detect_topic_shift(user_id, window_size=5)
print(f"\nTopic shift detected: {shift or 'None'}")

print("\n6. TAG SUGGESTION FOR EXISTING MEMORY")
print("-" * 70)

# Get a memory without many tags
sample_memory = stored_memories[0]

print(f"\nMemory: \"{sample_memory.text}\"")
print(f"Current tags: {sample_memory.tags}")

# Suggest additional tags
suggested = client.suggest_memory_tags(sample_memory, max_tags=8)
print(f"Suggested tags: {suggested}")

# Find new tag suggestions (not already present)
new_suggestions = [tag for tag in suggested if tag not in sample_memory.tags]
print(f"New suggestions: {new_suggestions}")

print("\n7. MEMORY STATISTICS BY TOPIC")
print("-" * 70)

# Group memories by their main topics
topic_groups = {}
for cluster in clusters:
    topic_groups[cluster.topic] = len(cluster.memories)

print("\nMemories per topic:")
for topic, count in sorted(topic_groups.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * count
    print(f"  {topic:15s} {bar} ({count})")

print("\n" + "=" * 70)
print("Semantic Clustering Demo Completed!")
print("=" * 70)
print("\nKey Takeaways:")
print("  • Memories are automatically enriched with tags and categories")
print("  • Similar memories are detected and can be merged/deduplicated")
print("  • Memories are clustered by semantic topics")
print("  • Topic shifts are automatically detected")
print("  • Perfect for organizing large memory collections!")
