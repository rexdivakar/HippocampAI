"""Basic HippocampAI usage example.

This demonstrates the core remember/recall workflow.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - Basic Usage Example")
print("=" * 60)

# Initialize memory client
print("\n1. Initializing memory client...")
client = MemoryClient()
print("   ✓ Client initialized")

# Store some memories
print("\n2. Storing memories...")

user_id = "alice"

# Store a preference
memory1 = client.remember(
    text="I prefer oat milk in my coffee",
    user_id=user_id,
    type="preference",
    importance=8.0,
)
print(f"   ✓ Stored preference: {memory1.id[:8]}...")

# Store a fact
memory2 = client.remember(
    text="I work as a software engineer at TechCorp",
    user_id=user_id,
    type="fact",
    importance=7.0,
)
print(f"   ✓ Stored fact: {memory2.id[:8]}...")

# Store a goal
memory3 = client.remember(
    text="I want to learn machine learning this year",
    user_id=user_id,
    type="goal",
    importance=9.0,
)
print(f"   ✓ Stored goal: {memory3.id[:8]}...")

# Store an event
memory4 = client.remember(
    text="I went hiking in Yosemite last weekend",
    user_id=user_id,
    type="event",
    importance=6.0,
)
print(f"   ✓ Stored event: {memory4.id[:8]}...")

# Recall memories
print("\n3. Recalling memories...")

# Query about coffee
results = client.recall(query="How does Alice like her coffee?", user_id=user_id, k=3)

print("\n   Query: 'How does Alice like her coffee?'")
print(f"   Found {len(results)} relevant memories:")
for i, result in enumerate(results, 1):
    print(f"\n   {i}. {result.memory.text}")
    print(f"      Type: {result.memory.type.value}")
    print(f"      Score: {result.score:.3f}")
    print(f"      Importance: {result.memory.importance}/10")

# Query about work
print("\n" + "-" * 60)
results = client.recall(query="Where does Alice work?", user_id=user_id, k=3)

print("\n   Query: 'Where does Alice work?'")
print(f"   Found {len(results)} relevant memories:")
for i, result in enumerate(results, 1):
    print(f"\n   {i}. {result.memory.text}")
    print(f"      Type: {result.memory.type.value}")
    print(f"      Score: {result.score:.3f}")

# Query about goals
print("\n" + "-" * 60)
results = client.recall(query="What are Alice's goals?", user_id=user_id, k=3)

print("\n   Query: 'What are Alice's goals?'")
print(f"   Found {len(results)} relevant memories:")
for i, result in enumerate(results, 1):
    print(f"\n   {i}. {result.memory.text}")
    print(f"      Type: {result.memory.type.value}")
    print(f"      Score: {result.score:.3f}")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)
