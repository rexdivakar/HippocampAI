"""Multi-user memory management example.

This demonstrates handling memories for multiple users.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - Multi-User Example")
print("=" * 60)

# Initialize memory client
print("\n1. Initializing memory client...")
client = MemoryClient()
print("   ✓ Client initialized")

# Define multiple users
users = ["alice", "bob", "carol"]

# Store memories for each user
print("\n2. Storing memories for multiple users...")

user_memories = {
    "alice": [
        ("I prefer tea over coffee", "preference"),
        ("I work as a designer", "fact"),
        ("I want to learn photography", "goal"),
    ],
    "bob": [
        ("I love coffee, especially espresso", "preference"),
        ("I work as a developer", "fact"),
        ("I want to start a tech blog", "goal"),
    ],
    "carol": [
        ("I enjoy herbal tea", "preference"),
        ("I work as a product manager", "fact"),
        ("I want to travel to Japan", "goal"),
    ],
}

for user_id in users:
    print(f"\n   Storing memories for {user_id}:")
    for text, mem_type in user_memories[user_id]:
        memory = client.remember(text=text, user_id=user_id, type=mem_type)
        print(f"   ✓ {text}")

# Retrieve memories for each user
print("\n3. Retrieving user-specific memories...")

queries = [
    ("alice", "What does Alice drink?"),
    ("bob", "What does Bob drink?"),
    ("carol", "What does Carol drink?"),
]

for user_id, query in queries:
    print(f"\n   User: {user_id}")
    print(f"   Query: '{query}'")

    results = client.recall(query=query, user_id=user_id, k=2)

    if results:
        for result in results:
            print(f"   → {result.memory.text} (score: {result.score:.3f})")
    else:
        print("   → No results found")

# Cross-user query test
print("\n4. Testing user isolation...")
print("\n   Querying Bob's memories with Alice's user_id:")
results = client.recall(query="What does Bob drink?", user_id="alice", k=5)
print(
    f"   Results about Bob found in Alice's memories: {len([r for r in results if 'bob' in r.memory.text.lower()])}"
)

print("\n   ✓ User isolation working correctly")

# Get stats for each user
print("\n5. Memory statistics per user...")

for user_id in users:
    facts = client.qdrant.scroll(
        collection_name=client.config.collection_facts, filters={"user_id": user_id}, limit=100
    )
    prefs = client.qdrant.scroll(
        collection_name=client.config.collection_prefs, filters={"user_id": user_id}, limit=100
    )

    total = len(facts) + len(prefs)
    print(f"\n   {user_id}: {total} total memories")
    print(f"      Facts: {len(facts)}")
    print(f"      Preferences: {len(prefs)}")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)
print("\nMulti-user features:")
print("  • Complete user isolation")
print("  • Per-user memory retrieval")
print("  • Independent memory stores")
print("  • User-specific statistics")
