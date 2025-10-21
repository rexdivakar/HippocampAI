"""Conversation memory extraction example.

This demonstrates how to extract memories from conversation text.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - Conversation Extraction Example")
print("=" * 60)

# Initialize memory client
print("\n1. Initializing memory client...")
client = MemoryClient()
print("   ✓ Client initialized")

user_id = "bob"
session_id = "session_001"

# Example conversations
conversations = [
    """
    User: I really enjoy drinking green tea in the morning.
    Assistant: That's great! Green tea is healthy.
    User: Yes, and I usually have it without sugar.
    """,
    """
    User: I work as a data scientist.
    Assistant: Interesting! What kind of projects do you work on?
    User: Mostly machine learning and predictive analytics.
    """,
    """
    User: I want to run a marathon next year.
    Assistant: That's an ambitious goal!
    User: I know, I need to start training soon.
    """,
]

# Extract memories from conversations
print("\n2. Extracting memories from conversations...")

for i, conversation in enumerate(conversations, 1):
    print(f"\n   Processing conversation {i}...")
    memories = client.extract_from_conversation(
        conversation=conversation, user_id=user_id, session_id=session_id
    )

    print(f"   ✓ Extracted {len(memories)} memories:")
    for mem in memories:
        print(f"      - [{mem.type.value}] {mem.text}")
        print(f"        Importance: {mem.importance}/10")

# Recall extracted memories
print("\n3. Recalling extracted memories...")

queries = [
    "What does Bob like to drink?",
    "What is Bob's profession?",
    "What are Bob's goals?",
]

for query in queries:
    print(f"\n   Query: '{query}'")
    results = client.recall(query=query, user_id=user_id, k=2)

    if results:
        for result in results:
            print(f"   → {result.memory.text} (score: {result.score:.3f})")
    else:
        print("   → No results found")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)
