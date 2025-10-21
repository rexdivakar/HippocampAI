"""Hybrid retrieval demonstration.

This shows how HippocampAI combines vector search, BM25, and reranking.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - Hybrid Retrieval Example")
print("=" * 60)

# Initialize memory client
print("\n1. Initializing memory client...")
client = MemoryClient()
print("   ✓ Client initialized")

user_id = "carol"

# Store diverse memories to test retrieval
print("\n2. Storing diverse memories...")

memories_data = [
    ("I prefer vegetarian food and avoid meat", "preference", 8.0),
    ("I am allergic to peanuts", "fact", 10.0),
    ("I love Italian cuisine, especially pasta", "preference", 7.0),
    ("I went to a great sushi restaurant last week", "event", 6.0),
    ("I want to learn to cook French food", "goal", 7.5),
    ("I work as a chef at a restaurant", "fact", 8.0),
    ("I enjoy baking desserts on weekends", "habit", 6.5),
    ("I am lactose intolerant", "fact", 9.0),
]

for text, mem_type, importance in memories_data:
    memory = client.remember(text=text, user_id=user_id, type=mem_type, importance=importance)
    print(f"   ✓ Stored: {text[:50]}...")

# Rebuild BM25 index for hybrid retrieval
print("\n3. Rebuilding BM25 index for hybrid retrieval...")
client.retriever.rebuild_bm25(user_id)
print("   ✓ BM25 index ready")

# Test different queries
print("\n4. Testing hybrid retrieval...")

queries = [
    "What food does Carol like?",
    "What are Carol's dietary restrictions?",
    "What does Carol do for work?",
    "What are Carol's cooking interests?",
]

for query in queries:
    print(f"\n   Query: '{query}'")
    results = client.recall(query=query, user_id=user_id, k=3)

    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. {result.memory.text}")
        print(f"      Type: {result.memory.type.value}")
        print(f"      Final Score: {result.score:.3f}")
        print("      Score Breakdown:")
        print(f"        - Similarity: {result.breakdown['sim']:.3f}")
        print(f"        - Rerank: {result.breakdown['rerank']:.3f}")
        print(f"        - Recency: {result.breakdown['recency']:.3f}")
        print(f"        - Importance: {result.breakdown['importance']:.3f}")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)
print("\nNote: Hybrid retrieval combines:")
print("  • Vector similarity (semantic search)")
print("  • BM25 (keyword matching)")
print("  • Cross-encoder reranking (precision)")
print("  • Recency and importance weighting")
