"""Example usage of the MemoryRetriever class."""

import sys

sys.path.append("..")

from src.embedding_service import EmbeddingService
from src.memory_retriever import MemoryRetriever
from src.memory_store import Category, MemoryStore, MemoryType
from src.qdrant_client import QdrantManager
from src.settings import get_settings


def main():
    print("=== Memory Retriever Example ===\n")

    # Load settings
    settings = get_settings()

    # Initialize services
    print("1. Initializing services...")
    qdrant = QdrantManager(host=settings.qdrant.host, port=settings.qdrant.port)
    qdrant.create_collections()

    embeddings = EmbeddingService(model_name=settings.embedding.model)
    memory_store = MemoryStore(qdrant_manager=qdrant, embedding_service=embeddings)
    retriever = MemoryRetriever(qdrant_manager=qdrant, embedding_service=embeddings)
    print("   Services initialized!\n")

    # Store some test memories
    print("2. Storing test memories...")
    test_memories = [
        {
            "text": "I prefer working in quiet environments with minimal distractions",
            "memory_type": MemoryType.PREFERENCE.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 8,
                "category": Category.WORK.value,
                "session_id": "session_001",
                "confidence": 0.9,
            },
        },
        {
            "text": "I love hiking in the mountains on weekends",
            "memory_type": MemoryType.HABIT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 7,
                "category": Category.PERSONAL.value,
                "session_id": "session_001",
                "confidence": 0.95,
            },
        },
        {
            "text": "I'm learning Python for machine learning and data science",
            "memory_type": MemoryType.GOAL.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 9,
                "category": Category.LEARNING.value,
                "session_id": "session_002",
                "confidence": 1.0,
            },
        },
        {
            "text": "I exercise every morning at 6 AM, usually running or yoga",
            "memory_type": MemoryType.HABIT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 8,
                "category": Category.HEALTH.value,
                "session_id": "session_002",
                "confidence": 0.85,
            },
        },
        {
            "text": "We discussed neural networks and deep learning architectures yesterday",
            "memory_type": MemoryType.CONTEXT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 6,
                "category": Category.LEARNING.value,
                "session_id": "session_003",
                "confidence": 0.8,
            },
        },
    ]

    memory_ids = memory_store.store_batch_memories(test_memories)
    print(f"   Stored {len(memory_ids)} test memories\n")

    # Example 1: Search by query
    print("3. Search by query (vector similarity)...")
    query = "What are my fitness habits?"
    results = retriever.search_memories(query, limit=3)
    print(f"   Query: '{query}'")
    print(f"   Found {len(results)} results:")
    for i, mem in enumerate(results, 1):
        print(f"     {i}. [{mem['similarity_score']:.3f}] {mem['text'][:60]}...")
        print(
            f"        Type: {mem['metadata']['memory_type']}, Category: {mem['metadata']['category']}"
        )
    print()

    # Example 2: Search with filters
    print("4. Search with user_id filter...")
    results = retriever.search_memories(
        query="learning programming",
        limit=5,
        filters={"user_id": "user_123", "category": Category.LEARNING.value},
    )
    print("   Query: 'learning programming' (filtered by user_123, category=learning)")
    print(f"   Found {len(results)} results:")
    for i, mem in enumerate(results, 1):
        print(f"     {i}. [{mem['similarity_score']:.3f}] {mem['text'][:60]}...")
    print()

    # Example 3: Get memory by ID
    print("5. Get memory by ID...")
    if memory_ids:
        memory = retriever.get_memory_by_id(memory_ids[0])
        if memory:
            print(f"   Memory ID: {memory['memory_id']}")
            print(f"   Text: {memory['text']}")
            print(f"   Type: {memory['metadata']['memory_type']}")
            print(f"   Importance: {memory['metadata']['importance']}")
    print()

    # Example 4: Get memories by filter (no vector search)
    print("6. Get memories by filter (high importance)...")
    results = retriever.get_memories_by_filter(
        filters={"user_id": "user_123", "min_importance": 7}, limit=10
    )
    print("   Filter: user_123, importance >= 7")
    print(f"   Found {len(results)} results:")
    for i, mem in enumerate(results, 1):
        print(f"     {i}. [Importance: {mem['metadata']['importance']}] {mem['text'][:60]}...")
    print()

    # Example 5: Get recent memories
    print("7. Get recent memories...")
    recent = retriever.get_recent_memories(user_id="user_123", limit=3)
    print(f"   Found {len(recent)} recent memories:")
    for i, mem in enumerate(recent, 1):
        print(f"     {i}. {mem['text'][:60]}...")
        print(f"        Timestamp: {mem['metadata']['timestamp']}")
    print()

    # Example 6: Get important memories
    print("8. Get important memories...")
    important = retriever.get_important_memories(user_id="user_123", min_importance=8, limit=5)
    print(f"   Found {len(important)} important memories (importance >= 8):")
    for i, mem in enumerate(important, 1):
        print(f"     {i}. [Importance: {mem['metadata']['importance']}] {mem['text'][:60]}...")
    print()

    # Example 7: Search across specific collections
    print("9. Search in specific collection...")
    results = retriever.search_memories(
        query="daily routine",
        limit=5,
        collections=["knowledge_base"],  # Only search in knowledge_base
    )
    print("   Query: 'daily routine' (only in knowledge_base collection)")
    print(f"   Found {len(results)} results:")
    for i, mem in enumerate(results, 1):
        print(f"     {i}. [{mem['similarity_score']:.3f}] {mem['text'][:60]}...")
    print()

    # Example 8: Filter by memory type
    print("10. Filter by memory type...")
    results = retriever.get_memories_by_filter(
        filters={"user_id": "user_123", "memory_type": MemoryType.HABIT.value}, limit=10
    )
    print("   Filter: user_123, memory_type=habit")
    print(f"   Found {len(results)} results:")
    for i, mem in enumerate(results, 1):
        print(f"     {i}. {mem['text'][:60]}...")
    print()

    # Close connection
    qdrant.close()
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
