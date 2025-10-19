"""Example usage of the MemoryStore class."""

import sys

sys.path.append("..")

from src.embedding_service import EmbeddingService
from src.memory_store import Category, MemoryStore, MemoryType
from src.qdrant_client import QdrantManager
from src.settings import get_settings


def main():
    print("=== Memory Store Example ===\n")

    # Load settings
    settings = get_settings()

    # Initialize services
    print("1. Initializing services...")
    qdrant = QdrantManager(host=settings.qdrant.host, port=settings.qdrant.port)
    qdrant.create_collections()

    embeddings = EmbeddingService(model_name=settings.embedding.model)
    memory_store = MemoryStore(qdrant_manager=qdrant, embedding_service=embeddings)
    print("   Services initialized!\n")

    # Store a single memory
    print("2. Storing a single memory...")
    memory_id = memory_store.store_memory(
        text="I prefer my coffee with oat milk and no sugar",
        memory_type=MemoryType.PREFERENCE.value,
        metadata={
            "user_id": "user_123",
            "importance": 7,
            "category": Category.PERSONAL.value,
            "session_id": "session_001",
            "confidence": 0.95,
        },
    )
    print(f"   Stored memory: {memory_id}\n")

    # Store batch memories
    print("3. Storing batch memories...")
    memories = [
        {
            "text": "User's birthday is on March 15th",
            "memory_type": MemoryType.FACT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 9,
                "category": Category.PERSONAL.value,
                "session_id": "session_001",
                "confidence": 1.0,
            },
        },
        {
            "text": "User wants to learn Python for data science",
            "memory_type": MemoryType.GOAL.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 8,
                "category": Category.LEARNING.value,
                "session_id": "session_001",
                "confidence": 0.9,
            },
        },
        {
            "text": "User exercises every morning at 6 AM",
            "memory_type": MemoryType.HABIT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 6,
                "category": Category.HEALTH.value,
                "session_id": "session_002",
                "confidence": 0.85,
            },
        },
        {
            "text": "We discussed machine learning algorithms in the last conversation",
            "memory_type": MemoryType.CONTEXT.value,
            "metadata": {
                "user_id": "user_123",
                "importance": 5,
                "category": Category.LEARNING.value,
                "session_id": "session_002",
                "confidence": 0.8,
            },
        },
    ]

    batch_ids = memory_store.store_batch_memories(memories)
    print(f"   Stored {len(batch_ids)} memories:")
    for i, mem_id in enumerate(batch_ids, 1):
        print(f"     {i}. {mem_id}")
    print()

    # Retrieve a memory
    print("4. Retrieving a memory...")
    retrieved = memory_store.get_memory(memory_id)
    if retrieved:
        print(f"   Text: {retrieved['text']}")
        print(f"   Type: {retrieved['memory_type']}")
        print(f"   Importance: {retrieved['importance']}")
        print(f"   Category: {retrieved['category']}")
        print(f"   Confidence: {retrieved['confidence']}")
        print(f"   Timestamp: {retrieved['timestamp']}")
    print()

    # List all collections and their stats
    print("5. Collection statistics...")
    collections = qdrant.list_collections()
    for coll in collections:
        info = qdrant.get_collection_info(coll)
        print(f"   {coll}:")
        print(f"     Points: {info['points_count']}")
        print(f"     Status: {info['status']}")
    print()

    # Embedding cache stats
    print("6. Embedding cache statistics...")
    cache_stats = embeddings.get_cache_stats()
    for key, value in cache_stats.items():
        print(f"   {key}: {value}")

    # Close connection
    qdrant.close()
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
