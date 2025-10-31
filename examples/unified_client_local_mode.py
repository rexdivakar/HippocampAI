"""Example: Using UnifiedMemoryClient in LOCAL mode.

This example shows how to use the UnifiedMemoryClient with local mode,
which connects directly to Qdrant/Redis/Ollama for maximum performance.
"""

from hippocampai import UnifiedMemoryClient


def main() -> None:
    """Demonstrate local mode usage."""
    print("=== UnifiedMemoryClient - LOCAL Mode ===\n")

    # Initialize client in LOCAL mode (default)
    # This connects directly to Qdrant/Redis/Ollama
    client = UnifiedMemoryClient(mode="local")

    print(f"Mode: {client.mode}")
    print("Connected to local Qdrant/Redis/Ollama\n")

    # 1. Store a memory
    print("1. Storing a memory...")
    memory = client.remember(
        text="User prefers dark mode and large fonts for better readability",
        user_id="user123",
        tags=["preferences", "ui"],
        importance=0.8,
    )
    print(f"   Created: {memory.id}")
    print(f"   Text: {memory.text}\n")

    # 2. Semantic search
    print("2. Semantic search...")
    results = client.recall(query="What are the user's UI preferences?", user_id="user123", limit=3)
    for i, result in enumerate(results, 1):
        print(f"   Result {i}: {result.memory.text}")
        print(f"   Score: {result.score:.3f}\n")

    # 3. Get memory by ID
    print("3. Get memory by ID...")
    retrieved = client.get_memory(memory.id)
    if retrieved:
        print(f"   Retrieved: {retrieved.text}\n")

    # 4. Update memory
    print("4. Updating memory...")
    updated = client.update_memory(
        memory_id=memory.id,
        text="User strongly prefers dark mode with extra large fonts",
        importance=0.9,
    )
    if updated:
        print(f"   Updated: {updated.text}")
        print(f"   New importance: {updated.importance}\n")

    # 5. Get all memories
    print("5. Getting all memories...")
    memories = client.get_memories(user_id="user123")
    print(f"   Total memories: {len(memories)}\n")

    # 6. Cleanup
    print("6. Cleanup...")
    deleted = client.delete_memory(memory.id)
    print(f"   Deleted: {deleted}\n")

    print("✓ Local mode demo complete!")
    print("  - Direct connection to storage")
    print("  - Maximum performance (5-15ms latency)")
    print("  - Full feature access")


if __name__ == "__main__":
    main()
