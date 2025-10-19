"""Advanced example showcasing deduplication, updates, and importance scoring."""

import sys

sys.path.append("..")

from hippocampai.embedding_service import EmbeddingService
from hippocampai.importance_scorer import ImportanceScorer
from hippocampai.memory_deduplicator import MemoryDeduplicator
from hippocampai.memory_retriever import MemoryRetriever
from hippocampai.memory_store import Category, MemoryStore, MemoryType
from hippocampai.memory_updater import MemoryUpdater
from hippocampai.qdrant_client import QdrantManager
from hippocampai.settings import get_settings


def main():
    print("=== Advanced Memory Management Example ===\n")

    # Load settings
    settings = get_settings()

    # Check for API key based on configured provider
    provider = settings.llm.provider.lower()
    has_api_key = False

    if provider == "anthropic" and settings.llm.anthropic_api_key:
        has_api_key = True
    elif provider == "openai" and settings.llm.openai_api_key:
        has_api_key = True
    elif provider == "groq" and settings.llm.groq_api_key:
        has_api_key = True

    if not has_api_key:
        print(f"ERROR: {provider.upper()}_API_KEY not set in .env file!")
        print("Please add your API key to .env or choose a different LLM_PROVIDER")
        return

    # Initialize services
    print("1. Initializing services...")
    qdrant = QdrantManager(host=settings.qdrant.host, port=settings.qdrant.port)
    qdrant.create_collections()

    embeddings = EmbeddingService(model_name=settings.embedding.model)
    memory_store = MemoryStore(qdrant_manager=qdrant, embedding_service=embeddings)
    retriever = MemoryRetriever(qdrant_manager=qdrant, embedding_service=embeddings)
    deduplicator = MemoryDeduplicator(retriever=retriever, embedding_service=embeddings)
    updater = MemoryUpdater(
        qdrant_manager=qdrant, retriever=retriever, embedding_service=embeddings
    )
    scorer = ImportanceScorer()
    print("   All services initialized!\n")

    # Store initial memory
    print("2. Storing initial memory...")
    memory_id = memory_store.store_memory(
        text="I prefer drinking coffee with almond milk",
        memory_type=MemoryType.PREFERENCE.value,
        metadata={
            "user_id": "user_789",
            "importance": 7,
            "category": Category.PERSONAL.value,
            "session_id": "session_adv_001",
            "confidence": 0.9,
        },
    )
    print(f"   Stored memory: {memory_id}\n")

    # Example 1: Check for duplicates before storing
    print("3. Testing deduplication...")
    new_memory = {
        "text": "I like coffee with almond milk",
        "memory_type": MemoryType.PREFERENCE.value,
        "importance": 7,
    }

    result = deduplicator.process_new_memory(
        new_memory=new_memory, user_id="user_789", similarity_threshold=0.85, auto_decide=True
    )

    print(f"   Action: {result['action']}")
    print(f"   Found {len(result['duplicates'])} duplicates")
    if result["decision_data"]:
        print(f"   Decision: {result['decision_data']['decision']}")
        print(f"   Reasoning: {result['decision_data']['reasoning']}")
    print()

    # Example 2: Update a memory
    print("4. Updating memory...")
    try:
        success = updater.update_memory(
            memory_id=memory_id,
            new_text="I prefer drinking coffee with oat milk now",
            reason="User changed preference from almond to oat milk",
            new_importance=8,
        )
        print(f"   Update successful: {success}\n")
    except Exception as e:
        print(f"   Update failed: {e}\n")

    # Retrieve updated memory
    updated_memory = retriever.get_memory_by_id(memory_id)
    if updated_memory:
        print("   Updated memory details:")
        print(f"   Text: {updated_memory['text']}")
        print(f"   Version: {updated_memory['metadata'].get('version', 0)}")
        print(f"   Update reason: {updated_memory['metadata'].get('update_reason', 'N/A')}")
        print()

    # Example 3: Test conflict resolution
    print("5. Testing conflict resolution...")

    # Store another memory
    memory_id_2 = memory_store.store_memory(
        text="I exercise in the morning at 7 AM",
        memory_type=MemoryType.HABIT.value,
        metadata={
            "user_id": "user_789",
            "importance": 6,
            "category": Category.HEALTH.value,
            "session_id": "session_adv_002",
            "confidence": 0.85,
        },
    )

    # Create conflicting memory
    conflicting_memory = {
        "text": "I exercise in the evening at 6 PM now",
        "memory_type": MemoryType.HABIT.value,
        "metadata": {
            "user_id": "user_789",
            "importance": 7,
            "category": Category.HEALTH.value,
            "session_id": "session_adv_003",
            "confidence": 0.9,
        },
    }

    old_memory = retriever.get_memory_by_id(memory_id_2)

    try:
        resolution = updater.resolve_conflict(old_memory=old_memory, new_memory=conflicting_memory)

        print(f"   Conflict resolution: {resolution['decision']}")
        print(f"   Reasoning: {resolution['reasoning']}")
        print()

        # Apply the resolution
        result_id = updater.apply_resolution(
            old_memory=old_memory, new_memory=conflicting_memory, resolution=resolution
        )

        if result_id:
            print(f"   Applied resolution, result ID: {result_id}")
        print()

    except Exception as e:
        print(f"   Conflict resolution failed: {e}\n")

    # Example 4: Calculate importance with AI
    print("6. Testing AI-powered importance scoring...")

    test_memory = "I am allergic to peanuts and must avoid them"

    try:
        importance_result = scorer.calculate_importance(
            memory_text=test_memory,
            memory_type=MemoryType.FACT.value,
            user_context="Health and safety information",
        )

        print(f"   Memory: {test_memory}")
        print(f"   AI Score: {importance_result['score']}/10")
        print(f"   Reasoning: {importance_result['reasoning']}")
        print()

    except Exception as e:
        print(f"   Importance scoring failed: {e}\n")

    # Example 5: Test importance decay
    print("7. Testing importance decay...")

    # Get all memories for user
    all_memories = retriever.get_memories_by_filter(filters={"user_id": "user_789"}, limit=20)

    print(f"   Found {len(all_memories)} memories for user_789")

    decay_results = scorer.batch_update_importance(
        memories=all_memories,
        access_counts={memory_id: 5, memory_id_2: 2},  # Simulate access counts
    )

    print(f"   Updated importance for {len(decay_results)} memories:")
    for mem_id, new_score in decay_results.items():
        print(f"     {mem_id[:8]}... -> {new_score:.2f}")
    print()

    # Example 6: Merge memories
    print("8. Testing memory merge...")

    # Create two related memories
    mem1_id = memory_store.store_memory(
        text="I work as a software engineer",
        memory_type=MemoryType.FACT.value,
        metadata={
            "user_id": "user_789",
            "importance": 8,
            "category": Category.WORK.value,
            "session_id": "session_adv_004",
            "confidence": 1.0,
        },
    )

    mem2_id = memory_store.store_memory(
        text="I specialize in backend development with Python",
        memory_type=MemoryType.FACT.value,
        metadata={
            "user_id": "user_789",
            "importance": 7,
            "category": Category.WORK.value,
            "session_id": "session_adv_004",
            "confidence": 0.95,
        },
    )

    try:
        merged_id = updater.merge_memories(
            memory_ids=[mem1_id, mem2_id],
            merged_text="I work as a software engineer specializing in backend development with Python",
            merged_importance=9,
            reason="Combined related work information",
        )

        print(f"   Merged memories into: {merged_id}")

        # Retrieve merged memory
        merged_mem = retriever.get_memory_by_id(merged_id)
        if merged_mem:
            print(f"   Merged text: {merged_mem['text']}")
            print(f"   Importance: {merged_mem['metadata']['importance']}")
            print(f"   Original count: {len(merged_mem['metadata'].get('original_texts', []))}")
        print()

    except Exception as e:
        print(f"   Merge failed: {e}\n")

    # Example 7: Mark memory as outdated
    print("9. Marking memory as outdated...")
    try:
        updater.mark_memory_outdated(memory_id=mem2_id, reason="Merged into another memory")
        print(f"   Marked {mem2_id} as outdated\n")
    except Exception as e:
        print(f"   Failed to mark as outdated: {e}\n")

    # Summary
    print("10. Final summary...")
    final_memories = retriever.get_memories_by_filter(filters={"user_id": "user_789"}, limit=50)

    active_memories = [m for m in final_memories if not m["metadata"].get("outdated", False)]
    outdated_memories = [m for m in final_memories if m["metadata"].get("outdated", False)]

    print(f"   Total memories: {len(final_memories)}")
    print(f"   Active: {len(active_memories)}")
    print(f"   Outdated: {len(outdated_memories)}")

    # Close connection
    print("\n11. Cleaning up...")
    qdrant.close()
    print("   Connection closed")
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
