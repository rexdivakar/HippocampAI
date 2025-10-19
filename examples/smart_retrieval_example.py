"""Example demonstrating smart retrieval, sessions, and consolidation."""

import sys

sys.path.append("..")

from src.embedding_service import EmbeddingService
from src.memory_consolidator import MemoryConsolidator
from src.memory_retriever import MemoryRetriever
from src.memory_store import Category, MemoryStore, MemoryType
from src.memory_updater import MemoryUpdater
from src.qdrant_client import QdrantManager
from src.session_manager import SessionManager
from src.settings import get_settings


def main():
    print("=== Smart Retrieval, Sessions & Consolidation Example ===\n")

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
    session_mgr = SessionManager(
        memory_store=memory_store, retriever=retriever, embedding_service=embeddings
    )
    updater = MemoryUpdater(
        qdrant_manager=qdrant, retriever=retriever, embedding_service=embeddings
    )
    consolidator = MemoryConsolidator(
        retriever=retriever, updater=updater, embedding_service=embeddings
    )
    print("   All services initialized!\n")

    # Store test memories with varied timestamps and importance
    print("2. Storing test memories...")
    test_memories = [
        {
            "text": "I work as a senior software engineer at TechCorp",
            "memory_type": MemoryType.FACT.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 9,
                "category": Category.WORK.value,
                "session_id": "session_001",
                "confidence": 1.0,
            },
        },
        {
            "text": "I specialize in Python backend development",
            "memory_type": MemoryType.FACT.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 8,
                "category": Category.WORK.value,
                "session_id": "session_001",
                "confidence": 0.95,
            },
        },
        {
            "text": "I prefer drinking green tea in the afternoon",
            "memory_type": MemoryType.PREFERENCE.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 5,
                "category": Category.PERSONAL.value,
                "session_id": "session_002",
                "confidence": 0.9,
            },
        },
        {
            "text": "I'm learning machine learning and AI",
            "memory_type": MemoryType.GOAL.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 9,
                "category": Category.LEARNING.value,
                "session_id": "session_002",
                "confidence": 1.0,
            },
        },
        {
            "text": "I run every morning at 6 AM for fitness",
            "memory_type": MemoryType.HABIT.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 7,
                "category": Category.HEALTH.value,
                "session_id": "session_003",
                "confidence": 0.85,
            },
        },
    ]

    memory_ids = memory_store.store_batch_memories(test_memories)
    print(f"   Stored {len(memory_ids)} memories\n")

    # Example 1: Smart search with multi-factor ranking
    print("3. Testing smart search with multi-factor ranking...")
    query = "Tell me about my work and career"

    # Simulate access counts (in production, track these in database)
    access_counts = {memory_ids[0]: 10, memory_ids[1]: 5}

    results = retriever.smart_search(
        query=query, user_id="user_smart", context_type="work", limit=3, access_counts=access_counts
    )

    print(f"   Query: '{query}'")
    print(f"   Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"   {i}. Final Score: {result['final_score']:.3f}")
        print(f"      Text: {result['text']}")
        breakdown = result["score_breakdown"]
        print("      Breakdown:")
        print(f"        - Similarity: {breakdown['similarity']:.3f} (50%)")
        print(f"        - Importance: {breakdown['importance']:.3f} (30%)")
        print(f"        - Recency: {breakdown['recency']:.3f} (20%)")
        print(f"        - Access Boost: {breakdown['access_boost']:.3f}")
        print(f"        - Category Boost: {breakdown['category_boost']:.3f}")
        print()

    # Example 2: Get context for query
    print("4. Getting optimal context for query...")
    context = retriever.get_context_for_query(
        query="What are my learning goals?", user_id="user_smart", max_memories=3
    )

    print(f"   Query: {context['query']}")
    print(f"   Context Type: {context['context_type']}")
    print(f"   Total Memories: {context['memory_count']}")
    print(f"   Total Relevance: {context['total_relevance']:.3f}")
    print("\n   Organized by type:")
    for mem_type, memories in context["memories"].items():
        if memories:
            print(f"     {mem_type}: {len(memories)} memories")
            for mem in memories:
                print(f"       - {mem['text'][:60]}...")
    print()

    # Example 3: Session management
    print("5. Testing session management...")

    session_id = session_mgr.start_session(
        user_id="user_smart", context="Work discussion about project planning"
    )
    print(f"   Started session: {session_id}")

    # Simulate conversation
    session_mgr.add_message(
        session_id=session_id,
        role="user",
        content="I need help planning my machine learning project",
    )

    session_mgr.add_message(
        session_id=session_id,
        role="assistant",
        content="I'd be happy to help! Since you're learning ML, let's start with defining your project goals.",
    )

    session_mgr.add_message(
        session_id=session_id,
        role="user",
        content="I want to build a recommendation system using collaborative filtering",
    )

    session_mgr.add_message(
        session_id=session_id,
        role="assistant",
        content="Great choice! For a recommendation system, you'll need user-item interaction data.",
    )

    # End session and generate summary
    try:
        summary_id = session_mgr.end_session(session_id)
        if summary_id:
            print(f"   Session ended, summary stored: {summary_id}")

            # Retrieve session summary
            summary = retriever.get_memory_by_id(summary_id)
            if summary:
                print(f"   Summary: {summary['text'][:100]}...")
                print(f"   Topics: {summary['metadata'].get('topics', [])}")
                print(f"   Tone: {summary['metadata'].get('tone', 'N/A')}")
        print()
    except Exception as e:
        print(f"   Session summary failed: {e}\n")

    # Example 4: Memory consolidation (dry run)
    print("6. Testing memory consolidation (dry run)...")

    # Add some similar memories first
    similar_memories = [
        {
            "text": "I'm a software engineer working in Python",
            "memory_type": MemoryType.FACT.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 8,
                "category": Category.WORK.value,
                "session_id": "session_004",
                "confidence": 0.9,
            },
        },
        {
            "text": "I develop backend systems using Python",
            "memory_type": MemoryType.FACT.value,
            "metadata": {
                "user_id": "user_smart",
                "importance": 7,
                "category": Category.WORK.value,
                "session_id": "session_004",
                "confidence": 0.85,
            },
        },
    ]

    memory_store.store_batch_memories(similar_memories)

    try:
        consolidation_result = consolidator.consolidate_memories(
            user_id="user_smart",
            similarity_threshold=0.80,
            max_clusters=2,
            dry_run=True,  # Don't actually consolidate
        )

        print(f"   Clusters found: {consolidation_result['clusters_found']}")
        print(f"   Memories analyzed: {consolidation_result['memories_analyzed']}")
        print(f"   Would consolidate: {consolidation_result['clusters_consolidated']} clusters")

        if consolidation_result["results"]:
            print("\n   Sample consolidation:")
            sample = consolidation_result["results"][0]
            print(f"     Cluster size: {sample['cluster_size']}")
            print(f"     Consolidated: {sample['consolidated_text'][:80]}...")
            print(f"     New importance: {sample['importance']}")
            print(f"     Reasoning: {sample['reasoning']}")
        print()
    except Exception as e:
        print(f"   Consolidation failed: {e}\n")

    # Example 5: Check consolidation schedule
    print("7. Checking consolidation schedule...")
    schedule = consolidator.schedule_consolidation(user_id="user_smart", frequency_days=7)

    print(f"   Active memories: {schedule.get('active_memories', 0)}")
    print(f"   Should consolidate: {schedule.get('should_consolidate', False)}")
    print(f"   Recommendation: {schedule.get('recommendation', 'N/A')}")
    print()

    # Summary
    print("8. Final statistics...")
    all_memories = retriever.get_memories_by_filter(filters={"user_id": "user_smart"}, limit=100)

    by_type = {}
    by_category = {}

    for mem in all_memories:
        mem_type = mem["metadata"].get("memory_type", "unknown")
        category = mem["metadata"].get("category", "unknown")

        by_type[mem_type] = by_type.get(mem_type, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1

    print(f"   Total memories: {len(all_memories)}")
    print(f"   By type: {dict(by_type)}")
    print(f"   By category: {dict(by_category)}")

    # Close connection
    print("\n9. Cleaning up...")
    qdrant.close()
    print("   Connection closed")
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
