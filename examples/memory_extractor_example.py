"""Example usage of the MemoryExtractor class."""

import sys
import os
sys.path.append('..')

from src.qdrant_client import QdrantManager
from src.embedding_service import EmbeddingService
from src.memory_store import MemoryStore
from src.memory_extractor import MemoryExtractor
from src.settings import get_settings


def main():
    print("=== Memory Extractor Example ===\n")

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
        print(f"Please add your API key to .env or choose a different LLM_PROVIDER")
        return

    # Initialize services
    print("1. Initializing services...")
    qdrant = QdrantManager(host=settings.qdrant.host, port=settings.qdrant.port)
    qdrant.create_collections()

    embeddings = EmbeddingService(model_name=settings.embedding.model)
    memory_store = MemoryStore(qdrant_manager=qdrant, embedding_service=embeddings)
    extractor = MemoryExtractor()  # Uses configured LLM provider from settings
    print("   Services initialized!\n")

    # Example conversation
    conversation = """
    User: Hi! I'm looking for help with my morning routine. I've been struggling to wake up early.

    Assistant: I'd be happy to help! What time do you currently wake up, and what time would you like to wake up?

    User: I usually wake up around 8:30 AM, but I want to start waking up at 6:00 AM so I can exercise before work.
    I work as a software engineer and my meetings usually start at 9 AM.

    Assistant: That's a great goal! Exercise in the morning can really boost your energy. What kind of exercise do you enjoy?

    User: I really love running and yoga. I used to run 5k three times a week, but I haven't been consistent lately.
    I also prefer working out outdoors when the weather is nice.

    Assistant: Those are excellent choices! Have you thought about what might help you wake up earlier?

    User: I think I need to stop drinking coffee after 3 PM - I've noticed it affects my sleep.
    Also, I tend to stay up late watching Netflix, which doesn't help.

    Assistant: Those are important insights. Creating better evening habits will definitely help with waking up earlier.

    User: Yeah, I really need to be more disciplined. My goal this year is to run a half marathon in October.
    I've never done one before, so I need to train consistently.
    """

    # Example 1: Extract memories
    print("2. Extracting memories from conversation...")
    print(f"   Conversation length: {len(conversation)} characters\n")

    try:
        memories = extractor.extract_memories(
            conversation_text=conversation,
            user_id="user_456",
            session_id="session_demo_001"
        )

        print(f"   Extracted {len(memories)} memories:\n")
        for i, memory in enumerate(memories, 1):
            print(f"   {i}. Text: {memory['text']}")
            print(f"      Type: {memory['memory_type']}")
            print(f"      Importance: {memory['metadata']['importance']}/10")
            print(f"      Category: {memory['metadata']['category']}")
            print(f"      Confidence: {memory['metadata']['confidence']}")
            print()

    except Exception as e:
        print(f"   ERROR: Failed to extract memories: {e}")
        qdrant.close()
        return

    # Example 2: Extract and store in one step
    print("3. Extracting and storing memories...")

    another_conversation = """
    User: I've been thinking about learning machine learning. Do you have any recommendations?

    Assistant: Absolutely! What's your background in programming?

    User: I'm comfortable with Python - I use it daily for data analysis at work.
    I work in the finance industry analyzing market trends.

    Assistant: Perfect! Python is great for machine learning. Are you interested in any specific area?

    User: I'm really interested in neural networks and deep learning.
    I want to build models that can predict stock market movements, though I know that's very challenging.

    Assistant: That's an ambitious goal! Have you worked with any ML libraries before?

    User: Not yet, but I'm planning to dedicate 2 hours every evening to learning.
    I learn best through hands-on projects rather than just theory.
    """

    try:
        memory_ids = extractor.extract_and_store(
            conversation_text=another_conversation,
            user_id="user_456",
            memory_store=memory_store,
            session_id="session_demo_002"
        )

        print(f"   Extracted and stored {len(memory_ids)} memories")
        print(f"   Memory IDs: {memory_ids[:3]}{'...' if len(memory_ids) > 3 else ''}\n")

    except Exception as e:
        print(f"   ERROR: Failed to extract and store memories: {e}")

    # Example 3: Show all memories for user
    print("4. Retrieving all stored memories for user_456...")
    from src.memory_retriever import MemoryRetriever

    retriever = MemoryRetriever(qdrant_manager=qdrant, embedding_service=embeddings)
    all_memories = retriever.get_memories_by_filter(
        filters={"user_id": "user_456"},
        limit=20
    )

    print(f"   Total memories for user_456: {len(all_memories)}\n")
    for i, mem in enumerate(all_memories, 1):
        print(f"   {i}. [{mem['metadata']['memory_type']}] {mem['text'][:70]}...")
        print(f"      Importance: {mem['metadata']['importance']}, "
              f"Category: {mem['metadata']['category']}")

    # Close connection
    print("\n5. Cleaning up...")
    qdrant.close()
    print("   Connection closed")
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
