"""Simple chatbot with memory - Minimal example.

This is a simplified version of chat.py showing core patterns
for integrating HippocampAI into your own chat applications.

Usage:
    python examples/16_chatbot_with_memory.py
"""

import os

from hippocampai import MemoryClient
from hippocampai.adapters import GroqLLM

print("=" * 60)
print("  Simple Chatbot with Memory Example")
print("=" * 60)

# 1. Initialize components
api_key = os.getenv("GROQ_API_KEY", "demo-key")
llm = GroqLLM(api_key=api_key, model="llama-3.1-8b-instant")
client = MemoryClient(llm_provider=llm)

user_id = "demo_user"
conversation_history = []

print("\n✓ Initialized chatbot with memory")
print(f"  User ID: {user_id}")
print("\n" + "-" * 60)


def chat(user_message: str) -> str:
    """Process message with memory context."""

    # 1. Retrieve relevant memories
    print("\n[1] Recalling relevant memories...")
    try:
        memories = client.recall(query=user_message, user_id=user_id, k=3)
        print(f"    Found {len(memories)} relevant memories")

        # Build context from memories
        context = ""
        if memories:
            context = "\n<memories>\n"
            for mem in memories:
                context += f"- {mem.memory.text}\n"
            context += "</memories>\n"
    except Exception as e:
        print(f"    Warning: Memory recall failed: {e}")
        context = ""

    # 2. Build prompt with context
    print("[2] Building prompt with context...")
    system_prompt = """You are a helpful AI assistant with memory.
Use the provided memories to give personalized responses."""

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({"role": "user", "content": context})

    # Add conversation history (last 3 turns)
    for msg in conversation_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    # 3. Generate response
    print("[3] Generating response...")
    try:
        response = llm.chat(messages, max_tokens=256, temperature=0.7)
    except Exception as e:
        response = f"Error generating response: {e}"

    # 4. Extract and store memories
    print("[4] Extracting memories from conversation...")
    try:
        conversation = f"User: {user_message}\nAssistant: {response}"
        new_memories = client.extract_from_conversation(conversation=conversation, user_id=user_id)
        print(f"    Stored {len(new_memories)} new memories")
    except Exception as e:
        print(f"    Warning: Memory extraction failed: {e}")

    # 5. Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": response})

    return response


# Example conversation
print("\nExample Conversation:")
print("=" * 60)

# Turn 1: Share information
print("\nUser: Hi! I'm Alice and I work as a data scientist at TechCorp.")
response = chat("Hi! I'm Alice and I work as a data scientist at TechCorp.")
print(f"Bot: {response}")

print("\n" + "-" * 60)

# Turn 2: Share preference
print("\nUser: I love Python and spend most of my time building ML models.")
response = chat("I love Python and spend most of my time building ML models.")
print(f"Bot: {response}")

print("\n" + "-" * 60)

# Turn 3: Share goal
print("\nUser: My goal is to learn more about LLMs and transformer architecture.")
response = chat("My goal is to learn more about LLMs and transformer architecture.")
print(f"Bot: {response}")

print("\n" + "-" * 60)

# Turn 4: Test memory recall
print("\nUser: What do you know about me?")
response = chat("What do you know about me?")
print(f"Bot: {response}")

print("\n" + "=" * 60)

# Show stored memories
print("\nStored Memories:")
print("-" * 60)
try:
    all_memories = client.get_memories(user_id=user_id, limit=10)
    for i, mem in enumerate(all_memories, 1):
        print(f"{i}. [{mem.type.value.upper()}] {mem.text}")
        print(f"   Importance: {mem.importance}/10")
except Exception as e:
    print(f"Failed to retrieve memories: {e}")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)

print("\nKey Patterns Demonstrated:")
print("  1. Memory retrieval before each response")
print("  2. Context injection into LLM prompt")
print("  3. Automatic memory extraction after response")
print("  4. Conversation history management")
print("  5. Error handling for robustness")

print("\nNext Steps:")
print("  • See chat.py for full interactive version")
print("  • Check docs/CHAT_DEMO_GUIDE.md for details")
print("  • Try other examples in examples/ directory")
