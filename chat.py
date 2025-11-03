#!/usr/bin/env python3
"""Interactive Chat with HippocampAI Memory Integration.

This script demonstrates a fully-functional chatbot that uses:
- Groq for fast LLM responses
- HippocampAI for persistent memory across sessions
- Rich memory features: facts, preferences, patterns, insights

Usage:
        python chat.py

Requirements:
        - GROQ_API_KEY environment variable set
        - Qdrant running (docker run -d -p 6333:6333 qdrant/qdrant)
        - Optional: Redis for caching
"""

import os
import sys
from datetime import datetime

try:
    from hippocampai import MemoryClient
    from hippocampai.adapters import GroqLLM
except ImportError:
    print("ERROR: hippocampai not installed. Install with: pip install hippocampai")
    sys.exit(1)


class MemoryChatBot:
    """Interactive chatbot with persistent memory."""

    def __init__(self, user_id: str = "demo_user"):
        """Initialize chatbot with memory and LLM."""
        self.user_id = user_id

        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY environment variable not set")
            print("Get your API key from: https://console.groq.com/keys")
            sys.exit(1)

        self.llm = GroqLLM(
            api_key=api_key,
            model="llama-3.1-8b-instant",  # Fast and capable
        )

        # Initialize HippocampAI memory client
        try:
            self.memory_client = MemoryClient(llm_provider=self.llm)
            print("✓ Connected to HippocampAI memory engine")
        except Exception as e:
            print(f"ERROR: Failed to connect to HippocampAI: {e}")
            print("Make sure Qdrant is running: docker run -d -p 6333:6333 qdrant/qdrant")
            sys.exit(1)

        # Create a session for this conversation
        self.session = self.memory_client.create_session(
            user_id=self.user_id,
            title=f"Chat Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"✓ Created session: {self.session.id[:8]}...")

        self.conversation_history = []

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant memories for context."""
        try:
            results = self.memory_client.recall(query=query, user_id=self.user_id, k=k)

            if not results:
                return ""

            context_parts = ["<relevant_memories>"]
            for result in results:
                context_parts.append(
                    f"- [{result.memory.type.value}] {result.memory.text} "
                    f"(relevance: {result.score:.2f})"
                )
            context_parts.append("</relevant_memories>")

            return "\n".join(context_parts)
        except Exception as e:
            print(f"Warning: Failed to retrieve memories: {e}")
            return ""

    def extract_and_store_memories(self, user_message: str, assistant_message: str):
        """Extract insights from conversation and store as memories."""
        try:
            # Extract memories from the conversation turn
            conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"
            memories = self.memory_client.extract_from_conversation(
                conversation=conversation_text, user_id=self.user_id
            )

            if memories:
                print(f"  💾 Stored {len(memories)} new memories")

            # Track this message in the session
            self.memory_client.track_session_message(
                session_id=self.session.id, role="user", content=user_message
            )
            self.memory_client.track_session_message(
                session_id=self.session.id, role="assistant", content=assistant_message
            )

        except Exception as e:
            print(f"  Warning: Memory extraction failed: {e}")

    def chat(self, user_message: str) -> str:
        """Process user message and generate response with memory."""
        # Get relevant context from past memories
        context = self.get_relevant_context(user_message)

        # Build system prompt with memory context
        system_prompt = """You are a helpful, friendly AI assistant with persistent memory.
You remember details about the user from previous conversations.

When relevant memories are provided, use them naturally in your responses.
Be conversational and personable while being helpful and accurate.

IMPORTANT: Only use information from memories if they are directly relevant to the current question.
Don't force memory usage if it doesn't fit naturally."""

        # Build the full prompt
        if context:
            full_prompt = f"{context}\n\nUser question: {user_message}"
        else:
            full_prompt = user_message

        # Add to conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 5 turns for context)
        for msg in self.conversation_history[-10:]:  # Last 5 turns = 10 messages
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": full_prompt})

        # Generate response
        try:
            response = self.llm.chat(messages, max_tokens=512, temperature=0.7)
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Extract and store memories asynchronously
        self.extract_and_store_memories(user_message, response)

        return response

    def show_memory_stats(self):
        """Display statistics about stored memories."""
        try:
            stats = self.memory_client.get_memory_statistics(user_id=self.user_id)
            print("\n" + "=" * 60)
            print("📊 MEMORY STATISTICS")
            print("=" * 60)
            print(f"Total memories: {stats.get('total_memories', 0)}")
            print(f"Memory types: {stats.get('memory_by_type', {})}")
            print("=" * 60)
        except Exception as e:
            print(f"Failed to get memory stats: {e}")

    def show_recent_memories(self, limit: int = 5):
        """Display recent memories."""
        try:
            # Get all memories and show the most recent
            memories = self.memory_client.get_memories(user_id=self.user_id, limit=limit)

            print("\n" + "=" * 60)
            print(f"🧠 RECENT MEMORIES (Last {limit})")
            print("=" * 60)

            for i, mem in enumerate(memories, 1):
                print(f"\n{i}. [{mem.type.value.upper()}] {mem.text}")
                print(
                    f"   Importance: {mem.importance}/10 | Created: {mem.created_at.strftime('%Y-%m-%d %H:%M')}"
                )

            print("=" * 60)
        except Exception as e:
            print(f"Failed to retrieve memories: {e}")

    def show_patterns(self):
        """Detect and show behavioral patterns."""
        try:
            patterns = self.memory_client.detect_patterns(user_id=self.user_id)

            print("\n" + "=" * 60)
            print("🔍 DETECTED PATTERNS")
            print("=" * 60)

            if patterns:
                for i, pattern in enumerate(patterns, 1):
                    print(f"\n{i}. {pattern.get('pattern', 'Unknown pattern')}")
                    print(f"   Confidence: {pattern.get('confidence', 0):.2f}")
                    print(f"   Description: {pattern.get('description', 'N/A')}")
            else:
                print("No patterns detected yet. Chat more to build up memory!")

            print("=" * 60)
        except Exception as e:
            print(f"Failed to detect patterns: {e}")

    def search_memories(self, query: str, k: int = 5):
        """Search and display memories."""
        try:
            results = self.memory_client.recall(query=query, user_id=self.user_id, k=k)

            print("\n" + "=" * 60)
            print(f"🔎 SEARCH RESULTS for: '{query}'")
            print("=" * 60)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.memory.text}")
                    print(f"   Type: {result.memory.type.value} | Score: {result.score:.3f}")
                    print(f"   Importance: {result.memory.importance}/10")
            else:
                print("No matching memories found.")

            print("=" * 60)
        except Exception as e:
            print(f"Failed to search memories: {e}")

    def complete_session(self):
        """Complete the session and generate summary."""
        try:
            # Complete session with summary
            self.memory_client.complete_session(session_id=self.session.id, generate_summary=True)

            # Retrieve the session to get the summary
            session = self.memory_client.get_session(self.session.id)

            print("\n" + "=" * 60)
            print("📝 SESSION SUMMARY")
            print("=" * 60)
            if session.summary:
                print(session.summary)
            else:
                print("Summary generation in progress...")
            print("=" * 60)

        except Exception as e:
            print(f"Failed to complete session: {e}")


def print_help():
    """Print available commands."""
    print("\n" + "=" * 60)
    print("💡 AVAILABLE COMMANDS")
    print("=" * 60)
    print("  /help       - Show this help message")
    print("  /stats      - Show memory statistics")
    print("  /memories   - Show recent memories")
    print("  /patterns   - Detect behavioral patterns")
    print("  /search     - Search memories (e.g., /search coffee)")
    print("  /clear      - Clear screen")
    print("  /exit       - Exit chat (with session summary)")
    print("=" * 60)


def main():
    """Main chat loop."""
    print("\n" + "=" * 70)
    print("  🤖 HippocampAI Interactive Chat")
    print("=" * 70)
    print("\n  A chatbot with persistent memory powered by:")
    print("    • Groq (llama-3.1-8b-instant) for fast responses")
    print("    • HippocampAI for intelligent memory management")
    print("\n  Try talking about yourself, your preferences, or your goals.")
    print("  The bot will remember everything across sessions!")

    # Get user ID (or use default)
    user_input = input("\n  Enter your name (or press Enter for 'demo_user'): ").strip()
    user_id = user_input if user_input else "demo_user"

    # Initialize chatbot
    print(f"\n  Initializing chatbot for user: {user_id}...")
    try:
        bot = MemoryChatBot(user_id=user_id)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print_help()

    print("\n  Type your message (or /help for commands, /exit to quit)")
    print("=" * 70)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_message = input(f"\n{user_id}> ").strip()

            if not user_message:
                continue

            # Handle commands
            if user_message.startswith("/"):
                cmd_parts = user_message.split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd == "/exit":
                    print("\n  Completing session and generating summary...")
                    bot.complete_session()
                    print("\n  Goodbye! Your memories are saved for next time. 👋")
                    break

                elif cmd == "/help":
                    print_help()

                elif cmd == "/stats":
                    bot.show_memory_stats()

                elif cmd == "/memories":
                    bot.show_recent_memories(limit=10)

                elif cmd == "/patterns":
                    bot.show_patterns()

                elif cmd == "/search":
                    if len(cmd_parts) > 1:
                        bot.search_memories(cmd_parts[1])
                    else:
                        print("  Usage: /search <query>")

                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    print("\n  Screen cleared. Type /help for commands.")

                else:
                    print(f"  Unknown command: {cmd}. Type /help for available commands.")

                continue

            # Generate response
            print("\n🤖 Assistant> ", end="", flush=True)
            response = bot.chat(user_message)
            print(response)

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Completing session...")
            bot.complete_session()
            print("  Goodbye! 👋\n")
            break

        except Exception as e:
            print(f"\n  ERROR: {e}")
            print("  Type /exit to quit or continue chatting.")


if __name__ == "__main__":
    main()
