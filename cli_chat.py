"""CLI interface for memory-enhanced AI chat.

This provides a command-line chat interface with memory features using HippocampAI.

Usage:
    python cli_chat.py [user_id]

Example:
    python cli_chat.py alice
    python cli_chat.py  # defaults to 'cli_user'

Commands:
    /help - Show help
    /stats - Show memory statistics
    /memories - View stored memories
    /clear - Clear current session
    /quit - Exit chat
"""

import sys
from typing import List

from hippocampai import MemoryClient
from hippocampai.config import get_config
from hippocampai.models.memory import RetrievalResult


# ANSI color codes for pretty output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")


def print_user(text: str) -> None:
    """Print user message."""
    print(f"{Colors.OKBLUE}You: {text}{Colors.ENDC}")


def print_assistant(text: str) -> None:
    """Print assistant message."""
    print(f"{Colors.OKGREEN}Assistant: {text}{Colors.ENDC}")


def print_system(text: str) -> None:
    """Print system message."""
    print(f"{Colors.OKCYAN}[System] {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}[Error] {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}[Warning] {text}{Colors.ENDC}")


def show_welcome(user_id: str, config: any) -> None:
    """Show welcome message."""
    print("=" * 70)
    print_header("🧠 HippocampAI - Memory-Enhanced AI Chat")
    print("=" * 70)
    print(f"\nUser: {Colors.BOLD}{user_id}{Colors.ENDC}")
    print(f"LLM Provider: {Colors.BOLD}{config.llm_provider}{Colors.ENDC}")
    print(f"Model: {Colors.BOLD}{config.llm_model}{Colors.ENDC}")
    print("\nI'm your AI assistant with memory. I remember our conversations")
    print("and learn from them to provide personalized responses.\n")
    print("Commands:")
    print("  /help      - Show this help")
    print("  /stats     - Show memory statistics")
    print("  /memories  - View stored memories")
    print("  /clear     - Clear current session")
    print("  /quit      - Exit chat")
    print("\n" + "=" * 70)


def show_help() -> None:
    """Show help message."""
    print_header("\n📚 Help")
    print("\nAvailable commands:")
    print("  /help      - Show this help message")
    print("  /stats     - Display your memory statistics")
    print("  /memories  - View all stored memories about you")
    print("  /clear     - Clear the current conversation context")
    print("  /quit      - Exit the chat application")
    print("\nJust type normally to chat and create memories!")


def show_stats(client: MemoryClient, user_id: str) -> None:
    """Show memory statistics."""
    print_header("\n📊 Memory Statistics")

    try:
        # Get all memories for user
        all_memories = client.qdrant.scroll(
            collection_name=client.config.collection_facts, filters={"user_id": user_id}, limit=1000
        )
        all_prefs = client.qdrant.scroll(
            collection_name=client.config.collection_prefs, filters={"user_id": user_id}, limit=1000
        )

        total = len(all_memories) + len(all_prefs)
        print(f"\n  Total Memories: {Colors.BOLD}{total}{Colors.ENDC}")
        print(f"  Facts/Events: {Colors.BOLD}{len(all_memories)}{Colors.ENDC}")
        print(f"  Preferences/Goals: {Colors.BOLD}{len(all_prefs)}{Colors.ENDC}")

        if all_memories or all_prefs:
            # Calculate average importance
            importances = []
            for mem in all_memories + all_prefs:
                imp = mem.get("payload", {}).get("importance", 5.0)
                importances.append(imp)

            if importances:
                avg_imp = sum(importances) / len(importances)
                print(f"  Average Importance: {Colors.BOLD}{avg_imp:.1f}/10{Colors.ENDC}")

    except Exception as e:
        print_error(f"Failed to load stats: {e}")


def show_memories(client: MemoryClient, user_id: str) -> None:
    """Show stored memories."""
    print_header("\n💭 Your Memories")

    try:
        # Get memories from both collections
        facts = client.qdrant.scroll(
            collection_name=client.config.collection_facts, filters={"user_id": user_id}, limit=50
        )
        prefs = client.qdrant.scroll(
            collection_name=client.config.collection_prefs, filters={"user_id": user_id}, limit=50
        )

        all_memories = facts + prefs

        if not all_memories:
            print("\n  No memories stored yet. Start chatting to create memories!")
            return

        print(f"\n  Showing {len(all_memories)} memories:\n")

        for i, mem in enumerate(all_memories, 1):
            payload = mem.get("payload", {})
            text = payload.get("text", "")
            mem_type = payload.get("type", "unknown")
            importance = payload.get("importance", 0)

            print(f"  {i}. [{mem_type.upper()}] {text}")
            print(f"     Importance: {importance}/10")
            print()

    except Exception as e:
        print_error(f"Failed to load memories: {e}")


def generate_response(client: MemoryClient, user_input: str, user_id: str, session_id: str) -> str:
    """Generate AI response with memory context."""
    # Retrieve relevant memories
    memories: List[RetrievalResult] = client.recall(query=user_input, user_id=user_id, k=5)

    # Build context from memories
    context = ""
    if memories:
        context = "Here's what I remember about you:\n"
        for mem in memories[:3]:
            context += f"- {mem.memory.text}\n"
        context += "\n"

    # Create simple response (in production, use LLM here)
    if client.llm:
        # Use LLM if available
        prompt = f"{context}User: {user_input}\n\nProvide a helpful response:"
        response = client.llm.generate(prompt, max_tokens=512, temperature=0.7)
        return response
    else:
        # Fallback response
        response = "I understand. I've noted that and will remember our conversation."
        if memories:
            response += f" I recall {len(memories)} related memories about you."
        return response


def main() -> None:
    """Main CLI chat loop."""
    # Get user ID from command line or use default
    user_id = sys.argv[1] if len(sys.argv) > 1 else "cli_user"
    session_id = f"cli_session_{user_id}"

    # Load configuration
    try:
        config = get_config()
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        print_warning("Make sure .env file exists and is configured correctly")
        sys.exit(1)

    # Show welcome
    show_welcome(user_id, config)

    # Initialize memory client
    print_system("Initializing memory client...")

    try:
        client = MemoryClient(config=config)
        print_system(f"Ready! Connected to Qdrant at {config.qdrant_url}\n")
    except Exception as e:
        print_error(f"Failed to initialize memory client: {e}")
        print_warning("Make sure Qdrant is running and accessible")
        sys.exit(1)

    # Main chat loop
    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.OKBLUE}You: {Colors.ENDC}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/quit" or command == "/exit":
                    print_system("Goodbye! 👋")
                    break

                elif command == "/help":
                    show_help()

                elif command == "/stats":
                    show_stats(client, user_id)

                elif command == "/memories":
                    show_memories(client, user_id)

                elif command == "/clear":
                    conversation_history.clear()
                    print_system("Conversation history cleared!")

                else:
                    print_warning(f"Unknown command: {command}")
                    print("Type /help for available commands")

                continue

            # Add to conversation history
            conversation_history.append(user_input)

            # Extract and store memories from conversation
            if len(conversation_history) >= 3:
                conv_text = "\n".join(conversation_history[-3:])
                try:
                    client.extract_from_conversation(conv_text, user_id, session_id)
                except Exception as e:
                    print_warning(f"Memory extraction failed: {e}")

            # Generate response
            print_system("Thinking...")

            try:
                response = generate_response(client, user_input, user_id, session_id)
                print_assistant(response)
            except Exception as e:
                print_error(f"Error generating response: {e}")

        except KeyboardInterrupt:
            print("\n")
            print_system("Goodbye! 👋")
            break

        except EOFError:
            print("\n")
            print_system("Goodbye! 👋")
            break

        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
