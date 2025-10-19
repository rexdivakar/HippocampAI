"""CLI interface for memory-enhanced AI chat.

This provides a command-line chat interface with memory features.

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
    /end - End session and save summary
    /quit - Exit chat
"""

import sys

from src.ai_chat import MemoryEnhancedChat
from src.settings import get_settings


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
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")


def print_user(text: str):
    """Print user message."""
    print(f"{Colors.OKBLUE}You: {text}{Colors.ENDC}")


def print_assistant(text: str):
    """Print assistant message."""
    print(f"{Colors.OKGREEN}Assistant: {text}{Colors.ENDC}")


def print_system(text: str):
    """Print system message."""
    print(f"{Colors.OKCYAN}[System] {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}[Error] {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}[Warning] {text}{Colors.ENDC}")


def show_welcome(user_id: str):
    """Show welcome message."""
    print("=" * 70)
    print_header("🧠 HippocampAI - Memory-Enhanced AI Chat")
    print("=" * 70)
    print(f"\nUser: {Colors.BOLD}{user_id}{Colors.ENDC}")
    print("\nI'm your AI assistant with memory. I remember our conversations")
    print("and learn from them to provide personalized responses.\n")
    print("Commands:")
    print("  /help      - Show this help")
    print("  /stats     - Show memory statistics")
    print("  /memories  - View stored memories")
    print("  /clear     - Clear current session")
    print("  /end       - End session and save summary")
    print("  /quit      - Exit chat")
    print("\n" + "=" * 70)


def show_help():
    """Show help message."""
    print_header("\n📚 Help")
    print("\nAvailable commands:")
    print("  /help      - Show this help message")
    print("  /stats     - Display your memory statistics")
    print("  /memories  - View all stored memories about you")
    print("  /clear     - Clear the current conversation (no summary)")
    print("  /end       - End session and save conversation summary")
    print("  /quit      - Exit the chat application")
    print("\nJust type normally to chat with the AI!")


def show_stats(chat: MemoryEnhancedChat):
    """Show memory statistics."""
    print_header("\n📊 Memory Statistics")

    try:
        stats = chat.get_memory_stats()

        print(f"\n  Total Memories: {Colors.BOLD}{stats['total_memories']}{Colors.ENDC}")
        print(f"  Average Importance: {Colors.BOLD}{stats['avg_importance']:.1f}/10{Colors.ENDC}")
        print(f"  Recent (7 days): {Colors.BOLD}{stats['recent_count']}{Colors.ENDC}")

        if stats["by_type"]:
            print("\n  By Type:")
            for mem_type, count in stats["by_type"].items():
                print(f"    - {mem_type}: {count}")

        if stats["by_category"]:
            print("\n  By Category:")
            for category, count in stats["by_category"].items():
                print(f"    - {category}: {count}")

    except Exception as e:
        print_error(f"Failed to load stats: {e}")


def show_memories(chat: MemoryEnhancedChat):
    """Show stored memories."""
    print_header("\n💭 Your Memories")

    try:
        memories = chat.get_user_memories(limit=50)

        if not memories:
            print("\n  No memories stored yet. Start chatting to create memories!")
            return

        print(f"\n  Showing {len(memories)} memories:\n")

        for i, mem in enumerate(memories, 1):
            mem_type = mem.get("memory_type", "unknown")
            text = mem.get("text", "")
            importance = mem.get("importance", 0)
            category = mem.get("category", "unknown")

            print(f"  {i}. [{mem_type.upper()}] {text}")
            print(f"     Importance: {importance}/10 | Category: {category}")
            print()

    except Exception as e:
        print_error(f"Failed to load memories: {e}")


def main():
    """Main CLI chat loop."""
    # Get user ID from command line or use default
    user_id = sys.argv[1] if len(sys.argv) > 1 else "cli_user"

    # Check API key
    settings = get_settings()
    provider = settings.llm.provider.lower()

    has_api_key = False
    if provider == "anthropic" and settings.llm.anthropic_api_key:
        has_api_key = True
    elif provider == "openai" and settings.llm.openai_api_key:
        has_api_key = True
    elif provider == "groq" and settings.llm.groq_api_key:
        has_api_key = True

    if not has_api_key:
        print_error(f"{provider.upper()}_API_KEY not set in .env file!")
        print("Please add your API key to .env to use the chat.")
        sys.exit(1)

    # Show welcome
    show_welcome(user_id)

    # Initialize chat
    print_system("Initializing memory-enhanced chat...")

    try:
        chat = MemoryEnhancedChat(
            user_id=user_id, auto_extract_memories=True, auto_consolidate=True
        )
        print_system(f"Ready! Using {provider} as LLM provider.\n")
    except Exception as e:
        print_error(f"Failed to initialize chat: {e}")
        sys.exit(1)

    # Main chat loop
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
                    print_system("Ending session...")
                    chat.end_conversation()
                    print_system("Goodbye! 👋")
                    break

                elif command == "/help":
                    show_help()

                elif command == "/stats":
                    show_stats(chat)

                elif command == "/memories":
                    show_memories(chat)

                elif command == "/clear":
                    chat.clear_session()
                    print_system("Session cleared!")

                elif command == "/end":
                    print_system("Ending session and saving summary...")
                    summary_id = chat.end_conversation()
                    if summary_id:
                        print_system(f"Session summary saved! (ID: {summary_id})")
                    else:
                        print_warning("No active session to end")

                else:
                    print_warning(f"Unknown command: {command}")
                    print("Type /help for available commands")

                continue

            # Send message to AI
            print_system("Thinking...")

            try:
                response = chat.send_message(user_input)
                print_assistant(response)
            except Exception as e:
                print_error(f"Error getting response: {e}")

        except KeyboardInterrupt:
            print("\n")
            print_system("Interrupted! Ending session...")
            chat.end_conversation()
            print_system("Goodbye! 👋")
            break

        except EOFError:
            print("\n")
            print_system("End of input. Ending session...")
            chat.end_conversation()
            print_system("Goodbye! 👋")
            break

        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
