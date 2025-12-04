#!/usr/bin/env python3
"""
Groq + HippocampAI Interactive CLI Demo

This demo shows how to build a conversational AI with persistent memory using:
- Groq API with llama-3.1-8b-instant model
- HippocampAI for memory storage and retrieval

Features:
- Interactive command-line chat interface
- Automatic memory extraction and storage
- Context-aware responses using retrieved memories
- Session-based conversation tracking
- Memory search and management commands

Requirements:
    pip install groq hippocampai rich

Environment Variables:
    GROQ_API_KEY: Your Groq API key
    HIPPOCAMPAI_API_KEY: Your HippocampAI API key (or use local mode)
"""

import os
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Add local source to path if needed
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), 'src')
if os.path.exists(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from groq import Groq
except ImportError:
    print("Error: groq package not installed. Run: pip install groq")
    sys.exit(1)

try:
    from hippocampai import UnifiedMemoryClient
except ImportError:
    print("Error: hippocampai package not installed and source not found.")
    print("Run: pip install hippocampai")
    print("Or ensure you're running from the HippocampAI project directory")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.table import Table
except ImportError:
    print("Error: rich package not installed. Run: pip install rich")
    sys.exit(1)


class GroqHippocampAIChat:
    """Interactive chat system with Groq LLM and HippocampAI memory."""

    def __init__(
        self,
        user_id: str = None,
        base_url: str = None,
        qdrant_url: str = None,
        redis_url: str = None
    ):
        """Initialize the chat system.

        Args:
            user_id: Unique user identifier (auto-generated if not provided)
            base_url: HippocampAI API base URL (uses local mode if not provided)
            qdrant_url: Qdrant server URL for local mode (default: http://localhost:6333)
            redis_url: Redis server URL for local mode (default: redis://localhost:6379)
        """
        self.console = Console()
        self.user_id = user_id or str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            self.console.print("[red]Error: GROQ_API_KEY environment variable not set[/red]")
            self.console.print("Get your API key from: https://console.groq.com/")
            sys.exit(1)

        self.groq_client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"

        # Initialize HippocampAI client
        try:
            if base_url:
                # Remote mode - using HippocampAI SaaS
                api_key = os.getenv("HIPPOCAMPAI_API_KEY")
                self.memory_client = UnifiedMemoryClient(
                    mode="remote",
                    api_url=base_url,
                    api_key=api_key
                )
                self.console.print(f"[green]✓ HippocampAI connected to {base_url}[/green]")
            else:
                # Local mode - direct connection to Qdrant/Redis
                local_kwargs = {}
                if qdrant_url:
                    local_kwargs['qdrant_url'] = qdrant_url
                if redis_url:
                    local_kwargs['redis_url'] = redis_url

                self.memory_client = UnifiedMemoryClient(mode="local", **local_kwargs)
                self.console.print("[green]✓ HippocampAI memory system initialized (local mode)[/green]")

        except Exception as e:
            self.console.print(f"[red]Error initializing HippocampAI: {e}[/red]")
            sys.exit(1)

        self.conversation_history: List[Dict[str, str]] = []

    def detect_memory_type(self, text: str) -> str:
        """Automatically detect the type of memory based on content analysis.

        Args:
            text: The text to analyze

        Returns:
            Detected memory type (fact, preference, goal, habit, event, or context)
        """
        text_lower = text.lower()

        # Fact patterns (identity, personal information, statements)
        fact_patterns = [
            "my name is", "i'm", "i am", "call me", "i work", "i live",
            "my job", "my age", "my birthday", "i have", "i own",
            "i was born", "i graduated", "my address", "my email"
        ]

        # Preference patterns (likes, dislikes, opinions)
        preference_patterns = [
            "i like", "i love", "i prefer", "i enjoy", "i hate",
            "i dislike", "my favorite", "i'd rather", "i don't like",
            "i appreciate", "i fancy", "i'm fond of"
        ]

        # Goal patterns (intentions, aspirations, plans)
        goal_patterns = [
            "i want to", "i plan to", "my goal is", "i hope to",
            "i aim to", "i intend to", "i wish to", "i'd like to",
            "i need to", "i should", "i will", "going to"
        ]

        # Habit patterns (routines, regular activities)
        habit_patterns = [
            "i usually", "i always", "i often", "i regularly",
            "every day", "every week", "every morning", "every night",
            "i tend to", "i typically", "my routine", "i never"
        ]

        # Event patterns (specific occurrences, meetings, happenings)
        event_patterns = [
            "happened", "occurred", "took place", "yesterday", "last week",
            "last month", "ago", "meeting", "appointment", "on", "at"
        ]

        # Check patterns in order of specificity
        for pattern in fact_patterns:
            if pattern in text_lower:
                return "fact"

        for pattern in preference_patterns:
            if pattern in text_lower:
                return "preference"

        for pattern in goal_patterns:
            if pattern in text_lower:
                return "goal"

        for pattern in habit_patterns:
            if pattern in text_lower:
                return "habit"

        for pattern in event_patterns:
            if pattern in text_lower:
                return "event"

        # Default to context for general conversation
        return "context"

    def extract_and_store_memories(self, user_message: str, assistant_message: str) -> None:
        """Extract important information and store as memories.

        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        try:
            # Always store every conversation turn
            conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"

            # Store the conversation exchange with automatic type detection
            self.memory_client.remember(
                text=conversation_text,
                user_id=self.user_id,
                session_id=self.session_id,
                importance=5.0,
                tags=["conversation", "exchange"],
                metadata={
                    "type": "context",  # Pass type in metadata, not as parameter
                    "timestamp": datetime.now().isoformat(),
                    "user_message": user_message,
                    "assistant_message": assistant_message
                }
            )

            # Detect memory type for the user message
            detected_type = self.detect_memory_type(user_message)

            # If it's not just generic context, store it separately with higher importance
            if detected_type != "context":
                self.memory_client.remember(
                    text=user_message,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    importance=8.0,
                    tags=[detected_type, "important"],
                    metadata={
                        "type": detected_type,  # Pass type in metadata
                        "timestamp": datetime.now().isoformat(),
                        "context": assistant_message[:200]
                    }
                )
                self.console.print(f"[dim]📝 Automatically detected and stored {detected_type} memory[/dim]")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to store memory: {e}[/yellow]")

    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for the current query.

        Args:
            query: User's query
            limit: Maximum number of memories to retrieve

        Returns:
            List of relevant memories
        """
        try:
            results = self.memory_client.recall(
                query=query,
                user_id=self.user_id,
                limit=limit
            )
            # Convert RetrievalResult objects to dicts if needed
            if results and hasattr(results[0], '__dict__'):
                return [
                    {
                        'text': r.text if hasattr(r, 'text') else str(r),
                        'score': r.score if hasattr(r, 'score') else 1.0,
                        'metadata': r.metadata if hasattr(r, 'metadata') else {}
                    }
                    for r in results
                ]
            return results
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to retrieve memories: {e}[/yellow]")
            return []

    def build_context_with_memories(self, user_message: str) -> str:
        """Build context string with relevant memories.

        Args:
            user_message: Current user message

        Returns:
            Context string to prepend to the conversation
        """
        memories = self.retrieve_relevant_memories(user_message)

        if not memories:
            return ""

        context_parts = ["Here's what I remember about our conversations:"]

        for i, memory in enumerate(memories, 1):
            mem_text = memory.get('text', '')
            mem_metadata = memory.get('metadata', {})
            mem_type = mem_metadata.get('type', 'memory')
            context_parts.append(f"{i}. [{mem_type}] {mem_text}")

        return "\n".join(context_parts) + "\n\n"

    def chat(self, user_message: str) -> str:
        """Send a message and get response with memory context.

        Args:
            user_message: User's message

        Returns:
            Assistant's response
        """
        # Retrieve relevant memories
        memory_context = self.build_context_with_memories(user_message)

        # Build messages for Groq
        messages = []

        # System message - DON'T mention memory unless we have actual retrieved memories
        if memory_context:
            # We have real memories - tell the LLM to use them
            system_prompt = (
                "You are a helpful AI assistant. "
                "Below are some relevant past interactions with this user:\n\n"
                f"{memory_context}\n\n"
                "IMPORTANT: Use ONLY the above information when referencing past conversations. "
                "DO NOT make up or hallucinate memories that are not shown above. "
                "If no relevant past context is provided, treat this as a fresh conversation."
            )
        else:
            # No memories retrieved - don't mention memory at all
            system_prompt = (
                "You are a helpful AI assistant. "
                "Be conversational and friendly. "
                "Answer questions to the best of your ability."
            )

        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (last 10 messages)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Get response from Groq
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )

            assistant_message = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Extract and store memories
            self.extract_and_store_memories(user_message, assistant_message)

            return assistant_message

        except Exception as e:
            return f"Error getting response from Groq: {e}"

    def show_memories(self, limit: int = 10) -> None:
        """Display stored memories in a table.

        Args:
            limit: Maximum number of memories to display
        """
        try:
            # Use recall with empty query to get recent memories
            results = self.memory_client.recall(
                query="",
                user_id=self.user_id,
                limit=limit
            )

            if not results:
                self.console.print("[yellow]No memories found[/yellow]")
                return

            # Convert to list of dicts if needed
            if results and hasattr(results[0], '__dict__'):
                memories = [
                    {
                        'text': r.text if hasattr(r, 'text') else str(r),
                        'metadata': r.metadata if hasattr(r, 'metadata') else {},
                        'score': r.score if hasattr(r, 'score') else 0
                    }
                    for r in results
                ]
            else:
                memories = results

            table = Table(title=f"Stored Memories (User: {self.user_id[:8]}...)")
            table.add_column("Type", style="cyan")
            table.add_column("Text", style="white", max_width=60)
            table.add_column("Score", style="yellow")

            for memory in memories:
                mem_metadata = memory.get('metadata', {})
                mem_type = mem_metadata.get('type', 'memory')
                mem_text = memory.get('text', '')

                table.add_row(
                    mem_type,
                    mem_text[:100] + ('...' if len(mem_text) > 100 else ''),
                    f"{memory.get('score', 0):.2f}"
                )

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Error fetching memories: {e}[/red]")

    def search_memories_interactive(self) -> None:
        """Interactive memory search."""
        query = Prompt.ask("🔍 [cyan]Enter search query[/cyan]")

        try:
            results = self.memory_client.recall(query=query, user_id=self.user_id, limit=5)

            if not results:
                self.console.print("[yellow]No matching memories found[/yellow]")
                return

            # Convert to list of dicts if needed
            if results and hasattr(results[0], '__dict__'):
                memories = [
                    {
                        'text': r.text if hasattr(r, 'text') else str(r),
                        'metadata': r.metadata if hasattr(r, 'metadata') else {},
                        'score': r.score if hasattr(r, 'score') else 0
                    }
                    for r in results
                ]
            else:
                memories = results

            self.console.print(f"\n[green]Found {len(memories)} matching memories:[/green]\n")

            for i, memory in enumerate(memories, 1):
                mem_metadata = memory.get('metadata', {})
                mem_type = mem_metadata.get('type', 'memory')
                mem_text = memory.get('text', '')
                mem_tags = mem_metadata.get('tags', [])
                mem_score = memory.get('score', 0)

                self.console.print(Panel(
                    f"[cyan]Type:[/cyan] {mem_type}\n"
                    f"[cyan]Text:[/cyan] {mem_text}\n"
                    f"[cyan]Score:[/cyan] {mem_score:.2f}\n"
                    f"[cyan]Tags:[/cyan] {', '.join(mem_tags) if mem_tags else 'None'}",
                    title=f"Memory {i}",
                    border_style="blue"
                ))

        except Exception as e:
            self.console.print(f"[red]Error searching memories: {e}[/red]")

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.console.print(Panel.fit(
            "[bold cyan]Groq + HippocampAI Chat Demo[/bold cyan]\n"
            f"Model: {self.model}\n"
            f"User ID: {self.user_id[:16]}...\n"
            f"Session ID: {self.session_id[:16]}...\n\n"
            "[dim]Commands:[/dim]\n"
            "  /memories - Show stored memories\n"
            "  /search - Search memories\n"
            "  /info - Show full session info\n"
            "  /clear - Clear conversation history\n"
            "  /help - Show this help\n"
            "  /quit - Exit\n",
            border_style="green"
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower().strip()

                    if command == '/quit' or command == '/exit':
                        self.console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    elif command == '/memories':
                        self.show_memories()
                        continue
                    elif command == '/search':
                        self.search_memories_interactive()
                        continue
                    elif command == '/clear':
                        self.conversation_history.clear()
                        self.console.print("[green]✓ Conversation history cleared[/green]")
                        continue
                    elif command == '/info':
                        self.console.print(Panel(
                            f"[cyan]User ID:[/cyan] {self.user_id}\n"
                            f"[cyan]Session ID:[/cyan] {self.session_id}\n"
                            f"[cyan]Model:[/cyan] {self.model}",
                            title="Session Information",
                            border_style="cyan"
                        ))
                        continue
                    elif command == '/help':
                        self.console.print(
                            "[cyan]Available commands:[/cyan]\n"
                            "  /memories - Show stored memories\n"
                            "  /search - Search memories\n"
                            "  /info - Show session information\n"
                            "  /clear - Clear conversation history\n"
                            "  /help - Show this help\n"
                            "  /quit - Exit"
                        )
                        continue
                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                        continue

                # Get response from chat
                with self.console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                    response = self.chat(user_input)

                # Display response
                self.console.print(f"\n[bold blue]Assistant[/bold blue]")
                self.console.print(Markdown(response))

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Groq + HippocampAI Interactive Chat Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode (embedded Qdrant)
  python groq_llama_chat_demo.py

  # Remote mode (HippocampAI SaaS)
  python groq_llama_chat_demo.py --base-url http://localhost:8000

  # With custom user ID
  python groq_llama_chat_demo.py --user-id my-user-123

Environment Variables:
  GROQ_API_KEY            Your Groq API key (required)
  HIPPOCAMPAI_API_KEY     Your HippocampAI API key (for remote mode)
        """
    )

    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID for memory storage (auto-generated if not provided)"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        help="HippocampAI API base URL (e.g., http://localhost:8000)"
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        help="Qdrant server URL for local mode (e.g., http://100.113.229.40:6333)"
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        help="Redis server URL for local mode (e.g., redis://localhost:6379)"
    )

    args = parser.parse_args()

    # Create and run chat
    chat = GroqHippocampAIChat(
        user_id=args.user_id,
        base_url=args.base_url,
        qdrant_url=args.qdrant_url,
        redis_url=args.redis_url
    )

    chat.run()


if __name__ == "__main__":
    main()
