#!/usr/bin/env python3
"""
Groq + HippocampAI Interactive CLI Demo

This demo shows how to build a conversational AI with persistent memory using:
- Groq API with llama-3.1-8b-instant model
- HippocampAI for memory storage and retrieval

Features Demonstrated:
1. Core Operations:
   - remember() - Store memories with metadata
   - recall() - Semantic search and retrieval
   - update_memory() - Update existing memories
   - delete_memory() - Remove memories
   - get_memory() - Retrieve by ID
   - get_memories() - List all memories

2. Batch Operations:
   - batch_remember() - Bulk memory creation
   - batch_get_memories() - Retrieve multiple memories
   - batch_delete_memories() - Bulk deletion

3. Advanced Features:
   - Advanced filtering (tags, importance, min_score)
   - Entity extraction (extract_entities, extract_facts)
   - Memory expiration (TTL with expires_at)
   - Memory consolidation (merge similar memories)
   - Cleanup expired memories

4. Analytics & Monitoring:
   - get_memory_analytics() - Memory statistics
   - health_check() - System health status
   - Session management
   - Automatic memory type detection

5. Interactive Commands:
   - /test - Run comprehensive feature tests
   - /analytics - Show memory analytics
   - /health - System health check
   - /memories - View stored memories
   - /search - Search memories

Requirements:
    pip install groq hippocampai rich

Environment Variables:
    GROQ_API_KEY: Your Groq API key (required)
    HIPPOCAMPAI_API_KEY: Your HippocampAI API key (optional for remote mode)

Usage:
    # Local mode
    python groq_llama_chat_demo.py --qdrant-url http://localhost:6333

    # Remote mode
    python groq_llama_chat_demo.py --base-url http://localhost:8000

    # Run feature tests
    Type '/test' in the chat to test all HippocampAI features
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List

# Add local source to path if needed
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), "src")
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
    from rich.markdown import Markdown
    from rich.panel import Panel
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
        session_id: str = None,
        base_url: str = None,
        qdrant_url: str = None,
        redis_url: str = None,
    ):
        """Initialize the chat system.

        Args:
            user_id: Unique user identifier (auto-generated if not provided)
            session_id: Session identifier (auto-generated if not provided)
            base_url: HippocampAI API base URL (uses local mode if not provided)
            qdrant_url: Qdrant server URL for local mode (default: http://localhost:6333)
            redis_url: Redis server URL for local mode (default: redis://localhost:6379)
        """
        self.console = Console()
        self.user_id = user_id or str(uuid.uuid4())
        self.session_id = session_id or str(uuid.uuid4())

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
                    mode="remote", api_url=base_url, api_key=api_key
                )
                self.console.print(f"[green]✓ HippocampAI connected to {base_url}[/green]")
            else:
                # Local mode - direct connection to Qdrant/Redis
                local_kwargs = {}
                if qdrant_url:
                    local_kwargs["qdrant_url"] = qdrant_url
                if redis_url:
                    local_kwargs["redis_url"] = redis_url

                self.memory_client = UnifiedMemoryClient(mode="local", **local_kwargs)
                self.console.print(
                    "[green]✓ HippocampAI memory system initialized (local mode)[/green]"
                )

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
            "my name is",
            "i'm",
            "i am",
            "call me",
            "i work",
            "i live",
            "my job",
            "my age",
            "my birthday",
            "i have",
            "i own",
            "i was born",
            "i graduated",
            "my address",
            "my email",
        ]

        # Preference patterns (likes, dislikes, opinions)
        preference_patterns = [
            "i like",
            "i love",
            "i prefer",
            "i enjoy",
            "i hate",
            "i dislike",
            "my favorite",
            "i'd rather",
            "i don't like",
            "i appreciate",
            "i fancy",
            "i'm fond of",
        ]

        # Goal patterns (intentions, aspirations, plans)
        goal_patterns = [
            "i want to",
            "i plan to",
            "my goal is",
            "i hope to",
            "i aim to",
            "i intend to",
            "i wish to",
            "i'd like to",
            "i need to",
            "i should",
            "i will",
            "going to",
        ]

        # Habit patterns (routines, regular activities)
        habit_patterns = [
            "i usually",
            "i always",
            "i often",
            "i regularly",
            "every day",
            "every week",
            "every morning",
            "every night",
            "i tend to",
            "i typically",
            "my routine",
            "i never",
        ]

        # Event patterns (specific occurrences, meetings, happenings)
        event_patterns = [
            "happened",
            "occurred",
            "took place",
            "yesterday",
            "last week",
            "last month",
            "ago",
            "meeting",
            "appointment",
            "on",
            "at",
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
                    "assistant_message": assistant_message,
                },
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
                        "context": assistant_message[:200],
                    },
                )
                self.console.print(
                    f"[dim]📝 Automatically detected and stored {detected_type} memory[/dim]"
                )

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
                session_id=self.session_id,  # Also filter by session_id
                limit=limit,
            )
            # Convert RetrievalResult objects to dicts if needed
            if results and hasattr(results[0], "__dict__"):
                return [
                    {
                        "text": r.memory.text
                        if hasattr(r, "memory") and r.memory
                        else (r.text if hasattr(r, "text") else str(r)),
                        "score": r.score if hasattr(r, "score") else 1.0,
                        "metadata": r.memory.metadata
                        if hasattr(r, "memory") and r.memory
                        else (r.metadata if hasattr(r, "metadata") else {}),
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
            mem_text = memory.get("text", "")
            mem_metadata = memory.get("metadata", {})
            mem_type = mem_metadata.get("type", "memory")
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
            results = self.memory_client.recall(query="", user_id=self.user_id, limit=limit)

            if not results:
                self.console.print("[yellow]No memories found[/yellow]")
                return

            # Convert to list of dicts if needed
            if results and hasattr(results[0], "__dict__"):
                memories = [
                    {
                        "text": r.text if hasattr(r, "text") else str(r),
                        "metadata": r.metadata if hasattr(r, "metadata") else {},
                        "score": r.score if hasattr(r, "score") else 0,
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
                mem_metadata = memory.get("metadata", {})
                mem_type = mem_metadata.get("type", "memory")
                mem_text = memory.get("text", "")

                table.add_row(
                    mem_type,
                    mem_text[:100] + ("..." if len(mem_text) > 100 else ""),
                    f"{memory.get('score', 0):.2f}",
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
            if results and hasattr(results[0], "__dict__"):
                memories = [
                    {
                        "text": r.text if hasattr(r, "text") else str(r),
                        "metadata": r.metadata if hasattr(r, "metadata") else {},
                        "score": r.score if hasattr(r, "score") else 0,
                    }
                    for r in results
                ]
            else:
                memories = results

            self.console.print(f"\n[green]Found {len(memories)} matching memories:[/green]\n")

            for i, memory in enumerate(memories, 1):
                mem_metadata = memory.get("metadata", {})
                mem_type = mem_metadata.get("type", "memory")
                mem_text = memory.get("text", "")
                mem_tags = mem_metadata.get("tags", [])
                mem_score = memory.get("score", 0)

                self.console.print(
                    Panel(
                        f"[cyan]Type:[/cyan] {mem_type}\n"
                        f"[cyan]Text:[/cyan] {mem_text}\n"
                        f"[cyan]Score:[/cyan] {mem_score:.2f}\n"
                        f"[cyan]Tags:[/cyan] {', '.join(mem_tags) if mem_tags else 'None'}",
                        title=f"Memory {i}",
                        border_style="blue",
                    )
                )

        except Exception as e:
            self.console.print(f"[red]Error searching memories: {e}[/red]")

    def test_all_features(self) -> None:
        """Comprehensive test of all HippocampAI features."""
        self.console.print("\n[bold cyan]🧪 Running Comprehensive Feature Tests[/bold cyan]\n")

        test_results = []

        # Test 1: Basic Memory CRUD
        self.console.print("[yellow]Test 1: Basic Memory CRUD Operations[/yellow]")
        try:
            # Create
            memory = self.memory_client.remember(
                text="Test memory for CRUD operations",
                user_id=self.user_id,
                session_id=self.session_id,
                tags=["test", "crud"],
                importance=7.5,
            )
            memory_id = memory.id if hasattr(memory, "id") else str(memory)

            # Read
            _retrieved = self.memory_client.get_memory(memory_id)

            # Update
            _updated = self.memory_client.update_memory(
                memory_id=memory_id, text="Updated test memory", tags=["test", "crud", "updated"]
            )

            # Delete
            _deleted = self.memory_client.delete_memory(memory_id)

            test_results.append(("CRUD Operations", "✅ PASS"))
            self.console.print("[green]  ✅ PASS: Create, Read, Update, Delete[/green]")
        except Exception as e:
            test_results.append(("CRUD Operations", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 2: Batch Operations
        self.console.print("\n[yellow]Test 2: Batch Operations[/yellow]")
        try:
            batch_memories = [
                {"text": f"Batch memory {i}", "user_id": self.user_id, "tags": ["batch"]}
                for i in range(3)
            ]
            created = self.memory_client.batch_remember(batch_memories)

            # Get the IDs
            batch_ids = [m.id if hasattr(m, "id") else str(m) for m in created]

            # Batch get
            _retrieved = self.memory_client.batch_get_memories(batch_ids)

            # Batch delete
            _deleted = self.memory_client.batch_delete_memories(batch_ids)

            test_results.append(("Batch Operations", "✅ PASS"))
            self.console.print(
                f"[green]  ✅ PASS: Created, retrieved, deleted {len(batch_ids)} memories[/green]"
            )
        except Exception as e:
            test_results.append(("Batch Operations", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 3: Advanced Filtering
        self.console.print("\n[yellow]Test 3: Advanced Filtering & Search[/yellow]")
        try:
            # Create test memories with different attributes
            self.memory_client.remember(
                text="Important project deadline",
                user_id=self.user_id,
                tags=["work", "urgent"],
                importance=9.0,
            )

            # Filter by tags
            results = self.memory_client.recall(
                query="project", user_id=self.user_id, filters={"tags": ["work"]}, limit=5
            )

            # Filter by importance
            _results_important = self.memory_client.recall(
                query="deadline", user_id=self.user_id, min_score=0.5, limit=5
            )

            test_results.append(("Advanced Filtering", "✅ PASS"))
            self.console.print(
                f"[green]  ✅ PASS: Retrieved {len(results)} filtered memories[/green]"
            )
        except Exception as e:
            test_results.append(("Advanced Filtering", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 4: Entity Extraction
        self.console.print("\n[yellow]Test 4: Entity & Fact Extraction[/yellow]")
        try:
            _memory_with_entities = self.memory_client.remember(
                text="Meeting with John at Google office in San Francisco on Monday",
                user_id=self.user_id,
                extract_entities=True,
                extract_facts=True,
                extract_relationships=True,
            )

            test_results.append(("Entity Extraction", "✅ PASS"))
            self.console.print("[green]  ✅ PASS: Entity extraction enabled[/green]")
        except Exception as e:
            test_results.append(("Entity Extraction", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 5: Memory with Expiration
        self.console.print("\n[yellow]Test 5: Memory Expiration (TTL)[/yellow]")
        try:
            from datetime import timedelta

            expiry_time = datetime.now() + timedelta(days=7)

            _expiring_memory = self.memory_client.remember(
                text="This memory expires in 7 days",
                user_id=self.user_id,
                expires_at=expiry_time,
                tags=["temporary"],
            )

            test_results.append(("Memory Expiration", "✅ PASS"))
            self.console.print("[green]  ✅ PASS: Memory with expiration created[/green]")
        except Exception as e:
            test_results.append(("Memory Expiration", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 6: Memory Consolidation
        self.console.print("\n[yellow]Test 6: Memory Consolidation[/yellow]")
        try:
            # Create duplicate-like memories
            for i in range(3):
                self.memory_client.remember(
                    text=f"I like coffee variation {i}",
                    user_id=self.user_id,
                    tags=["preference", "duplicate-test"],
                )

            # Consolidate
            _consolidated = self.memory_client.consolidate_memories(
                user_id=self.user_id, lookback_hours=24
            )

            test_results.append(("Memory Consolidation", "✅ PASS"))
            self.console.print("[green]  ✅ PASS: Memory consolidation executed[/green]")
        except Exception as e:
            test_results.append(("Memory Consolidation", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 7: Cleanup Expired Memories
        self.console.print("\n[yellow]Test 7: Cleanup Expired Memories[/yellow]")
        try:
            cleaned = self.memory_client.cleanup_expired_memories()
            test_results.append(("Cleanup Expired", "✅ PASS"))
            self.console.print(f"[green]  ✅ PASS: Cleaned up {cleaned} expired memories[/green]")
        except Exception as e:
            test_results.append(("Cleanup Expired", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 8: Get All Memories
        self.console.print("\n[yellow]Test 8: Get All Memories[/yellow]")
        try:
            all_memories = self.memory_client.get_memories(user_id=self.user_id, limit=100)
            test_results.append(("Get All Memories", "✅ PASS"))
            self.console.print(
                f"[green]  ✅ PASS: Retrieved {len(all_memories)} total memories[/green]"
            )
        except Exception as e:
            test_results.append(("Get All Memories", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Test 9: Health Check
        self.console.print("\n[yellow]Test 9: System Health Check[/yellow]")
        try:
            health = self.memory_client.health_check()
            test_results.append(("Health Check", "✅ PASS"))
            self.console.print(
                f"[green]  ✅ PASS: System health: {health.get('status', 'unknown')}[/green]"
            )
        except Exception as e:
            test_results.append(("Health Check", f"❌ FAIL: {str(e)[:50]}"))
            self.console.print(f"[red]  ❌ FAIL: {e}[/red]")

        # Summary Table
        self.console.print("\n[bold cyan]📊 Test Summary[/bold cyan]")
        summary_table = Table(title="Feature Test Results")
        summary_table.add_column("Feature", style="cyan")
        summary_table.add_column("Result", style="white")

        for feature, result in test_results:
            summary_table.add_row(feature, result)

        self.console.print(summary_table)

        passed = sum(1 for _, r in test_results if "PASS" in r)
        total = len(test_results)
        self.console.print(f"\n[bold]Total: {passed}/{total} tests passed[/bold]")

    def show_analytics(self) -> None:
        """Display memory analytics for the current user."""
        try:
            self.console.print("\n[bold cyan]📈 Memory Analytics[/bold cyan]\n")

            analytics = self.memory_client.get_memory_analytics(self.user_id)

            analytics_table = Table(title=f"Analytics for User {self.user_id[:16]}...")
            analytics_table.add_column("Metric", style="cyan")
            analytics_table.add_column("Value", style="yellow")

            for key, value in analytics.items():
                analytics_table.add_row(str(key), str(value))

            self.console.print(analytics_table)

        except Exception as e:
            self.console.print(f"[red]Error fetching analytics: {e}[/red]")

    def show_health(self) -> None:
        """Display system health status."""
        try:
            self.console.print("\n[bold cyan]🏥 System Health Check[/bold cyan]\n")

            health = self.memory_client.health_check()

            health_table = Table(title="System Health Status")
            health_table.add_column("Component", style="cyan")
            health_table.add_column("Status", style="white")

            for key, value in health.items():
                status_color = "green" if value == "healthy" or value == "ok" else "red"
                health_table.add_row(str(key), f"[{status_color}]{value}[/{status_color}]")

            self.console.print(health_table)

        except Exception as e:
            self.console.print(f"[red]Error checking health: {e}[/red]")

    def compact_memories(self) -> None:
        """Compact and consolidate conversation memories with detailed metadata."""
        try:
            self.console.print("\n[bold cyan]📦 Memory Compaction[/bold cyan]\n")

            # Ask for memory types to compact
            self.console.print(
                "[dim]Available memory types: fact, event, context, preference, goal, habit[/dim]"
            )
            types_input = Prompt.ask(
                "[yellow]Memory types to compact (comma-separated, or 'all')[/yellow]",
                default="all",
            )

            memory_types = None
            if types_input.lower() != "all":
                memory_types = [t.strip() for t in types_input.split(",")]

            # First do a dry run to show what would happen
            self.console.print("\n[yellow]Running preview...[/yellow]")

            result = self.memory_client.compact_conversations(
                user_id=self.user_id,
                session_id=self.session_id,
                lookback_hours=168,  # 1 week
                dry_run=True,
                memory_types=memory_types,
            )

            metrics = result.get("metrics", {})

            # Show preview table
            preview_table = Table(title="Compaction Preview (Dry Run)", show_header=True)
            preview_table.add_column("Metric", style="cyan", width=25)
            preview_table.add_column("Value", style="yellow")

            preview_table.add_row("Input Memories", str(metrics.get("input_memories", 0)))
            preview_table.add_row("Input Tokens", f"{metrics.get('input_tokens', 0):,}")
            preview_table.add_row("Output Summaries", str(metrics.get("output_memories", 0)))
            preview_table.add_row("Output Tokens", f"{metrics.get('output_tokens', 0):,}")
            preview_table.add_row(
                "Compression Ratio", f"{metrics.get('compression_ratio', 0) * 100:.1f}%"
            )
            preview_table.add_row("Tokens Saved", f"{metrics.get('tokens_saved', 0):,}")
            preview_table.add_row(
                "Storage Saved", f"{metrics.get('estimated_storage_saved_bytes', 0):,} bytes"
            )
            preview_table.add_row("Clusters Found", str(metrics.get("clusters_found", 0)))
            preview_table.add_row(
                "Est. Cost",
                f"${metrics.get('estimated_input_cost', 0) + metrics.get('estimated_output_cost', 0):.4f}",
            )

            self.console.print(preview_table)

            # Show types breakdown
            types_compacted = metrics.get("types_compacted", {})
            if types_compacted:
                self.console.print("\n[dim]Types breakdown:[/dim]")
                for t, count in types_compacted.items():
                    self.console.print(f"  • {t}: {count}")

            # Show insights
            insights = result.get("insights", [])
            if insights:
                self.console.print("\n[bold]💡 Insights:[/bold]")
                for insight in insights:
                    self.console.print(f"  {insight}")

            # Ask for confirmation
            confirm = Prompt.ask(
                "\n[yellow]Run compaction for real?[/yellow]", choices=["y", "n"], default="n"
            )

            if confirm.lower() == "y":
                self.console.print("\n[cyan]Running compaction...[/cyan]")

                result = self.memory_client.compact_conversations(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    lookback_hours=168,
                    dry_run=False,
                    memory_types=memory_types,
                )

                if result.get("status") == "completed":
                    self.console.print("\n[green]✓ Compaction complete![/green]\n")

                    # Show final metrics
                    final_metrics = result.get("metrics", {})
                    result_table = Table(title="Compaction Results", show_header=True)
                    result_table.add_column("Metric", style="cyan", width=25)
                    result_table.add_column("Value", style="green")

                    result_table.add_row(
                        "Memories Merged", str(final_metrics.get("memories_merged", 0))
                    )
                    result_table.add_row(
                        "Tokens Saved", f"{final_metrics.get('tokens_saved', 0):,}"
                    )
                    result_table.add_row(
                        "Storage Saved",
                        f"{final_metrics.get('estimated_storage_saved_bytes', 0):,} bytes",
                    )
                    result_table.add_row(
                        "Compression", f"{final_metrics.get('compression_ratio', 0) * 100:.1f}%"
                    )
                    result_table.add_row(
                        "Key Facts Preserved", str(final_metrics.get("key_facts_preserved", 0))
                    )
                    result_table.add_row(
                        "Entities Preserved", str(final_metrics.get("entities_preserved", 0))
                    )
                    result_table.add_row(
                        "Context Retention",
                        f"{final_metrics.get('context_retention_score', 0) * 100:.0f}%",
                    )
                    result_table.add_row(
                        "Duration", f"{final_metrics.get('duration_seconds', 0):.2f}s"
                    )

                    self.console.print(result_table)

                    # Show preserved facts
                    preserved_facts = result.get("preserved_facts", [])
                    if preserved_facts:
                        self.console.print("\n[bold]🧠 Preserved Key Facts:[/bold]")
                        for fact in preserved_facts[:5]:
                            self.console.print(f"  • {fact[:80]}{'...' if len(fact) > 80 else ''}")

                    # Show preserved entities
                    preserved_entities = result.get("preserved_entities", [])
                    if preserved_entities:
                        self.console.print(
                            f"\n[bold]👤 Preserved Entities:[/bold] {', '.join(preserved_entities[:10])}"
                        )

                    # Show final insights
                    final_insights = result.get("insights", [])
                    if final_insights:
                        self.console.print("\n[bold]💡 Final Insights:[/bold]")
                        for insight in final_insights:
                            self.console.print(f"  {insight}")

                    self.console.print(f"\n[dim]{result.get('summary', '')}[/dim]")

                elif result.get("status") == "skipped":
                    self.console.print(
                        f"[yellow]⚠ Compaction skipped: {result.get('summary', 'Not enough memories')}[/yellow]"
                    )
                else:
                    self.console.print(
                        f"[red]Compaction failed: {result.get('error', 'Unknown error')}[/red]"
                    )
            else:
                self.console.print("[dim]Compaction cancelled.[/dim]")

        except Exception as e:
            self.console.print(f"[red]Error during compaction: {e}[/red]")

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.console.print(
            Panel.fit(
                "[bold cyan]Groq + HippocampAI Chat Demo[/bold cyan]\n"
                f"Model: {self.model}\n"
                f"User ID: {self.user_id[:16]}...\n"
                f"Session ID: {self.session_id[:16]}...\n\n"
                "[dim]Commands:[/dim]\n"
                "  /memories - Show stored memories\n"
                "  /search - Search memories\n"
                "  /info - Show full session info\n"
                "  /clear - Clear conversation history\n"
                "  /test - Run feature tests\n"
                "  /analytics - Show memory analytics\n"
                "  /health - Check system health\n"
                "  /help - Show this help\n"
                "  /quit - Exit\n",
                border_style="green",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower().strip()

                    if command == "/quit" or command == "/exit":
                        self.console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    elif command == "/memories":
                        self.show_memories()
                        continue
                    elif command == "/search":
                        self.search_memories_interactive()
                        continue
                    elif command == "/clear":
                        self.conversation_history.clear()
                        self.console.print("[green]✓ Conversation history cleared[/green]")
                        continue
                    elif command == "/info":
                        self.console.print(
                            Panel(
                                f"[cyan]User ID:[/cyan] {self.user_id}\n"
                                f"[cyan]Session ID:[/cyan] {self.session_id}\n"
                                f"[cyan]Model:[/cyan] {self.model}",
                                title="Session Information",
                                border_style="cyan",
                            )
                        )
                        continue
                    elif command == "/test":
                        self.test_all_features()
                        continue
                    elif command == "/analytics":
                        self.show_analytics()
                        continue
                    elif command == "/health":
                        self.show_health()
                        continue
                    elif command == "/help":
                        self.console.print(
                            "[cyan]Available commands:[/cyan]\n"
                            "  /memories - Show stored memories\n"
                            "  /search - Search memories\n"
                            "  /info - Show session information\n"
                            "  /clear - Clear conversation history\n"
                            "  /test - Run comprehensive feature tests\n"
                            "  /analytics - Show memory analytics\n"
                            "  /health - Check system health\n"
                            "  /compact - Compact conversation memories\n"
                            "  /help - Show this help\n"
                            "  /quit - Exit"
                        )
                        continue
                    elif command == "/compact":
                        self.compact_memories()
                        continue
                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                        continue

                # Get response from chat
                with self.console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                    response = self.chat(user_input)

                # Display response
                self.console.print("\n[bold blue]Assistant[/bold blue]")
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

  # With specific session ID (continue previous session)
  python groq_llama_chat_demo.py --session-id sess-abc123

  # Full example with all options
  python groq_llama_chat_demo.py --user-id alice --session-id test-session-1 --qdrant-url http://localhost:6333

Environment Variables:
  GROQ_API_KEY            Your Groq API key (required)
  HIPPOCAMPAI_API_KEY     Your HippocampAI API key (for remote mode)
        """,
    )

    parser.add_argument(
        "--user-id", type=str, help="User ID for memory storage (auto-generated if not provided)"
    )

    parser.add_argument(
        "--session-id",
        type=str,
        help="Session ID for conversation tracking (auto-generated if not provided)",
    )

    parser.add_argument(
        "--base-url", type=str, help="HippocampAI API base URL (e.g., http://localhost:8000)"
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        help="Qdrant server URL for local mode (e.g., http://100.113.229.40:6333)",
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        help="Redis server URL for local mode (e.g., redis://localhost:6379)",
    )

    args = parser.parse_args()

    # If session_id is provided but user_id is not, try to look up the user_id
    user_id = args.user_id
    if args.session_id and not args.user_id:
        user_id = lookup_user_id_from_session(args.session_id, args.qdrant_url)

    # Create and run chat
    chat = GroqHippocampAIChat(
        user_id=user_id,
        session_id=args.session_id,
        base_url=args.base_url,
        qdrant_url=args.qdrant_url,
        redis_url=args.redis_url,
    )

    chat.run()


def lookup_user_id_from_session(session_id: str, qdrant_url: str = None) -> str:
    """Look up the user_id associated with a session_id from Qdrant.

    Args:
        session_id: The session ID to look up
        qdrant_url: Qdrant server URL

    Returns:
        The user_id if found, otherwise the session_id itself
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)

        collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts"]

        # First, look for signup record with metadata.session_id
        for collection in collections:
            try:
                if not client.collection_exists(collection):
                    continue

                signup_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.session_id", match=MatchValue(value=session_id)
                        )
                    ]
                )
                results, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=signup_filter,
                    limit=1,
                    with_payload=True,
                )
                if results:
                    user_id = results[0].payload.get("user_id")
                    if user_id:
                        print(
                            f"[green]✓ Found user_id '{user_id}' for session '{session_id[:16]}...'[/green]"
                        )
                        return user_id
            except Exception:
                continue

        # Fallback: look for any record with this session_id
        for collection in collections:
            try:
                if not client.collection_exists(collection):
                    continue

                session_filter = Filter(
                    must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
                results, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=session_filter,
                    limit=1,
                    with_payload=True,
                )
                if results:
                    user_id = results[0].payload.get("user_id")
                    if user_id:
                        print(f"[yellow]⚠ Using user_id '{user_id}' from session records[/yellow]")
                        return user_id
            except Exception:
                continue

        print(
            f"[yellow]⚠ Could not find user_id for session '{session_id}', using session_id as user_id[/yellow]"
        )
        return session_id

    except Exception as e:
        print(f"[yellow]⚠ Error looking up user_id: {e}[/yellow]")
        return session_id


if __name__ == "__main__":
    main()
