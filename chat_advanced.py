#!/usr/bin/env python3
"""Advanced Interactive Chat with Full HippocampAI Features.

This script demonstrates ALL HippocampAI features including:
- Memory conflict resolution
- Health monitoring and quality reports
- Auto-summarization and consolidation
- Metrics and tracing
- Advanced compression
- Pattern detection

Usage:
    python chat_advanced.py

Requirements:
    - GROQ_API_KEY environment variable set
    - Qdrant running (docker run -d -p 6333:6333 qdrant/qdrant)
"""

import os
import sys
from datetime import datetime

try:
    from hippocampai import MemoryClient
    from hippocampai.adapters import GroqLLM
    from hippocampai.embed.embedder import Embedder
    from hippocampai.monitoring import (
        MemoryHealthMonitor,
        MetricsCollector,
        MonitoringStorage,
        OperationType,
    )
    from hippocampai.pipeline import (
        AdvancedCompression,
        AutoConsolidation,
        AutoSummarization,
    )
except ImportError:
    print("ERROR: hippocampai not installed. Install with: pip install -e .")
    sys.exit(1)


class AdvancedMemoryChatBot:
    """Interactive chatbot with ALL advanced HippocampAI features."""

    def __init__(self, user_id: str = "demo_user"):
        """Initialize chatbot with full feature set."""
        self.user_id = user_id

        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY environment variable not set")
            print("Get your API key from: https://console.groq.com/keys")
            sys.exit(1)

        self.llm = GroqLLM(api_key=api_key, model="llama-3.1-8b-instant")

        # Initialize HippocampAI memory client
        try:
            self.memory_client = MemoryClient(llm_provider=self.llm)
            print("✓ Connected to HippocampAI memory engine")
        except Exception as e:
            print(f"ERROR: Failed to connect to HippocampAI: {e}")
            print("Make sure Qdrant is running: docker run -d -p 6333:6333 qdrant/qdrant")
            sys.exit(1)

        # Initialize embedder for health monitoring
        self.embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")

        # Initialize health monitor
        self.health_monitor = MemoryHealthMonitor(embedder=self.embedder)

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(enable_tracing=True)

        # Initialize monitoring storage
        self.monitoring_storage = MonitoringStorage(
            qdrant_store=self.memory_client.memory_service.qdrant_store
        )

        # Initialize auto-summarization
        self.auto_summarizer = AutoSummarization(
            llm=self.llm,
            embedder=self.embedder,
            memory_service=self.memory_client.memory_service,
        )

        # Initialize auto-consolidation
        self.auto_consolidator = AutoConsolidation(
            llm=self.llm,
            embedder=self.embedder,
            memory_service=self.memory_client.memory_service,
        )

        # Initialize advanced compression
        self.compressor = AdvancedCompression(
            llm=self.llm,
            embedder=self.embedder,
            memory_service=self.memory_client.memory_service,
        )

        # Create a session
        self.session = self.memory_client.create_session(
            user_id=self.user_id,
            title=f"Advanced Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"✓ Created session: {self.session.id[:8]}...")
        print("✓ All advanced features enabled!")

        self.conversation_history = []

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant memories with tracing."""
        with self.metrics_collector.trace_operation(
            OperationType.SEARCH,
            tags={"component": "chat", "user_id": self.user_id},
            user_id=self.user_id,
        ):
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
        """Extract insights from conversation with conflict detection."""
        with self.metrics_collector.trace_operation(
            OperationType.CREATE,
            tags={"component": "chat", "user_id": self.user_id},
            user_id=self.user_id,
        ):
            try:
                conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"
                memories = self.memory_client.extract_from_conversation(
                    conversation=conversation_text, user_id=self.user_id
                )

                if memories:
                    print(f"  💾 Stored {len(memories)} new memories")

                # Track in session
                self.memory_client.track_session_message(
                    session_id=self.session.id, role="user", content=user_message
                )
                self.memory_client.track_session_message(
                    session_id=self.session.id,
                    role="assistant",
                    content=assistant_message,
                )

            except Exception as e:
                print(f"  Warning: Memory extraction failed: {e}")

    def chat(self, user_message: str) -> str:
        """Process user message with full tracing."""
        # Get relevant context
        context = self.get_relevant_context(user_message)

        # Build system prompt
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

        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": full_prompt})

        # Generate response with tracing
        try:
            response = self.llm.chat(messages, max_tokens=512, temperature=0.7)
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Extract and store memories
        self.extract_and_store_memories(user_message, response)

        return response

    def show_health_report(self):
        """Generate and display comprehensive health report."""
        print("\n  Generating health report...")

        with self.metrics_collector.trace_operation(
            OperationType.SEARCH,
            tags={"component": "health", "user_id": self.user_id},
            user_id=self.user_id,
        ):
            try:
                # Get all memories
                memories = self.memory_client.get_memories(user_id=self.user_id, limit=1000)

                if not memories:
                    print("  No memories to analyze yet.")
                    return

                # Generate health report
                report = self.health_monitor.generate_quality_report(memories, user_id=self.user_id)

                # Store in Qdrant
                self.monitoring_storage.store_health_report(
                    report, tags={"source": "chat_advanced", "user_id": self.user_id}
                )

                print("\n" + "=" * 70)
                print("🏥 MEMORY HEALTH REPORT")
                print("=" * 70)
                print(f"Overall Health Score: {report.health_score.overall_score:.1f}/100")
                print(f"Status: {report.health_score.status}")
                print("\nComponent Scores:")
                print(f"  • Freshness: {report.health_score.freshness_score:.1f}/100")
                print(f"  • Diversity: {report.health_score.diversity_score:.1f}/100")
                print(f"  • Consistency: {report.health_score.consistency_score:.1f}/100")
                print(f"  • Coverage: {report.health_score.coverage_score:.1f}/100")
                print("\nMemory Counts:")
                print(f"  • Total: {report.health_score.total_memories}")
                print(f"  • Healthy: {report.health_score.healthy_memories}")
                print(f"  • Stale: {report.health_score.stale_memories}")
                print(f"  • Duplicate Clusters: {report.health_score.duplicate_clusters}")
                print(f"  • Low Quality: {report.health_score.low_quality_memories}")

                if report.health_score.recommendations:
                    print("\n💡 Recommendations:")
                    for rec in report.health_score.recommendations:
                        print(f"  • {rec}")

                print("=" * 70)

            except Exception as e:
                print(f"  Failed to generate health report: {e}")

    def run_auto_summarization(self):
        """Run auto-summarization on user's memories."""
        print("\n  Running auto-summarization...")

        try:
            # Get current memory count
            memories = self.memory_client.get_memories(user_id=self.user_id, limit=1000)
            count = len(memories)

            if count < 10:
                print(f"  Not enough memories to summarize ({count}). Need at least 10.")
                return

            # Run summarization
            result = self.auto_summarizer.summarize_memories(user_id=self.user_id)

            print("\n" + "=" * 70)
            print("📝 AUTO-SUMMARIZATION RESULTS")
            print("=" * 70)
            print(f"Memories processed: {result['memories_processed']}")
            print(f"Summaries created: {result['summaries_created']}")
            print(f"Memories archived: {result['memories_archived']}")
            print(f"Space saved: {result['space_saved']:.1%}")
            print("=" * 70)

        except Exception as e:
            print(f"  Failed to run summarization: {e}")

    def run_consolidation(self):
        """Run consolidation to merge related memories."""
        print("\n  Running memory consolidation...")

        try:
            result = self.auto_consolidator.consolidate_memories(user_id=self.user_id)

            print("\n" + "=" * 70)
            print("🔄 CONSOLIDATION RESULTS")
            print("=" * 70)
            print(f"Memories processed: {result['memories_processed']}")
            print(f"Clusters found: {result['clusters_found']}")
            print(f"Consolidated memories: {result['consolidated_memories']}")
            print(f"Original memories archived: {result['memories_archived']}")
            print(f"Reduction: {result['reduction_percentage']:.1%}")
            print("=" * 70)

        except Exception as e:
            print(f"  Failed to run consolidation: {e}")

    def run_compression(self):
        """Run advanced compression on memories."""
        print("\n  Running advanced compression...")

        try:
            result = self.compressor.compress_memories(user_id=self.user_id)

            print("\n" + "=" * 70)
            print("🗜️ COMPRESSION RESULTS")
            print("=" * 70)
            print(f"Memories processed: {result['memories_processed']}")
            print(f"Compressed memories: {result['compressed_memories']}")
            print(f"Compression ratio: {result['compression_ratio']:.2f}x")
            print(f"Space saved: {result['space_saved']:.1%}")
            print("=" * 70)

        except Exception as e:
            print(f"  Failed to run compression: {e}")

    def show_metrics(self):
        """Display metrics and traces."""
        print("\n" + "=" * 70)
        print("📊 METRICS & TRACING")
        print("=" * 70)

        summary = self.metrics_collector.get_metrics_summary()
        print(f"Total operations: {summary['total_traces']}")
        print(f"Successful: {summary['successful_operations']}")
        print(f"Failed: {summary['failed_operations']}")

        if summary["operation_stats"]:
            print("\nOperation breakdown:")
            for op, stats in summary["operation_stats"].items():
                print(
                    f"  • {op}: {stats['count']} ops, "
                    f"avg {stats['avg_duration_ms']:.1f}ms, "
                    f"error rate {stats['error_rate']:.1%}"
                )

        print("=" * 70)

    def show_trace_history(self, limit: int = 10):
        """Show recent operation traces from Qdrant storage."""
        print("\n  Querying trace history from Qdrant...")

        try:
            traces = self.monitoring_storage.query_traces(user_id=self.user_id, limit=limit)

            print("\n" + "=" * 70)
            print(f"🔍 RECENT TRACES (Last {limit})")
            print("=" * 70)

            for trace in traces[:limit]:
                print(f"\nOperation: {trace['operation']}")
                print(
                    f"Duration: {trace.get('duration_ms', 0):.1f}ms | Success: {trace['success']}"
                )
                print(f"Tags: {trace.get('tags', {})}")
                if trace.get("error"):
                    print(f"Error: {trace['error']}")

            print("=" * 70)

        except Exception as e:
            print(f"  Failed to query traces: {e}")

    def complete_session(self):
        """Complete the session and generate summary."""
        try:
            self.memory_client.complete_session(session_id=self.session.id, generate_summary=True)

            session = self.memory_client.get_session(self.session.id)

            print("\n" + "=" * 70)
            print("📝 SESSION SUMMARY")
            print("=" * 70)
            if session.summary:
                print(session.summary)
            else:
                print("Summary generation in progress...")
            print("=" * 70)

        except Exception as e:
            print(f"Failed to complete session: {e}")


def print_help():
    """Print available commands."""
    print("\n" + "=" * 70)
    print("💡 AVAILABLE COMMANDS")
    print("=" * 70)
    print("Basic Commands:")
    print("  /help       - Show this help message")
    print("  /stats      - Show memory statistics")
    print("  /memories   - Show recent memories")
    print("  /patterns   - Detect behavioral patterns")
    print("  /search     - Search memories (e.g., /search coffee)")
    print("\nAdvanced Features:")
    print("  /health     - Generate comprehensive health report")
    print("  /summarize  - Run auto-summarization")
    print("  /consolidate - Run memory consolidation")
    print("  /compress   - Run advanced compression")
    print("  /metrics    - Show metrics and tracing stats")
    print("  /traces     - Show recent operation traces")
    print("\nOther:")
    print("  /clear      - Clear screen")
    print("  /exit       - Exit chat (with session summary)")
    print("=" * 70)


def main():
    """Main chat loop."""
    print("\n" + "=" * 70)
    print("  🚀 HippocampAI Advanced Interactive Chat")
    print("=" * 70)
    print("\n  Featuring ALL advanced capabilities:")
    print("    • Memory conflict resolution")
    print("    • Health monitoring & quality reports")
    print("    • Auto-summarization & consolidation")
    print("    • Advanced compression")
    print("    • Metrics & distributed tracing")
    print("    • Qdrant storage for monitoring data")

    # Get user ID
    user_input = input("\n  Enter your name (or press Enter for 'demo_user'): ").strip()
    user_id = user_input if user_input else "demo_user"

    # Initialize chatbot
    print(f"\n  Initializing advanced chatbot for user: {user_id}...")
    try:
        bot = AdvancedMemoryChatBot(user_id=user_id)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print_help()

    print("\n  Type your message (or /help for commands, /exit to quit)")
    print("=" * 70)

    # Main chat loop
    while True:
        try:
            user_message = input(f"\n{user_id}> ").strip()

            if not user_message:
                continue

            # Handle commands
            if user_message.startswith("/"):
                cmd = user_message.split()[0].lower()

                if cmd == "/exit":
                    print("\n  Completing session and generating summary...")
                    bot.complete_session()
                    print("\n  Goodbye! Your memories are saved for next time. 👋")
                    break

                elif cmd == "/help":
                    print_help()

                elif cmd == "/stats":
                    bot.memory_client.get_memory_statistics(user_id=user_id)

                elif cmd == "/memories":
                    memories = bot.memory_client.get_memories(user_id=user_id, limit=10)
                    print(f"\n  Showing {len(memories)} recent memories...")
                    for i, mem in enumerate(memories, 1):
                        print(f"  {i}. [{mem.type.value}] {mem.text}")

                elif cmd == "/patterns":
                    patterns = bot.memory_client.detect_patterns(user_id=user_id)
                    print(f"\n  Detected {len(patterns)} patterns")

                elif cmd == "/search":
                    query = user_message[8:].strip()
                    if query:
                        results = bot.memory_client.recall(query=query, user_id=user_id, k=5)
                        print(f"\n  Found {len(results)} results")
                    else:
                        print("  Usage: /search <query>")

                elif cmd == "/health":
                    bot.show_health_report()

                elif cmd == "/summarize":
                    bot.run_auto_summarization()

                elif cmd == "/consolidate":
                    bot.run_consolidation()

                elif cmd == "/compress":
                    bot.run_compression()

                elif cmd == "/metrics":
                    bot.show_metrics()

                elif cmd == "/traces":
                    bot.show_trace_history(limit=10)

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
            import traceback

            traceback.print_exc()
            print("  Type /exit to quit or continue chatting.")


if __name__ == "__main__":
    main()
