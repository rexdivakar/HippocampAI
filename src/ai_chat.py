"""Memory-enhanced AI chat assistant.

This module provides a complete chat assistant that integrates all HippocampAI features:
- Smart memory retrieval before responses
- Memory extraction after conversations
- Session management
- Deduplication and consolidation
- Importance scoring and decay

Usage:
    # CLI mode
    from src.ai_chat import MemoryEnhancedChat

    chat = MemoryEnhancedChat(user_id="user_123")
    response = chat.send_message("Hello! I love hiking.")
    print(response)

    # Web API mode
    See web_chat.py for Flask integration
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from src.qdrant_client import QdrantManager
from src.embedding_service import EmbeddingService
from src.memory_store import MemoryStore, MemoryType, Category
from src.memory_retriever import MemoryRetriever
from src.memory_extractor import MemoryExtractor
from src.memory_deduplicator import MemoryDeduplicator
from src.memory_updater import MemoryUpdater
from src.importance_scorer import ImportanceScorer
from src.session_manager import SessionManager
from src.memory_consolidator import MemoryConsolidator
from src.llm_provider import get_llm_client
from src.settings import get_settings
from src.tools import create_default_registry, ToolRegistry


logger = logging.getLogger(__name__)


class MemoryEnhancedChat:
    """
    AI chat assistant with comprehensive memory management.

    Features:
    - Retrieves relevant memories before each response
    - Extracts new memories from conversations
    - Manages conversation sessions
    - Deduplicates and consolidates memories
    - Tracks importance and handles decay

    Example:
        chat = MemoryEnhancedChat(user_id="alice")

        # Start conversation
        response = chat.send_message("Hi! I'm planning a trip to Japan.")
        # AI knows about user's previous preferences, plans, etc.

        # Continue conversation
        response = chat.send_message("What should I pack?")
        # AI remembers the Japan trip from previous message

        # End session (optional, auto-ends on timeout)
        chat.end_conversation()
    """

    def __init__(
        self,
        user_id: str,
        auto_extract_memories: bool = True,
        auto_consolidate: bool = False,
        memory_retrieval_limit: int = 10,
        system_prompt: Optional[str] = None,
        enable_tools: bool = True
    ):
        """
        Initialize memory-enhanced chat assistant.

        Args:
            user_id: Unique identifier for the user
            auto_extract_memories: Automatically extract memories after each turn
            auto_consolidate: Automatically consolidate memories (runs periodically)
            memory_retrieval_limit: Number of memories to retrieve for context
            system_prompt: Custom system prompt (default: friendly assistant)
            enable_tools: Enable tool calling (web search, calculator, etc.)
        """
        self.user_id = user_id
        self.auto_extract = auto_extract_memories
        self.auto_consolidate = auto_consolidate
        self.memory_limit = memory_retrieval_limit
        self.enable_tools = enable_tools

        # Load settings
        self.settings = get_settings()

        logger.info(f"Initializing MemoryEnhancedChat for user: {user_id}")

        # Initialize core services
        self._initialize_services()

        # Initialize tools
        if self.enable_tools:
            self.tool_registry = create_default_registry()
            logger.info(f"Tools enabled: {[t.name for t in self.tool_registry.get_all_tools()]}")
        else:
            self.tool_registry = None

        # Session tracking
        self.current_session_id: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        logger.info(f"MemoryEnhancedChat initialized for user: {user_id}")

    def _initialize_services(self) -> None:
        """Initialize all HippocampAI services."""
        # Core infrastructure
        self.qdrant = QdrantManager(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port
        )
        self.qdrant.create_collections()

        self.embeddings = EmbeddingService(
            model_name=self.settings.embedding.model
        )

        # Memory services
        self.memory_store = MemoryStore(
            qdrant_manager=self.qdrant,
            embedding_service=self.embeddings
        )

        self.retriever = MemoryRetriever(
            qdrant_manager=self.qdrant,
            embedding_service=self.embeddings
        )

        self.extractor = MemoryExtractor()

        self.deduplicator = MemoryDeduplicator(
            retriever=self.retriever,
            embedding_service=self.embeddings
        )

        self.updater = MemoryUpdater(
            qdrant_manager=self.qdrant,
            retriever=self.retriever,
            embedding_service=self.embeddings
        )

        self.scorer = ImportanceScorer()

        # Session management
        self.session_manager = SessionManager(
            memory_store=self.memory_store,
            retriever=self.retriever,
            embedding_service=self.embeddings
        )

        # Consolidation (for periodic cleanup)
        self.consolidator = MemoryConsolidator(
            retriever=self.retriever,
            updater=self.updater,
            embedding_service=self.embeddings
        )

        # LLM client
        self.llm_client = get_llm_client()

        logger.info("All services initialized successfully")

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return """You are a helpful AI assistant with memory. Keep responses concise and natural.

When answering:
- Be direct and to the point
- Use user's remembered context when relevant
- Don't over-explain or offer unnecessary follow-up questions
- Keep responses under 3-4 sentences unless more detail is specifically requested

For factual queries (like weather, distances, calculations), provide the answer directly without extra elaboration."""

    def send_message(
        self,
        message: str,
        context_type: Optional[str] = None
    ) -> str:
        """
        Send a message and get AI response with memory context.

        Args:
            message: User's message
            context_type: Optional context hint ("work", "personal", "casual")

        Returns:
            AI assistant's response

        Example:
            response = chat.send_message("What are good restaurants in Tokyo?")
        """
        logger.info(f"Processing message from user {self.user_id}: {message[:50]}...")

        # Start session if not active
        if not self.current_session_id:
            self._start_session()

        # Add user message to session
        self.session_manager.add_message(
            session_id=self.current_session_id,
            role="user",
            content=message
        )

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # Step 1: Retrieve relevant memories
        memories = self._retrieve_relevant_memories(message, context_type)

        # Step 2: Build context with memories
        context = self._build_context(memories)

        # Step 3: Generate AI response
        response = self._generate_response(message, context)

        # Add assistant response to session
        self.session_manager.add_message(
            session_id=self.current_session_id,
            role="assistant",
            content=response
        )

        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # Step 4: Extract and store new memories
        if self.auto_extract:
            self._extract_and_store_memories(message, response)

        # Step 5: Periodic consolidation (every N messages)
        if self.auto_consolidate and len(self.conversation_history) % 20 == 0:
            self._consolidate_memories()

        logger.info(f"Response generated for user {self.user_id}")

        return response

    def _start_session(self) -> None:
        """Start a new conversation session."""
        self.current_session_id = self.session_manager.start_session(
            user_id=self.user_id
        )
        logger.info(f"Started session: {self.current_session_id}")

    def _retrieve_relevant_memories(
        self,
        query: str,
        context_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for the query.

        Args:
            query: User's message
            context_type: Optional context hint

        Returns:
            List of relevant memories
        """
        try:
            # Use smart search for multi-factor ranking
            memories = self.retriever.smart_search(
                query=query,
                user_id=self.user_id,
                context_type=context_type,
                limit=self.memory_limit
            )

            logger.info(f"Retrieved {len(memories)} relevant memories")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def _build_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Build context string from memories.

        Args:
            memories: List of memory objects

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        context_parts = ["\n=== CONTEXT ABOUT USER ==="]

        # Group memories by type
        by_type = {}
        for mem in memories:
            mem_type = mem.get('memory_type', 'other')
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(mem)

        # Format by type
        type_labels = {
            'preference': 'Preferences',
            'fact': 'Facts',
            'goal': 'Goals',
            'habit': 'Habits',
            'event': 'Recent Events',
            'context': 'Background'
        }

        for mem_type, mems in by_type.items():
            label = type_labels.get(mem_type, mem_type.title())
            context_parts.append(f"\n{label}:")
            for mem in mems[:5]:  # Limit per type
                text = mem.get('text', '')
                importance = mem.get('importance', 0)
                context_parts.append(f"  - {text} [importance: {importance}/10]")

        context_parts.append("\n=== END CONTEXT ===\n")

        return "\n".join(context_parts)

    def _generate_response(self, message: str, context: str) -> str:
        """
        Generate AI response using LLM.

        Args:
            message: User's message
            context: Memory context string

        Returns:
            AI response
        """
        # Build messages for chat
        messages = [
            {"role": "system", "content": self.system_prompt + context}
        ]

        # Add recent conversation history (last 10 messages)
        recent_history = self.conversation_history[-10:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current message
        messages.append({"role": "user", "content": message})

        # Get tools schema if enabled
        tools = None
        if self.enable_tools and self.tool_registry:
            tools = self.tool_registry.get_tools_schema()

        # Generate response
        try:
            response = self.llm_client.chat(messages=messages, tools=tools)

            # Check if tools were called
            if isinstance(response, dict) and response.get("type") == "tool_calls":
                # Handle tool calls
                tool_results = self._handle_tool_calls(response["tool_calls"])

                # Add assistant message with tool use
                assistant_message = response["message"]
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })

                # Add user message with tool results
                tool_result_content = []
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_result = tool_results.get(tool_name, {})

                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    })

                messages.append({
                    "role": "user",
                    "content": tool_result_content
                })

                # Regenerate response with tool results
                final_response = self.llm_client.chat(messages=messages)
                return final_response if isinstance(final_response, str) else final_response

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Dictionary mapping tool names to their results
        """
        results = {}

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Execute tool
            result = self.tool_registry.execute_tool(tool_name, **tool_args)
            results[tool_name] = result

            logger.info(f"Tool {tool_name} completed: {result.get('success', False)}")

        return results

    def _extract_and_store_memories(
        self,
        user_message: str,
        assistant_message: str
    ) -> None:
        """
        Extract and store memories from conversation turn.

        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        try:
            # Create conversation text
            conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"

            # Extract memories
            memories = self.extractor.extract_memories(
                conversation_text=conversation_text,
                user_id=self.user_id
            )

            if not memories:
                logger.debug("No memories extracted from conversation")
                return

            logger.info(f"Extracted {len(memories)} potential memories")

            # Process each memory through deduplication
            for memory in memories:
                result = self.deduplicator.process_new_memory(
                    new_memory=memory,
                    user_id=self.user_id,
                    auto_decide=True  # Automatically handle duplicates
                )

                action = result.get('action', 'unknown')

                if action == 'store':
                    # Store new memory
                    self.memory_store.store_memory(
                        text=memory['text'],
                        memory_type=memory['memory_type'],
                        metadata={
                            'user_id': self.user_id,
                            'importance': memory.get('importance', 5),
                            'category': memory.get('category', Category.PERSONAL.value),
                            'session_id': self.current_session_id,
                            'confidence': memory.get('confidence', 0.8)
                        }
                    )
                    logger.info(f"Stored new memory: {memory['text'][:50]}...")

                elif action == 'replace':
                    # Replace existing memory
                    existing_id = result.get('existing_memory_id')
                    if existing_id:
                        self.updater.update_memory(
                            memory_id=existing_id,
                            new_text=memory['text'],
                            reason="Updated from conversation"
                        )
                        logger.info(f"Updated existing memory: {existing_id}")

                elif action == 'merge':
                    # Merge with existing
                    existing_id = result.get('existing_memory_id')
                    if existing_id:
                        # Store as new and let consolidation handle merging later
                        self.memory_store.store_memory(
                            text=memory['text'],
                            memory_type=memory['memory_type'],
                            metadata={
                                'user_id': self.user_id,
                                'importance': memory.get('importance', 5),
                                'category': memory.get('category', Category.PERSONAL.value),
                                'session_id': self.current_session_id,
                                'confidence': memory.get('confidence', 0.8),
                                'merge_candidate': existing_id
                            }
                        )
                        logger.info(f"Stored memory for later merge with: {existing_id}")

                elif action == 'skip':
                    logger.debug(f"Skipped duplicate memory: {memory['text'][:50]}...")

        except Exception as e:
            logger.error(f"Error extracting/storing memories: {e}")

    def _consolidate_memories(self) -> None:
        """Consolidate similar memories to reduce redundancy."""
        try:
            logger.info(f"Running memory consolidation for user: {self.user_id}")

            # Find clusters of similar memories
            clusters = self.consolidator.find_similar_clusters(
                user_id=self.user_id,
                similarity_threshold=self.settings.memory.consolidation_threshold
            )

            if not clusters:
                logger.info("No memory clusters found for consolidation")
                return

            # Consolidate each cluster
            for cluster in clusters[:3]:  # Limit to 3 clusters per run
                self.consolidator.consolidate_cluster(
                    cluster=cluster,
                    user_id=self.user_id
                )

            logger.info(f"Consolidated {len(clusters[:3])} memory clusters")

        except Exception as e:
            logger.error(f"Error during consolidation: {e}")

    def end_conversation(self) -> Optional[str]:
        """
        End the current conversation session.

        Returns:
            Session summary memory ID if session was active

        Example:
            summary_id = chat.end_conversation()
        """
        if not self.current_session_id:
            logger.warning("No active session to end")
            return None

        logger.info(f"Ending session: {self.current_session_id}")

        # End session and get summary
        summary_id = self.session_manager.end_session(
            session_id=self.current_session_id
        )

        # Clear session state
        self.current_session_id = None
        self.conversation_history = []

        logger.info(f"Session ended, summary stored: {summary_id}")

        return summary_id

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get current conversation history.

        Returns:
            List of messages with role, content, timestamp
        """
        return self.conversation_history.copy()

    def get_user_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get user's stored memories.

        Args:
            memory_type: Filter by type (preference, fact, goal, etc.)
            limit: Maximum number of memories to return

        Returns:
            List of memory objects
        """
        filters = {"user_id": self.user_id}
        if memory_type:
            filters["memory_type"] = memory_type

        try:
            # Use get_memories_by_filter instead of search_memories
            # This retrieves by filter without requiring a search query
            memories = self.retriever.get_memories_by_filter(
                filters=filters,
                limit=limit
            )
            return memories
        except Exception as e:
            logger.error(f"Error retrieving user memories: {e}")
            return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about user's memories.

        Returns:
            Dictionary with memory counts by type and category
        """
        try:
            all_memories = self.get_user_memories(limit=1000)

            stats = {
                "total_memories": len(all_memories),
                "by_type": {},
                "by_category": {},
                "avg_importance": 0,
                "recent_count": 0
            }

            if not all_memories:
                return stats

            # Count by type
            for mem in all_memories:
                mem_type = mem.get('memory_type', 'unknown')
                stats['by_type'][mem_type] = stats['by_type'].get(mem_type, 0) + 1

                category = mem.get('category', 'unknown')
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

            # Average importance
            importances = [m.get('importance', 0) for m in all_memories]
            stats['avg_importance'] = sum(importances) / len(importances) if importances else 0

            # Recent memories (last 7 days)
            from datetime import datetime, timedelta
            week_ago = datetime.now() - timedelta(days=7)
            stats['recent_count'] = sum(
                1 for m in all_memories
                if datetime.fromisoformat(m.get('timestamp', '2000-01-01')) > week_ago
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    def clear_session(self) -> None:
        """Clear current session without saving summary."""
        self.current_session_id = None
        self.conversation_history = []
        logger.info("Session cleared")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self.current_session_id:
                self.end_conversation()
            self.qdrant.close()
        except:
            pass
