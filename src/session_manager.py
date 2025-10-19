"""Session management for tracking conversation sessions."""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import anthropic

from src.embedding_service import EmbeddingService
from src.memory_retriever import MemoryRetriever
from src.memory_store import Category, MemoryStore, MemoryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """Manager for conversation sessions and session summaries."""

    # Prompt template for session summarization
    SESSION_SUMMARY_PROMPT = """Summarize this conversation session for future reference.

Conversation History:
{conversation_history}

Provide a comprehensive summary including:
1. Main topics discussed (as bullet points)
2. Key decisions or outcomes
3. Important context for future conversations
4. Overall tone/purpose (problem-solving, casual chat, learning, planning, etc.)

Return ONLY a valid JSON object:
{{
  "main_topics": ["topic 1", "topic 2", ...],
  "key_decisions": ["decision 1", "decision 2", ...],
  "important_context": "Brief summary of important context",
  "tone": "problem-solving|casual|learning|planning|support|other",
  "summary": "One-paragraph overall summary"
}}

Do not include any other text."""

    def __init__(
        self,
        memory_store: MemoryStore,
        retriever: MemoryRetriever,
        embedding_service: EmbeddingService,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize the session manager.

        Args:
            memory_store: MemoryStore instance
            retriever: MemoryRetriever instance
            embedding_service: EmbeddingService instance
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.memory_store = memory_store
        self.retriever = retriever
        self.embeddings = embedding_service

        import os

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._last_request_time = 0
        self._min_request_interval = 0.1

        # In-memory session storage (in production, use database)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("SessionManager initialized")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def start_session(self, user_id: str, context: Optional[str] = None) -> str:
        """
        Start a new conversation session.

        Args:
            user_id: User identifier
            context: Optional initial context for the session

        Returns:
            Session ID
        """
        session_id = f"session_{user_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "context": context or "",
            "message_count": 0,
            "conversation_history": [],
            "topics": [],
            "status": "active",
        }

        self._active_sessions[session_id] = session_data

        logger.info(f"Started session {session_id} for user {user_id}")
        return session_id

    def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the session conversation history.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant)
            content: Message content
            metadata: Optional metadata
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found or already ended")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        session = self._active_sessions[session_id]
        session["conversation_history"].append(message)
        session["message_count"] += 1

        logger.debug(f"Added {role} message to session {session_id}")

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session information.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id].copy()

        # Check if it's a stored session (retrieve from memory store)
        try:
            session_memory = self.retriever.get_memory_by_id(session_id)
            if session_memory and session_memory.get("metadata", {}).get("is_session_summary"):
                return session_memory
        except Exception:
            pass

        return None

    def end_session(self, session_id: str, max_retries: int = 2) -> Optional[str]:
        """
        End a session and create summary.

        Args:
            session_id: Session identifier
            max_retries: Maximum retry attempts for AI summarization

        Returns:
            Memory ID of stored session summary, or None if failed

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._active_sessions[session_id]

        # Mark as ended
        session["end_time"] = datetime.utcnow().isoformat()
        session["status"] = "ended"

        # Calculate duration
        start_time = datetime.fromisoformat(session["start_time"])
        end_time = datetime.fromisoformat(session["end_time"])
        duration_seconds = (end_time - start_time).total_seconds()
        session["duration_seconds"] = duration_seconds

        try:
            # Generate conversation history text
            history_text = self._format_conversation_history(session["conversation_history"])

            # Generate summary using Claude
            summary_data = self._generate_session_summary(history_text, max_retries)

            if not summary_data:
                logger.warning(f"Failed to generate summary for session {session_id}")
                return None

            # Create session summary text
            summary_text = self._create_summary_text(summary_data, session)

            # Store session summary in conversation_history collection
            memory_id = self.memory_store.store_memory(
                text=summary_text,
                memory_type=MemoryType.CONTEXT.value,
                metadata={
                    "user_id": session["user_id"],
                    "importance": self._calculate_session_importance(session, summary_data),
                    "category": self._infer_category(summary_data.get("tone", "other")),
                    "session_id": session_id,
                    "confidence": 1.0,
                    # Additional session metadata
                    "is_session_summary": True,
                    "start_time": session["start_time"],
                    "end_time": session["end_time"],
                    "duration_seconds": duration_seconds,
                    "message_count": session["message_count"],
                    "topics": summary_data.get("main_topics", []),
                    "tone": summary_data.get("tone", "other"),
                },
            )

            # Remove from active sessions
            del self._active_sessions[session_id]

            logger.info(
                f"Session {session_id} ended and summarized "
                f"({session['message_count']} messages, {duration_seconds:.0f}s)"
            )

            return memory_id

        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            # Still remove from active sessions
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            raise

    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for AI processing."""
        lines = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _generate_session_summary(
        self, conversation_history: str, max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Generate session summary using Claude API."""
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()

                # Build prompt
                prompt = self.SESSION_SUMMARY_PROMPT.format(
                    conversation_history=conversation_history[:8000]  # Limit length
                )

                # Call Claude
                logger.debug("Generating session summary...")
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response
                response_text = message.content[0].text.strip()
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1

                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")

                json_text = response_text[start_idx:end_idx]
                summary = json.loads(json_text)

                logger.info("Session summary generated successfully")
                return summary

            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    time.sleep(2**attempt)
                else:
                    logger.error(f"Rate limit exceeded: {e}")
                    return None

            except anthropic.APIError as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    logger.error(f"Claude API error: {e}")
                    return None

            except (ValueError, json.JSONDecodeError) as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    logger.error(f"Failed to parse summary: {e}")
                    return None

        return None

    def _create_summary_text(self, summary_data: Dict[str, Any], session: Dict[str, Any]) -> str:
        """Create human-readable summary text."""
        parts = []

        # Overall summary
        if "summary" in summary_data:
            parts.append(summary_data["summary"])

        # Main topics
        if "main_topics" in summary_data and summary_data["main_topics"]:
            topics = ", ".join(summary_data["main_topics"])
            parts.append(f"Topics discussed: {topics}")

        # Key decisions
        if "key_decisions" in summary_data and summary_data["key_decisions"]:
            decisions = "; ".join(summary_data["key_decisions"])
            parts.append(f"Decisions: {decisions}")

        # Important context
        if "important_context" in summary_data:
            parts.append(summary_data["important_context"])

        return ". ".join(parts)

    def _calculate_session_importance(
        self, session: Dict[str, Any], summary_data: Dict[str, Any]
    ) -> int:
        """Calculate importance score for session."""
        base_importance = 5

        # Longer sessions are more important
        if session["message_count"] > 10:
            base_importance += 1
        if session["message_count"] > 20:
            base_importance += 1

        # Sessions with decisions are more important
        if summary_data.get("key_decisions"):
            base_importance += 2

        # Tone-based adjustment
        tone = summary_data.get("tone", "casual")
        if tone in ["problem-solving", "planning"]:
            base_importance += 1

        return min(10, base_importance)

    def _infer_category(self, tone: str) -> str:
        """Infer category from tone."""
        tone_category_map = {
            "problem-solving": Category.WORK.value,
            "learning": Category.LEARNING.value,
            "planning": Category.WORK.value,
            "support": Category.PERSONAL.value,
            "casual": Category.SOCIAL.value,
        }
        return tone_category_map.get(tone, Category.OTHER.value)

    def get_recent_sessions(
        self, user_id: str, days: int = 7, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent session summaries for a user.

        Args:
            user_id: User identifier
            days: Number of days to look back
            limit: Maximum number of sessions

        Returns:
            List of session summary memories
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            # Retrieve session summaries
            filters = {
                "user_id": user_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }

            sessions = self.retriever.get_memories_by_filter(
                filters=filters,
                limit=limit * 2,  # Get more for filtering
            )

            # Filter for session summaries only
            session_summaries = [
                s for s in sessions if s.get("metadata", {}).get("is_session_summary", False)
            ]

            # Sort by start time (most recent first)
            session_summaries.sort(key=lambda x: x["metadata"].get("start_time", ""), reverse=True)

            return session_summaries[:limit]

        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            raise RuntimeError("Failed to retrieve recent sessions") from e

    def get_active_session(self, user_id: str) -> Optional[str]:
        """
        Get active session ID for a user.

        Args:
            user_id: User identifier

        Returns:
            Session ID if active session exists, None otherwise
        """
        for session_id, session in self._active_sessions.items():
            if session["user_id"] == user_id and session["status"] == "active":
                return session_id
        return None
