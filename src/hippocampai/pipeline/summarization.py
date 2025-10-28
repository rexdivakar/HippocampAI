"""Session summarization and key information extraction.

This module provides:
- Automatic session summarization
- Key points extraction
- Action item detection
- Topic identification
- Sentiment analysis
- Different summary styles (concise, detailed, bullet points)
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SummaryStyle(str, Enum):
    """Summary generation styles."""

    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    NARRATIVE = "narrative"
    EXECUTIVE = "executive"


class SentimentType(str, Enum):
    """Sentiment types."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SessionSummary(BaseModel):
    """Complete session summary with metadata."""

    session_id: str
    summary: str = Field(..., description="Main summary text")
    key_points: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    questions_asked: int = 0
    questions_answered: int = 0
    sentiment: SentimentType = SentimentType.NEUTRAL
    duration_minutes: Optional[int] = None
    message_count: int = 0
    style: SummaryStyle = SummaryStyle.CONCISE
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Summarizer:
    """Generates summaries from conversations and sessions."""

    def __init__(self, llm=None):
        """Initialize summarizer.

        Args:
            llm: Language model for advanced summarization
        """
        self.llm = llm

        # Build detection patterns
        self.action_patterns = self._build_action_patterns()
        self.question_patterns = self._build_question_patterns()
        self.topic_keywords = self._build_topic_keywords()

    def _build_action_patterns(self) -> list[str]:
        """Build patterns for action item detection."""
        return [
            r"\b(?:need to|have to|must|should|will|going to)\s+([^.!?]+)",
            r"\b(?:todo|to-do|task|action item):\s*([^.!?\n]+)",
            r"\b(?:remember to|don\'t forget to)\s+([^.!?]+)",
            r"\b(?:follow up on|check|review|prepare|send|schedule)\s+([^.!?]+)",
        ]

    def _build_question_patterns(self) -> list[str]:
        """Build patterns for question detection."""
        return [
            r"\?",  # Simple question mark
            r"\b(?:what|when|where|who|why|how|which)\b.*\?",  # Wh-questions
            r"\b(?:do|does|did|can|could|would|should|is|are|was|were)\b.*\?",  # Yes/no questions
        ]

    def _build_topic_keywords(self) -> dict[str, list[str]]:
        """Build keywords for topic identification."""
        return {
            "work": ["work", "job", "career", "project", "meeting", "deadline", "task"],
            "technology": ["software", "code", "programming", "app", "tech", "system"],
            "personal": ["family", "friend", "home", "personal", "relationship"],
            "health": ["health", "fitness", "exercise", "diet", "wellness"],
            "finance": ["money", "budget", "investment", "salary", "financial"],
            "education": ["learn", "study", "course", "education", "training"],
            "travel": ["travel", "trip", "vacation", "visit", "destination"],
        }

    def summarize_session(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        style: SummaryStyle = SummaryStyle.CONCISE,
        entities: Optional[list[str]] = None,
    ) -> SessionSummary:
        """Generate summary for a conversation session.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            session_id: Session identifier
            style: Summary style
            entities: Pre-extracted entities (optional)

        Returns:
            SessionSummary object
        """
        # Combine messages into full conversation text
        conversation_text = self._format_conversation(messages)

        # Extract key components
        key_points = self._extract_key_points(messages)
        topics = self._identify_topics(conversation_text)
        action_items = self._extract_action_items(conversation_text)
        sentiment = self._analyze_sentiment(conversation_text)

        # Count questions
        questions_asked = self._count_questions(messages, role="user")
        questions_answered = self._count_questions(messages, role="assistant")

        # Generate main summary
        if self.llm:
            summary_text = self._generate_summary_llm(messages, style, key_points, topics)
        else:
            summary_text = self._generate_summary_template(key_points, topics, style)

        # Calculate duration if timestamps available
        duration_minutes = self._calculate_duration(messages)

        return SessionSummary(
            session_id=session_id,
            summary=summary_text,
            key_points=key_points,
            entities_mentioned=entities or [],
            topics=topics,
            action_items=action_items,
            questions_asked=questions_asked,
            questions_answered=questions_answered,
            sentiment=sentiment,
            duration_minutes=duration_minutes,
            message_count=len(messages),
            style=style,
            metadata={"has_llm": self.llm is not None, "message_count": len(messages)},
        )

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

    def _extract_key_points(self, messages: list[dict[str, Any]]) -> list[str]:
        """Extract key points from conversation."""
        key_points = []

        for msg in messages:
            content = msg.get("content", "")

            # Look for important statements
            # Sentences with specific keywords
            important_keywords = [
                "important",
                "key",
                "main",
                "crucial",
                "essential",
                "goal",
                "objective",
                "plan",
                "decision",
                "conclusion",
            ]

            sentences = re.split(r"[.!?]+", content)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue

                # Check if sentence contains important keywords
                if any(kw in sentence.lower() for kw in important_keywords):
                    key_points.append(sentence)
                # Or if it's a longer, substantive sentence
                elif len(sentence) > 50 and len(sentence.split()) > 8:
                    # Avoid questions
                    if not sentence.endswith("?"):
                        key_points.append(sentence)

        # Limit to most relevant
        return key_points[:5]

    def _identify_topics(self, text: str) -> list[str]:
        """Identify main topics discussed."""
        text_lower = text.lower()
        topics = []

        # Check each topic category
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score

        # Return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        topics = [topic for topic, score in sorted_topics[:3]]

        return topics

    def _extract_action_items(self, text: str) -> list[str]:
        """Extract action items and tasks."""
        action_items = []

        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.group(1) if match.lastindex >= 1 else match.group(0)
                action = action.strip()

                # Clean up action text
                action = re.sub(r"\s+", " ", action)

                if len(action) > 10 and action not in action_items:
                    action_items.append(action)

        return action_items[:10]  # Limit to 10

    def _count_questions(self, messages: list[dict[str, Any]], role: str) -> int:
        """Count questions asked by a specific role."""
        count = 0
        for msg in messages:
            if msg.get("role") == role:
                content = msg.get("content", "")
                # Simple question mark count
                count += content.count("?")

        return count

    def _analyze_sentiment(self, text: str) -> SentimentType:
        """Analyze overall sentiment of conversation."""
        text_lower = text.lower()

        # Simple keyword-based sentiment analysis
        positive_words = [
            "great",
            "good",
            "excellent",
            "love",
            "happy",
            "amazing",
            "fantastic",
            "wonderful",
            "perfect",
            "thanks",
            "appreciate",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "sad",
            "disappointed",
            "frustrated",
            "angry",
            "problem",
            "issue",
            "concern",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count * 1.5:
            return SentimentType.POSITIVE
        elif negative_count > positive_count * 1.5:
            return SentimentType.NEGATIVE
        elif positive_count > 0 and negative_count > 0:
            return SentimentType.MIXED
        else:
            return SentimentType.NEUTRAL

    def _calculate_duration(self, messages: list[dict[str, Any]]) -> Optional[int]:
        """Calculate conversation duration in minutes."""
        timestamps = []

        for msg in messages:
            if "timestamp" in msg:
                try:
                    ts = msg["timestamp"]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except (KeyError, ValueError, TypeError, AttributeError):
                    pass

        if len(timestamps) >= 2:
            duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60
            return int(duration)

        return None

    def _generate_summary_llm(
        self,
        messages: list[dict[str, Any]],
        style: SummaryStyle,
        key_points: list[str],
        topics: list[str],
    ) -> str:
        """Generate summary using LLM."""
        conversation = self._format_conversation(messages)

        # Build prompt based on style
        if style == SummaryStyle.CONCISE:
            style_instruction = "Write a concise 2-3 sentence summary."
        elif style == SummaryStyle.DETAILED:
            style_instruction = (
                "Write a detailed summary covering all main points (1-2 paragraphs)."
            )
        elif style == SummaryStyle.BULLET_POINTS:
            style_instruction = "Summarize in 5-7 bullet points."
        elif style == SummaryStyle.NARRATIVE:
            style_instruction = "Write a narrative summary as a story."
        else:  # EXECUTIVE
            style_instruction = (
                "Write an executive summary highlighting key decisions and outcomes."
            )

        prompt = f"""Summarize this conversation. {style_instruction}

Topics discussed: {", ".join(topics) if topics else "general"}

Conversation:
{conversation[:2000]}  # Limit context length

Summary:"""

        try:
            response = self.llm.generate(prompt, max_tokens=300)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._generate_summary_template(key_points, topics, style)

    def _generate_summary_template(
        self, key_points: list[str], topics: list[str], style: SummaryStyle
    ) -> str:
        """Generate summary using templates (fallback)."""
        if style == SummaryStyle.BULLET_POINTS:
            if key_points:
                return "• " + "\n• ".join(key_points)
            else:
                return f"• Discussion covered topics: {', '.join(topics) if topics else 'various topics'}"

        # For other styles, create a simple summary
        topic_str = f" about {', '.join(topics)}" if topics else ""
        point_str = f" Key points discussed: {'. '.join(key_points[:3])}" if key_points else ""

        summary = f"Conversation{topic_str}.{point_str}"
        return summary

    def create_rolling_summary(
        self,
        messages: list[dict[str, Any]],
        window_size: int = 10,
        style: SummaryStyle = SummaryStyle.CONCISE,
    ) -> str:
        """Create rolling summary of recent messages.

        Args:
            messages: All messages
            window_size: Number of recent messages to summarize
            style: Summary style

        Returns:
            Summary of recent messages
        """
        # Get recent messages
        recent = messages[-window_size:] if len(messages) > window_size else messages

        # Create summary
        temp_summary = self.summarize_session(recent, session_id="rolling", style=style)

        return temp_summary.summary

    def extract_insights(self, messages: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
        """Extract insights from conversation.

        Args:
            messages: Conversation messages
            user_id: User identifier

        Returns:
            Dictionary of insights
        """
        conversation = self._format_conversation(messages)

        insights = {
            "user_id": user_id,
            "message_count": len(messages),
            "topics": self._identify_topics(conversation),
            "sentiment": self._analyze_sentiment(conversation).value,
            "action_items": self._extract_action_items(conversation),
            "key_decisions": [],
            "learning_points": [],
            "patterns": [],
        }

        # Extract decisions
        decision_keywords = ["decided", "agreed", "concluded", "chose", "selected"]
        for msg in messages:
            content = msg.get("content", "").lower()
            for keyword in decision_keywords:
                if keyword in content:
                    # Extract sentence containing decision
                    sentences = re.split(r"[.!?]+", msg.get("content", ""))
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            insights["key_decisions"].append(sentence.strip())
                            break

        # Extract learning points
        learning_keywords = ["learned", "discovered", "found out", "realized", "understood"]
        for msg in messages:
            content = msg.get("content", "").lower()
            for keyword in learning_keywords:
                if keyword in content:
                    sentences = re.split(r"[.!?]+", msg.get("content", ""))
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            insights["learning_points"].append(sentence.strip())
                            break

        return insights
