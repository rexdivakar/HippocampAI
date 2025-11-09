"""
Conversation-Aware Memory System.

This module provides intelligent conversation tracking and memory creation:
- Automatic conversation turn detection
- Speaker attribution in multi-party conversations
- Dialogue flow tracking (topic shifts, interruptions)
- Conversation sentiment over time
- Key decision points extraction
- Action items from conversations
- Follow-up suggestions based on history
- Unresolved question tracking

Features:
- Turn-by-turn conversation parsing
- Topic segmentation and shift detection
- Entity and relationship extraction from dialogue
- Automatic summarization at multiple levels
- Action item and decision tracking
- Question-answer pair detection
- Sentiment analysis per turn
- Context-aware memory creation
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeakerRole(str, Enum):
    """Speaker roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    PARTICIPANT = "participant"  # Multi-party conversations
    UNKNOWN = "unknown"


class TurnType(str, Enum):
    """Types of conversation turns."""

    STATEMENT = "statement"
    QUESTION = "question"
    ANSWER = "answer"
    COMMAND = "command"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    INTERRUPTION = "interruption"


class TopicShiftType(str, Enum):
    """Types of topic transitions."""

    NATURAL = "natural"  # Smooth transition
    ABRUPT = "abrupt"  # Sudden change
    RETURN = "return"  # Back to previous topic
    TANGENT = "tangent"  # Temporary deviation


class SentimentType(str, Enum):
    """Sentiment classifications."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    turn_id: str
    speaker: str
    role: SpeakerRole
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_type: TurnType = TurnType.STATEMENT
    sentiment: SentimentType = SentimentType.NEUTRAL
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    references_turn_ids: list[str] = Field(
        default_factory=list, description="Turns this references/responds to"
    )
    entities_mentioned: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    is_question: bool = False
    is_action_item: bool = False
    is_decision: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopicSegment(BaseModel):
    """A segment of conversation focused on a topic."""

    segment_id: str
    topic: str
    turn_ids: list[str]
    start_turn: int
    end_turn: int
    summary: Optional[str] = None
    key_points: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    shift_type: Optional[TopicShiftType] = None


class ActionItem(BaseModel):
    """Action item extracted from conversation."""

    action_id: str
    text: str
    assignee: Optional[str] = None
    deadline: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed, cancelled
    from_turn_id: str
    related_topic: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Decision made during conversation."""

    decision_id: str
    text: str
    made_by: Optional[str] = None
    timestamp: datetime
    from_turn_id: str
    rationale: Optional[str] = None
    alternatives_considered: list[str] = Field(default_factory=list)
    related_topic: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnresolvedQuestion(BaseModel):
    """Question that hasn't been answered."""

    question_id: str
    text: str
    asker: str
    timestamp: datetime
    from_turn_id: str
    topic: Optional[str] = None
    attempted_answers: list[str] = Field(default_factory=list)
    priority: str = "medium"


class ConversationSummary(BaseModel):
    """Multi-level conversation summary."""

    conversation_id: str
    full_summary: str  # Comprehensive summary
    key_points: list[str]  # Main takeaways
    decisions: list[Decision]
    action_items: list[ActionItem]
    unresolved_questions: list[UnresolvedQuestion]
    participants: list[str]
    topics_discussed: list[str]
    sentiment_trajectory: list[tuple[int, str]]  # (turn_index, sentiment)
    total_turns: int
    duration_seconds: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """Active conversation context."""

    conversation_id: str
    session_id: str
    user_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    current_topic: Optional[str] = None
    topic_history: list[str] = Field(default_factory=list)
    active_entities: set[str] = Field(default_factory=set)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemoryManager:
    """Manage conversation-aware memory creation and tracking."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        topic_shift_threshold: float = 0.3,
        min_topic_turns: int = 3,
        enable_action_detection: bool = True,
        enable_decision_detection: bool = True,
    ):
        """
        Initialize conversation memory manager.

        Args:
            llm: Optional LLM for advanced analysis
            topic_shift_threshold: Similarity threshold for topic shifts
            min_topic_turns: Minimum turns to constitute a topic
            enable_action_detection: Detect action items
            enable_decision_detection: Detect decisions
        """
        self.llm = llm
        self.topic_shift_threshold = topic_shift_threshold
        self.min_topic_turns = min_topic_turns
        self.enable_action_detection = enable_action_detection
        self.enable_decision_detection = enable_decision_detection

        # Conversation contexts (in-memory, should be persisted)
        self.active_conversations: dict[str, ConversationContext] = {}

    def parse_turn(
        self,
        text: str,
        speaker: str,
        role: SpeakerRole = SpeakerRole.USER,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Parse a conversation turn.

        Args:
            text: Turn text
            speaker: Speaker identifier
            role: Speaker role
            conversation_id: Optional conversation ID for context
            metadata: Optional additional metadata

        Returns:
            ConversationTurn with analysis
        """
        import uuid

        turn_id = str(uuid.uuid4())

        # Detect turn type
        turn_type = self._detect_turn_type(text)

        # Detect sentiment (simple heuristic, can use LLM)
        sentiment = self._detect_sentiment(text)

        # Extract entities (simple regex-based, can use NER)
        entities = self._extract_entities(text)

        # Detect if question
        is_question = turn_type == TurnType.QUESTION

        # Detect if action item
        is_action_item = self.enable_action_detection and self._is_action_item(text)

        # Detect if decision
        is_decision = self.enable_decision_detection and self._is_decision(text)

        turn = ConversationTurn(
            turn_id=turn_id,
            speaker=speaker,
            role=role,
            text=text,
            turn_type=turn_type,
            sentiment=sentiment,
            entities_mentioned=entities,
            is_question=is_question,
            is_action_item=is_action_item,
            is_decision=is_decision,
            metadata=metadata or {},
        )

        return turn

    def add_turn(
        self,
        conversation_id: str,
        session_id: str,
        user_id: str,
        turn: ConversationTurn,
    ) -> ConversationContext:
        """
        Add a turn to conversation context.

        Args:
            conversation_id: Conversation identifier
            session_id: Session identifier
            user_id: User identifier
            turn: Conversation turn

        Returns:
            Updated conversation context
        """
        # Get or create context
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                session_id=session_id,
                user_id=user_id,
            )

        context = self.active_conversations[conversation_id]

        # Add turn
        context.turns.append(turn)

        # Update entities
        context.active_entities.update(turn.entities_mentioned)

        # Update timestamp
        context.last_update = datetime.now(timezone.utc)

        # Detect topic if enough turns
        if len(context.turns) >= self.min_topic_turns:
            current_topic = self._detect_current_topic(context.turns[-self.min_topic_turns:])
            if current_topic != context.current_topic:
                # Topic shift
                if context.current_topic:
                    context.topic_history.append(context.current_topic)
                context.current_topic = current_topic

        logger.debug(
            f"Added turn to conversation {conversation_id}: "
            f"{turn.turn_type.value} by {turn.speaker}"
        )

        return context

    def segment_by_topic(
        self, conversation_id: str, use_llm: bool = True
    ) -> list[TopicSegment]:
        """
        Segment conversation into topic-based sections.

        Args:
            conversation_id: Conversation identifier
            use_llm: Use LLM for better topic detection

        Returns:
            List of topic segments
        """
        if conversation_id not in self.active_conversations:
            return []

        context = self.active_conversations[conversation_id]
        turns = context.turns

        if len(turns) < self.min_topic_turns:
            return []

        segments: list[TopicSegment] = []
        current_segment_start = 0
        current_topic = None

        for i in range(self.min_topic_turns, len(turns) + 1):
            window = turns[max(0, i - self.min_topic_turns) : i]
            topic = self._detect_current_topic(window, use_llm=use_llm)

            # Check for topic shift
            if topic != current_topic:
                # Save previous segment
                if current_topic and current_segment_start < i - 1:
                    segment = TopicSegment(
                        segment_id=f"seg_{len(segments)}",
                        topic=current_topic,
                        turn_ids=[t.turn_id for t in turns[current_segment_start : i - 1]],
                        start_turn=current_segment_start,
                        end_turn=i - 2,
                        entities=list(
                            set(
                                e
                                for t in turns[current_segment_start : i - 1]
                                for e in t.entities_mentioned
                            )
                        ),
                    )
                    segments.append(segment)

                # Start new segment
                current_segment_start = i - 1
                current_topic = topic

        # Add final segment
        if current_topic:
            segment = TopicSegment(
                segment_id=f"seg_{len(segments)}",
                topic=current_topic,
                turn_ids=[t.turn_id for t in turns[current_segment_start:]],
                start_turn=current_segment_start,
                end_turn=len(turns) - 1,
                entities=list(
                    set(
                        e
                        for t in turns[current_segment_start:]
                        for e in t.entities_mentioned
                    )
                ),
            )
            segments.append(segment)

        logger.info(f"Segmented conversation {conversation_id} into {len(segments)} topics")

        return segments

    def extract_action_items(self, conversation_id: str) -> list[ActionItem]:
        """
        Extract action items from conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of action items
        """
        if conversation_id not in self.active_conversations:
            return []

        context = self.active_conversations[conversation_id]
        action_items: list[ActionItem] = []

        for turn in context.turns:
            if turn.is_action_item:
                # Extract action details
                action = self._parse_action_item(turn, context)
                if action:
                    action_items.append(action)

        logger.info(f"Extracted {len(action_items)} action items from {conversation_id}")

        return action_items

    def extract_decisions(self, conversation_id: str) -> list[Decision]:
        """
        Extract decisions from conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of decisions
        """
        if conversation_id not in self.active_conversations:
            return []

        context = self.active_conversations[conversation_id]
        decisions: list[Decision] = []

        for turn in context.turns:
            if turn.is_decision:
                # Extract decision details
                decision = self._parse_decision(turn, context)
                if decision:
                    decisions.append(decision)

        logger.info(f"Extracted {len(decisions)} decisions from {conversation_id}")

        return decisions

    def find_unresolved_questions(self, conversation_id: str) -> list[UnresolvedQuestion]:
        """
        Find questions that weren't adequately answered.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of unresolved questions
        """
        if conversation_id not in self.active_conversations:
            return []

        context = self.active_conversations[conversation_id]
        unresolved: list[UnresolvedQuestion] = []

        # Find all questions
        questions = [t for t in context.turns if t.is_question]

        for q_turn in questions:
            # Look for answers in subsequent turns
            q_index = context.turns.index(q_turn)
            following_turns = context.turns[q_index + 1 : q_index + 5]  # Check next 5 turns

            # Check if adequately answered
            has_answer = any(
                t.turn_type == TurnType.ANSWER or (t.speaker != q_turn.speaker and len(t.text) > 20)
                for t in following_turns
            )

            if not has_answer:
                import uuid

                unresolved.append(
                    UnresolvedQuestion(
                        question_id=str(uuid.uuid4()),
                        text=q_turn.text,
                        asker=q_turn.speaker,
                        timestamp=q_turn.timestamp,
                        from_turn_id=q_turn.turn_id,
                        topic=context.current_topic,
                        attempted_answers=[t.text for t in following_turns],
                    )
                )

        logger.info(f"Found {len(unresolved)} unresolved questions in {conversation_id}")

        return unresolved

    def summarize_conversation(
        self, conversation_id: str, use_llm: bool = True
    ) -> Optional[ConversationSummary]:
        """
        Generate multi-level conversation summary.

        Args:
            conversation_id: Conversation identifier
            use_llm: Use LLM for better summarization

        Returns:
            ConversationSummary or None if conversation not found
        """
        if conversation_id not in self.active_conversations:
            return None

        context = self.active_conversations[conversation_id]

        # Extract components
        action_items = self.extract_action_items(conversation_id)
        decisions = self.extract_decisions(conversation_id)
        unresolved = self.find_unresolved_questions(conversation_id)
        segments = self.segment_by_topic(conversation_id, use_llm=use_llm)

        # Generate summary
        if use_llm and self.llm:
            full_summary = self._generate_llm_summary(context)
            key_points = self._extract_llm_key_points(context)
        else:
            full_summary = self._generate_simple_summary(context)
            key_points = self._extract_simple_key_points(context, segments)

        # Participants
        participants = list(set(t.speaker for t in context.turns))

        # Topics
        topics = list(set(s.topic for s in segments))

        # Sentiment trajectory
        sentiment_trajectory = [
            (i, t.sentiment.value) for i, t in enumerate(context.turns)
        ]

        # Duration
        if context.turns:
            duration = (
                context.turns[-1].timestamp - context.turns[0].timestamp
            ).total_seconds()
        else:
            duration = 0.0

        summary = ConversationSummary(
            conversation_id=conversation_id,
            full_summary=full_summary,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            unresolved_questions=unresolved,
            participants=participants,
            topics_discussed=topics,
            sentiment_trajectory=sentiment_trajectory,
            total_turns=len(context.turns),
            duration_seconds=duration,
        )

        logger.info(f"Generated summary for conversation {conversation_id}")

        return summary

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _detect_turn_type(self, text: str) -> TurnType:
        """Detect type of turn based on text."""
        text_lower = text.lower().strip()

        # Question detection
        if "?" in text or any(
            text_lower.startswith(q)
            for q in ["what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "would", "should"]
        ):
            return TurnType.QUESTION

        # Command detection
        if any(
            text_lower.startswith(c)
            for c in ["please", "let's", "we should", "you should", "can you", "could you"]
        ):
            return TurnType.COMMAND

        # Acknowledgment
        if text_lower in ["ok", "okay", "yes", "no", "sure", "right", "i see", "got it", "understood"]:
            return TurnType.ACKNOWLEDGMENT

        # Clarification
        if any(
            phrase in text_lower
            for phrase in ["i mean", "in other words", "to clarify", "what i meant"]
        ):
            return TurnType.CLARIFICATION

        return TurnType.STATEMENT

    def _detect_sentiment(self, text: str) -> SentimentType:
        """Simple sentiment detection (can be enhanced with ML)."""
        text_lower = text.lower()

        # Simple keyword-based sentiment
        positive_words = ["good", "great", "excellent", "happy", "love", "thanks", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "hate", "wrong", "problem", "issue"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return SentimentType.POSITIVE
        elif neg_count > pos_count:
            return SentimentType.NEGATIVE
        elif pos_count > 0 and neg_count > 0:
            return SentimentType.MIXED
        else:
            return SentimentType.NEUTRAL

    def _extract_entities(self, text: str) -> list[str]:
        """Simple entity extraction (can be enhanced with NER)."""
        # Find capitalized words (proper nouns)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        # Remove common words
        common_words = {"I", "The", "A", "An", "This", "That", "These", "Those"}
        entities = [e for e in entities if e not in common_words]

        return list(set(entities))

    def _is_action_item(self, text: str) -> bool:
        """Detect if turn contains action item."""
        text_lower = text.lower()
        action_indicators = [
            "will",
            "going to",
            "need to",
            "should",
            "must",
            "have to",
            "todo",
            "to-do",
            "action:",
            "task:",
            "by",
            "deadline",
        ]
        return any(indicator in text_lower for indicator in action_indicators)

    def _is_decision(self, text: str) -> bool:
        """Detect if turn contains a decision."""
        text_lower = text.lower()
        decision_indicators = [
            "decided",
            "decision",
            "agreed",
            "will go with",
            "chose",
            "selected",
            "picking",
            "final answer",
        ]
        return any(indicator in text_lower for indicator in decision_indicators)

    def _detect_current_topic(
        self, turns: list[ConversationTurn], use_llm: bool = False
    ) -> str:
        """Detect topic from recent turns."""
        if use_llm and self.llm:
            # Use LLM for better topic detection
            return self._llm_detect_topic(turns)

        # Simple: most common entities or keywords
        all_entities = [e for t in turns for e in t.entities_mentioned]
        if all_entities:
            # Most frequent entity
            from collections import Counter

            topic = Counter(all_entities).most_common(1)[0][0]
            return topic

        # Fallback: first few words of recent turn
        if turns:
            words = turns[-1].text.split()[:3]
            return " ".join(words)

        return "general"

    def _parse_action_item(
        self, turn: ConversationTurn, context: ConversationContext
    ) -> Optional[ActionItem]:
        """Parse action item from turn."""
        import uuid

        # Simple extraction (can be enhanced with NLP)
        action_id = str(uuid.uuid4())

        # Try to extract assignee
        assignee = None
        if "you" in turn.text.lower():
            assignee = "other_party"
        elif "i" in turn.text.lower() or "i'll" in turn.text.lower():
            assignee = turn.speaker

        return ActionItem(
            action_id=action_id,
            text=turn.text,
            assignee=assignee,
            from_turn_id=turn.turn_id,
            related_topic=context.current_topic,
        )

    def _parse_decision(
        self, turn: ConversationTurn, context: ConversationContext
    ) -> Optional[Decision]:
        """Parse decision from turn."""
        import uuid

        decision_id = str(uuid.uuid4())

        return Decision(
            decision_id=decision_id,
            text=turn.text,
            made_by=turn.speaker,
            timestamp=turn.timestamp,
            from_turn_id=turn.turn_id,
            related_topic=context.current_topic,
        )

    def _generate_simple_summary(self, context: ConversationContext) -> str:
        """Generate simple summary without LLM."""
        if not context.turns:
            return "Empty conversation"

        participants = list(set(t.speaker for t in context.turns))
        turn_count = len(context.turns)
        topics = list(set(context.topic_history + [context.current_topic] if context.current_topic else context.topic_history))

        summary = f"Conversation between {', '.join(participants)} with {turn_count} turns."
        if topics:
            summary += f" Topics discussed: {', '.join(topics)}."

        return summary

    def _extract_simple_key_points(
        self, context: ConversationContext, segments: list[TopicSegment]
    ) -> list[str]:
        """Extract key points without LLM."""
        key_points = []

        # One key point per topic segment
        for segment in segments:
            key_points.append(f"{segment.topic}: {len(segment.turn_ids)} turns")

        return key_points

    def _generate_llm_summary(self, context: ConversationContext) -> str:
        """Generate summary using LLM."""
        if not self.llm:
            return self._generate_simple_summary(context)

        # Build prompt
        conversation_text = "\n".join(
            f"{t.speaker}: {t.text}" for t in context.turns
        )

        prompt = f"""Summarize the following conversation concisely:

{conversation_text}

Summary:"""

        try:
            response = self.llm.generate(prompt, max_tokens=200)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._generate_simple_summary(context)

    def _extract_llm_key_points(self, context: ConversationContext) -> list[str]:
        """Extract key points using LLM."""
        if not self.llm:
            return self._extract_simple_key_points(context, [])

        conversation_text = "\n".join(
            f"{t.speaker}: {t.text}" for t in context.turns
        )

        prompt = f"""Extract 3-5 key points from this conversation:

{conversation_text}

Key points (one per line):"""

        try:
            response = self.llm.generate(prompt, max_tokens=150)
            key_points = [
                line.strip("- ").strip()
                for line in response.strip().split("\n")
                if line.strip()
            ]
            return key_points[:5]
        except Exception as e:
            logger.error(f"LLM key point extraction failed: {e}")
            return []

    def _llm_detect_topic(self, turns: list[ConversationTurn]) -> str:
        """Use LLM to detect topic."""
        if not self.llm:
            return self._detect_current_topic(turns, use_llm=False)

        recent_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)

        prompt = f"""What is the main topic of this conversation segment? Respond with 2-3 words.

{recent_text}

Topic:"""

        try:
            response = self.llm.generate(prompt, max_tokens=10)
            return response.strip().lower()
        except Exception as e:
            logger.error(f"LLM topic detection failed: {e}")
            return self._detect_current_topic(turns, use_llm=False)
