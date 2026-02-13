"""Prospective memory: remembering to perform intended actions at the right time or context.

Supports time-based triggers (fire at datetime / cron), event-based triggers
(fire when recall query matches keywords/regex/embedding), and hybrid triggers.
"""

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProspectiveStatus(str, Enum):
    """Lifecycle status of a prospective intent."""

    PENDING = "pending"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ProspectiveTriggerType(str, Enum):
    """How the intent is activated."""

    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    HYBRID = "hybrid"


class RecurrencePattern(str, Enum):
    """Recurrence schedule for repeating intents."""

    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM_CRON = "custom_cron"


class ProspectiveIntent(BaseModel):
    """A single prospective-memory intent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    memory_id: Optional[str] = None  # link to Qdrant Memory object
    intent_text: str
    action_description: str = ""

    # Trigger configuration
    trigger_type: ProspectiveTriggerType = ProspectiveTriggerType.EVENT_BASED

    # Time-based fields
    trigger_at: Optional[datetime] = None
    trigger_cron: Optional[str] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # Event-based fields
    context_keywords: list[str] = Field(default_factory=list)
    context_pattern: Optional[str] = None  # regex
    context_embedding: Optional[list[float]] = None
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # Recurrence
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_cron: Optional[str] = None
    remaining_occurrences: Optional[int] = None

    # Priority & status
    priority: int = Field(default=5, ge=0, le=10)
    status: ProspectiveStatus = ProspectiveStatus.PENDING

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Metadata
    tags: list[str] = Field(default_factory=list)
    source_conversation: Optional[str] = None
    trigger_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProspectiveMemoryManager:
    """Manages prospective memory intents — creation, evaluation, lifecycle.

    Intents are stored in-memory (dict keyed by user_id) for fast evaluation.
    Optionally backed by Qdrant via the MemoryClient for persistence/vector search.
    """

    def __init__(
        self,
        max_intents_per_user: int = 100,
        eval_interval_seconds: int = 60,
        llm: Optional[Any] = None,
    ) -> None:
        self.max_intents_per_user = max_intents_per_user
        self.eval_interval_seconds = eval_interval_seconds
        self.llm = llm
        # user_id -> list of intents
        self._intents: dict[str, list[ProspectiveIntent]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_intent(
        self,
        user_id: str,
        intent_text: str,
        trigger_type: ProspectiveTriggerType = ProspectiveTriggerType.EVENT_BASED,
        action_description: str = "",
        trigger_at: Optional[datetime] = None,
        trigger_cron: Optional[str] = None,
        time_window_start: Optional[datetime] = None,
        time_window_end: Optional[datetime] = None,
        context_keywords: Optional[list[str]] = None,
        context_pattern: Optional[str] = None,
        context_embedding: Optional[list[float]] = None,
        similarity_threshold: float = 0.75,
        recurrence: RecurrencePattern = RecurrencePattern.NONE,
        recurrence_cron: Optional[str] = None,
        remaining_occurrences: Optional[int] = None,
        priority: int = 5,
        expires_at: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        source_conversation: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProspectiveIntent:
        """Create a prospective intent with explicit fields."""
        intent = ProspectiveIntent(
            user_id=user_id,
            intent_text=intent_text,
            action_description=action_description,
            trigger_type=trigger_type,
            trigger_at=trigger_at,
            trigger_cron=trigger_cron,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            context_keywords=context_keywords or [],
            context_pattern=context_pattern,
            context_embedding=context_embedding,
            similarity_threshold=similarity_threshold,
            recurrence=recurrence,
            recurrence_cron=recurrence_cron,
            remaining_occurrences=remaining_occurrences,
            priority=priority,
            expires_at=expires_at,
            tags=tags or [],
            source_conversation=source_conversation,
            metadata=metadata or {},
        )

        if user_id not in self._intents:
            self._intents[user_id] = []

        # Enforce per-user limit
        if len(self._intents[user_id]) >= self.max_intents_per_user:
            logger.warning(
                f"User {user_id} reached max intents ({self.max_intents_per_user}). "
                "Dropping oldest pending intent."
            )
            # Remove oldest pending intent
            pending = [i for i in self._intents[user_id] if i.status == ProspectiveStatus.PENDING]
            if pending:
                self._intents[user_id].remove(pending[0])

        self._intents[user_id].append(intent)
        logger.info(
            f"Created prospective intent {intent.id} for user {user_id}: "
            f"type={trigger_type.value}, priority={priority}"
        )
        return intent

    def create_intent_from_natural_language(
        self,
        user_id: str,
        text: str,
    ) -> ProspectiveIntent:
        """Parse natural language into a structured intent using LLM.

        Falls back to heuristic parsing when no LLM is available.
        """
        if self.llm:
            return self._parse_intent_llm(user_id, text)
        return self._parse_intent_heuristic(user_id, text)

    def get_intent(self, intent_id: str) -> Optional[ProspectiveIntent]:
        """Get a single intent by ID."""
        for intents in self._intents.values():
            for intent in intents:
                if intent.id == intent_id:
                    return intent
        return None

    def list_intents(
        self, user_id: str, status: Optional[ProspectiveStatus] = None
    ) -> list[ProspectiveIntent]:
        """List intents for a user, optionally filtered by status."""
        intents = self._intents.get(user_id, [])
        if status is not None:
            intents = [i for i in intents if i.status == status]
        return intents

    def cancel_intent(self, intent_id: str, user_id: str) -> Optional[ProspectiveIntent]:
        """Cancel a pending intent."""
        intent = self.get_intent(intent_id)
        if intent is None or intent.user_id != user_id:
            return None
        if intent.status != ProspectiveStatus.PENDING:
            logger.warning(f"Cannot cancel intent {intent_id} in status {intent.status.value}")
            return None
        intent.status = ProspectiveStatus.CANCELLED
        intent.updated_at = datetime.now(timezone.utc)
        logger.info(f"Cancelled intent {intent_id}")
        return intent

    def complete_intent(self, intent_id: str, user_id: str) -> Optional[ProspectiveIntent]:
        """Mark an intent as completed (action was taken)."""
        intent = self.get_intent(intent_id)
        if intent is None or intent.user_id != user_id:
            return None
        if intent.status not in {ProspectiveStatus.PENDING, ProspectiveStatus.TRIGGERED}:
            logger.warning(f"Cannot complete intent {intent_id} in status {intent.status.value}")
            return None
        intent.status = ProspectiveStatus.COMPLETED
        intent.completed_at = datetime.now(timezone.utc)
        intent.updated_at = datetime.now(timezone.utc)
        logger.info(f"Completed intent {intent_id}")
        return intent

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_context(
        self,
        user_id: str,
        context_text: str,
        context_embedding: Optional[list[float]] = None,
    ) -> list[ProspectiveIntent]:
        """Check pending event-based/hybrid intents against a recall query.

        Returns triggered intents sorted by priority (descending).
        """
        pending = [
            i
            for i in self._intents.get(user_id, [])
            if i.status == ProspectiveStatus.PENDING
            and i.trigger_type
            in {ProspectiveTriggerType.EVENT_BASED, ProspectiveTriggerType.HYBRID}
        ]

        triggered: list[ProspectiveIntent] = []
        for intent in pending:
            if self._matches_context(intent, context_text, context_embedding):
                self._fire_intent(intent)
                triggered.append(intent)

        triggered.sort(key=lambda i: i.priority, reverse=True)
        return triggered

    def evaluate_time_triggers(self) -> list[ProspectiveIntent]:
        """Check all pending time-based/hybrid intents for time conditions.

        Called by the background evaluation loop.
        """
        now = datetime.now(timezone.utc)
        triggered: list[ProspectiveIntent] = []

        for intents in self._intents.values():
            for intent in intents:
                if intent.status != ProspectiveStatus.PENDING:
                    continue
                if intent.trigger_type not in {
                    ProspectiveTriggerType.TIME_BASED,
                    ProspectiveTriggerType.HYBRID,
                }:
                    continue

                # Check trigger_at
                if intent.trigger_at and now >= intent.trigger_at:
                    # Check time window if specified
                    if intent.time_window_end and now > intent.time_window_end:
                        continue  # outside window
                    self._fire_intent(intent)
                    triggered.append(intent)

        return triggered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_intent(self, intent: ProspectiveIntent) -> None:
        """Transition intent to TRIGGERED, handle recurrence."""
        intent.status = ProspectiveStatus.TRIGGERED
        intent.triggered_at = datetime.now(timezone.utc)
        intent.trigger_count += 1
        intent.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Fired prospective intent {intent.id} "
            f"(trigger_count={intent.trigger_count})"
        )

        # Handle recurrence: reset to PENDING with next trigger_at
        if intent.recurrence != RecurrencePattern.NONE:
            if intent.remaining_occurrences is not None:
                intent.remaining_occurrences -= 1
                if intent.remaining_occurrences <= 0:
                    logger.info(f"Intent {intent.id} exhausted all occurrences")
                    return

            next_trigger = self._compute_next_trigger(intent)
            if next_trigger and (intent.expires_at is None or next_trigger < intent.expires_at):
                intent.trigger_at = next_trigger
                intent.status = ProspectiveStatus.PENDING
                logger.info(f"Recurring intent {intent.id} reset to PENDING, next={next_trigger}")

    def _compute_next_trigger(self, intent: ProspectiveIntent) -> Optional[datetime]:
        """Compute the next trigger datetime for a recurring intent."""
        base = intent.triggered_at or datetime.now(timezone.utc)

        if intent.recurrence == RecurrencePattern.DAILY:
            return base + timedelta(days=1)
        if intent.recurrence == RecurrencePattern.WEEKLY:
            return base + timedelta(weeks=1)
        if intent.recurrence == RecurrencePattern.MONTHLY:
            # Approximate: 30 days
            return base + timedelta(days=30)
        if intent.recurrence == RecurrencePattern.CUSTOM_CRON and intent.recurrence_cron:
            return self._next_cron_time(intent.recurrence_cron, base)
        return None

    @staticmethod
    def _next_cron_time(cron_expr: str, after: datetime) -> Optional[datetime]:
        """Compute next cron fire time. Uses croniter if available, else returns None."""
        try:
            from croniter import croniter

            cron = croniter(cron_expr, after)
            next_dt: datetime = cron.get_next(datetime)
            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(tzinfo=timezone.utc)
            return next_dt
        except ImportError:
            logger.warning("croniter not installed; custom_cron recurrence unavailable")
            return None
        except Exception as e:
            logger.error(f"Cron parsing failed for '{cron_expr}': {e}")
            return None

    def _matches_context(
        self,
        intent: ProspectiveIntent,
        text: str,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """Check whether a context text matches an intent's event-based conditions.

        Any matching condition is sufficient (OR logic).
        """
        if intent.context_keywords and self._match_keywords(intent, text):
            return True
        if intent.context_pattern and self._match_pattern(intent, text):
            return True
        if (
            intent.context_embedding
            and embedding
            and self._match_embedding_similarity(intent, embedding)
        ):
            return True
        return False

    @staticmethod
    def _match_keywords(intent: ProspectiveIntent, text: str) -> bool:
        """Return True if any keyword is found in text (case-insensitive)."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in intent.context_keywords)

    @staticmethod
    def _match_pattern(intent: ProspectiveIntent, text: str) -> bool:
        """Return True if the regex pattern matches the text."""
        if not intent.context_pattern:
            return False
        try:
            return bool(re.search(intent.context_pattern, text, re.IGNORECASE))
        except re.error as e:
            logger.warning(f"Invalid regex in intent {intent.id}: {e}")
            return False

    @staticmethod
    def _match_embedding_similarity(
        intent: ProspectiveIntent, embedding: list[float]
    ) -> bool:
        """Return True if cosine similarity exceeds threshold."""
        if not intent.context_embedding or not embedding:
            return False

        a = intent.context_embedding
        b = embedding
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return False
        similarity = dot / (norm_a * norm_b)
        return similarity >= intent.similarity_threshold

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def expire_stale_intents(self, user_id: Optional[str] = None) -> int:
        """Expire intents that have passed their expires_at deadline.

        Returns the number of intents expired.
        """
        now = datetime.now(timezone.utc)
        count = 0
        users = [user_id] if user_id else list(self._intents.keys())

        for uid in users:
            for intent in self._intents.get(uid, []):
                if (
                    intent.status == ProspectiveStatus.PENDING
                    and intent.expires_at
                    and now > intent.expires_at
                ):
                    intent.status = ProspectiveStatus.EXPIRED
                    intent.updated_at = now
                    count += 1

        if count:
            logger.info(f"Expired {count} stale prospective intents")
        return count

    def consolidate_intents(self, user_id: str) -> list[ProspectiveIntent]:
        """Deduplicate/merge redundant intents for a user via LLM or simple dedup.

        Returns the consolidated list.
        """
        intents = [
            i for i in self._intents.get(user_id, []) if i.status == ProspectiveStatus.PENDING
        ]
        if len(intents) <= 1:
            return intents

        if self.llm:
            consolidated = self._consolidate_intents_llm(user_id, intents)
        else:
            consolidated = self._consolidate_intents_simple(intents)

        # Replace pending intents with consolidated, keep non-pending as-is
        non_pending = [
            i for i in self._intents.get(user_id, []) if i.status != ProspectiveStatus.PENDING
        ]
        self._intents[user_id] = non_pending + consolidated
        logger.info(
            f"Consolidated intents for user {user_id}: {len(intents)} -> {len(consolidated)}"
        )
        return consolidated

    # ------------------------------------------------------------------
    # NL parsing
    # ------------------------------------------------------------------

    def _parse_intent_llm(self, user_id: str, text: str) -> ProspectiveIntent:
        """Use LLM to parse natural language into a structured intent."""
        prompt = (
            "Parse the following into a prospective memory intent.\n"
            "Return fields as KEY: VALUE, one per line.\n"
            "Required fields: intent_text, action_description, trigger_type "
            "(time_based, event_based, or hybrid)\n"
            "Optional fields: context_keywords (comma-separated), context_pattern (regex), "
            "priority (0-10), recurrence (none, daily, weekly, monthly)\n\n"
            f"Input: {text}\n\nParsed:"
        )

        trigger_type = ProspectiveTriggerType.EVENT_BASED
        action_description = text
        context_keywords: list[str] = []
        priority = 5
        recurrence = RecurrencePattern.NONE

        try:
            if self.llm is None:
                return self._parse_intent_heuristic(user_id, text)
            response = self.llm.generate(prompt, max_tokens=300)

            for line in response.strip().split("\n"):
                line = line.strip()
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key == "action_description":
                    action_description = value
                elif key == "trigger_type" and value in {t.value for t in ProspectiveTriggerType}:
                    trigger_type = ProspectiveTriggerType(value)
                elif key == "context_keywords":
                    context_keywords = [k.strip() for k in value.split(",") if k.strip()]
                elif key == "priority":
                    try:
                        priority = max(0, min(10, int(value)))
                    except ValueError:
                        pass
                elif key == "recurrence" and value in {r.value for r in RecurrencePattern}:
                    recurrence = RecurrencePattern(value)
        except Exception as e:
            logger.warning(f"LLM intent parsing failed: {e}")

        return self.create_intent(
            user_id=user_id,
            intent_text=text,
            action_description=action_description,
            trigger_type=trigger_type,
            context_keywords=context_keywords,
            priority=priority,
            recurrence=recurrence,
            source_conversation=text,
        )

    def _parse_intent_heuristic(self, user_id: str, text: str) -> ProspectiveIntent:
        """Heuristic fallback for NL parsing."""
        text_lower = text.lower()
        trigger_type = ProspectiveTriggerType.EVENT_BASED
        context_keywords: list[str] = []
        recurrence = RecurrencePattern.NONE
        priority = 5

        # Detect time-based keywords
        time_indicators = ["remind me", "at ", "every ", "tomorrow", "next ", "in "]
        if any(ind in text_lower for ind in time_indicators):
            trigger_type = ProspectiveTriggerType.TIME_BASED

        # Detect recurrence
        if "every day" in text_lower or "daily" in text_lower:
            recurrence = RecurrencePattern.DAILY
        elif "every week" in text_lower or "weekly" in text_lower:
            recurrence = RecurrencePattern.WEEKLY
        elif "every month" in text_lower or "monthly" in text_lower:
            recurrence = RecurrencePattern.MONTHLY

        # Detect event-based keywords ("when ... mention ...")
        when_match = re.search(r"when\s+.*?\b(mention|talk|discuss|bring up|ask about)\b\s+(.*)", text_lower)
        if when_match:
            trigger_type = ProspectiveTriggerType.EVENT_BASED
            topic = when_match.group(2).strip().rstrip(".,!?")
            context_keywords = [w.strip() for w in topic.split() if len(w.strip()) > 2]

        return self.create_intent(
            user_id=user_id,
            intent_text=text,
            action_description=text,
            trigger_type=trigger_type,
            context_keywords=context_keywords,
            priority=priority,
            recurrence=recurrence,
            source_conversation=text,
        )

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    def _consolidate_intents_llm(
        self, user_id: str, intents: list[ProspectiveIntent]
    ) -> list[ProspectiveIntent]:
        """Consolidate intents using LLM."""
        intents_text = "\n".join(
            f"{i + 1}. [{intent.trigger_type.value}] {intent.intent_text} "
            f"(priority={intent.priority})"
            for i, intent in enumerate(intents)
        )
        prompt = (
            "Review these prospective memory intents and identify duplicates or "
            "redundancies. Return the IDs (1-indexed) to KEEP, comma-separated.\n\n"
            f"Intents:\n{intents_text}\n\nKeep:"
        )

        try:
            if self.llm is None:
                return intents
            response = self.llm.generate(prompt, max_tokens=200)
            keep_indices: set[int] = set()
            for token in response.strip().replace(",", " ").split():
                try:
                    idx = int(token) - 1
                    if 0 <= idx < len(intents):
                        keep_indices.add(idx)
                except ValueError:
                    continue
            if keep_indices:
                return [intents[i] for i in sorted(keep_indices)]
        except Exception as e:
            logger.warning(f"LLM intent consolidation failed: {e}")

        return intents

    @staticmethod
    def _consolidate_intents_simple(
        intents: list[ProspectiveIntent],
    ) -> list[ProspectiveIntent]:
        """Simple deduplication by intent_text."""
        seen: set[str] = set()
        unique: list[ProspectiveIntent] = []
        for intent in intents:
            key = intent.intent_text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(intent)
        return unique
