"""Cross-session insights and behavioral analysis.

This module provides:
- Pattern detection across sessions
- Behavioral change tracking
- Preference drift analysis
- Habit formation detection
- Long-term trend analysis
- User evolution tracking
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.models.session import Session

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of behavioral changes."""

    PREFERENCE_SHIFT = "preference_shift"
    HABIT_FORMED = "habit_formed"
    HABIT_BROKEN = "habit_broken"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_ABANDONED = "goal_abandoned"
    INTEREST_GAINED = "interest_gained"
    INTEREST_LOST = "interest_lost"
    BEHAVIOR_PATTERN = "behavior_pattern"


class Pattern(BaseModel):
    """Represents a detected pattern in user behavior."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    pattern_type: str  # "recurring", "sequential", "correlational"
    description: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    memory_ids: List[str] = Field(default_factory=list)
    session_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Temporal info
    frequency: Optional[str] = None  # "daily", "weekly", etc.
    time_of_day: Optional[int] = None  # Hour of day (0-23)
    day_of_week: Optional[str] = None


class BehaviorChange(BaseModel):
    """Represents a detected change in user behavior."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    change_type: ChangeType
    description: str
    confidence: float
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Before/after comparison
    before_value: Optional[str] = None
    after_value: Optional[str] = None
    change_magnitude: Optional[float] = None  # 0.0 to 1.0

    # Supporting evidence
    evidence_memory_ids: List[str] = Field(default_factory=list)
    evidence_session_ids: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreferenceDrift(BaseModel):
    """Tracks evolution of a preference over time."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    category: str  # What the preference is about
    original_preference: str
    current_preference: str
    drift_score: float  # 0.0 (no drift) to 1.0 (complete change)

    timeline: List[Tuple[datetime, str]] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)

    first_recorded: datetime
    last_updated: datetime

    metadata: Dict[str, Any] = Field(default_factory=dict)


class HabitScore(BaseModel):
    """Scores likelihood that a behavior is a habit."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    behavior: str
    habit_score: float  # 0.0 to 1.0 (higher = more habitual)

    # Habit characteristics
    consistency: float  # How regular the behavior is
    frequency: int  # Occurrences in time period
    recency: float  # How recently observed
    duration: int  # Days the behavior has been tracked

    status: str  # "forming", "established", "breaking", "broken"

    # Evidence
    occurrences: List[datetime] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)

    first_occurrence: datetime
    last_occurrence: datetime

    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trend(BaseModel):
    """Represents a trend in user behavior or preferences."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    category: str
    trend_type: str  # "increasing", "decreasing", "stable", "cyclical"
    description: str
    confidence: float

    # Trend data
    data_points: List[Tuple[datetime, float]] = Field(default_factory=list)
    direction: str  # "up", "down", "flat"
    strength: float  # 0.0 to 1.0

    start_time: datetime
    end_time: datetime

    metadata: Dict[str, Any] = Field(default_factory=dict)


class InsightAnalyzer:
    """Analyzes cross-session patterns and behavioral insights."""

    def __init__(self, llm=None, min_pattern_occurrences: int = 3):
        """Initialize insight analyzer.

        Args:
            llm: Optional language model for advanced analysis
            min_pattern_occurrences: Minimum occurrences to consider as pattern
        """
        self.llm = llm
        self.min_pattern_occurrences = min_pattern_occurrences

    def detect_patterns(
        self,
        memories: List[Memory],
        user_id: str,
        sessions: Optional[List[Session]] = None,
    ) -> List[Pattern]:
        """Detect behavioral patterns across memories and sessions.

        Args:
            memories: List of memories to analyze
            user_id: User ID
            sessions: Optional list of sessions for context

        Returns:
            List of detected patterns
        """
        patterns = []

        # Detect recurring patterns
        recurring = self._detect_recurring_patterns(memories, user_id)
        patterns.extend(recurring)

        # Detect sequential patterns
        sequential = self._detect_sequential_patterns(memories, user_id)
        patterns.extend(sequential)

        # Detect correlational patterns (if sessions provided)
        if sessions:
            correlational = self._detect_correlational_patterns(memories, sessions, user_id)
            patterns.extend(correlational)

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        logger.info(f"Detected {len(patterns)} patterns for user {user_id}")
        return patterns

    def _detect_recurring_patterns(
        self, memories: List[Memory], user_id: str
    ) -> List[Pattern]:
        """Detect recurring patterns in memories."""
        patterns = []

        # Group by content similarity (simple keyword matching)
        keyword_groups = defaultdict(list)

        for memory in memories:
            # Extract keywords (simple approach)
            words = set(memory.text.lower().split())
            # Remove common words
            words = words - {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}

            for word in words:
                if len(word) > 3:  # Only meaningful words
                    keyword_groups[word].append(memory)

        # Find groups with recurring memories
        for keyword, group_memories in keyword_groups.items():
            if len(group_memories) >= self.min_pattern_occurrences:
                # Calculate time frequency
                if len(group_memories) > 1:
                    sorted_mems = sorted(group_memories, key=lambda m: m.created_at)
                    time_diffs = [
                        (sorted_mems[i + 1].created_at - sorted_mems[i].created_at).days
                        for i in range(len(sorted_mems) - 1)
                    ]
                    avg_gap = sum(time_diffs) / len(time_diffs) if time_diffs else 0

                    # Determine frequency
                    if avg_gap < 2:
                        frequency = "daily"
                    elif avg_gap < 8:
                        frequency = "weekly"
                    elif avg_gap < 35:
                        frequency = "monthly"
                    else:
                        frequency = "occasional"

                    pattern = Pattern(
                        user_id=user_id,
                        pattern_type="recurring",
                        description=f"Recurring mentions of '{keyword}'",
                        confidence=min(len(group_memories) / 10, 1.0),
                        occurrences=len(group_memories),
                        first_seen=sorted_mems[0].created_at,
                        last_seen=sorted_mems[-1].created_at,
                        memory_ids=[m.id for m in group_memories],
                        frequency=frequency,
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_sequential_patterns(
        self, memories: List[Memory], user_id: str
    ) -> List[Pattern]:
        """Detect sequential patterns (A followed by B)."""
        patterns = []

        # Sort memories by time
        sorted_mems = sorted(memories, key=lambda m: m.created_at)

        # Look for sequences with tags
        tag_sequences = defaultdict(int)

        for i in range(len(sorted_mems) - 1):
            mem1 = sorted_mems[i]
            mem2 = sorted_mems[i + 1]

            # Check if close in time (within 1 hour)
            time_diff = (mem2.created_at - mem1.created_at).total_seconds() / 3600
            if time_diff <= 1:
                # Create sequence from tags
                if mem1.tags and mem2.tags:
                    seq_key = f"{mem1.tags[0]} -> {mem2.tags[0]}"
                    tag_sequences[seq_key] += 1

        # Create patterns for frequent sequences
        for seq, count in tag_sequences.items():
            if count >= self.min_pattern_occurrences:
                pattern = Pattern(
                    user_id=user_id,
                    pattern_type="sequential",
                    description=f"Sequential pattern: {seq}",
                    confidence=min(count / 5, 1.0),
                    occurrences=count,
                    first_seen=sorted_mems[0].created_at,
                    last_seen=sorted_mems[-1].created_at,
                )
                patterns.append(pattern)

        return patterns

    def _detect_correlational_patterns(
        self, memories: List[Memory], sessions: List[Session], user_id: str
    ) -> List[Pattern]:
        """Detect correlational patterns across sessions."""
        patterns = []

        # Group memories by session
        session_memories = defaultdict(list)
        for memory in memories:
            if memory.session_id:
                session_memories[memory.session_id].append(memory)

        # Look for session-level patterns
        session_tags = defaultdict(list)
        for session_id, mems in session_memories.items():
            all_tags = [tag for m in mems for tag in m.tags]
            if all_tags:
                most_common = Counter(all_tags).most_common(1)[0][0]
                session_tags[most_common].append(session_id)

        # Create patterns for tags appearing in multiple sessions
        for tag, sess_ids in session_tags.items():
            if len(sess_ids) >= self.min_pattern_occurrences:
                pattern = Pattern(
                    user_id=user_id,
                    pattern_type="correlational",
                    description=f"'{tag}' appears across multiple sessions",
                    confidence=min(len(sess_ids) / 5, 1.0),
                    occurrences=len(sess_ids),
                    first_seen=min(m.created_at for s_id in sess_ids for m in session_memories[s_id]),
                    last_seen=max(m.created_at for s_id in sess_ids for m in session_memories[s_id]),
                    session_ids=sess_ids,
                )
                patterns.append(pattern)

        return patterns

    def track_behavior_changes(
        self,
        old_memories: List[Memory],
        new_memories: List[Memory],
        user_id: str,
    ) -> List[BehaviorChange]:
        """Track changes in user behavior between time periods.

        Args:
            old_memories: Memories from earlier period
            new_memories: Memories from recent period
            user_id: User ID

        Returns:
            List of detected behavior changes
        """
        changes = []

        # Compare preferences
        preference_changes = self._compare_preferences(old_memories, new_memories, user_id)
        changes.extend(preference_changes)

        # Compare habits
        habit_changes = self._compare_habits(old_memories, new_memories, user_id)
        changes.extend(habit_changes)

        # Compare goals
        goal_changes = self._compare_goals(old_memories, new_memories, user_id)
        changes.extend(goal_changes)

        # Compare interests (via tags)
        interest_changes = self._compare_interests(old_memories, new_memories, user_id)
        changes.extend(interest_changes)

        logger.info(f"Detected {len(changes)} behavior changes for user {user_id}")
        return changes

    def _compare_preferences(
        self, old_mems: List[Memory], new_mems: List[Memory], user_id: str
    ) -> List[BehaviorChange]:
        """Compare preferences between time periods."""
        changes = []

        # Get preference memories
        old_prefs = [m for m in old_mems if m.type == MemoryType.PREFERENCE]
        new_prefs = [m for m in new_mems if m.type == MemoryType.PREFERENCE]

        # Simple keyword-based comparison
        old_keywords = set()
        for m in old_prefs:
            old_keywords.update(m.text.lower().split())

        new_keywords = set()
        for m in new_prefs:
            new_keywords.update(m.text.lower().split())

        # Find gained/lost preferences
        gained = new_keywords - old_keywords
        lost = old_keywords - new_keywords

        if gained:
            change = BehaviorChange(
                user_id=user_id,
                change_type=ChangeType.PREFERENCE_SHIFT,
                description=f"New preferences: {', '.join(list(gained)[:5])}",
                confidence=0.7,
                before_value="",
                after_value=", ".join(list(gained)[:5]),
                evidence_memory_ids=[m.id for m in new_prefs],
            )
            changes.append(change)

        return changes

    def _compare_habits(
        self, old_mems: List[Memory], new_mems: List[Memory], user_id: str
    ) -> List[BehaviorChange]:
        """Compare habits between time periods."""
        changes = []

        old_habits = [m for m in old_mems if m.type == MemoryType.HABIT]
        new_habits = [m for m in new_mems if m.type == MemoryType.HABIT]

        # Check for new habits
        if len(new_habits) > len(old_habits):
            change = BehaviorChange(
                user_id=user_id,
                change_type=ChangeType.HABIT_FORMED,
                description=f"New habits forming ({len(new_habits) - len(old_habits)} detected)",
                confidence=0.75,
                evidence_memory_ids=[m.id for m in new_habits],
            )
            changes.append(change)

        return changes

    def _compare_goals(
        self, old_mems: List[Memory], new_mems: List[Memory], user_id: str
    ) -> List[BehaviorChange]:
        """Compare goals between time periods."""
        changes = []

        old_goals = {m.text.lower() for m in old_mems if m.type == MemoryType.GOAL}
        new_goals = {m.text.lower() for m in new_mems if m.type == MemoryType.GOAL}

        # Detect abandoned goals
        abandoned = old_goals - new_goals
        if abandoned:
            change = BehaviorChange(
                user_id=user_id,
                change_type=ChangeType.GOAL_ABANDONED,
                description=f"Goals no longer mentioned: {len(abandoned)}",
                confidence=0.65,
                before_value=", ".join(list(abandoned)[:3]),
            )
            changes.append(change)

        return changes

    def _compare_interests(
        self, old_mems: List[Memory], new_mems: List[Memory], user_id: str
    ) -> List[BehaviorChange]:
        """Compare interests based on tags."""
        changes = []

        old_tags = Counter([tag for m in old_mems for tag in m.tags])
        new_tags = Counter([tag for m in new_mems for tag in m.tags])

        # Find emerging interests (tags more frequent in new period)
        for tag, new_count in new_tags.most_common(5):
            old_count = old_tags.get(tag, 0)
            if new_count > old_count * 2:  # 2x increase
                change = BehaviorChange(
                    user_id=user_id,
                    change_type=ChangeType.INTEREST_GAINED,
                    description=f"Increased interest in '{tag}'",
                    confidence=0.7,
                    change_magnitude=(new_count - old_count) / max(new_count, 1),
                )
                changes.append(change)

        return changes

    def analyze_preference_drift(
        self, memories: List[Memory], user_id: str, category: Optional[str] = None
    ) -> List[PreferenceDrift]:
        """Analyze how preferences have changed over time.

        Args:
            memories: List of preference memories
            user_id: User ID
            category: Optional category to filter

        Returns:
            List of preference drift analyses
        """
        drifts = []

        # Filter to preferences
        prefs = [m for m in memories if m.type == MemoryType.PREFERENCE]

        # Group by category (using first keyword as category proxy)
        categories = defaultdict(list)
        for pref in prefs:
            words = pref.text.lower().split()
            if words:
                cat = words[0]
                categories[cat].append(pref)

        # Analyze each category
        for cat, cat_prefs in categories.items():
            if category and cat != category:
                continue

            if len(cat_prefs) >= 2:
                # Sort by time
                sorted_prefs = sorted(cat_prefs, key=lambda m: m.created_at)

                # Build timeline
                timeline = [(m.created_at, m.text) for m in sorted_prefs]

                # Calculate drift score (simple: how different is latest from first)
                first_words = set(sorted_prefs[0].text.lower().split())
                last_words = set(sorted_prefs[-1].text.lower().split())

                overlap = len(first_words & last_words)
                total = len(first_words | last_words)
                drift_score = 1.0 - (overlap / total if total > 0 else 0)

                drift = PreferenceDrift(
                    user_id=user_id,
                    category=cat,
                    original_preference=sorted_prefs[0].text,
                    current_preference=sorted_prefs[-1].text,
                    drift_score=drift_score,
                    timeline=timeline,
                    memory_ids=[m.id for m in cat_prefs],
                    first_recorded=sorted_prefs[0].created_at,
                    last_updated=sorted_prefs[-1].created_at,
                )
                drifts.append(drift)

        logger.info(f"Analyzed {len(drifts)} preference drifts for user {user_id}")
        return drifts

    def detect_habit_formation(
        self, memories: List[Memory], user_id: str, min_occurrences: int = 5
    ) -> List[HabitScore]:
        """Detect and score potential habits.

        Args:
            memories: List of memories
            user_id: User ID
            min_occurrences: Minimum occurrences to consider

        Returns:
            List of habit scores
        """
        habit_scores = []

        # Group by behavior (using tags/text similarity)
        behaviors = defaultdict(list)

        for memory in memories:
            # Use tags or keywords as behavior identifier
            if memory.tags:
                behavior = memory.tags[0]
            else:
                words = memory.text.lower().split()
                behavior = words[0] if words else "unknown"

            behaviors[behavior].append(memory)

        # Score each behavior
        for behavior, behavior_mems in behaviors.items():
            if len(behavior_mems) >= min_occurrences:
                # Sort by time
                sorted_mems = sorted(behavior_mems, key=lambda m: m.created_at)

                # Calculate metrics
                first_time = sorted_mems[0].created_at
                last_time = sorted_mems[-1].created_at
                duration_days = (last_time - first_time).days + 1

                # Frequency
                frequency = len(behavior_mems)

                # Consistency (regularity of occurrences)
                if len(sorted_mems) > 1:
                    gaps = [
                        (sorted_mems[i + 1].created_at - sorted_mems[i].created_at).days
                        for i in range(len(sorted_mems) - 1)
                    ]
                    avg_gap = sum(gaps) / len(gaps)
                    gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
                    consistency = 1.0 / (1.0 + gap_variance)  # Lower variance = higher consistency
                else:
                    consistency = 0.5

                # Recency
                days_since_last = (datetime.now(timezone.utc) - last_time).days
                recency = max(0, 1.0 - (days_since_last / 30))  # Decay over 30 days

                # Overall habit score
                habit_score = (consistency * 0.4 + (min(frequency / 20, 1.0)) * 0.4 + recency * 0.2)

                # Determine status
                if habit_score > 0.7 and recency > 0.7:
                    status = "established"
                elif habit_score > 0.5:
                    status = "forming"
                elif recency < 0.3:
                    status = "breaking"
                else:
                    status = "forming"

                score_obj = HabitScore(
                    user_id=user_id,
                    behavior=behavior,
                    habit_score=habit_score,
                    consistency=consistency,
                    frequency=frequency,
                    recency=recency,
                    duration=duration_days,
                    status=status,
                    occurrences=[m.created_at for m in sorted_mems],
                    memory_ids=[m.id for m in sorted_mems],
                    first_occurrence=first_time,
                    last_occurrence=last_time,
                )
                habit_scores.append(score_obj)

        # Sort by habit score
        habit_scores.sort(key=lambda h: h.habit_score, reverse=True)

        logger.info(f"Detected {len(habit_scores)} potential habits for user {user_id}")
        return habit_scores

    def analyze_trends(
        self, memories: List[Memory], user_id: str, window_days: int = 30
    ) -> List[Trend]:
        """Analyze long-term trends in user behavior.

        Args:
            memories: List of memories
            user_id: User ID
            window_days: Analysis window in days

        Returns:
            List of trends
        """
        trends = []

        if not memories:
            return trends

        # Group by tags
        tag_timeline = defaultdict(list)

        for memory in memories:
            for tag in memory.tags:
                tag_timeline[tag].append(memory.created_at)

        # Analyze each tag's trend
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)

        for tag, timestamps in tag_timeline.items():
            # Filter to window
            recent_times = [t for t in timestamps if t >= cutoff]

            if len(recent_times) >= 3:
                # Sort timestamps
                sorted_times = sorted(recent_times)

                # Split into 3 periods
                period_size = len(sorted_times) // 3
                period1 = sorted_times[:period_size]
                period2 = sorted_times[period_size:2*period_size]
                period3 = sorted_times[2*period_size:]

                # Count per period
                count1 = len(period1)
                count2 = len(period2)
                count3 = len(period3)

                # Determine trend direction
                if count3 > count1 * 1.5:
                    trend_type = "increasing"
                    direction = "up"
                    strength = min((count3 - count1) / max(count1, 1), 1.0)
                elif count3 < count1 * 0.5:
                    trend_type = "decreasing"
                    direction = "down"
                    strength = min((count1 - count3) / max(count1, 1), 1.0)
                else:
                    trend_type = "stable"
                    direction = "flat"
                    strength = 0.5

                trend = Trend(
                    user_id=user_id,
                    category=tag,
                    trend_type=trend_type,
                    description=f"'{tag}' mentions are {trend_type}",
                    confidence=min(len(recent_times) / 10, 1.0),
                    data_points=[(t, 1.0) for t in sorted_times],
                    direction=direction,
                    strength=strength,
                    start_time=sorted_times[0],
                    end_time=sorted_times[-1],
                )
                trends.append(trend)

        logger.info(f"Analyzed {len(trends)} trends for user {user_id}")
        return trends
