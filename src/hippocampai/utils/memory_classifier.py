"""Automatic memory type classification based on content analysis."""

import re
from typing import Optional

from hippocampai.models.memory import MemoryType


class MemoryClassifier:
    """
    Automatically classify memories based on content patterns.

    This classifier analyzes text content to determine the most appropriate
    memory type: fact, preference, goal, habit, event, or context.

    Example:
        classifier = MemoryClassifier()
        memory_type = classifier.classify("My name is Alex")  # Returns MemoryType.FACT
    """

    # Pattern definitions for each memory type
    FACT_PATTERNS = [
        # Identity and personal information
        r"\bmy name is\b",
        r"\bi'?m\s+\w+",  # "I'm Alex", "I am Alex"
        r"\bcall me\b",
        r"\bi am\b",
        r"\bmy (job|work|position|role|title)\b",
        r"\bi (work|study|live)\b",
        r"\bmy (age|birthday|birth)\b",
        r"\bi (have|own|possess)\b",
        r"\bmy (address|email|phone|number)\b",
        r"\bi (was born|graduated|studied)\b",
        r"\bmy (family|parents|siblings|children)\b",
    ]

    PREFERENCE_PATTERNS = [
        # Likes, dislikes, opinions
        r"\bi (like|love|enjoy|adore)\b",
        r"\bi (hate|dislike|detest)\b",
        r"\bmy favorite\b",
        r"\bi prefer\b",
        r"\bi'?d rather\b",
        r"\bi don'?t like\b",
        r"\bi appreciate\b",
        r"\bi fancy\b",
        r"\bi'?m fond of\b",
        r"\bi can'?t stand\b",
        r"\bi think .+ is (good|bad|great|terrible|amazing)\b",
    ]

    GOAL_PATTERNS = [
        # Intentions, aspirations, plans
        r"\bi want to\b",
        r"\bi plan to\b",
        r"\bmy goal is\b",
        r"\bi hope to\b",
        r"\bi aim to\b",
        r"\bi intend to\b",
        r"\bi wish to\b",
        r"\bi'?d like to\b",
        r"\bi need to\b",
        r"\bi (should|must|have to)\b",
        r"\bi will\b",
        r"\bi'?m going to\b",
        r"\bmy (dream|aspiration|ambition)\b",
        r"\bsomeday i\b",
    ]

    HABIT_PATTERNS = [
        # Routines, regular activities
        r"\bi (usually|always|often|regularly|frequently|typically)\b",
        r"\bevery (day|week|month|morning|night|evening)\b",
        r"\bi tend to\b",
        r"\bmy routine\b",
        r"\bi never\b",
        r"\bi rarely\b",
        r"\bi sometimes\b",
        r"\bi (habit|custom|practice)\b",
        r"\bon (mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays)\b",
        r"\beach (day|week|month|year)\b",
    ]

    EVENT_PATTERNS = [
        # Specific occurrences, meetings, happenings
        r"\b(yesterday|tomorrow)\b",  # Removed "today" to avoid false positives
        r"\blast (week|month|year|night)\b",
        r"\bnext (week|month|year)\b",
        r"\b\d+ (days?|weeks?|months?|years?) ago\b",
        r"\b(happened|occurred|took place)\b",
        r"\b(meeting|appointment|event|conference)\b",
        r"\bon (january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\bat \d+:\d+\b",  # Time references like "at 3:30"
        r"\bwhen i (was|went|saw|met|did)\b",  # More specific than just "when i"
        r"\bwhile i was\b",
        r"\bi (met|saw|attended|went to)\b",  # Specific event actions
    ]

    def __init__(self) -> None:
        """Initialize the memory classifier with compiled regex patterns."""
        self._fact_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.FACT_PATTERNS]
        self._preference_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PREFERENCE_PATTERNS
        ]
        self._goal_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.GOAL_PATTERNS]
        self._habit_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.HABIT_PATTERNS]
        self._event_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.EVENT_PATTERNS]

    def classify(self, text: str, default: Optional[MemoryType] = None) -> MemoryType:
        """
        Classify a memory based on its text content.

        The classifier checks patterns in order of specificity:
        1. Fact - Identity and personal information
        2. Preference - Likes, dislikes, opinions
        3. Goal - Intentions and aspirations
        4. Habit - Routines and regular activities
        5. Event - Specific occurrences
        6. Context (default) - General conversation

        Args:
            text: The memory text to classify
            default: Default type if no pattern matches (default: MemoryType.CONTEXT)

        Returns:
            The detected MemoryType enum value

        Example:
            >>> classifier = MemoryClassifier()
            >>> classifier.classify("My name is Alex")
            <MemoryType.FACT: 'fact'>
            >>> classifier.classify("I love pizza")
            <MemoryType.PREFERENCE: 'preference'>
            >>> classifier.classify("I want to learn Python")
            <MemoryType.GOAL: 'goal'>
        """
        if not text or not text.strip():
            return default or MemoryType.CONTEXT

        # Check patterns in order of specificity
        if any(regex.search(text) for regex in self._fact_regex):
            return MemoryType.FACT

        if any(regex.search(text) for regex in self._preference_regex):
            return MemoryType.PREFERENCE

        if any(regex.search(text) for regex in self._goal_regex):
            return MemoryType.GOAL

        if any(regex.search(text) for regex in self._habit_regex):
            return MemoryType.HABIT

        if any(regex.search(text) for regex in self._event_regex):
            return MemoryType.EVENT

        # Default to context for general conversation
        return default or MemoryType.CONTEXT

    def classify_with_confidence(self, text: str) -> tuple[MemoryType, float]:
        """
        Classify a memory and return a confidence score.

        The confidence score is based on the number of matching patterns:
        - 3+ matches: 1.0 (very confident)
        - 2 matches: 0.8 (confident)
        - 1 match: 0.6 (somewhat confident)
        - 0 matches: 0.3 (default classification)

        Args:
            text: The memory text to classify

        Returns:
            Tuple of (MemoryType, confidence_score)

        Example:
            >>> classifier = MemoryClassifier()
            >>> memory_type, confidence = classifier.classify_with_confidence("My name is Alex and I work at Google")
            >>> print(f"{memory_type.value} (confidence: {confidence})")
            fact (confidence: 1.0)
        """
        if not text or not text.strip():
            return MemoryType.CONTEXT, 0.3

        # Count matches for each type
        type_scores = {
            MemoryType.FACT: sum(1 for regex in self._fact_regex if regex.search(text)),
            MemoryType.PREFERENCE: sum(1 for regex in self._preference_regex if regex.search(text)),
            MemoryType.GOAL: sum(1 for regex in self._goal_regex if regex.search(text)),
            MemoryType.HABIT: sum(1 for regex in self._habit_regex if regex.search(text)),
            MemoryType.EVENT: sum(1 for regex in self._event_regex if regex.search(text)),
        }

        # Find the type with the highest score
        max_score = max(type_scores.values())

        if max_score == 0:
            return MemoryType.CONTEXT, 0.3

        # Get the memory type with the highest score
        memory_type = max(type_scores.items(), key=lambda x: x[1])[0]

        # Calculate confidence based on number of matches
        if max_score >= 3:
            confidence = 1.0
        elif max_score == 2:
            confidence = 0.8
        else:  # max_score == 1
            confidence = 0.6

        return memory_type, confidence


# Global singleton instance for convenience
_classifier: Optional[MemoryClassifier] = None


def get_classifier() -> MemoryClassifier:
    """
    Get the global MemoryClassifier instance.

    Returns:
        The singleton MemoryClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = MemoryClassifier()
    return _classifier


def classify_memory(text: str, default: Optional[MemoryType] = None) -> MemoryType:
    """
    Convenience function to classify a memory using the global classifier.

    Args:
        text: The memory text to classify
        default: Default type if no pattern matches

    Returns:
        The detected MemoryType enum value

    Example:
        >>> from hippocampai.utils.memory_classifier import classify_memory
        >>> classify_memory("I love pizza")
        <MemoryType.PREFERENCE: 'preference'>
    """
    return get_classifier().classify(text, default)


def classify_memory_with_confidence(text: str) -> tuple[MemoryType, float]:
    """
    Convenience function to classify with confidence using the global classifier.

    Args:
        text: The memory text to classify

    Returns:
        Tuple of (MemoryType, confidence_score)

    Example:
        >>> from hippocampai.utils.memory_classifier import classify_memory_with_confidence
        >>> memory_type, confidence = classify_memory_with_confidence("I love pizza")
        >>> print(f"{memory_type.value} (confidence: {confidence})")
        preference (confidence: 0.6)
    """
    return get_classifier().classify_with_confidence(text)
