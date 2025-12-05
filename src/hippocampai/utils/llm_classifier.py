"""LLM-based dynamic memory classification for intelligent and consistent type detection."""

import hashlib
import logging
from typing import Optional, Tuple

from cachetools import TTLCache

from hippocampai.models.memory import MemoryType

logger = logging.getLogger(__name__)

# Cache for consistent classification (1 hour TTL, max 1000 entries)
_classification_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)


class LLMMemoryClassifier:
    """
    LLM-powered memory classifier with consistency guarantees.

    Uses LLM to intelligently classify memories based on semantic understanding,
    with caching to ensure consistent classification for the same input.

    Features:
    - Dynamic classification without hardcoded patterns
    - Semantic understanding of context
    - Consistent results through caching
    - Confidence scoring
    - Fallback to pattern-based classification
    """

    # Classification prompt template
    CLASSIFICATION_PROMPT = """You are a memory classification expert. Analyze the following text and classify it into ONE of these memory types:

**Memory Types:**
- **fact**: Personal information, identity statements, biographical data (e.g., "My name is Alex", "I work at Google", "I live in NYC")
- **preference**: Likes, dislikes, opinions, favorites (e.g., "I love pizza", "I prefer dark mode", "My favorite color is blue")
- **goal**: Intentions, aspirations, plans, objectives (e.g., "I want to learn Python", "My goal is to run a marathon", "I plan to visit Japan")
- **habit**: Routines, regular activities, repeated behaviors (e.g., "I usually wake up at 7am", "I always drink coffee in the morning", "I exercise every day")
- **event**: Specific occurrences, meetings, past/future happenings (e.g., "I met John yesterday", "The meeting happened last week", "I have a dentist appointment tomorrow")
- **context**: General conversation, neutral statements, observations (e.g., "The weather is nice", "That's interesting", "I understand")

**Text to classify:**
"{text}"

**Important instructions:**
1. Respond with ONLY the memory type word (fact, preference, goal, habit, event, or context)
2. Choose the MOST appropriate type based on semantic meaning
3. If uncertain, prefer more specific types over "context"
4. Be consistent - the same text should always get the same classification

**Your classification:**"""

    def __init__(self, llm_provider=None, llm_model=None, use_cache: bool = True):
        """
        Initialize LLM-based classifier.

        Args:
            llm_provider: LLM provider instance (optional, will use config default if not provided)
            llm_model: LLM model name (optional, will use config default if not provided)
            use_cache: Whether to cache classifications for consistency (default: True)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_cache = use_cache

        # Import here to avoid circular dependency
        try:
            from hippocampai.llm import get_llm_provider

            from hippocampai.config import get_config

            self.config = get_config()
            if not self.llm_provider:
                self.llm_provider = get_llm_provider()
        except ImportError as e:
            logger.warning(f"Could not import LLM dependencies: {e}")
            self.config = None
            self.llm_provider = None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for consistent lookups."""
        # Normalize text and hash for consistent keys
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    def _classify_with_llm(self, text: str) -> Tuple[MemoryType, float]:
        """
        Classify using LLM.

        Args:
            text: Text to classify

        Returns:
            Tuple of (MemoryType, confidence)
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not available")

        # Format prompt with text
        prompt = self.CLASSIFICATION_PROMPT.format(text=text)

        try:
            # Call LLM with low temperature for consistency
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.1,  # Very low temperature for consistent results
                max_tokens=20,  # We only need a single word response
            )

            # Extract classification from response
            classification_text = response.strip().lower()

            # Map response to MemoryType
            type_mapping = {
                "fact": MemoryType.FACT,
                "preference": MemoryType.PREFERENCE,
                "goal": MemoryType.GOAL,
                "habit": MemoryType.HABIT,
                "event": MemoryType.EVENT,
                "context": MemoryType.CONTEXT,
            }

            # Find exact match or best partial match
            for key, mem_type in type_mapping.items():
                if key in classification_text:
                    confidence = 0.9  # High confidence for LLM classification
                    logger.debug(
                        f"LLM classified '{text[:50]}...' as {mem_type.value} (confidence: {confidence})"
                    )
                    return mem_type, confidence

            # Fallback to context if no match
            logger.warning(
                f"Could not parse LLM response '{classification_text}', defaulting to context"
            )
            return MemoryType.CONTEXT, 0.5

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise

    def classify(self, text: str, default: Optional[MemoryType] = None) -> MemoryType:
        """
        Classify a memory with caching for consistency.

        Args:
            text: The memory text to classify
            default: Default type if classification fails

        Returns:
            The detected MemoryType enum value
        """
        memory_type, _ = self.classify_with_confidence(text, default)
        return memory_type

    def classify_with_confidence(
        self, text: str, default: Optional[MemoryType] = None
    ) -> Tuple[MemoryType, float]:
        """
        Classify with confidence score and caching for consistency.

        Args:
            text: The memory text to classify
            default: Default type if classification fails

        Returns:
            Tuple of (MemoryType, confidence_score)
        """
        if not text or not text.strip():
            return default or MemoryType.CONTEXT, 0.3

        # Check cache first for consistency
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in _classification_cache:
                cached_result = _classification_cache[cache_key]
                logger.debug(f"Using cached classification for '{text[:50]}...'")
                return cached_result

        # Try LLM classification
        try:
            if self.llm_provider:
                result = self._classify_with_llm(text)

                # Cache result for consistency
                if self.use_cache:
                    cache_key = self._get_cache_key(text)
                    _classification_cache[cache_key] = result

                return result
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to pattern-based")

        # Fallback to pattern-based classification
        from hippocampai.utils.memory_classifier import get_classifier as get_pattern_classifier

        pattern_classifier = get_pattern_classifier()
        result = pattern_classifier.classify_with_confidence(text)

        # Cache fallback result for consistency
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            _classification_cache[cache_key] = result

        return result


# Global singleton instance
_llm_classifier: Optional[LLMMemoryClassifier] = None


def get_llm_classifier(use_cache: bool = True) -> LLMMemoryClassifier:
    """
    Get the global LLMMemoryClassifier instance.

    Args:
        use_cache: Whether to use caching for consistency

    Returns:
        The singleton LLMMemoryClassifier instance
    """
    global _llm_classifier
    if _llm_classifier is None:
        _llm_classifier = LLMMemoryClassifier(use_cache=use_cache)
    return _llm_classifier


def classify_memory_with_llm(text: str, default: Optional[MemoryType] = None) -> MemoryType:
    """
    Convenience function to classify using LLM-based classifier.

    Args:
        text: The memory text to classify
        default: Default type if classification fails

    Returns:
        The detected MemoryType enum value

    Example:
        >>> from hippocampai.utils.llm_classifier import classify_memory_with_llm
        >>> classify_memory_with_llm("I love pizza")
        <MemoryType.PREFERENCE: 'preference'>
    """
    return get_llm_classifier().classify(text, default)


def classify_memory_with_llm_and_confidence(text: str) -> Tuple[MemoryType, float]:
    """
    Convenience function to classify with confidence using LLM.

    Args:
        text: The memory text to classify

    Returns:
        Tuple of (MemoryType, confidence_score)

    Example:
        >>> from hippocampai.utils.llm_classifier import classify_memory_with_llm_and_confidence
        >>> memory_type, confidence = classify_memory_with_llm_and_confidence("I love pizza")
        >>> print(f"{memory_type.value} (confidence: {confidence})")
        preference (confidence: 0.9)
    """
    return get_llm_classifier().classify_with_confidence(text)


def clear_classification_cache() -> None:
    """Clear the classification cache. Useful for testing or when consistency needs to be reset."""
    _classification_cache.clear()
    logger.info("Classification cache cleared")
