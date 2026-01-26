"""
Agentic Memory Classification System.

.. deprecated::
    This module is deprecated. Use :mod:`hippocampai.utils.classifier_service` instead:

    >>> from hippocampai.utils.classifier_service import (
    ...     get_classifier_service,
    ...     ClassificationStrategy,
    ... )
    >>> service = get_classifier_service(strategy=ClassificationStrategy.AGENTIC)
    >>> service.classify(text)

Uses a multi-step LLM-based approach for accurate memory type classification:
1. Initial classification with reasoning
2. Confidence assessment
3. Validation against examples
4. Final decision with explanation

This module now delegates to the unified ClassifierService.
"""

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from cachetools import TTLCache

from hippocampai.models.memory import MemoryType

logger = logging.getLogger(__name__)

# Cache for consistent classification (2 hour TTL, max 2000 entries)
_agentic_cache: TTLCache[str, "ClassificationResult"] = TTLCache(maxsize=2000, ttl=7200)


class ClassificationConfidence(Enum):
    """Confidence levels for classification."""

    HIGH = "high"  # 0.9+
    MEDIUM = "medium"  # 0.7-0.9
    LOW = "low"  # 0.5-0.7
    UNCERTAIN = "uncertain"  # <0.5


@dataclass
class ClassificationResult:
    """Result of agentic classification."""

    memory_type: MemoryType
    confidence: float
    confidence_level: ClassificationConfidence
    reasoning: str
    alternative_type: Optional[MemoryType] = None
    alternative_confidence: Optional[float] = None


# Detailed type definitions with examples for the agent (compact version)
MEMORY_TYPE_DEFINITIONS = """
## Memory Types

FACT: Personal info, identity, biographical data
- "My name is Alex", "I work at Google", "I live in SF"

PREFERENCE: Likes, dislikes, opinions, favorites
- "I love pizza", "I prefer dark mode", "Python is my favorite"

GOAL: Intentions, aspirations, plans
- "I want to learn Python", "My goal is to run a marathon"

HABIT: Routines, regular activities, patterns
- "I usually wake up at 7am", "I exercise every day"

EVENT: Specific occurrences, meetings, time-bound activities
- "I met John yesterday", "The meeting is at 3pm"

CONTEXT: General conversation, observations, acknowledgments
- "The weather is nice", "That's interesting", "I understand"
"""


class AgenticMemoryClassifier:
    """
    Agentic memory classifier using multi-step LLM reasoning.

    This classifier uses a structured approach:
    1. Analyze the text for key indicators
    2. Consider multiple possible classifications
    3. Reason about the best fit
    4. Provide confidence and explanation
    """

    CLASSIFICATION_PROMPT = """You are an expert memory classification agent. Your task is to classify the given text into exactly ONE memory type.

{type_definitions}

## Text to Classify
"{text}"

## Instructions
1. Analyze the text carefully
2. Identify key indicators (verbs, temporal markers, sentiment, etc.)
3. Consider which type best captures the INTENT and CONTENT
4. If multiple types could apply, choose the MOST SPECIFIC one
5. Provide your reasoning

## Response Format (JSON)
{{
    "analysis": "Brief analysis of key indicators in the text",
    "primary_type": "fact|preference|goal|habit|event|context",
    "primary_confidence": 0.0-1.0,
    "reasoning": "Why this type is the best fit",
    "alternative_type": "second best type or null",
    "alternative_confidence": 0.0-1.0 or null
}}

Respond with ONLY the JSON object, no other text."""

    VALIDATION_PROMPT = """You are validating a memory classification. Given the text and proposed classification, determine if it's correct.

Text: "{text}"
Proposed Type: {proposed_type}
Reasoning: {reasoning}

Is this classification correct? If not, what should it be?

Response Format (JSON):
{{
    "is_correct": true|false,
    "corrected_type": "type if incorrect, null if correct",
    "explanation": "brief explanation"
}}

Respond with ONLY the JSON object."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        use_cache: bool = True,
        validate_classifications: bool = False,
    ):
        """
        Initialize the agentic classifier.

        Args:
            llm: LLM instance (BaseLLM). If None, will try to get from config.
            use_cache: Whether to cache classifications for consistency.
            validate_classifications: Whether to run a validation step (slower but more accurate).
        """
        self.llm = llm
        self.use_cache = use_cache
        self.validate_classifications = validate_classifications

        # Try to get LLM from config if not provided
        if self.llm is None:
            self._init_llm_from_config()

    def _init_llm_from_config(self) -> None:
        """Initialize LLM from configuration."""
        try:
            import os

            from hippocampai.adapters.provider_groq import GroqLLM
            from hippocampai.adapters.provider_ollama import OllamaLLM
            from hippocampai.config import get_config

            config = get_config()

            if config.llm_provider == "groq" and config.allow_cloud:
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self.llm = GroqLLM(api_key=api_key, model=config.llm_model)
                    logger.info(f"Initialized Groq LLM for classification: {config.llm_model}")
            elif config.llm_provider == "ollama":
                self.llm = OllamaLLM(model=config.llm_model, base_url=config.llm_base_url)
                logger.info(f"Initialized Ollama LLM for classification: {config.llm_model}")
        except Exception as e:
            logger.warning(f"Could not initialize LLM from config: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for consistent lookups."""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling common issues."""
        # Clean up response
        response = response.strip()

        # Try to extract JSON from markdown code blocks (safe string-based approach)
        code_start = response.find("```")
        if code_start != -1:
            # Skip past the opening ``` and optional language identifier
            content_start = response.find("\n", code_start)
            if content_start != -1:
                code_end = response.find("```", content_start)
                if code_end != -1:
                    response = response[content_start:code_end].strip()

        # Try to find JSON object (safe approach: find first { and last })
        brace_start = response.find("{")
        brace_end = response.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            response = response[brace_start : brace_end + 1]

        try:
            parsed: dict[str, Any] = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None

    def _map_type_string(self, type_str: str) -> MemoryType:
        """Map string to MemoryType enum."""
        type_mapping = {
            "fact": MemoryType.FACT,
            "preference": MemoryType.PREFERENCE,
            "goal": MemoryType.GOAL,
            "habit": MemoryType.HABIT,
            "event": MemoryType.EVENT,
            "context": MemoryType.CONTEXT,
        }
        return type_mapping.get(type_str.lower().strip(), MemoryType.CONTEXT)

    def _get_confidence_level(self, confidence: float) -> ClassificationConfidence:
        """Map confidence score to level."""
        if confidence >= 0.9:
            return ClassificationConfidence.HIGH
        elif confidence >= 0.7:
            return ClassificationConfidence.MEDIUM
        elif confidence >= 0.5:
            return ClassificationConfidence.LOW
        else:
            return ClassificationConfidence.UNCERTAIN

    def _classify_with_llm(self, text: str) -> ClassificationResult:
        """Perform LLM-based classification."""
        if not self.llm:
            raise RuntimeError("LLM not available for classification")

        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            type_definitions=MEMORY_TYPE_DEFINITIONS, text=text
        )

        try:
            # Call LLM with low temperature for consistency
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )

            # Parse response
            parsed = self._parse_json_response(response)

            if not parsed:
                logger.warning("Could not parse LLM response, using fallback")
                return self._fallback_classification(text)

            # Extract classification
            primary_type = self._map_type_string(parsed.get("primary_type", "context"))
            primary_confidence = float(parsed.get("primary_confidence", 0.7))
            reasoning = parsed.get("reasoning", "LLM classification")

            alternative_type = None
            alternative_confidence = None
            if parsed.get("alternative_type"):
                alternative_type = self._map_type_string(parsed["alternative_type"])
                alternative_confidence = float(parsed.get("alternative_confidence", 0.0))

            result = ClassificationResult(
                memory_type=primary_type,
                confidence=primary_confidence,
                confidence_level=self._get_confidence_level(primary_confidence),
                reasoning=reasoning,
                alternative_type=alternative_type,
                alternative_confidence=alternative_confidence,
            )

            # Optional validation step
            if self.validate_classifications and primary_confidence < 0.9:
                result = self._validate_classification(text, result)

            logger.debug(
                f"Classified '{text[:50]}...' as {result.memory_type.value} "
                f"(confidence: {result.confidence:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._fallback_classification(text)

    def _validate_classification(
        self, text: str, result: ClassificationResult
    ) -> ClassificationResult:
        """Validate and potentially correct a classification."""
        if not self.llm:
            return result

        prompt = self.VALIDATION_PROMPT.format(
            text=text,
            proposed_type=result.memory_type.value,
            reasoning=result.reasoning,
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=100,
            )

            parsed = self._parse_json_response(response)

            if parsed and not parsed.get("is_correct", True):
                corrected_type = parsed.get("corrected_type")
                if corrected_type:
                    return ClassificationResult(
                        memory_type=self._map_type_string(corrected_type),
                        confidence=result.confidence
                        * 0.9,  # Slightly lower confidence after correction
                        confidence_level=self._get_confidence_level(result.confidence * 0.9),
                        reasoning=f"Corrected: {parsed.get('explanation', 'validation correction')}",
                        alternative_type=result.memory_type,
                        alternative_confidence=result.confidence * 0.5,
                    )

        except Exception as e:
            logger.warning(f"Validation failed: {e}")

        return result

    def _fallback_classification(self, text: str) -> ClassificationResult:
        """Fallback to pattern-based classification."""
        try:
            from hippocampai.utils.memory_classifier import get_classifier

            pattern_classifier = get_classifier()
            memory_type, confidence = pattern_classifier.classify_with_confidence(text)

            return ClassificationResult(
                memory_type=memory_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                reasoning="Pattern-based fallback classification",
            )
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            return ClassificationResult(
                memory_type=MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ClassificationConfidence.UNCERTAIN,
                reasoning="Default classification due to errors",
            )

    def classify(self, text: str, default: Optional[MemoryType] = None) -> MemoryType:
        """
        Classify a memory text into a type.

        Args:
            text: The memory text to classify.
            default: Default type if classification fails.

        Returns:
            The classified MemoryType.
        """
        result = self.classify_with_details(text, default)
        return result.memory_type

    def classify_with_confidence(
        self, text: str, default: Optional[MemoryType] = None
    ) -> tuple[MemoryType, float]:
        """
        Classify with confidence score.

        Args:
            text: The memory text to classify.
            default: Default type if classification fails.

        Returns:
            Tuple of (MemoryType, confidence_score).
        """
        result = self.classify_with_details(text, default)
        return result.memory_type, result.confidence

    def classify_with_details(
        self, text: str, default: Optional[MemoryType] = None
    ) -> ClassificationResult:
        """
        Classify with full details including reasoning.

        Args:
            text: The memory text to classify.
            default: Default type if classification fails.

        Returns:
            ClassificationResult with full details.
        """
        if not text or not text.strip():
            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ClassificationConfidence.UNCERTAIN,
                reasoning="Empty text",
            )

        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in _agentic_cache:
                logger.debug(f"Using cached classification for '{text[:50]}...'")
                cached_result: ClassificationResult = _agentic_cache[cache_key]
                return cached_result

        # Perform classification
        if self.llm:
            result = self._classify_with_llm(text)
        else:
            result = self._fallback_classification(text)

        # Cache result
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            _agentic_cache[cache_key] = result

        return result

    def classify_batch(
        self, texts: list[str], default: Optional[MemoryType] = None
    ) -> list[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify.
            default: Default type if classification fails.

        Returns:
            List of ClassificationResults.
        """
        return [self.classify_with_details(text, default) for text in texts]


# Global singleton instance
_agentic_classifier: Optional[AgenticMemoryClassifier] = None


def get_agentic_classifier(
    use_cache: bool = True,
    validate: bool = False,
) -> AgenticMemoryClassifier:
    """
    Get the global AgenticMemoryClassifier instance.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.get_classifier_service` instead
        with ``strategy=ClassificationStrategy.AGENTIC``.

    Args:
        use_cache: Whether to use caching for consistency.
        validate: Whether to run validation step.

    Returns:
        The singleton AgenticMemoryClassifier instance.
    """
    warnings.warn(
        "get_agentic_classifier is deprecated. "
        "Use classifier_service.get_classifier_service(strategy=ClassificationStrategy.AGENTIC) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _agentic_classifier
    if _agentic_classifier is None:
        _agentic_classifier = AgenticMemoryClassifier(
            use_cache=use_cache,
            validate_classifications=validate,
        )
    return _agentic_classifier


def classify_memory_agentic(text: str, default: Optional[MemoryType] = None) -> MemoryType:
    """
    Convenience function to classify using agentic classifier.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.classify_memory` instead
        with ``strategy=ClassificationStrategy.AGENTIC``.

    Args:
        text: The memory text to classify.
        default: Default type if classification fails.

    Returns:
        The detected MemoryType.
    """
    warnings.warn(
        "classify_memory_agentic is deprecated. "
        "Use classifier_service.classify_memory(text, strategy=ClassificationStrategy.AGENTIC) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import (
        ClassificationStrategy,
        classify_memory,
    )
    return classify_memory(text, default, ClassificationStrategy.AGENTIC)


def classify_memory_agentic_with_confidence(
    text: str,
) -> tuple[MemoryType, float]:
    """
    Convenience function to classify with confidence.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.classify_memory_with_confidence` instead
        with ``strategy=ClassificationStrategy.AGENTIC``.

    Args:
        text: The memory text to classify.

    Returns:
        Tuple of (MemoryType, confidence_score).
    """
    warnings.warn(
        "classify_memory_agentic_with_confidence is deprecated. "
        "Use classifier_service.classify_memory_with_confidence(text, strategy=ClassificationStrategy.AGENTIC) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import (
        ClassificationStrategy,
        classify_memory_with_confidence,
    )
    result = classify_memory_with_confidence(text, strategy=ClassificationStrategy.AGENTIC)
    return (result[0], result[1])


def classify_memory_agentic_with_details(
    text: str,
) -> ClassificationResult:
    """
    Convenience function to classify with full details.

    .. deprecated::
        Use the unified :class:`hippocampai.utils.classifier_service.ClassifierService` instead.

    Args:
        text: The memory text to classify.

    Returns:
        ClassificationResult with full details.
    """
    warnings.warn(
        "classify_memory_agentic_with_details is deprecated. "
        "Use classifier_service.get_classifier_service().classify_with_details() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Still use the local classifier for the detailed result format
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return get_agentic_classifier().classify_with_details(text)


def clear_agentic_cache() -> None:
    """Clear the agentic classification cache.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.clear_classification_cache` instead.
    """
    warnings.warn(
        "clear_agentic_cache is deprecated. "
        "Use classifier_service.clear_classification_cache() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import clear_classification_cache
    clear_classification_cache()
