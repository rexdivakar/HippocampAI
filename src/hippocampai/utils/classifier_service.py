"""Unified classifier service with strategy pattern for memory type classification.

This module consolidates the three classifier implementations:
- Pattern-based (fast, rule-based)
- LLM-based (simple LLM call)
- Agentic (multi-step LLM reasoning)

Usage:
    from hippocampai.utils.classifier_service import get_classifier_service

    service = get_classifier_service()
    memory_type = service.classify(text)
    memory_type, confidence = service.classify_with_confidence(text)
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from cachetools import TTLCache

from hippocampai.models.memory import MemoryType

logger = logging.getLogger(__name__)

# Unified cache with 2-hour TTL, max 2000 entries
_unified_cache: TTLCache[str, "ClassificationResult"] = TTLCache(maxsize=2000, ttl=7200)


class ClassificationStrategy(Enum):
    """Available classification strategies."""

    PATTERN = "pattern"  # Fast pattern-based (no LLM)
    LLM = "llm"  # Simple LLM classification
    AGENTIC = "agentic"  # Multi-step LLM reasoning
    AUTO = "auto"  # Automatic selection (agentic -> llm -> pattern)


class ConfidenceLevel(Enum):
    """Confidence levels for classification results."""

    HIGH = "high"  # 0.9+
    MEDIUM = "medium"  # 0.7-0.9
    LOW = "low"  # 0.5-0.7
    UNCERTAIN = "uncertain"  # <0.5


@dataclass
class ClassificationResult:
    """Result of memory classification."""

    memory_type: MemoryType
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: str
    strategy_used: ClassificationStrategy
    alternative_type: Optional[MemoryType] = None
    alternative_confidence: Optional[float] = None

    def to_tuple(self) -> tuple[MemoryType, float]:
        """Return (memory_type, confidence) tuple for backward compatibility."""
        return (self.memory_type, self.confidence)


def _get_confidence_level(confidence: float) -> ConfidenceLevel:
    """Map confidence score to level."""
    if confidence >= 0.9:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.7:
        return ConfidenceLevel.MEDIUM
    elif confidence >= 0.5:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNCERTAIN


class ClassifierStrategy(ABC):
    """Abstract base class for classifier strategies."""

    @abstractmethod
    def classify(
        self, text: str, default: Optional[MemoryType] = None
    ) -> ClassificationResult:
        """Classify text into a memory type."""
        pass


class PatternClassifierStrategy(ClassifierStrategy):
    """Fast pattern-based classification strategy."""

    def __init__(self) -> None:
        # Import here to avoid circular dependency
        from hippocampai.utils.memory_classifier import MemoryClassifier
        self._classifier = MemoryClassifier()

    def classify(
        self, text: str, default: Optional[MemoryType] = None
    ) -> ClassificationResult:
        """Classify using pattern matching."""
        if not text or not text.strip():
            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                reasoning="Empty text",
                strategy_used=ClassificationStrategy.PATTERN,
            )

        memory_type, confidence = self._classifier.classify_with_confidence(text)

        return ClassificationResult(
            memory_type=memory_type,
            confidence=confidence,
            confidence_level=_get_confidence_level(confidence),
            reasoning="Pattern-based classification",
            strategy_used=ClassificationStrategy.PATTERN,
        )


class LLMClassifierStrategy(ClassifierStrategy):
    """Simple LLM-based classification strategy."""

    PROMPT_TEMPLATE = """Classify this text into ONE memory type:
- fact: Personal info, identity, biographical data
- preference: Likes, dislikes, opinions, favorites
- goal: Intentions, aspirations, plans
- habit: Routines, regular activities
- event: Specific occurrences, meetings
- context: General conversation

Text: "{text}"

Respond with ONLY the type word (fact, preference, goal, habit, event, or context)."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm
        if self.llm is None:
            self._init_llm()

    def _init_llm(self) -> None:
        """Try to initialize LLM from config."""
        import os

        try:
            from hippocampai.config import get_config

            config = get_config()

            if config.llm_provider == "groq" and config.allow_cloud:
                from hippocampai.adapters.provider_groq import GroqLLM
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self.llm = GroqLLM(api_key=api_key, model=config.llm_model)
            elif config.llm_provider == "ollama":
                from hippocampai.adapters.provider_ollama import OllamaLLM
                self.llm = OllamaLLM(model=config.llm_model, base_url=config.llm_base_url)
        except Exception as e:
            logger.debug(f"Could not initialize LLM: {e}")

    def classify(
        self, text: str, default: Optional[MemoryType] = None
    ) -> ClassificationResult:
        """Classify using LLM."""
        if not text or not text.strip():
            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                reasoning="Empty text",
                strategy_used=ClassificationStrategy.LLM,
            )

        if not self.llm:
            raise RuntimeError("LLM not available")

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=20,
            )

            type_mapping = {
                "fact": MemoryType.FACT,
                "preference": MemoryType.PREFERENCE,
                "goal": MemoryType.GOAL,
                "habit": MemoryType.HABIT,
                "event": MemoryType.EVENT,
                "context": MemoryType.CONTEXT,
            }

            response_lower = response.strip().lower()
            for key, mem_type in type_mapping.items():
                if key in response_lower:
                    return ClassificationResult(
                        memory_type=mem_type,
                        confidence=0.85,
                        confidence_level=ConfidenceLevel.MEDIUM,
                        reasoning="LLM classification",
                        strategy_used=ClassificationStrategy.LLM,
                    )

            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.5,
                confidence_level=ConfidenceLevel.LOW,
                reasoning=f"LLM response unparseable: {response[:50]}",
                strategy_used=ClassificationStrategy.LLM,
            )

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise


class AgenticClassifierStrategy(ClassifierStrategy):
    """Multi-step LLM reasoning classification strategy."""

    PROMPT_TEMPLATE = """You are an expert memory classification agent. Classify the given text into exactly ONE memory type.

## Memory Types
FACT: Personal info, identity, biographical data ("My name is Alex", "I work at Google")
PREFERENCE: Likes, dislikes, opinions ("I love pizza", "I prefer dark mode")
GOAL: Intentions, aspirations, plans ("I want to learn Python", "My goal is to run a marathon")
HABIT: Routines, regular activities ("I usually wake up at 7am", "I exercise every day")
EVENT: Specific occurrences, meetings ("I met John yesterday", "The meeting is at 3pm")
CONTEXT: General conversation, observations ("The weather is nice", "That's interesting")

## Text to Classify
"{text}"

## Response Format (JSON)
{{
    "primary_type": "fact|preference|goal|habit|event|context",
    "primary_confidence": 0.0-1.0,
    "reasoning": "Brief explanation why this type fits best",
    "alternative_type": "second best type or null",
    "alternative_confidence": 0.0-1.0 or null
}}

Respond with ONLY the JSON object."""

    def __init__(self, llm: Optional[Any] = None, validate: bool = False) -> None:
        self.llm = llm
        self.validate = validate
        if self.llm is None:
            self._init_llm()

    def _init_llm(self) -> None:
        """Try to initialize LLM from config."""
        import os

        try:
            from hippocampai.config import get_config

            config = get_config()

            if config.llm_provider == "groq" and config.allow_cloud:
                from hippocampai.adapters.provider_groq import GroqLLM
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self.llm = GroqLLM(api_key=api_key, model=config.llm_model)
            elif config.llm_provider == "ollama":
                from hippocampai.adapters.provider_ollama import OllamaLLM
                self.llm = OllamaLLM(model=config.llm_model, base_url=config.llm_base_url)
        except Exception as e:
            logger.debug(f"Could not initialize LLM: {e}")

    def _parse_json_response(self, response: str) -> Optional[dict[str, Any]]:
        """Parse JSON from LLM response."""
        import json

        response = response.strip()

        # Extract from markdown code blocks
        code_start = response.find("```")
        if code_start != -1:
            content_start = response.find("\n", code_start)
            if content_start != -1:
                code_end = response.find("```", content_start)
                if code_end != -1:
                    response = response[content_start:code_end].strip()

        # Find JSON object
        brace_start = response.find("{")
        brace_end = response.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            response = response[brace_start:brace_end + 1]

        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    def classify(
        self, text: str, default: Optional[MemoryType] = None
    ) -> ClassificationResult:
        """Classify using multi-step LLM reasoning."""
        if not text or not text.strip():
            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                reasoning="Empty text",
                strategy_used=ClassificationStrategy.AGENTIC,
            )

        if not self.llm:
            raise RuntimeError("LLM not available for agentic classification")

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )

            parsed = self._parse_json_response(response)

            if not parsed:
                logger.warning("Could not parse agentic LLM response")
                raise RuntimeError("Failed to parse LLM response")

            type_mapping = {
                "fact": MemoryType.FACT,
                "preference": MemoryType.PREFERENCE,
                "goal": MemoryType.GOAL,
                "habit": MemoryType.HABIT,
                "event": MemoryType.EVENT,
                "context": MemoryType.CONTEXT,
            }

            primary_type_str = parsed.get("primary_type", "context").lower().strip()
            primary_type = type_mapping.get(primary_type_str, MemoryType.CONTEXT)
            confidence = float(parsed.get("primary_confidence", 0.7))
            reasoning = parsed.get("reasoning", "Agentic classification")

            alternative_type = None
            alternative_confidence = None
            if parsed.get("alternative_type"):
                alt_str = parsed["alternative_type"].lower().strip()
                alternative_type = type_mapping.get(alt_str)
                alternative_confidence = float(parsed.get("alternative_confidence", 0.0))

            return ClassificationResult(
                memory_type=primary_type,
                confidence=confidence,
                confidence_level=_get_confidence_level(confidence),
                reasoning=reasoning,
                strategy_used=ClassificationStrategy.AGENTIC,
                alternative_type=alternative_type,
                alternative_confidence=alternative_confidence,
            )

        except Exception as e:
            logger.error(f"Agentic classification failed: {e}")
            raise


class ClassifierService:
    """Unified classifier service with strategy pattern and caching.

    This is the recommended entry point for memory classification.
    It handles strategy selection, caching, and fallbacks automatically.

    Example:
        service = get_classifier_service()

        # Simple classification
        memory_type = service.classify(text)

        # With confidence
        memory_type, confidence = service.classify_with_confidence(text)

        # Full details
        result = service.classify_with_details(text)
    """

    def __init__(
        self,
        default_strategy: ClassificationStrategy = ClassificationStrategy.AUTO,
        use_cache: bool = True,
        llm: Optional[Any] = None,
    ) -> None:
        """Initialize the classifier service.

        Args:
            default_strategy: Default classification strategy to use.
            use_cache: Whether to cache results for consistency.
            llm: Optional LLM instance to use for LLM-based strategies.
        """
        self.default_strategy = default_strategy
        self.use_cache = use_cache
        self.llm = llm

        # Lazy-initialize strategies
        self._pattern_strategy: Optional[PatternClassifierStrategy] = None
        self._llm_strategy: Optional[LLMClassifierStrategy] = None
        self._agentic_strategy: Optional[AgenticClassifierStrategy] = None

    def _get_strategy(self, strategy: ClassificationStrategy) -> ClassifierStrategy:
        """Get or create the requested strategy."""
        if strategy == ClassificationStrategy.PATTERN:
            if self._pattern_strategy is None:
                self._pattern_strategy = PatternClassifierStrategy()
            return self._pattern_strategy

        elif strategy == ClassificationStrategy.LLM:
            if self._llm_strategy is None:
                self._llm_strategy = LLMClassifierStrategy(llm=self.llm)
            return self._llm_strategy

        elif strategy == ClassificationStrategy.AGENTIC:
            if self._agentic_strategy is None:
                self._agentic_strategy = AgenticClassifierStrategy(llm=self.llm)
            return self._agentic_strategy

        raise ValueError(f"Unknown strategy: {strategy}")

    def _get_cache_key(self, text: str, strategy: ClassificationStrategy) -> str:
        """Generate cache key for consistent lookups."""
        normalized = text.strip().lower()
        key_data = f"{normalized}:{strategy.value}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def classify_with_details(
        self,
        text: str,
        default: Optional[MemoryType] = None,
        strategy: Optional[ClassificationStrategy] = None,
    ) -> ClassificationResult:
        """Classify text with full details.

        Args:
            text: Text to classify.
            default: Default memory type if classification fails.
            strategy: Strategy to use (defaults to service default).

        Returns:
            ClassificationResult with full details.
        """
        strategy = strategy or self.default_strategy

        if not text or not text.strip():
            return ClassificationResult(
                memory_type=default or MemoryType.CONTEXT,
                confidence=0.3,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                reasoning="Empty text",
                strategy_used=strategy,
            )

        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(text, strategy)
            if cache_key in _unified_cache:
                logger.debug(f"Cache hit for '{text[:30]}...'")
                return _unified_cache[cache_key]

        result: Optional[ClassificationResult] = None

        # AUTO strategy: try agentic -> llm -> pattern
        if strategy == ClassificationStrategy.AUTO:
            # Try agentic first
            try:
                agentic = self._get_strategy(ClassificationStrategy.AGENTIC)
                result = agentic.classify(text, default)
            except Exception as e:
                logger.debug(f"Agentic failed, trying LLM: {e}")

            # Fall back to simple LLM
            if result is None:
                try:
                    llm = self._get_strategy(ClassificationStrategy.LLM)
                    result = llm.classify(text, default)
                except Exception as e:
                    logger.debug(f"LLM failed, using pattern: {e}")

            # Final fallback to pattern
            if result is None:
                pattern = self._get_strategy(ClassificationStrategy.PATTERN)
                result = pattern.classify(text, default)
        else:
            # Use specific strategy with pattern fallback
            try:
                strat = self._get_strategy(strategy)
                result = strat.classify(text, default)
            except Exception as e:
                logger.warning(f"{strategy.value} classification failed: {e}")
                pattern = self._get_strategy(ClassificationStrategy.PATTERN)
                result = pattern.classify(text, default)

        # Cache result
        if self.use_cache and result:
            cache_key = self._get_cache_key(text, strategy)
            _unified_cache[cache_key] = result

        return result

    def classify(
        self,
        text: str,
        default: Optional[MemoryType] = None,
        strategy: Optional[ClassificationStrategy] = None,
    ) -> MemoryType:
        """Classify text into a memory type.

        Args:
            text: Text to classify.
            default: Default memory type if classification fails.
            strategy: Strategy to use (defaults to service default).

        Returns:
            The classified MemoryType.
        """
        result = self.classify_with_details(text, default, strategy)
        return result.memory_type

    def classify_with_confidence(
        self,
        text: str,
        default: Optional[MemoryType] = None,
        strategy: Optional[ClassificationStrategy] = None,
    ) -> tuple[MemoryType, float]:
        """Classify text with confidence score.

        Args:
            text: Text to classify.
            default: Default memory type if classification fails.
            strategy: Strategy to use (defaults to service default).

        Returns:
            Tuple of (MemoryType, confidence_score).
        """
        result = self.classify_with_details(text, default, strategy)
        return result.to_tuple()

    def classify_batch(
        self,
        texts: list[str],
        default: Optional[MemoryType] = None,
        strategy: Optional[ClassificationStrategy] = None,
    ) -> list[ClassificationResult]:
        """Classify multiple texts.

        Args:
            texts: List of texts to classify.
            default: Default memory type if classification fails.
            strategy: Strategy to use (defaults to service default).

        Returns:
            List of ClassificationResults.
        """
        return [self.classify_with_details(text, default, strategy) for text in texts]


# Global singleton instance
_classifier_service: Optional[ClassifierService] = None


def get_classifier_service(
    strategy: ClassificationStrategy = ClassificationStrategy.AUTO,
    use_cache: bool = True,
) -> ClassifierService:
    """Get the global ClassifierService instance.

    Args:
        strategy: Default classification strategy.
        use_cache: Whether to use caching.

    Returns:
        The singleton ClassifierService instance.
    """
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService(
            default_strategy=strategy,
            use_cache=use_cache,
        )
    return _classifier_service


def classify_memory(
    text: str,
    default: Optional[MemoryType] = None,
    strategy: ClassificationStrategy = ClassificationStrategy.AUTO,
) -> MemoryType:
    """Convenience function to classify a memory.

    Args:
        text: Text to classify.
        default: Default memory type if classification fails.
        strategy: Classification strategy to use.

    Returns:
        The classified MemoryType.
    """
    return get_classifier_service().classify(text, default, strategy)


def classify_memory_with_confidence(
    text: str,
    default: Optional[MemoryType] = None,
    strategy: ClassificationStrategy = ClassificationStrategy.AUTO,
) -> tuple[MemoryType, float]:
    """Convenience function to classify with confidence.

    Args:
        text: Text to classify.
        default: Default memory type if classification fails.
        strategy: Classification strategy to use.

    Returns:
        Tuple of (MemoryType, confidence_score).
    """
    return get_classifier_service().classify_with_confidence(text, default, strategy)


def clear_classification_cache() -> None:
    """Clear the unified classification cache."""
    _unified_cache.clear()
    logger.info("Unified classification cache cleared")
