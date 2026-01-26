"""LLM-based dynamic memory classification for intelligent and consistent type detection.

.. deprecated::
    This module is deprecated. Use :mod:`hippocampai.utils.classifier_service` instead:

    >>> from hippocampai.utils.classifier_service import classify_memory
    >>> classify_memory(text)

This module now delegates to the unified ClassifierService for all classification.
"""

import logging
import warnings
from typing import Optional, Tuple

from hippocampai.models.memory import MemoryType

logger = logging.getLogger(__name__)


class LLMMemoryClassifier:
    """
    LLM-powered memory classifier with consistency guarantees.

    .. deprecated::
        Use :class:`hippocampai.utils.classifier_service.ClassifierService` instead.

    This class now delegates to the unified ClassifierService.
    """

    def __init__(self, llm_provider=None, llm_model=None, use_cache: bool = True):
        """
        Initialize LLM-based classifier.

        Args:
            llm_provider: LLM provider instance (ignored, kept for compatibility)
            llm_model: LLM model name (ignored, kept for compatibility)
            use_cache: Whether to cache classifications for consistency (default: True)
        """
        warnings.warn(
            "LLMMemoryClassifier is deprecated. Use classifier_service.ClassifierService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_cache = use_cache
        self._service = None

    def _get_service(self):
        """Get the unified classifier service lazily."""
        if self._service is None:
            from hippocampai.utils.classifier_service import (
                ClassificationStrategy,
                get_classifier_service,
            )
            self._service = get_classifier_service(
                strategy=ClassificationStrategy.AUTO,
                use_cache=self.use_cache,
            )
        return self._service

    def classify(self, text: str, default: Optional[MemoryType] = None) -> MemoryType:
        """
        Classify a memory with caching for consistency.

        Args:
            text: The memory text to classify
            default: Default type if classification fails

        Returns:
            The detected MemoryType enum value
        """
        return self._get_service().classify(text, default)

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
        return self._get_service().classify_with_confidence(text, default)


# Global singleton instance
_llm_classifier: Optional[LLMMemoryClassifier] = None


def get_llm_classifier(use_cache: bool = True) -> LLMMemoryClassifier:
    """
    Get the global LLMMemoryClassifier instance.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.get_classifier_service` instead.

    Args:
        use_cache: Whether to use caching for consistency

    Returns:
        The singleton LLMMemoryClassifier instance
    """
    warnings.warn(
        "get_llm_classifier is deprecated. Use classifier_service.get_classifier_service instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _llm_classifier
    if _llm_classifier is None:
        # Suppress the deprecation warning from __init__ since we're already warned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _llm_classifier = LLMMemoryClassifier(use_cache=use_cache)
    return _llm_classifier


def classify_memory_with_llm(text: str, default: Optional[MemoryType] = None) -> MemoryType:
    """
    Convenience function to classify using LLM-based classifier.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.classify_memory` instead.

    Args:
        text: The memory text to classify
        default: Default type if classification fails

    Returns:
        The detected MemoryType enum value
    """
    warnings.warn(
        "classify_memory_with_llm is deprecated. Use classifier_service.classify_memory instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import classify_memory
    return classify_memory(text, default)


def classify_memory_with_llm_and_confidence(text: str) -> Tuple[MemoryType, float]:
    """
    Convenience function to classify with confidence using LLM.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.classify_memory_with_confidence` instead.

    Args:
        text: The memory text to classify

    Returns:
        Tuple of (MemoryType, confidence_score)
    """
    warnings.warn(
        "classify_memory_with_llm_and_confidence is deprecated. "
        "Use classifier_service.classify_memory_with_confidence instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import classify_memory_with_confidence
    return classify_memory_with_confidence(text)


def clear_classification_cache() -> None:
    """Clear the classification cache.

    .. deprecated::
        Use :func:`hippocampai.utils.classifier_service.clear_classification_cache` instead.
    """
    warnings.warn(
        "clear_classification_cache from llm_classifier is deprecated. "
        "Use classifier_service.clear_classification_cache instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hippocampai.utils.classifier_service import clear_classification_cache as clear_cache
    clear_cache()
