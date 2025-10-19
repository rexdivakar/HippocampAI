"""Importance scorer with heuristics and optional LLM."""

import re
from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM


class ImportanceScorer:
    """Score memory importance using heuristics + optional LLM."""

    KEYWORDS_HIGH = {"goal", "important", "critical", "must", "always", "never", "hate", "love"}
    KEYWORDS_MED = {"prefer", "like", "want", "need", "usually", "often"}
    KEYWORDS_LOW = {"maybe", "might", "sometimes", "occasionally"}

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    def score(self, text: str, memory_type: str = "fact") -> float:
        """
        Score importance [0-10].

        Uses heuristics:
        - High keywords: +2
        - Medium keywords: +1
        - Caps/exclamation: +1
        - Length > 100: +0.5
        - Type boost: preference/goal +1
        """
        score = 5.0  # baseline

        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Keyword boost
        if words & self.KEYWORDS_HIGH:
            score += 2
        elif words & self.KEYWORDS_MED:
            score += 1
        elif words & self.KEYWORDS_LOW:
            score -= 0.5

        # Emphasis
        if any(c.isupper() for c in text) or "!" in text:
            score += 1

        # Length
        if len(text) > 100:
            score += 0.5

        # Type
        if memory_type in {"preference", "goal"}:
            score += 1

        # Clamp
        score = max(1.0, min(10.0, score))

        return score
