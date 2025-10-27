"""Importance scorer with heuristics and optional LLM."""

import logging
import re
from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Score memory importance using heuristics + optional LLM."""

    KEYWORDS_HIGH = {"goal", "important", "critical", "must", "always", "never", "hate", "love"}
    KEYWORDS_MED = {"prefer", "like", "want", "need", "usually", "often"}
    KEYWORDS_LOW = {"maybe", "might", "sometimes", "occasionally"}

    LLM_PROMPT = """Rate the importance of remembering this information long-term on a scale of 0-10.

0 = Trivial/useless (e.g., "hi", "ok")
5 = Somewhat useful (e.g., "User likes coffee")
10 = Critical to remember (e.g., "User is allergic to peanuts", "User's goal is to become a doctor")

Memory: "{text}"
Type: {memory_type}

Return ONLY a number between 0 and 10:"""

    def __init__(self, llm: Optional[BaseLLM] = None, use_llm: bool = False):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def score(self, text: str, memory_type: str = "fact") -> float:
        """
        Score importance [0-10].

        First tries LLM if available, falls back to heuristics.
        """
        # Try LLM scoring if enabled
        if self.use_llm and self.llm:
            try:
                score = self._score_llm(text, memory_type)
                if score is not None:
                    return score
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}, using heuristics")

        # Fallback to heuristic scoring
        return self._score_heuristic(text, memory_type)

    def _score_llm(self, text: str, memory_type: str) -> Optional[float]:
        """Score using LLM."""
        if not self.llm:
            return None

        try:
            prompt = self.LLM_PROMPT.format(text=text[:500], memory_type=memory_type)
            response = self.llm.generate(prompt, max_tokens=10, temperature=0.0)

            # Extract number from response
            response = response.strip()
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))

            return None

        except Exception as e:
            logger.warning(f"LLM scoring error: {e}")
            return None

    def _score_heuristic(self, text: str, memory_type: str) -> float:
        """
        Heuristic scoring [0-10].

        - High keywords: +2
        - Medium keywords: +1
        - Caps/exclamation: +1
        - Length > 100: +0.5
        - Type boost: preference/goal +1
        - Short text penalty: -2 for very short
        """
        score = 5.0  # baseline

        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Penalty for very short/trivial text
        if len(text) < 5 or text_lower in {"hi", "hello", "ok", "yes", "no", "yeah", "sure"}:
            score -= 3

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

        # Type boost
        if memory_type in {"preference", "goal"}:
            score += 1
        elif memory_type in {"context"}:
            score += 0.5

        # Clamp
        score = max(1.0, min(10.0, score))

        return score
