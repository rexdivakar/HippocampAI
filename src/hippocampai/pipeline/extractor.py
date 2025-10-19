"""Memory extractor with heuristic + optional LLM modes."""

import json
import logging
import re
from typing import List, Optional

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.importance import ImportanceScorer

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extract memories from conversation using LLM or heuristics."""

    HEURISTIC_PATTERNS = {
        MemoryType.PREFERENCE: r"(?:i (?:prefer|like|love|hate|dislike|enjoy|want))\s+(.+)",
        MemoryType.GOAL: r"(?:i (?:want to|plan to|will|goal|aim to))\s+(.+)",
        MemoryType.FACT: r"(?:i (?:live in|work at|am from|studied|graduated))\s+(.+)",
    }

    LLM_PROMPT = """Extract important memories from this conversation. Return ONLY a JSON array.

Example: [{"text": "User prefers vegetarian food", "type": "preference", "importance": 7, "confidence": 0.9}]

Conversation:
{conversation}

JSON array:"""

    def __init__(self, llm: Optional[BaseLLM] = None, mode: str = "hybrid"):
        self.llm = llm
        self.mode = mode  # "llm", "heuristic", "hybrid"
        self.scorer = ImportanceScorer(llm=llm)

    def extract(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[Memory]:
        """Extract memories from conversation."""
        if self.mode == "llm" and self.llm:
            return self._extract_llm(conversation, user_id, session_id)
        elif self.mode == "heuristic":
            return self._extract_heuristic(conversation, user_id, session_id)
        else:  # hybrid
            heuristic_mems = self._extract_heuristic(conversation, user_id, session_id)
            if self.llm:
                llm_mems = self._extract_llm(conversation, user_id, session_id)
                # Merge, prefer LLM
                return llm_mems if llm_mems else heuristic_mems
            return heuristic_mems

    def _extract_heuristic(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[Memory]:
        """Simple regex-based extraction."""
        memories = []
        conv_lower = conversation.lower()

        for mem_type, pattern in self.HEURISTIC_PATTERNS.items():
            matches = re.findall(pattern, conv_lower, re.IGNORECASE)
            for match in matches:
                text = match.strip()
                if len(text) < 5:
                    continue

                importance = self.scorer.score(text, mem_type.value)
                memories.append(
                    Memory(
                        text=text,
                        user_id=user_id,
                        session_id=session_id,
                        type=mem_type,
                        importance=importance,
                        confidence=0.7,
                    )
                )

        return memories

    def _extract_llm(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[Memory]:
        """LLM-based extraction."""
        if not self.llm:
            return []

        try:
            prompt = self.LLM_PROMPT.format(conversation=conversation[:2000])
            response = self.llm.generate(prompt, max_tokens=1024, temperature=0.0)

            # Parse JSON
            response = response.strip()
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                return []

            json_str = response[start:end]
            data = json.loads(json_str)

            memories = []
            for item in data:
                mem_type = MemoryType(item.get("type", "fact"))
                memories.append(
                    Memory(
                        text=item["text"],
                        user_id=user_id,
                        session_id=session_id,
                        type=mem_type,
                        importance=item.get("importance", 5.0),
                        confidence=item.get("confidence", 0.9),
                    )
                )

            return memories

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to heuristic")
            return self._extract_heuristic(conversation, user_id, session_id)
