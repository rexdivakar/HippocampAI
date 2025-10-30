"""Memory extractor with heuristic + optional LLM modes."""

import json
import logging
import re
from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.importance import ImportanceScorer

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extract memories from conversation using LLM or heuristics."""

    HEURISTIC_PATTERNS = {
        MemoryType.PREFERENCE: [
            r"(?:i|user) (?:prefer|like|love|hate|dislike|enjoy|want)s?\s+(.+)",
            r"(?:my|user's) (?:favorite|preferred)\s+(.+)",
        ],
        MemoryType.GOAL: [
            r"(?:i|user) (?:want to|plan to|will|goal is to|aim to|trying to)\s+(.+)",
            r"(?:my|user's) goal is\s+(.+)",
        ],
        MemoryType.FACT: [
            r"(?:i|user) (?:live in|work at|am from|studied at|graduated from|am a)\s+(.+)",
            r"(?:my|user's) (?:name is|job is|work is)\s+(.+)",
        ],
        MemoryType.CONTEXT: [
            r"(?:looking for|searching for|need|want)\s+(.+)",
            r"(?:interested in|curious about)\s+(.+)",
        ],
    }

    LLM_PROMPT = """You are a memory extraction system. Extract important, actionable memories from the conversation below.

RULES:
1. Extract facts, preferences, goals, and context that would be useful to remember long-term
2. Rephrase user statements into complete, context-aware memories
3. Assign appropriate types: "preference", "fact", "goal", "habit", "event", "context"
4. Rate importance 0-10 (0=trivial, 10=critical)
5. Return ONLY valid JSON array, no markdown, no explanations

EXAMPLES:
User: "I love coffee" → {"text": "User loves coffee", "type": "preference", "importance": 6}
User: "hotels" (in context of Raleigh) → {"text": "User is looking for hotels in Raleigh", "type": "context", "importance": 7}
User: "I work at Google" → {"text": "User works at Google", "type": "fact", "importance": 8}

CONVERSATION:
{conversation}

Extract memories as JSON array:"""

    def __init__(self, llm: Optional[BaseLLM] = None, mode: str = "hybrid"):
        self.llm = llm
        self.mode = mode  # "llm", "heuristic", "hybrid"
        self.scorer = ImportanceScorer(llm=llm)

    def extract(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> list[Memory]:
        """Extract memories from conversation."""
        if self.mode == "llm" and self.llm:
            return self._extract_llm(conversation, user_id, session_id)
        if self.mode == "heuristic":
            return self._extract_heuristic(conversation, user_id, session_id)
        # hybrid
        heuristic_mems = self._extract_heuristic(conversation, user_id, session_id)
        if self.llm:
            llm_mems = self._extract_llm(conversation, user_id, session_id)
            # Merge, prefer LLM
            return llm_mems if llm_mems else heuristic_mems
        return heuristic_mems

    def _extract_heuristic(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> list[Memory]:
        """Simple regex-based extraction with improved patterns."""
        memories = []
        conv_lower = conversation.lower()

        for mem_type, patterns in self.HEURISTIC_PATTERNS.items():
            # Handle both list of patterns and single pattern
            if isinstance(patterns, str):
                patterns = [patterns]

            for pattern in patterns:
                matches = re.findall(pattern, conv_lower, re.IGNORECASE)
                for match in matches:
                    text = match.strip()
                    if len(text) < 3:  # Allow shorter matches
                        continue

                    # Clean up the text
                    text = text.rstrip(".,!?")

                    importance = self.scorer.score(text, mem_type.value)
                    memory = Memory(
                        text=text,
                        user_id=user_id,
                        session_id=session_id,
                        type=mem_type,
                        importance=importance,
                        confidence=0.7,
                    )
                    memory.calculate_size_metrics()
                    memories.append(memory)

        logger.info(f"Heuristic extracted {len(memories)} memories")
        return memories

    def _extract_llm(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> list[Memory]:
        """LLM-based extraction with robust JSON parsing."""
        if not self.llm:
            return []

        try:
            prompt = self.LLM_PROMPT.format(conversation=conversation[:2000])
            response = self.llm.generate(prompt, max_tokens=1024, temperature=0.0)

            # Robust JSON extraction
            json_str = self._extract_json_array(response)
            if not json_str:
                logger.warning("No valid JSON array found in LLM response")
                return self._extract_heuristic(conversation, user_id, session_id)

            data = json.loads(json_str)

            # Validate it's a list
            if not isinstance(data, list):
                logger.warning(f"Expected JSON array, got {type(data)}")
                return self._extract_heuristic(conversation, user_id, session_id)

            memories = []
            for item in data:
                if not isinstance(item, dict) or "text" not in item:
                    continue

                # Validate and normalize type
                mem_type_str = item.get("type", "fact").lower()
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    # Default to fact if invalid type
                    mem_type = MemoryType.FACT

                # Validate importance
                importance = float(item.get("importance", 5.0))
                importance = max(0.0, min(10.0, importance))

                # Validate confidence
                confidence = float(item.get("confidence", 0.9))
                confidence = max(0.0, min(1.0, confidence))

                memory = Memory(
                    text=item["text"].strip(),
                    user_id=user_id,
                    session_id=session_id,
                    type=mem_type,
                    importance=importance,
                    confidence=confidence,
                )
                memory.calculate_size_metrics()
                memories.append(memory)

            logger.info(f"LLM extracted {len(memories)} memories")
            return memories

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, falling back to heuristic")
            return self._extract_heuristic(conversation, user_id, session_id)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to heuristic")
            return self._extract_heuristic(conversation, user_id, session_id)

    def _extract_json_array(self, text: str) -> Optional[str]:
        """Extract JSON array from text, handling markdown and extra content.

        Args:
            text: Raw text that might contain JSON

        Returns:
            Clean JSON array string or None
        """
        # Remove markdown code blocks
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.strip()

        # Find JSON array boundaries
        start = text.find("[")
        if start == -1:
            return None

        # Find matching closing bracket
        bracket_count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    end = i + 1
                    break

        if end <= start:
            return None

        return text[start:end]
