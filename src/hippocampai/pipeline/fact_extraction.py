"""Fact extraction pipeline for structured knowledge extraction.

This module provides:
- Automatic fact extraction from conversations and text
- Entity and relationship extraction
- Temporal information extraction
- Fact categorization and confidence scoring
- Integration with LLM for advanced extraction
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FactCategory(str, Enum):
    """Categories for extracted facts."""

    EMPLOYMENT = "employment"
    OCCUPATION = "occupation"
    LOCATION = "location"
    EDUCATION = "education"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    SKILL = "skill"
    EXPERIENCE = "experience"
    CONTACT = "contact"
    EVENT = "event"
    GOAL = "goal"
    HABIT = "habit"
    OPINION = "opinion"
    ATTRIBUTE = "attribute"
    POSSESSION = "possession"
    OTHER = "other"


class TemporalType(str, Enum):
    """Types of temporal information."""

    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    START_DATE = "start_date"
    END_DATE = "end_date"
    DURATION = "duration"
    RECURRING = "recurring"


class ExtractedFact(BaseModel):
    """Represents an extracted fact."""

    fact: str = Field(..., description="The extracted fact statement")
    category: FactCategory = Field(..., description="Category of the fact")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: List[str] = Field(default_factory=list, description="Entities mentioned")
    temporal: Optional[str] = Field(None, description="Temporal information")
    temporal_type: TemporalType = Field(default=TemporalType.PRESENT)
    source: str = Field(..., description="Source of extraction (conversation, text, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactExtractionPipeline:
    """Pipeline for extracting structured facts from text and conversations."""

    def __init__(self, llm=None):
        """Initialize fact extraction pipeline.

        Args:
            llm: Language model for advanced extraction (optional)
        """
        self.llm = llm

        # Pattern-based extraction rules
        self.fact_patterns = self._build_fact_patterns()
        self.entity_patterns = self._build_entity_patterns()
        self.temporal_patterns = self._build_temporal_patterns()

    def _build_fact_patterns(self) -> Dict[FactCategory, List[Dict[str, Any]]]:
        """Build pattern matching rules for fact extraction."""
        return {
            FactCategory.EMPLOYMENT: [
                {
                    "pattern": r"\b(?:work|works|working|worked)\s+(?:at|for)\s+([A-Z][A-Za-z\s&]+)",
                    "extract": lambda m: f"works at {m.group(1).strip()}",
                    "entity_group": 1,
                },
                {
                    "pattern": r"\b(?:employed|employee)\s+(?:at|by)\s+([A-Z][A-Za-z\s&]+)",
                    "extract": lambda m: f"employed at {m.group(1).strip()}",
                    "entity_group": 1,
                },
            ],
            FactCategory.OCCUPATION: [
                {
                    "pattern": r"\b(?:I am|I\'m|is)\s+(?:a|an)\s+([a-z\s]+(?:engineer|developer|designer|manager|analyst|scientist|teacher|doctor|lawyer|consultant))",
                    "extract": lambda m: f"is a {m.group(1).strip()}",
                    "entity_group": 1,
                }
            ],
            FactCategory.LOCATION: [
                {
                    "pattern": r"\b(?:live|lives|living|lived)\s+in\s+([A-Z][A-Za-z\s]+)",
                    "extract": lambda m: f"lives in {m.group(1).strip()}",
                    "entity_group": 1,
                },
                {
                    "pattern": r"\b(?:from|based in)\s+([A-Z][A-Za-z\s]+)",
                    "extract": lambda m: f"from {m.group(1).strip()}",
                    "entity_group": 1,
                },
            ],
            FactCategory.EDUCATION: [
                {
                    "pattern": r"\b(?:studied|studying|study)\s+([a-z\s]+)\s+at\s+([A-Z][A-Za-z\s]+)",
                    "extract": lambda m: f"studied {m.group(1).strip()} at {m.group(2).strip()}",
                    "entity_group": 2,
                },
                {
                    "pattern": r"\b(?:graduated|degree)\s+(?:from|in)\s+([A-Z][A-Za-z\s]+)",
                    "extract": lambda m: f"graduated from {m.group(1).strip()}",
                    "entity_group": 1,
                },
            ],
            FactCategory.SKILL: [
                {
                    "pattern": r"\b(?:know|knows|skilled in|proficient in|expert in)\s+([A-Za-z\s,]+)",
                    "extract": lambda m: f"skilled in {m.group(1).strip()}",
                    "entity_group": 1,
                }
            ],
            FactCategory.PREFERENCE: [
                {
                    "pattern": r"\b(?:love|loves|like|likes|prefer|prefers|enjoy|enjoys)\s+([a-z\s]+)",
                    "extract": lambda m: f"likes {m.group(1).strip()}",
                    "entity_group": 1,
                }
            ],
            FactCategory.GOAL: [
                {
                    "pattern": r"\b(?:want to|plan to|planning to|goal is to|aiming to)\s+([a-z\s]+)",
                    "extract": lambda m: f"wants to {m.group(1).strip()}",
                    "entity_group": 1,
                }
            ],
        }

    def _build_entity_patterns(self) -> Dict[str, str]:
        """Build patterns for entity extraction."""
        return {
            "PERSON": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # First Last name
            "ORGANIZATION": r"\b(?:[A-Z][A-Za-z]*\s*)+(?:Inc|Corp|LLC|Ltd|Company|University|College)\b",
            "LOCATION": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Avenue|Street|Road))?\b",
            "DATE": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        }

    def _build_temporal_patterns(self) -> Dict[TemporalType, List[str]]:
        """Build patterns for temporal extraction."""
        return {
            TemporalType.PRESENT: [
                r"\b(?:currently|now|today|these days|at present)\b",
                r"\b(?:am|is|are)\b",
            ],
            TemporalType.PAST: [
                r"\b(?:was|were|had|used to|previously|formerly|in the past)\b",
                r"\b(?:yesterday|last week|last month|last year|ago)\b",
            ],
            TemporalType.FUTURE: [
                r"\b(?:will|going to|planning to|next week|next month|next year|soon|tomorrow)\b"
            ],
            TemporalType.START_DATE: [
                r"\b(?:started|began|joined|since)\s+(?:in\s+)?(\d{4}|[A-Z][a-z]+\s+\d{4})\b"
            ],
            TemporalType.END_DATE: [
                r"\b(?:ended|finished|left|until)\s+(?:in\s+)?(\d{4}|[A-Z][a-z]+\s+\d{4})\b"
            ],
        }

    def extract_facts(
        self, text: str, source: str = "text", user_id: Optional[str] = None
    ) -> List[ExtractedFact]:
        """Extract facts from text using pattern matching and LLM.

        Args:
            text: Input text to extract facts from
            source: Source identifier
            user_id: Optional user ID for context

        Returns:
            List of extracted facts
        """
        facts = []

        # Pattern-based extraction
        pattern_facts = self._extract_facts_pattern_based(text, source)
        facts.extend(pattern_facts)

        # LLM-based extraction (if available and not enough facts found)
        if self.llm and len(facts) < 3:
            llm_facts = self._extract_facts_llm(text, source, user_id)
            facts.extend(llm_facts)

        # Deduplicate facts
        facts = self._deduplicate_facts(facts)

        return facts

    def _extract_facts_pattern_based(self, text: str, source: str) -> List[ExtractedFact]:
        """Extract facts using pattern matching."""
        facts = []

        for category, patterns in self.fact_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    # Extract fact text
                    fact_text = pattern_info["extract"](match)

                    # Extract entities
                    entities = []
                    if "entity_group" in pattern_info:
                        entity = match.group(pattern_info["entity_group"]).strip()
                        entities.append(entity)

                    # Determine temporal information
                    temporal_info = self._extract_temporal_info(text, match.start(), match.end())

                    fact = ExtractedFact(
                        fact=fact_text,
                        category=category,
                        confidence=0.85,  # Pattern-based confidence
                        entities=entities,
                        temporal=temporal_info.get("temporal"),
                        temporal_type=temporal_info.get("temporal_type", TemporalType.PRESENT),
                        source=source,
                        metadata={
                            "extraction_method": "pattern",
                            "pattern": pattern,
                            "match_position": (match.start(), match.end()),
                        },
                    )
                    facts.append(fact)

        return facts

    def _extract_temporal_info(self, text: str, start: int, end: int) -> Dict[str, Any]:
        """Extract temporal information from context around a match."""
        # Look at context around the match (50 chars before and after)
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end]

        # Check for temporal patterns
        for temporal_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    # Try to extract specific date/time
                    date_match = re.search(r"\b(\d{4})\b", context)
                    temporal_value = date_match.group(1) if date_match else None

                    return {"temporal": temporal_value, "temporal_type": temporal_type}

        return {"temporal": None, "temporal_type": TemporalType.PRESENT}

    def _extract_facts_llm(
        self, text: str, source: str, user_id: Optional[str] = None
    ) -> List[ExtractedFact]:
        """Extract facts using LLM."""
        if not self.llm:
            return []

        prompt = f"""Extract structured facts from this text. For each fact, provide:
1. The fact statement (clear and concise)
2. Category (employment, occupation, location, education, skill, preference, goal, etc.)
3. Entities mentioned (people, organizations, locations)
4. Temporal information if any (present, past, future, specific dates)

Text: {text}

Return facts in this format:
FACT: [fact statement] | CATEGORY: [category] | ENTITIES: [entity1, entity2] | TEMPORAL: [temporal info]

Facts:"""

        try:
            response = self.llm.generate(prompt, max_tokens=500)

            # Parse LLM response
            facts = []
            for line in response.strip().split("\n"):
                if not line.strip() or not line.startswith("FACT:"):
                    continue

                # Parse fact components
                parts = line.split("|")
                if len(parts) < 2:
                    continue

                fact_text = parts[0].replace("FACT:", "").strip()
                category_str = (
                    parts[1].replace("CATEGORY:", "").strip().lower() if len(parts) > 1 else "other"
                )
                entities_str = parts[2].replace("ENTITIES:", "").strip() if len(parts) > 2 else ""
                temporal_str = parts[3].replace("TEMPORAL:", "").strip() if len(parts) > 3 else ""

                # Map category
                category = FactCategory.OTHER
                for cat in FactCategory:
                    if cat.value in category_str:
                        category = cat
                        break

                # Parse entities
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]

                # Determine temporal type
                temporal_type = TemporalType.PRESENT
                if any(word in temporal_str.lower() for word in ["past", "was", "were", "ago"]):
                    temporal_type = TemporalType.PAST
                elif any(word in temporal_str.lower() for word in ["future", "will", "going to"]):
                    temporal_type = TemporalType.FUTURE
                elif "start" in temporal_str.lower() or "since" in temporal_str.lower():
                    temporal_type = TemporalType.START_DATE

                fact = ExtractedFact(
                    fact=fact_text,
                    category=category,
                    confidence=0.75,  # LLM-based confidence
                    entities=entities,
                    temporal=temporal_str if temporal_str else None,
                    temporal_type=temporal_type,
                    source=source,
                    metadata={"extraction_method": "llm", "user_id": user_id},
                )
                facts.append(fact)

            return facts

        except Exception as e:
            logger.warning(f"LLM fact extraction failed: {e}")
            return []

    def _deduplicate_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Remove duplicate or very similar facts."""
        if not facts:
            return []

        unique_facts = []
        seen_facts = set()

        for fact in facts:
            # Create a simple hash of the fact
            fact_key = f"{fact.category}:{fact.fact.lower()}"

            if fact_key not in seen_facts:
                seen_facts.add(fact_key)
                unique_facts.append(fact)

        return unique_facts

    def extract_from_conversation(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[ExtractedFact]:
        """Extract facts from a conversation.

        Args:
            conversation: Full conversation text or messages
            user_id: User ID
            session_id: Optional session ID

        Returns:
            List of extracted facts
        """
        # Split conversation into turns if needed
        turns = self._split_conversation(conversation)

        all_facts = []
        for i, turn in enumerate(turns):
            facts = self.extract_facts(text=turn, source=f"conversation_turn_{i}", user_id=user_id)

            # Add conversation context to metadata
            for fact in facts:
                fact.metadata["session_id"] = session_id
                fact.metadata["turn"] = i

            all_facts.extend(facts)

        # Post-process: link related facts
        all_facts = self._link_related_facts(all_facts)

        return all_facts

    def _split_conversation(self, conversation: str) -> List[str]:
        """Split conversation into individual turns."""
        # Simple split by common patterns
        # Could be enhanced to parse structured formats (JSON, etc.)

        # Try to split by "User:" and "Assistant:" markers
        turns = []
        current_turn = []

        for line in conversation.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if new speaker
            if line.startswith(("User:", "Assistant:", "System:")):
                if current_turn:
                    turns.append(" ".join(current_turn))
                    current_turn = []
                # Remove speaker marker
                line = re.sub(r"^(?:User|Assistant|System):\s*", "", line)

            if line:
                current_turn.append(line)

        if current_turn:
            turns.append(" ".join(current_turn))

        # If no speaker markers found, return as single turn
        return turns if turns else [conversation]

    def _link_related_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Link related facts together."""
        # Group facts by entities
        entity_groups = {}

        for fact in facts:
            for entity in fact.entities:
                if entity not in entity_groups:
                    entity_groups[entity] = []
                entity_groups[entity].append(fact)

        # Add related fact IDs to metadata
        for entity, related_facts in entity_groups.items():
            if len(related_facts) > 1:
                fact_ids = [f.fact for f in related_facts]
                for fact in related_facts:
                    fact.metadata["related_facts"] = [f for f in fact_ids if f != fact.fact]

        return facts
