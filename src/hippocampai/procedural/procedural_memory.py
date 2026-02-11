"""Procedural memory: behavioral rules extracted from interactions.

Rules evolve based on feedback and improve the system prompt over time.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProceduralRule(BaseModel):
    """A behavioral rule extracted from user interactions."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    rule_text: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    source_interactions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProceduralMemoryManager:
    """Manages procedural rules extracted from user interactions.

    - Extract rules from recent interactions + feedback stats
    - Retrieve active rules for prompt injection
    - Update rule effectiveness via EMA
    - Consolidate redundant rules
    """

    def __init__(
        self,
        max_rules: int = 50,
        llm: Optional[Any] = None,
    ) -> None:
        self.max_rules = max_rules
        self.llm = llm
        self._rules: dict[str, list[ProceduralRule]] = {}  # user_id -> rules

    def extract_rules(
        self,
        user_id: str,
        recent_interactions: list[str],
    ) -> list[ProceduralRule]:
        """Extract behavioral rules from recent interactions.

        Uses LLM if available, otherwise pattern-based heuristics.

        Args:
            user_id: User ID.
            recent_interactions: List of recent interaction texts.

        Returns:
            List of newly extracted ProceduralRule objects.
        """
        if not recent_interactions:
            return []

        rules: list[ProceduralRule] = []

        if self.llm:
            rules = self._extract_rules_llm(user_id, recent_interactions)
        else:
            rules = self._extract_rules_heuristic(user_id, recent_interactions)

        # Store new rules
        if user_id not in self._rules:
            self._rules[user_id] = []

        for rule in rules:
            # Avoid duplicates by checking text similarity
            if not any(
                r.rule_text.lower() == rule.rule_text.lower()
                for r in self._rules[user_id]
            ):
                self._rules[user_id].append(rule)

        # Enforce max rules
        self._rules[user_id] = self._rules[user_id][-self.max_rules :]

        logger.info(
            f"Extracted {len(rules)} procedural rules for user {user_id}, "
            f"total={len(self._rules[user_id])}"
        )
        return rules

    def _extract_rules_llm(
        self, user_id: str, interactions: list[str]
    ) -> list[ProceduralRule]:
        """Extract rules using LLM."""
        combined = "\n---\n".join(interactions[-10:])
        prompt = (
            "Analyze these user interactions and extract behavioral rules "
            "(user preferences, communication style, common requests). "
            "Return each rule on its own line, prefixed with 'RULE: '.\n\n"
            f"Interactions:\n{combined}\n\nRules:"
        )

        try:
            if self.llm is None:
                return []
            response = self.llm.generate(prompt, max_tokens=500)
            rules: list[ProceduralRule] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("RULE:"):
                    rule_text = line[5:].strip()
                    if rule_text:
                        rules.append(
                            ProceduralRule(
                                user_id=user_id,
                                rule_text=rule_text,
                                confidence=0.7,
                                source_interactions=interactions[-3:],
                                metadata={"extraction_method": "llm"},
                            )
                        )
            return rules
        except Exception as e:
            logger.warning(f"LLM rule extraction failed: {e}")
            return []

    @staticmethod
    def _extract_rules_heuristic(
        user_id: str, interactions: list[str]
    ) -> list[ProceduralRule]:
        """Extract simple rules using heuristics."""
        rules: list[ProceduralRule] = []

        # Check for repeated patterns
        word_freq: dict[str, int] = {}
        for text in interactions:
            for word in text.lower().split():
                if len(word) > 4:
                    word_freq[word] = word_freq.get(word, 0) + 1

        frequent_topics = [
            w for w, c in word_freq.items() if c >= 3
        ]

        if frequent_topics:
            topics_str = ", ".join(frequent_topics[:5])
            rules.append(
                ProceduralRule(
                    user_id=user_id,
                    rule_text=f"User frequently discusses: {topics_str}",
                    confidence=0.5,
                    source_interactions=interactions[-3:],
                    metadata={"extraction_method": "heuristic"},
                )
            )

        return rules

    def get_active_rules(
        self, user_id: str, context: Optional[str] = None
    ) -> list[ProceduralRule]:
        """Get active rules sorted by confidence * success_rate.

        Args:
            user_id: User ID.
            context: Optional context string to filter relevant rules.

        Returns:
            Active rules sorted by effectiveness.
        """
        rules = self._rules.get(user_id, [])
        active = [r for r in rules if r.active]
        active.sort(key=lambda r: r.confidence * r.success_rate, reverse=True)
        return active

    def inject_rules_into_prompt(
        self, user_id: str, base_prompt: str, max_rules: int = 5
    ) -> str:
        """Prepend top rules to a base prompt.

        Args:
            user_id: User ID.
            base_prompt: The original prompt.
            max_rules: Maximum number of rules to inject.

        Returns:
            Prompt with rules prepended.
        """
        active = self.get_active_rules(user_id)[:max_rules]
        if not active:
            return base_prompt

        rules_text = "\n".join(
            f"- {rule.rule_text}" for rule in active
        )
        return (
            f"## User behavioral rules (learned from past interactions):\n"
            f"{rules_text}\n\n{base_prompt}"
        )

    def update_rule_effectiveness(
        self, rule_id: str, was_successful: bool
    ) -> Optional[ProceduralRule]:
        """Update a rule's success_rate via EMA.

        success_rate = 0.9 * old + 0.1 * new

        Args:
            rule_id: Rule ID.
            was_successful: Whether the rule-guided action was successful.

        Returns:
            Updated rule, or None if not found.
        """
        for rules in self._rules.values():
            for rule in rules:
                if rule.id == rule_id:
                    new_val = 1.0 if was_successful else 0.0
                    rule.success_rate = 0.9 * rule.success_rate + 0.1 * new_val
                    rule.updated_at = datetime.now(timezone.utc)
                    logger.debug(
                        f"Updated rule {rule_id} success_rate={rule.success_rate:.3f}"
                    )
                    return rule
        return None

    def consolidate_rules(self, user_id: str) -> list[ProceduralRule]:
        """Consolidate (merge redundant/contradicting) rules for a user.

        Uses LLM if available, otherwise simple deduplication.

        Args:
            user_id: User ID.

        Returns:
            The consolidated list of rules.
        """
        rules = self._rules.get(user_id, [])
        if len(rules) <= 1:
            return rules

        if self.llm:
            consolidated = self._consolidate_rules_llm(user_id, rules)
        else:
            consolidated = self._consolidate_rules_simple(rules)

        self._rules[user_id] = consolidated
        logger.info(
            f"Consolidated rules for user {user_id}: "
            f"{len(rules)} -> {len(consolidated)}"
        )
        return consolidated

    def _consolidate_rules_llm(
        self, user_id: str, rules: list[ProceduralRule]
    ) -> list[ProceduralRule]:
        """Consolidate rules using LLM."""
        rules_text = "\n".join(
            f"{i+1}. {r.rule_text} (confidence={r.confidence:.2f}, success={r.success_rate:.2f})"
            for i, r in enumerate(rules)
        )
        prompt = (
            "Consolidate these behavioral rules by merging redundant ones "
            "and removing contradictions. Keep the most useful rules. "
            "Return consolidated rules, one per line, prefixed with 'RULE: '.\n\n"
            f"Current rules:\n{rules_text}\n\nConsolidated:"
        )

        try:
            if self.llm is None:
                return rules
            response = self.llm.generate(prompt, max_tokens=500)
            consolidated: list[ProceduralRule] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("RULE:"):
                    rule_text = line[5:].strip()
                    if rule_text:
                        consolidated.append(
                            ProceduralRule(
                                user_id=user_id,
                                rule_text=rule_text,
                                confidence=0.8,
                                metadata={"consolidation": "llm"},
                            )
                        )
            return consolidated if consolidated else rules
        except Exception as e:
            logger.warning(f"LLM rule consolidation failed: {e}")
            return rules

    @staticmethod
    def _consolidate_rules_simple(
        rules: list[ProceduralRule],
    ) -> list[ProceduralRule]:
        """Simple deduplication by exact text match."""
        seen: set[str] = set()
        unique: list[ProceduralRule] = []
        for rule in rules:
            key = rule.rule_text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(rule)
        return unique

    def get_all_rules(self, user_id: str) -> list[ProceduralRule]:
        """Get all rules (active and inactive) for a user."""
        return self._rules.get(user_id, [])
