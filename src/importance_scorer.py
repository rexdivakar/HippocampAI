"""Importance scoring service for calculating and updating memory importance."""

import json
import logging
import math
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import anthropic


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Service for calculating and managing memory importance scores."""

    # Prompt template for importance scoring
    IMPORTANCE_SCORING_PROMPT = """You are helping rate the importance of memories for a personal AI assistant.

Rate the importance of remembering this information long-term:

Memory Text: {memory_text}
Memory Type: {memory_type}
User Context: {user_context}

Use this scale:
10 = Critical (core identity, fundamental preferences, life goals, health conditions)
9 = Very high (key relationships, major projects, critical deadlines)
8 = High (important preferences, ongoing commitments, significant facts)
7 = Moderately high (regular habits, work information, useful context)
6 = Moderate (occasional preferences, minor goals, helpful details)
5 = Medium (useful background, contextual information)
4 = Moderately low (nice-to-know facts, minor preferences)
3 = Low (trivial details, one-time mentions)
2 = Very low (fleeting comments, minimal value)
1 = Minimal (barely worth remembering)

Guidelines:
- Information about health, identity, and core values should score 9-10
- Ongoing goals and important projects score 7-9
- Regular habits and preferences score 6-8
- Contextual information scores 4-6
- Trivial or temporary information scores 1-3

Return ONLY a valid JSON object:
{{
  "score": X,
  "reasoning": "brief explanation of the score"
}}

Do not include any other text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the importance scorer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._last_request_time = 0
        self._min_request_interval = 0.1

        logger.info("ImportanceScorer initialized")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def calculate_importance(
        self,
        memory_text: str,
        memory_type: str,
        user_context: str = "",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Calculate importance score for a memory using Claude API.

        Args:
            memory_text: The memory text
            memory_type: Type of memory
            user_context: Optional context about the user
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with 'score' and 'reasoning'

        Raises:
            ValueError: If inputs invalid
            RuntimeError: If API call fails
        """
        if not memory_text:
            raise ValueError("memory_text is required")

        if not user_context:
            user_context = "General user"

        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()

                # Build prompt
                prompt = self.IMPORTANCE_SCORING_PROMPT.format(
                    memory_text=memory_text,
                    memory_type=memory_type,
                    user_context=user_context
                )

                # Call Claude
                logger.debug(f"Calculating importance for: {memory_text[:50]}...")
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Parse response
                response_text = message.content[0].text.strip()
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")

                json_text = response_text[start_idx:end_idx]
                result = json.loads(json_text)

                # Validate score
                if "score" not in result:
                    raise ValueError("Missing score field")

                score = result["score"]
                if not isinstance(score, (int, float)) or not 1 <= score <= 10:
                    raise ValueError(f"Invalid score: {score}")

                logger.info(
                    f"Importance score: {score}/10 - {result.get('reasoning', '')[:50]}..."
                )

                return result

            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError("Rate limit exceeded") from e

            except anthropic.APIError as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError("Claude API error") from e

            except (ValueError, json.JSONDecodeError) as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to get valid importance score") from e

        raise RuntimeError("Importance calculation failed")

    def decay_importance(
        self,
        memory_age_days: float,
        current_importance: float,
        access_count: int = 0,
        memory_type: str = "fact",
        last_update_days: float = None
    ) -> float:
        """
        Calculate decayed importance based on age and usage.

        Args:
            memory_age_days: Age of memory in days
            current_importance: Current importance score
            access_count: Number of times accessed
            memory_type: Type of memory (affects decay rate)
            last_update_days: Days since last update (None = never updated)

        Returns:
            Adjusted importance score (1-10)

        Algorithm:
        - Permanent facts (identity, preferences) decay slowly
        - Events and context decay faster
        - Frequently accessed memories resist decay
        - Recently updated memories get importance boost
        """
        if current_importance < 1 or current_importance > 10:
            raise ValueError("current_importance must be between 1 and 10")

        # Base decay rates by memory type
        decay_rates = {
            "preference": 0.001,   # Very slow decay
            "fact": 0.002,         # Slow decay
            "goal": 0.003,         # Moderate decay
            "habit": 0.004,        # Moderate-fast decay
            "context": 0.008,      # Fast decay
            "event": 0.01          # Very fast decay
        }

        decay_rate = decay_rates.get(memory_type, 0.005)

        # Calculate base decay using exponential decay
        # Formula: importance * e^(-decay_rate * age)
        decayed_importance = current_importance * math.exp(-decay_rate * memory_age_days)

        # Access count boost (logarithmic)
        # Frequently accessed memories are more important
        if access_count > 0:
            access_boost = min(2.0, math.log(access_count + 1) * 0.3)
            decayed_importance += access_boost

        # Recent update boost
        if last_update_days is not None and last_update_days < 7:
            # Boost if updated within last week
            update_boost = (7 - last_update_days) / 7 * 1.5
            decayed_importance += update_boost

        # Ensure score stays in valid range
        adjusted_score = max(1.0, min(10.0, decayed_importance))

        logger.debug(
            f"Decay calculation: {current_importance:.1f} -> {adjusted_score:.1f} "
            f"(age: {memory_age_days:.1f}d, access: {access_count}, type: {memory_type})"
        )

        return adjusted_score

    def should_decay_memory(self, memory_type: str, importance: float) -> bool:
        """
        Determine if a memory type should have importance decay applied.

        Args:
            memory_type: Type of memory
            importance: Current importance score

        Returns:
            True if decay should be applied
        """
        # Very high importance memories (9-10) rarely decay
        if importance >= 9:
            return memory_type in ["event", "context"]  # Only temporary types

        # High importance (7-8) decay slowly
        if importance >= 7:
            return memory_type in ["event", "context", "habit"]

        # All other memories decay
        return True

    def calculate_age_in_days(self, timestamp: str) -> float:
        """
        Calculate age of memory in days from ISO timestamp.

        Args:
            timestamp: ISO format timestamp string

        Returns:
            Age in days (float)
        """
        try:
            memory_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.utcnow() - memory_time
            return age.total_seconds() / 86400  # Convert to days

        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return 0.0

    def update_importance_for_memory(
        self,
        memory: Dict[str, Any],
        access_count: int = 0
    ) -> float:
        """
        Update importance score for a memory based on decay.

        Args:
            memory: Memory dictionary with metadata
            access_count: Number of times accessed

        Returns:
            New importance score
        """
        metadata = memory.get("metadata", {})
        current_importance = metadata.get("importance", 5)
        memory_type = metadata.get("memory_type", "fact")
        timestamp = metadata.get("timestamp", datetime.utcnow().isoformat())

        # Calculate age
        age_days = self.calculate_age_in_days(timestamp)

        # Check for last update
        last_update_days = None
        if "update_timestamp" in metadata:
            last_update_days = self.calculate_age_in_days(metadata["update_timestamp"])

        # Check if should decay
        if not self.should_decay_memory(memory_type, current_importance):
            logger.debug(f"Skipping decay for {memory_type} with importance {current_importance}")
            return current_importance

        # Calculate decayed importance
        new_importance = self.decay_importance(
            memory_age_days=age_days,
            current_importance=current_importance,
            access_count=access_count,
            memory_type=memory_type,
            last_update_days=last_update_days
        )

        return new_importance

    def batch_update_importance(
        self,
        memories: list,
        access_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Update importance for a batch of memories.

        Args:
            memories: List of memory dictionaries
            access_counts: Optional dict mapping memory_id to access count

        Returns:
            Dictionary mapping memory_id to new importance score
        """
        if access_counts is None:
            access_counts = {}

        results = {}

        for memory in memories:
            memory_id = memory.get("memory_id")
            if not memory_id:
                continue

            access_count = access_counts.get(memory_id, 0)

            try:
                new_importance = self.update_importance_for_memory(
                    memory=memory,
                    access_count=access_count
                )
                results[memory_id] = new_importance

            except Exception as e:
                logger.error(f"Failed to update importance for {memory_id}: {e}")
                continue

        logger.info(f"Updated importance for {len(results)}/{len(memories)} memories")
        return results
