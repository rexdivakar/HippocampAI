"""Memory deduplication service to prevent storing duplicate memories."""

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import anthropic

from hippocampai.embedding_service import EmbeddingService
from hippocampai.memory_retriever import MemoryRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeduplicationDecision(str, Enum):
    """Possible decisions for handling duplicate memories."""

    REPLACE = "replace"  # New memory replaces old one
    MERGE = "merge"  # Combine both memories
    SKIP = "skip"  # Don't store new memory (true duplicate)
    STORE_NEW = "store_new"  # Store as separate memory


class MemoryDeduplicator:
    """Service for detecting and handling duplicate memories."""

    # Prompt template for deduplication decision
    DEDUPLICATION_PROMPT = """You are helping manage a personal memory system. Compare these two memories and decide how to handle the potential duplicate.

Existing Memory:
Text: {existing_text}
Type: {existing_type}
Importance: {existing_importance}
Timestamp: {existing_timestamp}

New Memory:
Text: {new_text}
Type: {new_type}
Importance: {new_importance}

Your task is to decide what action to take. Choose one of:
1. "replace" - The new memory updates or supersedes the old one (e.g., preference changed, fact updated)
2. "merge" - Both contain useful complementary information that should be combined
3. "skip" - They're essentially the same; don't store the new one
4. "store_new" - They're different enough to warrant separate storage

Guidelines:
- If the new memory contradicts the old one, choose "replace"
- If they say the same thing in different words, choose "skip"
- If they both add unique valuable information, choose "merge"
- If they're related but distinctly different facts, choose "store_new"

Return ONLY a valid JSON object with this format:
{{
  "decision": "replace|merge|skip|store_new",
  "reasoning": "brief explanation of why",
  "merged_text": "combined text (only if decision is merge, otherwise null)"
}}

Do not include any other text or explanation."""

    def __init__(
        self,
        retriever: MemoryRetriever,
        embedding_service: EmbeddingService,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize the memory deduplicator.

        Args:
            retriever: MemoryRetriever instance for searching
            embedding_service: EmbeddingService for generating embeddings
            api_key: Anthropic API key (if None, uses environment variable)
            model: Claude model to use for decisions

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.retriever = retriever
        self.embeddings = embedding_service

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

        logger.info("MemoryDeduplicator initialized")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def check_for_duplicates(
        self, new_memory_text: str, user_id: str, similarity_threshold: float = 0.88, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Check for duplicate memories using vector similarity.

        Args:
            new_memory_text: Text of the new memory to check
            user_id: User identifier to filter by
            similarity_threshold: Minimum similarity score to consider (0.0-1.0)
            limit: Maximum number of duplicates to return

        Returns:
            List of potential duplicates with similarity scores

        Raises:
            ValueError: If inputs are invalid
        """
        if not new_memory_text or not isinstance(new_memory_text, str):
            raise ValueError("new_memory_text must be a non-empty string")

        if not user_id:
            raise ValueError("user_id must be provided")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        try:
            # Search for similar memories
            logger.debug(f"Searching for duplicates: {new_memory_text[:50]}...")
            results = self.retriever.search_memories(
                query=new_memory_text, limit=limit, filters={"user_id": user_id}
            )

            # Filter by similarity threshold
            duplicates = [
                result
                for result in results
                if result.get("similarity_score", 0) >= similarity_threshold
            ]

            logger.info(
                f"Found {len(duplicates)} potential duplicates (threshold: {similarity_threshold})"
            )

            return duplicates

        except Exception as e:
            logger.error(f"Failed to check for duplicates: {e}")
            raise

    def should_update_or_skip(
        self, new_memory: Dict[str, Any], existing_memory: Dict[str, Any], max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Use Claude API to decide how to handle a duplicate memory.

        Args:
            new_memory: Dictionary with 'text', 'memory_type', 'importance'
            existing_memory: Existing memory dictionary from retriever
            max_retries: Maximum retry attempts on API errors

        Returns:
            Dictionary with 'decision', 'reasoning', 'merged_text'

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If API call fails
        """
        # Validate inputs
        required_new = ["text", "memory_type", "importance"]
        for field in required_new:
            if field not in new_memory:
                raise ValueError(f"new_memory missing required field: {field}")

        if "text" not in existing_memory or "metadata" not in existing_memory:
            raise ValueError("existing_memory missing required fields")

        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Build the prompt
                prompt = self.DEDUPLICATION_PROMPT.format(
                    existing_text=existing_memory["text"],
                    existing_type=existing_memory["metadata"].get("memory_type", "unknown"),
                    existing_importance=existing_memory["metadata"].get("importance", 5),
                    existing_timestamp=existing_memory["metadata"].get("timestamp", "unknown"),
                    new_text=new_memory["text"],
                    new_type=new_memory["memory_type"],
                    new_importance=new_memory["importance"],
                )

                # Call Claude API
                logger.debug(f"Requesting deduplication decision (attempt {attempt + 1})")
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response
                response_text = message.content[0].text.strip()

                # Extract JSON
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1

                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")

                json_text = response_text[start_idx:end_idx]
                decision_data = json.loads(json_text)

                # Validate decision
                if "decision" not in decision_data:
                    raise ValueError("Response missing 'decision' field")

                try:
                    DeduplicationDecision(decision_data["decision"])
                except ValueError:
                    raise ValueError(f"Invalid decision: {decision_data['decision']}")

                logger.info(
                    f"Deduplication decision: {decision_data['decision']} - "
                    f"{decision_data.get('reasoning', 'no reason')}"
                )

                return decision_data

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                if attempt < max_retries:
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("Rate limit exceeded") from e

            except anthropic.APIError as e:
                logger.error(f"Claude API error: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError("Claude API error") from e

            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse decision response: {e}")
                if attempt < max_retries:
                    logger.info("Retrying...")
                    time.sleep(1)
                else:
                    raise RuntimeError("Failed to get valid deduplication decision") from e

        raise RuntimeError("Deduplication decision failed")

    def process_new_memory(
        self,
        new_memory: Dict[str, Any],
        user_id: str,
        similarity_threshold: float = 0.88,
        auto_decide: bool = True,
    ) -> Dict[str, Any]:
        """
        Check for duplicates and determine action for a new memory.

        Args:
            new_memory: New memory to process
            user_id: User identifier
            similarity_threshold: Similarity threshold for duplicate detection
            auto_decide: Whether to automatically use Claude to make decision

        Returns:
            Dictionary with 'action', 'duplicates', 'decision_data' (if auto_decide)

        Raises:
            ValueError: If inputs are invalid
        """
        if "text" not in new_memory:
            raise ValueError("new_memory must have 'text' field")

        # Check for duplicates
        duplicates = self.check_for_duplicates(
            new_memory_text=new_memory["text"],
            user_id=user_id,
            similarity_threshold=similarity_threshold,
        )

        result = {
            "action": "store",  # Default action
            "duplicates": duplicates,
            "decision_data": None,
        }

        # If duplicates found and auto_decide is True
        if duplicates and auto_decide:
            # Get decision for the highest-similarity duplicate
            highest_duplicate = duplicates[0]

            decision_data = self.should_update_or_skip(
                new_memory=new_memory, existing_memory=highest_duplicate
            )

            result["decision_data"] = decision_data
            result["action"] = decision_data["decision"]

        elif duplicates:
            # Duplicates found but not auto-deciding
            result["action"] = "manual_review"

        logger.info(
            f"Process new memory: action={result['action']}, duplicates_found={len(duplicates)}"
        )

        return result
