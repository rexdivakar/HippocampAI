"""Memory extraction service using Claude API to extract memories from conversations."""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import anthropic

from src.memory_store import MemoryType, Category


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extractor for identifying and extracting memories from conversations using Claude API."""

    # Prompt template for memory extraction
    EXTRACTION_PROMPT = """Analyze this conversation and extract important memories that should be remembered long-term.

For each memory you extract, provide:
- text: The actual memory statement (be concise but complete)
- memory_type: Must be one of: preference, fact, goal, habit, event, context
- importance: A score from 1-10 (10 being most important)
- category: Must be one of: work, personal, learning, health, social, finance, other
- confidence: A score from 0.0-1.0 indicating how certain you are about this memory

Guidelines:
- Only extract genuinely important information worth remembering long-term
- Focus on facts, preferences, goals, habits, and significant events
- Avoid extracting trivial or temporary information
- Be specific and concrete in the memory text
- Ensure each memory is self-contained and understandable without additional context

Return ONLY a valid JSON array of memory objects. Do not include any other text or explanation.

Example format:
[
  {{
    "text": "User prefers meetings scheduled in the afternoon",
    "memory_type": "preference",
    "importance": 7,
    "category": "work",
    "confidence": 0.9
  }}
]

Conversation to analyze:
{conversation}

Remember: Return ONLY the JSON array, nothing else."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the memory extractor.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Claude model to use for extraction

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests for basic rate limiting

        logger.info(f"MemoryExtractor initialized with model: {model}")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _validate_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Validate a single extracted memory.

        Args:
            memory: Memory dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["text", "memory_type", "importance", "category", "confidence"]

        # Check required fields
        for field in required_fields:
            if field not in memory:
                logger.warning(f"Memory missing required field: {field}")
                return False

        # Validate text
        if not isinstance(memory["text"], str) or not memory["text"].strip():
            logger.warning("Memory text is invalid or empty")
            return False

        # Validate memory_type
        try:
            MemoryType(memory["memory_type"])
        except ValueError:
            logger.warning(f"Invalid memory_type: {memory['memory_type']}")
            return False

        # Validate importance (1-10)
        importance = memory["importance"]
        if not isinstance(importance, (int, float)) or not 1 <= importance <= 10:
            logger.warning(f"Invalid importance: {importance}")
            return False

        # Validate category
        try:
            Category(memory["category"])
        except ValueError:
            logger.warning(f"Invalid category: {memory['category']}")
            return False

        # Validate confidence (0.0-1.0)
        confidence = memory["confidence"]
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            logger.warning(f"Invalid confidence: {confidence}")
            return False

        return True

    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse Claude's JSON response.

        Args:
            response_text: Raw response text from Claude

        Returns:
            List of parsed memory dictionaries

        Raises:
            ValueError: If response cannot be parsed as JSON
        """
        try:
            # Try to find JSON array in the response
            response_text = response_text.strip()

            # Handle case where response might have extra text
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON array found in response, returning empty list")
                return []

            json_text = response_text[start_idx:end_idx]

            # Clean up any markdown code blocks
            json_text = json_text.replace('```json', '').replace('```', '')

            memories = json.loads(json_text)

            if not isinstance(memories, list):
                logger.warning("Response is not a JSON array, returning empty list")
                return []

            logger.info(f"Parsed {len(memories)} memories from response")
            return memories

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}, returning empty list")
            logger.debug(f"Response text: {response_text[:500]}...")
            # Return empty list instead of raising error for weather queries
            return []

    def extract_memories(
        self,
        conversation_text: str,
        user_id: str,
        session_id: Optional[str] = None,
        max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Extract memories from a conversation using Claude API.

        Args:
            conversation_text: The conversation text to analyze
            user_id: User identifier for the memories
            session_id: Optional session identifier
            max_retries: Maximum number of retry attempts on API errors

        Returns:
            List of extracted memory objects ready for storage

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If API call fails after retries
        """
        if not conversation_text or not isinstance(conversation_text, str):
            raise ValueError("conversation_text must be a non-empty string")

        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not session_id:
            session_id = f"session_{int(time.time())}"

        logger.info(f"Extracting memories from conversation (user: {user_id})")

        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Create the prompt
                prompt = self.EXTRACTION_PROMPT.format(conversation=conversation_text)

                # Call Claude API
                logger.debug(f"Calling Claude API (attempt {attempt + 1}/{max_retries + 1})")
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.0,  # Use deterministic extraction
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                # Extract response text
                response_text = message.content[0].text
                logger.debug(f"Received response: {len(response_text)} characters")
                logger.debug(f"Response preview: {response_text[:200]}")

                # Parse the response
                memories = self._parse_response(response_text)

                # Validate and enrich each memory
                valid_memories = []
                for i, memory in enumerate(memories):
                    if self._validate_memory(memory):
                        # Add user_id and session_id to metadata
                        enriched_memory = {
                            "text": memory["text"].strip(),
                            "memory_type": memory["memory_type"],
                            "metadata": {
                                "user_id": user_id,
                                "importance": memory["importance"],
                                "category": memory["category"],
                                "session_id": session_id,
                                "confidence": memory["confidence"]
                            }
                        }
                        valid_memories.append(enriched_memory)
                    else:
                        logger.warning(f"Skipping invalid memory at index {i}")

                logger.info(
                    f"Extracted {len(valid_memories)} valid memories "
                    f"(out of {len(memories)} total)"
                )

                return valid_memories

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("Rate limit exceeded, max retries reached") from e

            except anthropic.APIError as e:
                logger.error(f"Claude API error: {e}")
                if attempt < max_retries:
                    wait_time = 1
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Claude API error after {max_retries} retries") from e

            except ValueError as e:
                # Validation or parsing errors - return empty list for weather queries
                logger.warning(f"Extraction parsing error: {e}, returning empty list")
                return []

            except Exception as e:
                logger.error(f"Unexpected error during extraction: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if attempt < max_retries:
                    wait_time = 1
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Return empty list instead of failing
                    logger.warning(f"Memory extraction failed after {max_retries} retries, returning empty list")
                    return []

        # Should not reach here, but return empty list just in case
        logger.warning("Memory extraction loop completed unexpectedly, returning empty list")
        return []

    def extract_and_store(
        self,
        conversation_text: str,
        user_id: str,
        memory_store,
        session_id: Optional[str] = None
    ) -> List[str]:
        """
        Extract memories from conversation and store them directly.

        Args:
            conversation_text: The conversation text to analyze
            user_id: User identifier
            memory_store: MemoryStore instance for storage
            session_id: Optional session identifier

        Returns:
            List of stored memory IDs

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If extraction or storage fails
        """
        try:
            # Extract memories
            memories = self.extract_memories(conversation_text, user_id, session_id)

            if not memories:
                logger.info("No memories extracted from conversation")
                return []

            # Store memories
            memory_ids = memory_store.store_batch_memories(memories)

            logger.info(f"Extracted and stored {len(memory_ids)} memories")
            return memory_ids

        except Exception as e:
            logger.error(f"Failed to extract and store memories: {e}")
            raise
