"""Memory update service for modifying and merging existing memories."""

import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import anthropic

from src.embedding_service import EmbeddingService
from src.memory_retriever import MemoryRetriever
from src.qdrant_client import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictResolution(str, Enum):
    """Possible conflict resolution strategies."""

    UPDATE = "update"  # New replaces old
    MERGE = "merge"  # Combine information
    SEPARATE = "separate"  # Both are valid, keep separate


class MemoryUpdater:
    """Service for updating and merging existing memories."""

    # Prompt template for conflict resolution
    CONFLICT_RESOLUTION_PROMPT = """You are helping manage a personal memory system. Two memories may conflict with each other.

Old Memory:
Text: {old_text}
Type: {old_type}
Importance: {old_importance}
Timestamp: {old_timestamp}

New Memory:
Text: {new_text}
Type: {new_type}
Importance: {new_importance}
Timestamp: {new_timestamp}

Your task: Determine if these memories conflict and how to handle it.

Choose one of:
1. "update" - New memory replaces old (e.g., preference changed, outdated information)
2. "merge" - Combine both into a single comprehensive memory (complementary info)
3. "separate" - No conflict, both are valid and should be kept separate

Guidelines:
- If they directly contradict, choose "update"
- If they provide complementary details about the same thing, choose "merge"
- If they're about different things or time periods, choose "separate"

Return ONLY a valid JSON object:
{{
  "decision": "update|merge|separate",
  "reasoning": "brief explanation",
  "merged_text": "combined text (only if merge, otherwise null)",
  "merged_importance": importance_score (only if merge, otherwise null)
}}

Do not include any other text."""

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        retriever: MemoryRetriever,
        embedding_service: EmbeddingService,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize the memory updater.

        Args:
            qdrant_manager: QdrantManager instance
            retriever: MemoryRetriever instance
            embedding_service: EmbeddingService instance
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.qdrant = qdrant_manager
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

        logger.info("MemoryUpdater initialized")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_collection_for_memory(self, memory_id: str) -> Optional[str]:
        """Find which collection contains the memory."""
        for collection in self.qdrant.COLLECTIONS:
            try:
                points = self.qdrant.client.retrieve(collection_name=collection, ids=[memory_id])
                if points:
                    return collection
            except Exception:
                continue
        return None

    def update_memory(
        self,
        memory_id: str,
        new_text: str,
        reason: str,
        new_importance: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Update an existing memory with version history.

        Args:
            memory_id: ID of the memory to update
            new_text: New text for the memory
            reason: Reason for the update
            new_importance: Optional new importance score
            collection_name: Optional collection name (auto-detected if None)

        Returns:
            True if successful

        Raises:
            ValueError: If memory not found or inputs invalid
            RuntimeError: If update fails
        """
        if not memory_id or not new_text or not reason:
            raise ValueError("memory_id, new_text, and reason are required")

        try:
            # Find the collection if not provided
            if not collection_name:
                collection_name = self._get_collection_for_memory(memory_id)
                if not collection_name:
                    raise ValueError(f"Memory {memory_id} not found")

            # Retrieve existing memory
            existing = self.retriever.get_memory_by_id(memory_id, collection_name)
            if not existing:
                raise ValueError(f"Memory {memory_id} not found")

            # Prepare updated metadata
            metadata = existing["metadata"].copy()

            # Store version history
            if "version_history" not in metadata:
                metadata["version_history"] = []

            # Add current version to history
            metadata["version_history"].append(
                {
                    "text": existing["text"],
                    "importance": metadata.get("importance"),
                    "timestamp": metadata.get("timestamp"),
                    "version": metadata.get("version", 0),
                }
            )

            # Update metadata
            metadata["text"] = new_text
            metadata["previous_text"] = existing["text"]
            metadata["update_reason"] = reason
            metadata["update_timestamp"] = datetime.utcnow().isoformat()
            metadata["version"] = metadata.get("version", 0) + 1

            if new_importance is not None:
                metadata["importance"] = new_importance

            # Generate new embedding
            new_embedding = self.embeddings.generate_embedding(new_text)

            # Update in Qdrant
            from qdrant_client.models import PointStruct

            point = PointStruct(id=memory_id, vector=new_embedding.tolist(), payload=metadata)

            self.qdrant.client.upsert(collection_name=collection_name, points=[point])

            logger.info(
                f"Updated memory {memory_id} (version {metadata['version']}, reason: {reason})"
            )

            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise RuntimeError("Memory update failed") from e

    def merge_memories(
        self,
        memory_ids: List[str],
        merged_text: str,
        merged_importance: int,
        reason: str = "Merged duplicate/related memories",
    ) -> str:
        """
        Merge multiple memories into one.

        Args:
            memory_ids: List of memory IDs to merge
            merged_text: Combined text for merged memory
            merged_importance: Importance score for merged memory
            reason: Reason for merging

        Returns:
            ID of the primary (first) memory that now contains merged data

        Raises:
            ValueError: If inputs invalid or memories not found
            RuntimeError: If merge fails
        """
        if not memory_ids or len(memory_ids) < 2:
            raise ValueError("At least 2 memory IDs required for merging")

        if not merged_text:
            raise ValueError("merged_text is required")

        try:
            # Retrieve all memories
            memories = []
            for mem_id in memory_ids:
                mem = self.retriever.get_memory_by_id(mem_id)
                if not mem:
                    raise ValueError(f"Memory {mem_id} not found")
                memories.append(mem)

            # Use first memory as the base
            primary_id = memory_ids[0]
            primary_collection = self._get_collection_for_memory(primary_id)

            # Prepare merged metadata
            merged_metadata = memories[0]["metadata"].copy()
            merged_metadata["text"] = merged_text
            merged_metadata["importance"] = merged_importance
            merged_metadata["merge_timestamp"] = datetime.utcnow().isoformat()
            merged_metadata["merge_reason"] = reason
            merged_metadata["merged_from"] = memory_ids[1:]  # Other IDs

            # Store original texts
            merged_metadata["original_texts"] = [
                {
                    "memory_id": mem["memory_id"],
                    "text": mem["text"],
                    "importance": mem["metadata"]["importance"],
                }
                for mem in memories
            ]

            merged_metadata["version"] = merged_metadata.get("version", 0) + 1

            # Generate embedding for merged text
            merged_embedding = self.embeddings.generate_embedding(merged_text)

            # Update primary memory
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=primary_id, vector=merged_embedding.tolist(), payload=merged_metadata
            )

            self.qdrant.client.upsert(collection_name=primary_collection, points=[point])

            # Mark other memories as outdated (or delete them)
            for mem_id in memory_ids[1:]:
                try:
                    self.mark_memory_outdated(memory_id=mem_id, reason=f"Merged into {primary_id}")
                except Exception as e:
                    logger.warning(f"Failed to mark {mem_id} as outdated: {e}")

            logger.info(f"Merged {len(memory_ids)} memories into {primary_id}")

            return primary_id

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to merge memories: {e}")
            raise RuntimeError("Memory merge failed") from e

    def mark_memory_outdated(
        self, memory_id: str, reason: str, collection_name: Optional[str] = None
    ) -> bool:
        """
        Mark a memory as outdated without deleting it.

        Args:
            memory_id: ID of the memory
            reason: Reason for marking as outdated
            collection_name: Optional collection name

        Returns:
            True if successful
        """
        try:
            if not collection_name:
                collection_name = self._get_collection_for_memory(memory_id)
                if not collection_name:
                    raise ValueError(f"Memory {memory_id} not found")

            # Retrieve memory
            memory = self.retriever.get_memory_by_id(memory_id, collection_name)
            if not memory:
                raise ValueError(f"Memory {memory_id} not found")

            # Update metadata
            metadata = memory["metadata"].copy()
            metadata["outdated"] = True
            metadata["outdated_reason"] = reason
            metadata["outdated_timestamp"] = datetime.utcnow().isoformat()

            # Get original embedding (no need to regenerate)
            points = self.qdrant.client.retrieve(
                collection_name=collection_name, ids=[memory_id], with_vectors=True
            )

            if not points:
                raise ValueError(f"Failed to retrieve vector for {memory_id}")

            from qdrant_client.models import PointStruct

            point = PointStruct(id=memory_id, vector=points[0].vector, payload=metadata)

            self.qdrant.client.upsert(collection_name=collection_name, points=[point])

            logger.info(f"Marked memory {memory_id} as outdated: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark memory as outdated: {e}")
            raise RuntimeError("Failed to mark memory as outdated") from e

    def resolve_conflict(
        self, old_memory: Dict[str, Any], new_memory: Dict[str, Any], max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Use Claude to determine how to resolve a conflict between memories.

        Args:
            old_memory: Existing memory dictionary
            new_memory: New memory dictionary
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with 'decision', 'reasoning', 'merged_text', 'merged_importance'

        Raises:
            RuntimeError: If API call fails
        """
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()

                # Build prompt
                prompt = self.CONFLICT_RESOLUTION_PROMPT.format(
                    old_text=old_memory.get("text", ""),
                    old_type=old_memory.get("metadata", {}).get("memory_type", "unknown"),
                    old_importance=old_memory.get("metadata", {}).get("importance", 5),
                    old_timestamp=old_memory.get("metadata", {}).get("timestamp", "unknown"),
                    new_text=new_memory.get("text", ""),
                    new_type=new_memory.get("metadata", {}).get("memory_type", "unknown"),
                    new_importance=new_memory.get("metadata", {}).get("importance", 5),
                    new_timestamp=datetime.utcnow().isoformat(),
                )

                # Call Claude
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response
                response_text = message.content[0].text.strip()
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1

                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")

                json_text = response_text[start_idx:end_idx]
                resolution = json.loads(json_text)

                # Validate
                if "decision" not in resolution:
                    raise ValueError("Missing decision field")

                try:
                    ConflictResolution(resolution["decision"])
                except ValueError:
                    raise ValueError(f"Invalid decision: {resolution['decision']}")

                logger.info(
                    f"Conflict resolution: {resolution['decision']} - "
                    f"{resolution.get('reasoning', '')}"
                )

                return resolution

            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    time.sleep(2**attempt)
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
                    raise RuntimeError("Failed to get valid resolution") from e

        raise RuntimeError("Conflict resolution failed")

    def apply_resolution(
        self, old_memory: Dict[str, Any], new_memory: Dict[str, Any], resolution: Dict[str, Any]
    ) -> Optional[str]:
        """
        Apply the conflict resolution decision.

        Args:
            old_memory: Existing memory
            new_memory: New memory
            resolution: Resolution decision from resolve_conflict

        Returns:
            Memory ID of the result (or None if no action taken)

        Raises:
            RuntimeError: If application fails
        """
        try:
            decision = resolution["decision"]

            if decision == ConflictResolution.UPDATE.value:
                # Replace old with new
                self.update_memory(
                    memory_id=old_memory["memory_id"],
                    new_text=new_memory["text"],
                    reason=resolution.get("reasoning", "Conflict resolution: update"),
                    new_importance=new_memory.get("metadata", {}).get("importance"),
                )
                return old_memory["memory_id"]

            elif decision == ConflictResolution.MERGE.value:
                # Merge both
                merged_text = resolution.get("merged_text")
                merged_importance = resolution.get(
                    "merged_importance", old_memory["metadata"]["importance"]
                )

                if not merged_text:
                    raise ValueError("Merged text not provided in resolution")

                return self.merge_memories(
                    memory_ids=[old_memory["memory_id"]],
                    merged_text=merged_text,
                    merged_importance=merged_importance,
                    reason=resolution.get("reasoning", "Conflict resolution: merge"),
                )

            elif decision == ConflictResolution.SEPARATE.value:
                # Keep both separate - no action needed
                logger.info("Keeping memories separate as per resolution")
                return None

            else:
                raise ValueError(f"Unknown decision: {decision}")

        except Exception as e:
            logger.error(f"Failed to apply resolution: {e}")
            raise RuntimeError("Failed to apply conflict resolution") from e
