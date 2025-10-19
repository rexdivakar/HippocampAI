"""Memory storage service for managing memories in Qdrant."""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from qdrant_client.models import PointStruct

from src.qdrant_client import QdrantManager
from src.embedding_service import EmbeddingService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Valid memory types."""
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"


class Category(str, Enum):
    """Valid memory categories."""
    WORK = "work"
    PERSONAL = "personal"
    LEARNING = "learning"
    HEALTH = "health"
    SOCIAL = "social"
    FINANCE = "finance"
    OTHER = "other"


class MemoryStore:
    """Store for managing memories with embeddings in Qdrant."""

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService
    ):
        """
        Initialize the memory store.

        Args:
            qdrant_manager: Initialized QdrantManager instance
            embedding_service: Initialized EmbeddingService instance
        """
        self.qdrant = qdrant_manager
        self.embeddings = embedding_service
        logger.info("MemoryStore initialized")

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate memory metadata.

        Args:
            metadata: Metadata dictionary to validate

        Raises:
            ValueError: If metadata is invalid
        """
        # Required fields
        required = ["user_id", "memory_type", "importance", "category", "session_id", "confidence"]
        for field in required:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")

        # Validate user_id
        if not isinstance(metadata["user_id"], str) or not metadata["user_id"]:
            raise ValueError("user_id must be a non-empty string")

        # Validate memory_type
        try:
            MemoryType(metadata["memory_type"])
        except ValueError:
            valid_types = [t.value for t in MemoryType]
            raise ValueError(f"Invalid memory_type. Must be one of: {valid_types}")

        # Validate importance (1-10)
        importance = metadata["importance"]
        if not isinstance(importance, (int, float)) or not 1 <= importance <= 10:
            raise ValueError("importance must be a number between 1 and 10")

        # Validate category
        try:
            Category(metadata["category"])
        except ValueError:
            valid_categories = [c.value for c in Category]
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")

        # Validate session_id
        if not isinstance(metadata["session_id"], str) or not metadata["session_id"]:
            raise ValueError("session_id must be a non-empty string")

        # Validate confidence (0.0-1.0)
        confidence = metadata["confidence"]
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a number between 0.0 and 1.0")

        # Validate timestamp if provided
        if "timestamp" in metadata:
            if not isinstance(metadata["timestamp"], (datetime, str)):
                raise ValueError("timestamp must be a datetime object or ISO string")

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for storage.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Prepared metadata with timestamp
        """
        prepared = metadata.copy()

        # Add timestamp if not present
        if "timestamp" not in prepared:
            prepared["timestamp"] = datetime.utcnow().isoformat()
        elif isinstance(prepared["timestamp"], datetime):
            prepared["timestamp"] = prepared["timestamp"].isoformat()

        return prepared

    def _get_collection_name(self, memory_type: str) -> str:
        """
        Get the appropriate collection name based on memory type.

        Args:
            memory_type: Type of memory

        Returns:
            Collection name to use
        """
        # Map memory types to collections
        if memory_type in [MemoryType.PREFERENCE.value, MemoryType.FACT.value, MemoryType.GOAL.value]:
            return "personal_facts"
        elif memory_type in [MemoryType.CONTEXT.value, MemoryType.EVENT.value]:
            return "conversation_history"
        else:
            return "knowledge_base"

    def store_memory(
        self,
        text: str,
        memory_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a single memory with its embedding.

        Args:
            text: The memory text to store
            memory_type: Type of memory (preference, fact, goal, habit, event, context)
            metadata: Dictionary containing user_id, importance, timestamp, category, session_id, confidence

        Returns:
            UUID of the stored memory

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If storage fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        # Validate metadata
        full_metadata = {**metadata, "memory_type": memory_type}
        self._validate_metadata(full_metadata)

        try:
            # Generate embedding
            logger.debug(f"Generating embedding for memory: {text[:50]}...")
            embedding = self.embeddings.generate_embedding(text)

            # Prepare metadata
            prepared_metadata = self._prepare_metadata(full_metadata)
            prepared_metadata["text"] = text  # Store original text in metadata

            # Generate unique ID
            memory_id = str(uuid.uuid4())

            # Determine collection
            collection_name = self._get_collection_name(memory_type)

            # Create point
            point = PointStruct(
                id=memory_id,
                vector=embedding.tolist(),
                payload=prepared_metadata
            )

            # Store in Qdrant
            self.qdrant.client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            logger.info(
                f"Stored memory {memory_id} in '{collection_name}' "
                f"(type: {memory_type}, user: {metadata['user_id']})"
            )

            return memory_id

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise RuntimeError("Memory storage failed") from e

    def store_batch_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple memories efficiently.

        Args:
            memories: List of dictionaries, each containing:
                - text: Memory text
                - memory_type: Type of memory
                - metadata: Metadata dictionary

        Returns:
            List of memory UUIDs

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If batch storage fails
        """
        if not memories or not isinstance(memories, list):
            raise ValueError("memories must be a non-empty list")

        # Validate all memories first
        for i, memory in enumerate(memories):
            if not isinstance(memory, dict):
                raise ValueError(f"Memory at index {i} must be a dictionary")
            if "text" not in memory or "memory_type" not in memory or "metadata" not in memory:
                raise ValueError(f"Memory at index {i} missing required fields (text, memory_type, metadata)")

            # Validate individual memory
            full_metadata = {**memory["metadata"], "memory_type": memory["memory_type"]}
            self._validate_metadata(full_metadata)

        try:
            # Extract texts for batch embedding
            texts = [m["text"] for m in memories]

            # Generate all embeddings at once
            logger.info(f"Generating embeddings for {len(texts)} memories...")
            embeddings = self.embeddings.generate_batch_embeddings(texts)

            # Group memories by collection
            collections_points = {}
            memory_ids = []

            for memory, embedding in zip(memories, embeddings):
                # Prepare metadata
                full_metadata = {**memory["metadata"], "memory_type": memory["memory_type"]}
                prepared_metadata = self._prepare_metadata(full_metadata)
                prepared_metadata["text"] = memory["text"]

                # Generate ID
                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)

                # Get collection
                collection_name = self._get_collection_name(memory["memory_type"])

                # Create point
                point = PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload=prepared_metadata
                )

                # Group by collection
                if collection_name not in collections_points:
                    collections_points[collection_name] = []
                collections_points[collection_name].append(point)

            # Batch upsert to each collection
            for collection_name, points in collections_points.items():
                self.qdrant.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Stored {len(points)} memories in '{collection_name}'")

            logger.info(f"Successfully stored {len(memory_ids)} memories")
            return memory_ids

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to store batch memories: {e}")
            raise RuntimeError("Batch memory storage failed") from e

    def get_memory(self, memory_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: UUID of the memory
            collection_name: Optional collection name to search in

        Returns:
            Memory dictionary or None if not found
        """
        try:
            collections_to_search = [collection_name] if collection_name else self.qdrant.COLLECTIONS

            for coll in collections_to_search:
                try:
                    points = self.qdrant.client.retrieve(
                        collection_name=coll,
                        ids=[memory_id]
                    )
                    if points:
                        return points[0].payload
                except Exception:
                    continue

            logger.warning(f"Memory {memory_id} not found")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise RuntimeError("Memory retrieval failed") from e
