"""Qdrant client wrapper for managing vector collections."""

import logging
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantManager:
    """Wrapper class for Qdrant client operations."""

    VECTOR_SIZE = 384  # for sentence-transformers
    COLLECTIONS = ["personal_facts", "conversation_history", "knowledge_base"]

    def __init__(self, host: str = "192.168.1.120", port: int = 6334):
        """
        Initialize Qdrant client connection.

        Args:
            host: Qdrant server host address
            port: Qdrant server port

        Raises:
            ConnectionError: If unable to connect to Qdrant server
        """
        self.host = host
        self.port = port
        self.client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Qdrant server."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test connection
            self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Unable to connect to Qdrant at {self.host}:{self.port}") from e

    def create_collections(self) -> None:
        """
        Create all required collections if they don't exist.

        Creates: personal_facts, conversation_history, knowledge_base
        Each with 384-dimensional vectors using Cosine distance.

        Raises:
            RuntimeError: If collection creation fails
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected")

        for collection_name in self.COLLECTIONS:
            try:
                # Check if collection exists
                collections = self.client.get_collections().collections
                exists = any(col.name == collection_name for col in collections)

                if exists:
                    logger.info(f"Collection '{collection_name}' already exists")
                    continue

                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info(f"Created collection '{collection_name}'")

            except UnexpectedResponse as e:
                logger.error(f"Failed to create collection '{collection_name}': {e}")
                raise RuntimeError(f"Collection creation failed for '{collection_name}'") from e
            except Exception as e:
                logger.error(f"Unexpected error creating collection '{collection_name}': {e}")
                raise RuntimeError(f"Unexpected error for '{collection_name}'") from e

    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a specific collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information

        Raises:
            ValueError: If collection doesn't exist
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected")

        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            raise ValueError(f"Collection '{collection_name}' not found or error occurred") from e

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected")

        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise RuntimeError("Failed to retrieve collections list") from e

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant client connection closed")
