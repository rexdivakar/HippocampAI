"""Qdrant vector store with HNSW tuning."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    SearchParams,
    VectorParams,
    WalConfigDiff,
)

from hippocampai.utils.retry import get_qdrant_retry_decorator

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant store with HNSW optimization."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_facts: str = "hippocampai_facts",
        collection_prefs: str = "hippocampai_prefs",
        dimension: int = 384,
        hnsw_m: int = 48,
        ef_construction: int = 256,
        ef_search: int = 128,
    ):
        self.client = QdrantClient(url=url, timeout=60.0)
        self.collection_facts = collection_facts
        self.collection_prefs = collection_prefs
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        logger.info(f"Connected to Qdrant at {url}")
        self._ensure_collections()

    def _ensure_collections(self, collection_name: Optional[str] = None):
        """Create collections if they don't exist.
        
        Args:
            collection_name: If provided, only ensure this specific collection exists.
                           Otherwise, ensure both default collections exist.
        """
        collections_to_ensure = [collection_name] if collection_name else [self.collection_facts, self.collection_prefs]
        
        for coll_name in collections_to_ensure:
            # Use idempotent create_collection to avoid race conditions
            # If collection exists with same params, this is a no-op
            try:
                self.client.create_collection(
                    collection_name=coll_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                    hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.ef_construction),
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
                    wal_config=WalConfigDiff(wal_capacity_mb=32),
                )

                # Create payload indices for 5-10x faster filtered queries
                # Index user_id (KEYWORD for exact match)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="user_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                # Index type (KEYWORD for memory type filtering)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                # Index tags (KEYWORD for tag filtering)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="tags",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                # Index importance (FLOAT for range queries)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="importance",
                    field_schema=PayloadSchemaType.FLOAT,
                )
                # Index created_at (DATETIME for date range filtering)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="created_at",
                    field_schema=PayloadSchemaType.DATETIME,
                )
                # Index updated_at (DATETIME for update date filtering)
                self.client.create_payload_index(
                    collection_name=coll_name,
                    field_name="updated_at",
                    field_schema=PayloadSchemaType.DATETIME,
                )

                logger.info(
                    f"Created collection '{coll_name}' with 6 payload indices (user_id, type, tags, importance, created_at, updated_at)"
                )

                # Wait for collection to be fully ready (avoid race conditions in tests)
                self._wait_for_collection_ready(coll_name)
            except Exception as e:
                # If collection already exists with different params, log and continue
                # This is idempotent behavior - we don't fail if collection exists
                if "already exists" in str(e).lower():
                    logger.debug(f"Collection '{coll_name}' already exists, skipping creation")
                else:
                    logger.error(f"Error creating collection '{coll_name}': {e}")
                    raise

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: str = "Cosine",
    ):
        """
        Public method to ensure a collection exists.

        Args:
            collection_name: Name of the collection to ensure exists
            vector_size: Dimension of vectors (uses instance dimension if not provided)
            distance: Distance metric (Cosine, Euclid, or Dot)
        """
        # Use instance dimension if not provided
        dimension = vector_size if vector_size is not None else self.dimension

        # Map string distance to Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
            "dot": Distance.DOT,
        }
        dist_metric = distance_map.get(distance, Distance.COSINE)

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=dist_metric),
                hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.ef_construction),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
                wal_config=WalConfigDiff(wal_capacity_mb=32),
            )

            # Create payload indices for 5-10x faster filtered queries
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="tags",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="importance",
                field_schema=PayloadSchemaType.FLOAT,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="created_at",
                field_schema=PayloadSchemaType.DATETIME,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="updated_at",
                field_schema=PayloadSchemaType.DATETIME,
            )

            logger.info(
                f"Created collection '{collection_name}' with 6 payload indices (user_id, type, tags, importance, created_at, updated_at)"
            )

            # Wait for collection to be fully ready
            self._wait_for_collection_ready(collection_name)
        except Exception as e:
            # If collection already exists, log and continue
            if "already exists" in str(e).lower():
                logger.debug(f"Collection '{collection_name}' already exists, skipping creation")
            else:
                logger.error(f"Error creating collection '{collection_name}': {e}")
                raise

    def _wait_for_collection_ready(self, collection_name: str, max_attempts: int = 10):
        """Wait for collection to be fully initialized and queryable."""
        for attempt in range(max_attempts):
            try:
                # Try to get collection info to verify it's ready
                info = self.client.get_collection(collection_name)
                if info.status == "green":
                    logger.debug(f"Collection {collection_name} is ready")
                    return
                logger.debug(f"Collection {collection_name} status: {info.status}, waiting...")
            except Exception as e:
                logger.debug(f"Collection {collection_name} not ready (attempt {attempt + 1}): {e}")

            if attempt < max_attempts - 1:
                time.sleep(0.2)  # Short wait between attempts

        logger.warning(
            f"Collection {collection_name} may not be fully ready after {max_attempts} attempts"
        )

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def upsert(self, collection_name: str, id: str, vector: np.ndarray, payload: dict[str, Any]):
        """Insert or update a point (with automatic retry on transient failures)."""
        # Ensure collection exists before upserting (idempotent)
        self._ensure_collections(collection_name)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=payload,
                )
            ],
        )

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def bulk_upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[np.ndarray],
        payloads: list[dict[str, Any]],
    ):
        """
        Bulk insert or update multiple points (3-5x faster than individual upserts).

        Args:
            collection_name: Name of the collection
            ids: List of point IDs
            vectors: List of vectors
            payloads: List of payloads

        Raises:
            ValueError: If lengths don't match
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have the same length")

        points = [
            PointStruct(
                id=id_val,
                vector=vec.tolist() if isinstance(vec, np.ndarray) else vec,
                payload=payload,
            )
            for id_val, vec, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(collection_name=collection_name, points=points)
        logger.debug(f"Bulk upserted {len(points)} points to {collection_name}")

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def search(
        self,
        collection_name: str,
        vector: np.ndarray,
        limit: int = 100,
        filters: Optional[dict[str, Any]] = None,
        ef: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search (with automatic retry on transient failures)."""
        query_filter = None
        if filters:
            conditions = []
            if "user_id" in filters:
                conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=filters["user_id"]))
                )
            if "type" in filters:
                conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=filters["type"]))
                )
            if "tags" in filters:
                # Support both single tag and list of tags
                tags = filters["tags"]
                if isinstance(tags, str):
                    tags = [tags]
                conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
            if conditions:
                query_filter = Filter(must=conditions)

        # Build search params
        hnsw_ef = ef if ef else self.ef_search
        search_params = SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None

        # Use query_points instead of deprecated search
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                limit=limit,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
            ).points
            return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]
        except UnexpectedResponse as e:
            # Handle the specific case where the collection does not exist
            if e.status_code == 404:
                logger.warning(f"Collection {collection_name} does not exist. Returning empty list.")
                return []
            # Re-raise other unexpected errors
            raise

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def scroll(
        self, collection_name: str, filters: Optional[dict[str, Any]] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Scroll through points (with automatic retry on transient failures)."""
        query_filter = None
        if filters:
            conditions = []
            if "user_id" in filters:
                conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=filters["user_id"]))
                )
            if "type" in filters:
                conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=filters["type"]))
                )
            if "tags" in filters:
                # Support both single tag and list of tags
                tags = filters["tags"]
                if isinstance(tags, str):
                    tags = [tags]
                conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
            if conditions:
                query_filter = Filter(must=conditions)

        try:
            results, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [{"id": str(r.id), "payload": r.payload} for r in results]
        except UnexpectedResponse as e:
            # Handle the specific case where the collection does not exist
            if e.status_code == 404:
                logger.warning(f"Collection {collection_name} does not exist. Returning empty list.")
                return []
            # Re-raise other unexpected errors
            raise

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def delete(self, collection_name: str, ids: list[str]):
        """Delete points by IDs (with automatic retry on transient failures)."""
        # If collection doesn't exist, nothing to delete
        if not self.client.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist. Skipping delete.")
            return
        
        self.client.delete(collection_name=collection_name, points_selector=ids)

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def get(self, collection_name: str, id: str) -> Optional[dict[str, Any]]:
        """Get a single point by ID (with automatic retry on transient failures)."""
        # If collection doesn't exist, nothing to get
        if not self.client.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist.")
            return None
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[id],
                with_payload=True,
                with_vectors=False,
            )
            if result:
                return {"id": str(result[0].id), "payload": result[0].payload}
            return None
        except Exception as e:
            logger.error(f"Failed to get point {id}: {e}")
            return None

    @get_qdrant_retry_decorator(max_attempts=3, min_wait=1, max_wait=5)
    def update(self, collection_name: str, id: str, payload: dict[str, Any]) -> bool:
        """Update payload of an existing point (with automatic retry on transient failures)."""
        # If collection doesn't exist, can't update
        if not self.client.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist. Skipping update.")
            return False
        
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[id],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update point {id}: {e}")
            return False

    def create_snapshot(self, collection_name: str) -> str:
        """Create collection snapshot."""
        result = self.client.create_snapshot(collection_name=collection_name)
        return result.name
