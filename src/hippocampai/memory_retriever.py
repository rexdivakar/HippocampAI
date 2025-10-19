"""Memory retrieval service for searching and fetching memories from Qdrant."""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchValue, Range

from hippocampai.embedding_service import EmbeddingService
from hippocampai.qdrant_client import QdrantManager
from hippocampai.utils.time import now_utc, parse_iso_datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retriever for searching and fetching memories from Qdrant."""

    def __init__(self, qdrant_manager: QdrantManager, embedding_service: EmbeddingService):
        """
        Initialize the memory retriever.

        Args:
            qdrant_manager: Initialized QdrantManager instance
            embedding_service: Initialized EmbeddingService instance
        """
        self.qdrant = qdrant_manager
        self.embeddings = embedding_service
        logger.info("MemoryRetriever initialized")

    def _build_filter(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
        """
        Build Qdrant filter from filter dictionary.

        Args:
            filters: Dictionary of filters to apply

        Returns:
            Qdrant Filter object or None
        """
        if not filters:
            return None

        conditions = []

        # User ID filter
        if "user_id" in filters:
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=filters["user_id"]))
            )

        # Memory type filter
        if "memory_type" in filters:
            memory_type = filters["memory_type"]
            if isinstance(memory_type, list):
                # Multiple memory types (OR condition would require nested structure)
                # For simplicity, we'll use the first one
                memory_type = memory_type[0]
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
            )

        # Category filter
        if "category" in filters:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=filters["category"]))
            )

        # Session ID filter
        if "session_id" in filters:
            conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=filters["session_id"]))
            )

        # Importance range filter
        if "min_importance" in filters or "max_importance" in filters:
            importance_range = Range(
                gte=filters.get("min_importance"), lte=filters.get("max_importance")
            )
            conditions.append(FieldCondition(key="importance", range=importance_range))

        # Confidence range filter
        if "min_confidence" in filters or "max_confidence" in filters:
            confidence_range = Range(
                gte=filters.get("min_confidence"), lte=filters.get("max_confidence")
            )
            conditions.append(FieldCondition(key="confidence", range=confidence_range))

        # Date range filter (timestamp)
        if "start_date" in filters or "end_date" in filters:
            date_range = DatetimeRange(gte=filters.get("start_date"), lte=filters.get("end_date"))
            conditions.append(FieldCondition(key="timestamp", range=date_range))

        if not conditions:
            return None

        return Filter(must=conditions)

    def _format_result(self, point, include_score: bool = True) -> Dict[str, Any]:
        """
        Format a search result into a clean dictionary.

        Args:
            point: Qdrant point object
            include_score: Whether to include similarity score

        Returns:
            Formatted memory dictionary
        """
        result = {
            "memory_id": point.id,
            "text": point.payload.get("text", ""),
            "metadata": {
                "user_id": point.payload.get("user_id"),
                "memory_type": point.payload.get("memory_type"),
                "importance": point.payload.get("importance"),
                "timestamp": point.payload.get("timestamp"),
                "category": point.payload.get("category"),
                "session_id": point.payload.get("session_id"),
                "confidence": point.payload.get("confidence"),
            },
        }

        if include_score and hasattr(point, "score"):
            result["similarity_score"] = point.score

        return result

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collections: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return (default: 10)
            filters: Optional filters to apply (user_id, memory_type, category, etc.)
            collections: Optional list of collections to search (default: all)

        Returns:
            List of memory dictionaries with similarity scores

        Raises:
            ValueError: If query is invalid
            RuntimeError: If search fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embeddings.generate_embedding(query)

            # Build filter
            search_filter = self._build_filter(filters)

            # Determine collections to search
            collections_to_search = collections or self.qdrant.COLLECTIONS

            # Search across all specified collections
            all_results = []

            for collection_name in collections_to_search:
                try:
                    logger.debug(f"Searching in collection '{collection_name}'...")
                    results = self.qdrant.client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding.tolist(),
                        query_filter=search_filter,
                        limit=limit,
                    )

                    # Format results
                    for point in results:
                        all_results.append(self._format_result(point, include_score=True))

                except Exception as e:
                    logger.warning(f"Error searching collection '{collection_name}': {e}")
                    continue

            # Sort all results by similarity score and limit
            all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            final_results = all_results[:limit]

            logger.info(
                f"Found {len(final_results)} memories for query: '{query[:50]}...' "
                f"(searched {len(collections_to_search)} collections)"
            )

            return final_results

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise RuntimeError("Memory search failed") from e

    def get_memory_by_id(
        self, memory_id: str, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its ID.

        Args:
            memory_id: UUID of the memory
            collection_name: Optional collection name to search in

        Returns:
            Memory dictionary or None if not found
        """
        if not memory_id:
            raise ValueError("memory_id must be provided")

        try:
            collections_to_search = (
                [collection_name] if collection_name else self.qdrant.COLLECTIONS
            )

            for coll in collections_to_search:
                try:
                    points = self.qdrant.client.retrieve(collection_name=coll, ids=[memory_id])

                    if points:
                        logger.info(f"Retrieved memory {memory_id} from '{coll}'")
                        return self._format_result(points[0], include_score=False)

                except Exception as e:
                    logger.debug(f"Memory {memory_id} not found in '{coll}': {e}")
                    continue

            logger.warning(f"Memory {memory_id} not found in any collection")
            return None

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise RuntimeError("Memory retrieval failed") from e

    def get_memories_by_filter(
        self, filters: Dict[str, Any], limit: int = 50, collections: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories by metadata filters without vector search.

        Args:
            filters: Filters to apply (user_id, memory_type, category, date range, etc.)
            limit: Maximum number of results to return (default: 50)
            collections: Optional list of collections to search (default: all)

        Returns:
            List of memory dictionaries (without similarity scores)

        Raises:
            ValueError: If filters are invalid
            RuntimeError: If retrieval fails
        """
        if not filters or not isinstance(filters, dict):
            raise ValueError("filters must be a non-empty dictionary")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        try:
            # Build filter
            search_filter = self._build_filter(filters)

            if not search_filter:
                raise ValueError("No valid filters provided")

            # Determine collections to search
            collections_to_search = collections or self.qdrant.COLLECTIONS

            # Scroll through collections with filters
            all_results = []

            for collection_name in collections_to_search:
                try:
                    logger.debug(f"Filtering in collection '{collection_name}'...")

                    # Use scroll to get filtered results
                    results, _ = self.qdrant.client.scroll(
                        collection_name=collection_name, scroll_filter=search_filter, limit=limit
                    )

                    # Format results
                    for point in results:
                        all_results.append(self._format_result(point, include_score=False))

                except Exception as e:
                    logger.warning(f"Error filtering collection '{collection_name}': {e}")
                    continue

            # Limit total results
            final_results = all_results[:limit]

            logger.info(
                f"Found {len(final_results)} memories with filters "
                f"(searched {len(collections_to_search)} collections)"
            )

            return final_results

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Memory filter retrieval failed: {e}")
            raise RuntimeError("Memory filter retrieval failed") from e

    def get_recent_memories(
        self, user_id: str, limit: int = 20, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most recent memories for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of results (default: 20)
            memory_type: Optional memory type filter

        Returns:
            List of recent memories sorted by timestamp
        """
        filters = {"user_id": user_id}
        if memory_type:
            filters["memory_type"] = memory_type

        try:
            memories = self.get_memories_by_filter(filters, limit=limit * 2)  # Get more for sorting

            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)

            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve recent memories: {e}")
            raise RuntimeError("Recent memory retrieval failed") from e

    def get_important_memories(
        self, user_id: str, min_importance: int = 7, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get important memories for a user.

        Args:
            user_id: User identifier
            min_importance: Minimum importance score (default: 7)
            limit: Maximum number of results (default: 20)

        Returns:
            List of important memories sorted by importance
        """
        filters = {"user_id": user_id, "min_importance": min_importance}

        try:
            memories = self.get_memories_by_filter(filters, limit=limit * 2)

            # Sort by importance (highest first)
            memories.sort(key=lambda x: x["metadata"].get("importance", 0), reverse=True)

            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve important memories: {e}")
            raise RuntimeError("Important memory retrieval failed") from e

    def smart_search(
        self,
        query: str,
        user_id: str,
        context_type: Optional[str] = None,
        limit: int = 10,
        access_counts: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Intelligent multi-factor search with re-ranking.

        Scoring formula:
        final_score = (similarity * 0.50) + (normalized_importance * 0.30) +
                     (recency_score * 0.20) + access_boost

        Args:
            query: Search query
            user_id: User identifier
            context_type: Optional context (work, personal, casual)
            limit: Number of results to return
            access_counts: Optional dict of memory_id -> access count

        Returns:
            Re-ranked memories with final scores
        """
        if access_counts is None:
            access_counts = {}

        try:
            # Step 1: Get candidates from semantic search (3x limit for re-ranking)
            candidate_limit = min(limit * 3, 30)

            filters = {"user_id": user_id}

            # Add context-aware filtering
            if context_type:
                context_map = {  # noqa: F841
                    "work": ["work", "finance"],
                    "personal": ["personal", "social", "health"],
                    "casual": ["personal", "social", "other"],
                }
                # Note: Qdrant doesn't support OR on categories easily,
                # so we'll filter during re-ranking instead

            candidates = self.search_memories(query=query, limit=candidate_limit, filters=filters)

            if not candidates:
                return []

            logger.debug(f"Got {len(candidates)} candidates for re-ranking")

            # Step 2: Calculate recency and importance scores
            now = now_utc()

            scored_results = []

            for candidate in candidates:
                metadata = candidate["metadata"]

                # Context-aware category boosting
                category_boost = 0.0
                if context_type:
                    category = metadata.get("category", "other")
                    context_categories = {
                        "work": ["work", "finance"],
                        "personal": ["personal", "social", "health"],
                        "casual": ["personal", "social"],
                    }
                    if category in context_categories.get(context_type, []):
                        category_boost = 0.15

                # 1. Similarity score (0-1, already normalized)
                similarity = candidate.get("similarity_score", 0.5)

                # 2. Importance score (normalize 1-10 to 0-1)
                importance = metadata.get("importance", 5)
                normalized_importance = (importance - 1) / 9.0  # Scale 1-10 to 0-1

                # 3. Recency score (exponential decay)
                timestamp_str = metadata.get("timestamp", "")
                try:
                    timestamp = parse_iso_datetime(timestamp_str)
                    age_days = (now - timestamp).total_seconds() / 86400

                    # Exponential decay: score = e^(-age/30)
                    # 30 days half-life
                    import math

                    recency_score = math.exp(-age_days / 30.0)
                except Exception:
                    recency_score = 0.5  # Default if timestamp invalid

                # 4. Access boost (logarithmic)
                memory_id = candidate.get("memory_id", "")
                access_count = access_counts.get(memory_id, 0)
                if access_count > 0:
                    import math

                    access_boost = min(0.2, math.log(access_count + 1) * 0.05)
                else:
                    access_boost = 0.0

                # Calculate final score with weights
                # Similarity: 50%, Importance: 30%, Recency: 20%
                base_score = similarity * 0.50 + normalized_importance * 0.30 + recency_score * 0.20

                final_score = base_score + access_boost + category_boost

                # Add scoring details to result
                result = candidate.copy()
                result["final_score"] = final_score
                result["score_breakdown"] = {
                    "similarity": similarity,
                    "importance": normalized_importance,
                    "recency": recency_score,
                    "access_boost": access_boost,
                    "category_boost": category_boost,
                    "base_score": base_score,
                }

                scored_results.append(result)

            # Step 3: Sort by final score
            scored_results.sort(key=lambda x: x["final_score"], reverse=True)

            # Step 4: Return top N
            final_results = scored_results[:limit]

            logger.info(
                f"Smart search complete: {len(final_results)} results "
                f"(top score: {final_results[0]['final_score']:.3f})"
            )

            # Log top result breakdown for debugging
            if final_results:
                top = final_results[0]
                logger.debug(
                    f"Top result breakdown: sim={top['score_breakdown']['similarity']:.3f}, "
                    f"imp={top['score_breakdown']['importance']:.3f}, "
                    f"rec={top['score_breakdown']['recency']:.3f}, "
                    f"access={top['score_breakdown']['access_boost']:.3f}"
                )

            return final_results

        except Exception as e:
            logger.error(f"Smart search failed: {e}")
            raise RuntimeError("Smart search failed") from e

    def get_context_for_query(
        self, query: str, user_id: str, max_memories: int = 5
    ) -> Dict[str, Any]:
        """
        Get optimal memory context for a specific query.

        Returns organized memories by type for easy AI context building.

        Args:
            query: User's query
            user_id: User identifier
            max_memories: Maximum memories per category

        Returns:
            Dictionary with categorized memories and metadata
        """
        try:
            # Determine context type from query keywords
            query_lower = query.lower()
            context_type = None

            if any(
                word in query_lower for word in ["work", "job", "project", "meeting", "deadline"]
            ):
                context_type = "work"
            elif any(word in query_lower for word in ["hobby", "personal", "family", "friend"]):
                context_type = "personal"

            # Get smart search results
            all_memories = self.smart_search(
                query=query,
                user_id=user_id,
                context_type=context_type,
                limit=max_memories * 3,  # Get more for categorization
            )

            # Organize by memory type
            organized = {
                "preferences": [],
                "facts": [],
                "goals": [],
                "habits": [],
                "context": [],
                "events": [],
            }

            for memory in all_memories:
                mem_type = memory["metadata"].get("memory_type", "fact")
                if mem_type in organized:
                    if len(organized[mem_type]) < max_memories:
                        organized[mem_type].append(memory)

            # Calculate total relevance
            total_score = sum(m.get("final_score", 0) for m in all_memories[:max_memories])

            result = {
                "query": query,
                "context_type": context_type,
                "memories": organized,
                "total_relevance": total_score,
                "memory_count": sum(len(v) for v in organized.values()),
            }

            logger.info(
                f"Context for query: {result['memory_count']} memories, "
                f"relevance: {total_score:.3f}, type: {context_type}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to get context for query: {e}")
            raise RuntimeError("Context retrieval failed") from e
