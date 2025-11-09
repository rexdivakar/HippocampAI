"""Comprehensive memory management service with CRUD, batch, dedup, and consolidation."""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.pipeline.conflict_resolution import (
    ConflictResolutionStrategy,
    MemoryConflictResolver,
)
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.conversation_memory import (
    ConversationMemoryManager,
    ConversationSummary,
    ConversationTurn,
    SpeakerRole,
)
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.memory_lifecycle import (
    LifecycleConfig,
    MemoryLifecycleManager,
    MemoryTier,
)
from hippocampai.pipeline.memory_merge import (
    MemoryMergeEngine,
    MergeCandidate,
    MergeResult,
    MergeStrategy,
)
from hippocampai.pipeline.memory_quality import (
    DuplicateCluster,
    MemoryHealthScore,
    MemoryQualityMonitor,
    MemoryStoreHealth,
    TopicCoverage,
)
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class MemoryManagementService:
    """
    Comprehensive memory management service.

    Provides:
    - CRUD operations (create, read, update, delete)
    - Batch operations (batch create, update, delete)
    - Automatic extraction from conversation logs
    - Hybrid search with customizable weights
    - Deduplication service
    - Consolidation service
    - Conflict resolution for contradictory memories
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker,
        redis_store: AsyncMemoryKVStore,
        llm: Optional[BaseLLM] = None,
        weights: Optional[dict[str, float]] = None,
        half_lives: Optional[dict[str, int]] = None,
        dedup_threshold: float = 0.88,
        enable_conflict_resolution: bool = True,
        conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TEMPORAL,
        conflict_similarity_threshold: float = 0.75,
        lifecycle_config: Optional[LifecycleConfig] = None,
        enable_lifecycle_management: bool = True,
    ):
        """
        Initialize memory management service.

        Args:
            qdrant_store: Vector store for memories
            embedder: Embedding model
            reranker: Reranking model
            redis_store: Redis KV store for caching
            llm: Optional LLM for extraction and consolidation
            weights: Optional custom scoring weights
            half_lives: Optional custom half-lives for memory types
            dedup_threshold: Similarity threshold for deduplication
            enable_conflict_resolution: Enable automatic conflict detection and resolution
            conflict_resolution_strategy: Default strategy for resolving conflicts
            conflict_similarity_threshold: Minimum similarity to check for conflicts
            lifecycle_config: Optional lifecycle management configuration
            enable_lifecycle_management: Enable automatic memory tiering and lifecycle management
        """
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.redis = redis_store
        self.llm = llm
        self.enable_conflict_resolution = enable_conflict_resolution
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.enable_lifecycle_management = enable_lifecycle_management

        # Initialize retriever
        self.retriever = HybridRetriever(
            qdrant_store=qdrant_store,
            embedder=embedder,
            reranker=reranker,
            weights=weights,
            half_lives=half_lives,
        )

        # Initialize deduplicator
        self.deduplicator = MemoryDeduplicator(
            qdrant_store=qdrant_store,
            embedder=embedder,
            reranker=reranker,
            similarity_threshold=dedup_threshold,
        )

        # Initialize consolidator
        self.consolidator = MemoryConsolidator(llm=llm)

        # Initialize conflict resolver
        self.conflict_resolver = MemoryConflictResolver(
            embedder=embedder,
            llm=llm,
            default_strategy=conflict_resolution_strategy,
            similarity_threshold=conflict_similarity_threshold,
        )

        # Initialize lifecycle manager
        self.lifecycle_manager = MemoryLifecycleManager(
            config=lifecycle_config or LifecycleConfig()
        )

        # Initialize quality monitor
        self.quality_monitor = MemoryQualityMonitor(
            stale_threshold_days=90,
            duplicate_threshold=0.85,
            near_duplicate_threshold=0.70,
        )

        # Initialize conversation memory manager
        self.conversation_manager = ConversationMemoryManager(
            kv_store=redis_store,
            llm=llm,
        )

        # Initialize memory merge engine
        self.merge_engine = MemoryMergeEngine(
            similarity_threshold=0.85,
            merge_confidence_threshold=0.7,
        )

        # Query cache TTL (60 seconds)
        self.query_cache_ttl = 60

    def _generate_query_cache_key(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str],
        k: int,
        filters: Optional[dict[str, Any]],
        custom_weights: Optional[dict[str, float]],
    ) -> str:
        """Generate a cache key for query results."""
        cache_data = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "k": k,
            "filters": filters,
            "weights": custom_weights,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode(), usedforsecurity=False).hexdigest()
        return f"query_cache:{cache_hash}"

    async def create_memory(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        ttl_days: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        check_duplicate: bool = True,
        check_conflicts: bool = True,
        auto_resolve_conflicts: bool = True,
    ) -> Memory:
        """
        Create a new memory with optional deduplication and conflict resolution.

        Args:
            text: Memory content
            user_id: User identifier
            session_id: Optional session identifier
            memory_type: Type of memory
            importance: Importance score (0-10)
            tags: Optional tags
            ttl_days: Optional time-to-live in days
            metadata: Optional metadata
            check_duplicate: Whether to check for duplicates
            check_conflicts: Whether to check for conflicting memories
            auto_resolve_conflicts: Whether to automatically resolve conflicts

        Returns:
            Created memory
        """
        # Create memory object
        memory = Memory(
            text=text,
            user_id=user_id,
            session_id=session_id,
            type=MemoryType(memory_type),
            importance=importance or 5.0,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Set expiration if TTL provided
        if ttl_days:
            memory.expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)

        # Calculate size metrics
        memory.calculate_size_metrics()

        # Check for duplicates
        if check_duplicate:
            action, duplicate_ids = self.deduplicator.check_duplicate(memory, user_id)

            if action == "skip":
                logger.info(f"Skipping duplicate memory: {memory.id}")
                # Return existing memory
                existing_id = duplicate_ids[0]
                existing_memory = await self.get_memory(existing_id)
                if existing_memory is None:
                    raise ValueError(f"Duplicate memory {existing_id} not found")
                return existing_memory

            if action == "update":
                logger.info(f"Updating existing memory: {duplicate_ids[0]}")
                # Update the first duplicate
                existing_id = duplicate_ids[0]
                updated_memory = await self.update_memory(
                    memory_id=existing_id,
                    text=text,
                    importance=memory.importance,
                    tags=memory.tags,
                )
                if updated_memory is None:
                    raise ValueError(f"Failed to update memory {existing_id}")
                return updated_memory

        # Check for conflicts
        if check_conflicts and self.enable_conflict_resolution:
            # Get existing memories of the same type
            existing_memories = await self.get_memories(
                user_id=user_id,
                memory_type=memory_type,
                limit=100,  # Check recent memories
            )

            if existing_memories:
                conflicts = self.conflict_resolver.detect_conflicts(
                    memory, existing_memories, check_llm=bool(self.llm)
                )

                if conflicts:
                    logger.info(
                        f"Detected {len(conflicts)} conflict(s) for new memory: '{text[:50]}...'"
                    )

                    # Auto-resolve if enabled
                    if auto_resolve_conflicts:
                        for conflict in conflicts:
                            resolution = self.conflict_resolver.resolve_conflict(
                                conflict, strategy=self.conflict_resolution_strategy
                            )

                            # Handle resolution
                            if resolution.action == "keep_first":
                                # Keep existing, don't create new
                                logger.info(
                                    f"Conflict resolved: keeping existing memory {conflict.memory_1.id}"
                                )
                                return conflict.memory_1

                            elif resolution.action == "keep_second":
                                # Continue to create new memory (will delete old below)
                                logger.info(
                                    f"Conflict resolved: keeping new memory, will delete {conflict.memory_1.id}"
                                )
                                # Delete conflicting old memory
                                await self.delete_memory(conflict.memory_1.id)

                            elif resolution.action == "merge":
                                # Use merged memory instead
                                logger.info("Conflict resolved: merged memories into new memory")
                                memory = resolution.updated_memory
                                # Delete both original memories
                                for mem_id in resolution.deleted_memory_ids:
                                    await self.delete_memory(mem_id)

                            elif resolution.action in ["flag", "keep_both"]:
                                # Flag both memories for review
                                logger.info(f"Conflict flagged for review: {conflict.id}")
                                # Update existing memory with conflict flag
                                await self.update_memory(
                                    memory_id=conflict.memory_1.id,
                                    metadata=conflict.memory_1.metadata,
                                )
                                # Add conflict flag to new memory
                                memory.metadata.update(conflict.memory_2.metadata)
                    else:
                        # Just flag for review without auto-resolving
                        logger.info(f"Flagging {len(conflicts)} conflict(s) for user review")
                        memory.metadata["has_conflicts"] = True
                        memory.metadata["conflict_count"] = len(conflicts)
                        memory.metadata["conflict_ids"] = [c.id for c in conflicts]

        # Store in vector database
        collection = memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )
        vector = self.embedder.encode_single(text)
        self.qdrant.upsert(
            collection_name=collection,
            id=memory.id,
            vector=vector,
            payload=memory.model_dump(mode="json"),
        )

        # Cache in Redis
        await self.redis.set_memory(memory.id, memory.model_dump(mode="json"))

        # Track Prometheus metrics
        try:
            from hippocampai.monitoring.prometheus_metrics import (
                memories_created_total,
                memory_operations_total,
                memory_size_bytes,
            )

            memory_operations_total.labels(operation="create", status="success").inc()
            memories_created_total.labels(memory_type=memory.type.value).inc()
            memory_size_bytes.labels(memory_type=memory.type.value).observe(len(text.encode("utf-8")))
        except Exception as metrics_err:
            logger.warning(f"Failed to track Prometheus metrics: {metrics_err}")

        logger.info(f"Created memory {memory.id} for user {user_id}")
        return memory

    async def get_memory(self, memory_id: str, track_access: bool = True) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory identifier
            track_access: Whether to track this access and update lifecycle metrics

        Returns:
            Memory object or None if not found
        """
        # Try Redis cache first
        cached = await self.redis.get_memory(memory_id)
        if cached:
            memory = Memory(**cached)
            if track_access and self.enable_lifecycle_management:
                # Update access count and track in background
                memory.access_count += 1
                await self._update_memory_access(memory)
            return memory

        # Fall back to vector store
        # Try both collections
        for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
            try:
                point = self.qdrant.get(collection_name=collection, id=memory_id)
                if point:
                    payload = point["payload"]
                    memory = Memory(**payload)

                    if track_access and self.enable_lifecycle_management:
                        # Update access count
                        memory.access_count += 1
                        await self._update_memory_access(memory)

                    # Cache in Redis
                    await self.redis.set_memory(memory_id, memory.model_dump())
                    return memory
            except Exception as e:
                logger.debug(f"Memory {memory_id} not found in {collection}: {e}")

        return None

    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """
        Update an existing memory.

        Args:
            memory_id: Memory identifier
            text: Optional new text
            importance: Optional new importance
            tags: Optional new tags
            metadata: Optional new metadata
            expires_at: Optional new expiration date

        Returns:
            Updated memory or None if not found
        """
        # Get existing memory
        memory = await self.get_memory(memory_id)
        if not memory:
            return None

        # Update fields
        if text is not None:
            memory.text = text
            memory.calculate_size_metrics()
        if importance is not None:
            memory.importance = importance
        if tags is not None:
            memory.tags = tags
        if metadata is not None:
            memory.metadata.update(metadata)
        if expires_at is not None:
            memory.expires_at = expires_at

        memory.updated_at = datetime.now(timezone.utc)

        # Update in vector store
        collection = memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )

        # Always re-upsert with re-embedding (simpler and ensures consistency)
        vector = self.embedder.encode_single(memory.text)
        self.qdrant.upsert(
            collection_name=collection,
            id=memory.id,
            vector=vector,
            payload=memory.model_dump(mode="json"),
        )

        # Update cache
        await self.redis.set_memory(memory.id, memory.model_dump(mode="json"))

        logger.info(f"Updated memory {memory_id}")
        return memory

    async def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier
            user_id: Optional user ID for authorization

        Returns:
            True if deleted, False otherwise
        """
        # Get memory to check ownership and determine collection
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        # Check authorization
        if user_id and memory.user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to delete memory {memory_id} of user {memory.user_id}"
            )
            return False

        # Delete from vector store
        collection = memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )
        self.qdrant.delete(collection_name=collection, ids=[memory_id])

        # Delete from cache
        await self.redis.delete_memory(memory_id)

        logger.info(f"Deleted memory {memory_id}")
        return True

    async def get_memories(
        self,
        user_id: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        memory_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance_min: Optional[float] = None,
        importance_max: Optional[float] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        search_text: Optional[str] = None,
    ) -> list[Memory]:
        """
        Get memories for a user with advanced filtering.

        Args:
            user_id: User identifier
            filters: Optional filters (type, tags, etc.)
            limit: Maximum number of memories to return
            memory_type: Filter by memory type
            tags: Filter by tags (memories with ANY of these tags)
            importance_min: Minimum importance score
            importance_max: Maximum importance score
            created_after: Filter memories created after this date
            created_before: Filter memories created before this date
            updated_after: Filter memories updated after this date
            updated_before: Filter memories updated before this date
            search_text: Text search in memory content

        Returns:
            List of memories
        """
        memories = []

        # Build filter
        query_filter = {"user_id": user_id}
        if filters:
            query_filter.update(filters)
        if memory_type:
            query_filter["type"] = memory_type

        # Query both collections
        for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
            try:
                results = self.qdrant.scroll(
                    collection_name=collection,
                    filters=query_filter,
                    limit=limit * 2,  # Get more for filtering
                )
                for result in results:
                    payload = result["payload"]
                    memory = Memory(**payload)

                    # Apply advanced filters
                    if tags and not any(tag in memory.tags for tag in tags):
                        continue

                    if importance_min is not None and memory.importance < importance_min:
                        continue

                    if importance_max is not None and memory.importance > importance_max:
                        continue

                    if created_after and memory.created_at < created_after:
                        continue

                    if created_before and memory.created_at > created_before:
                        continue

                    if updated_after and memory.updated_at < updated_after:
                        continue

                    if updated_before and memory.updated_at > updated_before:
                        continue

                    if search_text and search_text.lower() not in memory.text.lower():
                        continue

                    memories.append(memory)

            except Exception as e:
                logger.debug(f"Error querying {collection}: {e}")

        # Sort by created_at descending
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    async def batch_create_memories(
        self,
        memories: list[dict[str, Any]],
        check_duplicates: bool = True,
    ) -> list[Memory]:
        """
        Batch create multiple memories with parallel embeddings and bulk upsert (5-10x faster).

        Args:
            memories: List of memory data dictionaries
            check_duplicates: Whether to check for duplicates

        Returns:
            List of created memories
        """
        if not memories:
            return []

        # Phase 1: Create Memory objects
        memory_objects = []
        for mem_data in memories:
            # Build kwargs, only including optional fields if provided
            mem_kwargs = {
                "text": mem_data["text"],
                "user_id": mem_data["user_id"],
                "type": MemoryType(mem_data.get("type", "fact")),
                "tags": mem_data.get("tags", []),
                "metadata": mem_data.get("metadata", {}),
            }

            # Only include optional fields if they are provided
            if "session_id" in mem_data:
                mem_kwargs["session_id"] = mem_data["session_id"]
            if "importance" in mem_data:
                mem_kwargs["importance"] = mem_data["importance"]

            memory = Memory(**mem_kwargs)

            # Handle TTL
            if mem_data.get("ttl_days"):
                memory.expires_at = datetime.now(timezone.utc) + timedelta(
                    days=mem_data["ttl_days"]
                )

            memory_objects.append(memory)

        # Phase 2: Parallel embedding generation (5-10x faster than sequential)
        texts = [m.text for m in memory_objects]
        vectors = self.embedder.encode(texts)  # Batch encoding in parallel

        # Phase 3: Optionally check for duplicates
        if check_duplicates:
            # Note: Bulk duplicate checking can be optimized in future
            filtered_memories: list[Memory] = []
            filtered_vectors: list[np.ndarray] = []
            for memory, vector in zip(memory_objects, vectors):
                action, _ = self.deduplicator.check_duplicate(memory, memory.user_id)
                if action == "add":
                    filtered_memories.append(memory)
                    filtered_vectors.append(vector)
                else:
                    logger.debug(f"Skipping duplicate memory for user {memory.user_id}")
            memory_objects = filtered_memories
            vectors = np.array(filtered_vectors)

        if not memory_objects:
            logger.info("All memories were duplicates, skipping batch create")
            return []

        # Phase 4: Bulk upsert to Qdrant (3-5x faster than individual upserts)
        # Group by collection
        facts_memories = []
        facts_vectors = []
        facts_ids = []
        prefs_memories = []
        prefs_vectors = []
        prefs_ids = []

        for memory, vector in zip(memory_objects, vectors):
            collection = memory.collection_name(
                self.qdrant.collection_facts, self.qdrant.collection_prefs
            )
            if collection == self.qdrant.collection_facts:
                facts_memories.append(memory)
                facts_vectors.append(vector)
                facts_ids.append(memory.id)
            else:
                prefs_memories.append(memory)
                prefs_vectors.append(vector)
                prefs_ids.append(memory.id)

        # Bulk upsert facts
        if facts_memories:
            self.qdrant.bulk_upsert(
                collection_name=self.qdrant.collection_facts,
                ids=facts_ids,
                vectors=facts_vectors,
                payloads=[m.model_dump(mode="json") for m in facts_memories],
            )

        # Bulk upsert preferences
        if prefs_memories:
            self.qdrant.bulk_upsert(
                collection_name=self.qdrant.collection_prefs,
                ids=prefs_ids,
                vectors=prefs_vectors,
                payloads=[m.model_dump(mode="json") for m in prefs_memories],
            )

        # Phase 5: Bulk cache to Redis
        await self.redis.batch_set_memories(
            [(m.id, m.model_dump(mode="json")) for m in memory_objects]
        )

        logger.info(
            f"Batch created {len(memory_objects)} memories ({len(facts_memories)} facts, {len(prefs_memories)} prefs) using parallel embeddings + bulk upsert"
        )
        return memory_objects

    async def batch_update_memories(self, updates: list[dict[str, Any]]) -> list[Optional[Memory]]:
        """
        Batch update multiple memories.

        Args:
            updates: List of update data (must include memory_id)

        Returns:
            List of updated memories
        """
        updated_memories = []

        for update_data in updates:
            memory_id = update_data.pop("memory_id")
            memory = await self.update_memory(memory_id=memory_id, **update_data)
            updated_memories.append(memory)

        logger.info(f"Batch updated {len(updated_memories)} memories")
        return updated_memories

    async def batch_delete_memories(
        self, memory_ids: list[str], user_id: Optional[str] = None
    ) -> dict[str, bool]:
        """
        Batch delete multiple memories.

        Args:
            memory_ids: List of memory IDs to delete
            user_id: Optional user ID for authorization

        Returns:
            Dictionary mapping memory_id to deletion success
        """
        results = {}

        for memory_id in memory_ids:
            success = await self.delete_memory(memory_id, user_id)
            results[memory_id] = success

        logger.info(f"Batch deleted {sum(results.values())}/{len(memory_ids)} memories")
        return results

    async def recall_memories(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> list[RetrievalResult]:
        """
        Recall memories using hybrid search with customizable weights.
        Results are cached for 60 seconds to improve performance on repeated queries.

        Args:
            query: Search query
            user_id: User identifier
            session_id: Optional session identifier
            k: Number of results to return
            filters: Optional filters
            custom_weights: Optional custom scoring weights

        Returns:
            List of retrieval results
        """
        # Generate cache key
        cache_key = self._generate_query_cache_key(
            query, user_id, session_id, k, filters, custom_weights
        )

        # Try to get from cache
        cached_results = await self.redis.store.get(cache_key)
        if cached_results:
            logger.debug(f"Query cache hit for key {cache_key}")
            # Deserialize results
            return [RetrievalResult(**result) for result in cached_results]

        # Cache miss - perform search
        logger.debug(f"Query cache miss for key {cache_key}")

        # Update weights if provided
        if custom_weights:
            original_weights = self.retriever.weights.copy()
            self.retriever.weights.update(custom_weights)

        try:
            results = self.retriever.retrieve(
                query=query,
                user_id=user_id,
                session_id=session_id,
                k=k,
                filters=filters,
            )

            # Cache the results
            serialized_results = [result.model_dump(mode="json") for result in results]
            await self.redis.store.set(
                cache_key, serialized_results, ttl_seconds=self.query_cache_ttl
            )

            # Track Prometheus metrics
            try:
                from hippocampai.monitoring.prometheus_metrics import (
                    search_requests_total,
                    search_results_count,
                )

                search_requests_total.labels(search_type="hybrid", status="success").inc()
                search_results_count.observe(len(results))
            except Exception as metrics_err:
                logger.warning(f"Failed to track Prometheus search metrics: {metrics_err}")

            return results
        finally:
            # Restore original weights
            if custom_weights:
                self.retriever.weights = original_weights

    async def deduplicate_user_memories(
        self, user_id: str, dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Deduplicate memories for a user.

        Args:
            user_id: User identifier
            dry_run: If True, only report duplicates without deleting

        Returns:
            Dictionary with deduplication results
        """
        # Get all memories for user
        memories = await self.get_memories(user_id, limit=10000)

        duplicates_found = []
        memories_removed = 0

        for memory in memories:
            action, duplicate_ids = self.deduplicator.check_duplicate(memory, user_id)

            if action in ["skip", "update"] and duplicate_ids:
                duplicates_found.append(
                    {"memory_id": memory.id, "action": action, "duplicates": duplicate_ids}
                )

                if not dry_run and action == "skip":
                    # Delete the duplicate
                    await self.delete_memory(memory.id)
                    memories_removed += 1

        result = {
            "user_id": user_id,
            "total_memories": len(memories),
            "duplicates_found": len(duplicates_found),
            "memories_removed": memories_removed if not dry_run else 0,
            "dry_run": dry_run,
            "details": duplicates_found[:100],  # Limit details
        }

        logger.info(f"Deduplication for user {user_id}: {result}")
        return result

    async def consolidate_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Consolidate similar memories for a user.

        Args:
            user_id: User identifier
            similarity_threshold: Threshold for considering memories similar
            dry_run: If True, only report consolidations without executing

        Returns:
            Dictionary with consolidation results
        """
        # Get all memories for user
        memories = await self.get_memories(user_id, limit=10000)

        consolidation_groups = []
        consolidated_count = 0

        # Group similar memories
        processed = set()
        for i, memory in enumerate(memories):
            if memory.id in processed:
                continue

            similar_group = [memory]
            for j, other_memory in enumerate(memories[i + 1 :], start=i + 1):
                if other_memory.id in processed:
                    continue

                # Check similarity using embeddings
                vec1 = self.embedder.encode_single(memory.text)
                vec2 = self.embedder.encode_single(other_memory.text)

                # Simple cosine similarity
                import numpy as np

                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                if similarity >= similarity_threshold:
                    similar_group.append(other_memory)
                    processed.add(other_memory.id)

            if len(similar_group) >= 2:
                consolidation_groups.append(similar_group)

                if not dry_run:
                    # Consolidate the group
                    consolidated = self.consolidator.consolidate(similar_group)
                    if consolidated:
                        # Create consolidated memory
                        await self.create_memory(
                            text=consolidated.text,
                            user_id=user_id,
                            memory_type=consolidated.type.value,
                            importance=consolidated.importance,
                            tags=consolidated.tags,
                            metadata=consolidated.metadata,
                            check_duplicate=False,
                        )

                        # Delete original memories
                        for mem in similar_group:
                            await self.delete_memory(mem.id)

                        consolidated_count += 1

            processed.add(memory.id)

        result = {
            "user_id": user_id,
            "total_memories": len(memories),
            "groups_found": len(consolidation_groups),
            "memories_consolidated": consolidated_count if not dry_run else 0,
            "dry_run": dry_run,
        }

        logger.info(f"Consolidation for user {user_id}: {result}")
        return result

    async def extract_from_conversation(
        self,
        conversation: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
        """
        Extract memories from a conversation log.

        Args:
            conversation: Conversation text
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            List of extracted memories
        """
        if not self.llm:
            raise ValueError("LLM required for conversation extraction")

        # Extract memories using LLM
        prompt = f"""Extract important memories from this conversation.
Return a JSON array of memories with fields: text, type (preference/fact/goal/habit/event/context), importance (0-10), tags.

Conversation:
{conversation[:2000]}

JSON:"""

        try:
            import json

            response = self.llm.generate(prompt, max_tokens=1000, temperature=0.3)
            memories_data = json.loads(response)

            extracted_memories = []
            for mem_data in memories_data:
                memory = await self.create_memory(
                    text=mem_data["text"],
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=mem_data.get("type", "fact"),
                    importance=mem_data.get("importance", 5.0),
                    tags=mem_data.get("tags", []),
                    metadata={"extracted_from": "conversation"},
                )
                extracted_memories.append(memory)

            logger.info(f"Extracted {len(extracted_memories)} memories from conversation")
            return extracted_memories

        except Exception as e:
            logger.error(f"Failed to extract memories from conversation: {e}")
            return []

    async def expire_memories(self, user_id: Optional[str] = None) -> int:
        """
        Expire memories based on TTL.

        Args:
            user_id: Optional user ID to limit expiration

        Returns:
            Number of expired memories
        """
        if user_id:
            memories = await self.get_memories(user_id, limit=10000)
        else:
            # Get all memories (expensive operation)
            all_memories = []
            for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
                results = self.qdrant.scroll(collection_name=collection, limit=100000)
                all_memories.extend([Memory(**r["payload"]) for r in results])
            memories = all_memories

        expired_count = 0
        now = datetime.now(timezone.utc)

        for memory in memories:
            if memory.expires_at and now > memory.expires_at:
                await self.delete_memory(memory.id)
                expired_count += 1

        logger.info(f"Expired {expired_count} memories")
        return expired_count

    async def detect_memory_conflicts(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        check_llm: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect conflicts in existing memories for a user.

        Args:
            user_id: User identifier
            memory_type: Optional memory type filter
            check_llm: Whether to use LLM for deep contradiction analysis

        Returns:
            List of detected conflicts with details
        """
        # Get user memories
        memories = await self.get_memories(user_id=user_id, memory_type=memory_type, limit=1000)

        if len(memories) < 2:
            logger.info(f"No conflicts possible with {len(memories)} memories")
            return []

        # Detect conflicts
        conflicts = self.conflict_resolver.batch_detect_conflicts(memories, check_llm=check_llm)

        # Format results
        conflict_details = []
        for conflict in conflicts:
            conflict_details.append(
                {
                    "conflict_id": conflict.id,
                    "memory_1": {
                        "id": conflict.memory_1.id,
                        "text": conflict.memory_1.text,
                        "created_at": conflict.memory_1.created_at.isoformat(),
                        "confidence": conflict.memory_1.confidence,
                        "importance": conflict.memory_1.importance,
                    },
                    "memory_2": {
                        "id": conflict.memory_2.id,
                        "text": conflict.memory_2.text,
                        "created_at": conflict.memory_2.created_at.isoformat(),
                        "confidence": conflict.memory_2.confidence,
                        "importance": conflict.memory_2.importance,
                    },
                    "conflict_type": conflict.conflict_type,
                    "confidence_score": conflict.confidence_score,
                    "similarity_score": conflict.similarity_score,
                    "detected_at": conflict.detected_at.isoformat(),
                }
            )

        logger.info(f"Detected {len(conflicts)} conflicts for user {user_id}")
        return conflict_details

    async def resolve_memory_conflicts(
        self,
        user_id: str,
        strategy: Optional[ConflictResolutionStrategy] = None,
        memory_type: Optional[str] = None,
        dry_run: bool = False,
        check_llm: bool = True,
    ) -> dict[str, Any]:
        """
        Detect and resolve conflicts in existing memories.

        Args:
            user_id: User identifier
            strategy: Resolution strategy (uses default if not specified)
            memory_type: Optional memory type filter
            dry_run: If True, only report conflicts without resolving
            check_llm: Whether to use LLM for deep contradiction analysis

        Returns:
            Dictionary with resolution results
        """
        # Get user memories
        memories = await self.get_memories(user_id=user_id, memory_type=memory_type, limit=1000)

        if len(memories) < 2:
            return {
                "user_id": user_id,
                "total_memories": len(memories),
                "conflicts_found": 0,
                "conflicts_resolved": 0,
                "dry_run": dry_run,
            }

        # Detect conflicts
        conflicts = self.conflict_resolver.batch_detect_conflicts(memories, check_llm=check_llm)

        if not conflicts:
            return {
                "user_id": user_id,
                "total_memories": len(memories),
                "conflicts_found": 0,
                "conflicts_resolved": 0,
                "dry_run": dry_run,
            }

        logger.info(f"Found {len(conflicts)} conflicts for user {user_id}")

        resolved_count = 0
        memories_deleted = 0
        memories_updated = 0
        memories_flagged = 0
        resolution_details = []

        if not dry_run:
            for conflict in conflicts:
                resolution = self.conflict_resolver.resolve_conflict(
                    conflict, strategy=strategy or self.conflict_resolution_strategy
                )

                # Apply resolution
                if resolution.action == "keep_first":
                    # Delete second memory
                    await self.delete_memory(conflict.memory_2.id)
                    memories_deleted += 1
                    resolved_count += 1

                elif resolution.action == "keep_second":
                    # Delete first memory
                    await self.delete_memory(conflict.memory_1.id)
                    memories_deleted += 1
                    resolved_count += 1

                elif resolution.action == "merge":
                    # Create merged memory and delete originals
                    if resolution.updated_memory:
                        await self.create_memory(
                            text=resolution.updated_memory.text,
                            user_id=user_id,
                            memory_type=resolution.updated_memory.type.value,
                            importance=resolution.updated_memory.importance,
                            tags=resolution.updated_memory.tags,
                            metadata=resolution.updated_memory.metadata,
                            check_duplicate=False,
                            check_conflicts=False,
                        )
                        memories_updated += 1

                    # Delete original memories
                    for mem_id in resolution.deleted_memory_ids:
                        await self.delete_memory(mem_id)
                        memories_deleted += 1

                    resolved_count += 1

                elif resolution.action in ["flag", "keep_both"]:
                    # Update both memories with conflict flags
                    await self.update_memory(
                        memory_id=conflict.memory_1.id,
                        metadata=conflict.memory_1.metadata,
                    )
                    await self.update_memory(
                        memory_id=conflict.memory_2.id,
                        metadata=conflict.memory_2.metadata,
                    )
                    memories_flagged += 2
                    resolved_count += 1

                resolution_details.append(
                    {
                        "conflict_id": conflict.id,
                        "action": resolution.action,
                        "notes": resolution.notes,
                    }
                )

        result = {
            "user_id": user_id,
            "total_memories": len(memories),
            "conflicts_found": len(conflicts),
            "conflicts_resolved": resolved_count if not dry_run else 0,
            "memories_deleted": memories_deleted if not dry_run else 0,
            "memories_updated": memories_updated if not dry_run else 0,
            "memories_flagged": memories_flagged if not dry_run else 0,
            "dry_run": dry_run,
            "resolution_strategy": (strategy or self.conflict_resolution_strategy).value,
            "details": resolution_details[:50],  # Limit details
        }

        logger.info(f"Conflict resolution for user {user_id}: {result}")
        return result

    async def _update_memory_access(self, memory: Memory) -> None:
        """
        Update memory access patterns and lifecycle tier.

        Args:
            memory: Memory object with updated access_count
        """
        try:
            # Calculate temperature
            temperature = self.lifecycle_manager.calculate_temperature(
                memory_id=memory.id,
                created_at=memory.created_at,
                access_count=memory.access_count,
                last_access=datetime.now(timezone.utc),
                importance=memory.importance,
            )

            # Get current tier from metadata
            current_tier_str = memory.metadata.get("lifecycle", {}).get("tier")
            current_tier = (
                MemoryTier(current_tier_str) if current_tier_str else MemoryTier.WARM
            )

            # Update metadata with lifecycle info
            memory_data = memory.model_dump()
            memory_data = self.lifecycle_manager.update_access_metadata(
                memory_data, temperature
            )

            # Check if tier migration is needed
            if self.lifecycle_manager.should_migrate(current_tier, temperature.tier):
                logger.info(
                    f"Memory {memory.id} tier change: {current_tier.value} -> {temperature.tier.value} "
                    f"(temp: {temperature.temperature_score:.1f})"
                )

            # Update in vector store
            collection = memory.collection_name(
                self.qdrant.collection_facts, self.qdrant.collection_prefs
            )
            self.qdrant.update(
                collection_name=collection,
                id=memory.id,
                payload=memory_data,
            )

            # Update cache
            await self.redis.set_memory(memory.id, memory_data)

            logger.debug(
                f"Updated memory {memory.id} access: count={memory.access_count}, "
                f"tier={temperature.tier.value}, temp={temperature.temperature_score:.1f}"
            )

        except Exception as e:
            logger.error(f"Failed to update memory access for {memory.id}: {e}")

    async def get_memory_temperature(self, memory_id: str) -> Optional[dict[str, Any]]:
        """
        Get temperature metrics for a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Temperature metrics dictionary or None if memory not found
        """
        memory = await self.get_memory(memory_id, track_access=False)
        if not memory:
            return None

        temperature = self.lifecycle_manager.calculate_temperature(
            memory_id=memory.id,
            created_at=memory.created_at,
            access_count=memory.access_count,
            last_access=memory.metadata.get("lifecycle", {}).get("last_tier_update"),
            importance=memory.importance,
        )

        return temperature.model_dump()

    async def migrate_memory_tier(
        self, memory_id: str, target_tier: MemoryTier
    ) -> bool:
        """
        Manually migrate a memory to a specific tier.

        Args:
            memory_id: Memory identifier
            target_tier: Target storage tier

        Returns:
            True if migration successful
        """
        memory = await self.get_memory(memory_id, track_access=False)
        if not memory:
            logger.warning(f"Memory {memory_id} not found for tier migration")
            return False

        # Calculate current temperature
        temperature = self.lifecycle_manager.calculate_temperature(
            memory_id=memory.id,
            created_at=memory.created_at,
            access_count=memory.access_count,
            last_access=memory.metadata.get("lifecycle", {}).get("last_tier_update"),
            importance=memory.importance,
        )

        # Override recommended tier with target tier
        temperature.tier = target_tier

        # Update metadata
        memory_data = memory.model_dump()
        memory_data = self.lifecycle_manager.update_access_metadata(
            memory_data, temperature
        )

        # Handle compression for archived/hibernated tiers
        if target_tier in {MemoryTier.ARCHIVED, MemoryTier.HIBERNATED}:
            if self.lifecycle_manager.config.compress_archived or (
                target_tier == MemoryTier.HIBERNATED
                and self.lifecycle_manager.config.compress_hibernated
            ):
                # Store compression flag in metadata
                memory_data["metadata"]["compressed"] = True
                logger.info(f"Memory {memory_id} will be compressed in tier {target_tier.value}")

        # Update in vector store
        collection = memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )
        self.qdrant.update(
            collection_name=collection, id=memory.id, payload=memory_data
        )

        # Update cache
        await self.redis.set_memory(memory.id, memory_data)

        logger.info(f"Migrated memory {memory_id} to tier {target_tier.value}")
        return True

    async def get_tier_statistics(self, user_id: str) -> dict[str, Any]:
        """
        Get statistics about memory tiers for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with tier statistics
        """
        # Get all user memories
        memories = await self.get_memories(user_id=user_id, limit=10000)

        tier_counts = {tier.value: 0 for tier in MemoryTier}
        tier_temperatures = {tier.value: [] for tier in MemoryTier}
        total_size = 0
        total_accesses = 0

        for mem in memories:
            # Calculate temperature
            temperature = self.lifecycle_manager.calculate_temperature(
                memory_id=mem.id,
                created_at=mem.created_at,
                access_count=mem.access_count,
                last_access=mem.metadata.get("lifecycle", {}).get("last_tier_update"),
                importance=mem.importance,
            )

            tier = temperature.tier.value
            tier_counts[tier] += 1
            tier_temperatures[tier].append(temperature.temperature_score)
            total_size += mem.text_length
            total_accesses += mem.access_count

        # Calculate average temperatures
        tier_avg_temps = {}
        for tier, temps in tier_temperatures.items():
            tier_avg_temps[tier] = sum(temps) / len(temps) if temps else 0.0

        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "total_size_bytes": total_size,
            "total_accesses": total_accesses,
            "tier_counts": tier_counts,
            "tier_average_temperatures": tier_avg_temps,
            "lifecycle_config": self.lifecycle_manager.config.model_dump(),
        }

    # ========================================================================
    # MEMORY QUALITY & HEALTH MONITORING
    # ========================================================================

    async def assess_memory_health(self, memory_id: str) -> Optional[MemoryHealthScore]:
        """
        Assess health of a single memory.

        Args:
            memory_id: Memory identifier

        Returns:
            MemoryHealthScore or None if memory not found
        """
        try:
            # Get memory from cache or vector store
            memory_data = await self.redis.get_memory(memory_id)
            if not memory_data:
                # Try vector store
                results = await self.qdrant.get(self.qdrant.collection_facts, memory_id)
                if not results:
                    results = await self.qdrant.get(
                        self.qdrant.collection_prefs, memory_id
                    )
                if not results:
                    return None
                memory_data = results

            # Parse to Memory object
            memory = Memory(**memory_data)

            # Assess health
            health = self.quality_monitor.assess_memory_health(
                memory_id=memory.id,
                text=memory.text,
                confidence=memory.confidence,
                importance=memory.importance,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
                tags=memory.tags,
                metadata=memory.metadata,
            )

            return health

        except Exception as e:
            logger.error(f"Failed to assess memory health for {memory_id}: {e}")
            return None

    async def detect_duplicates(
        self, user_id: str, similarity_threshold: Optional[float] = None
    ) -> list[DuplicateCluster]:
        """
        Detect duplicate and near-duplicate memory clusters.

        Args:
            user_id: User identifier
            similarity_threshold: Custom threshold (uses default if not provided)

        Returns:
            List of duplicate clusters
        """
        try:
            # Get all user memories
            memories = await self.get_memories(user_id=user_id, limit=10000)

            if len(memories) < 2:
                return []

            # Build similarity matrix using embeddings
            similarity_matrix: dict[tuple[str, str], float] = {}

            logger.info(
                f"Computing similarity matrix for {len(memories)} memories..."
            )

            for i, mem1 in enumerate(memories):
                for j, mem2 in enumerate(memories[i + 1 :], start=i + 1):
                    # Use embeddings if available
                    if mem1.embedding and mem2.embedding:
                        # Cosine similarity
                        dot_product = np.dot(mem1.embedding, mem2.embedding)
                        norm1 = np.linalg.norm(mem1.embedding)
                        norm2 = np.linalg.norm(mem2.embedding)
                        similarity = (
                            dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                        )
                    else:
                        # Fallback: use text similarity via reranker
                        sim_result = self.deduplicator.reranker.rerank(
                            query=mem1.text, documents=[mem2.text]
                        )
                        similarity = sim_result[0].score if sim_result else 0.0

                    if similarity >= (
                        similarity_threshold
                        or self.quality_monitor.near_duplicate_threshold
                    ):
                        similarity_matrix[(mem1.id, mem2.id)] = similarity

            # Detect clusters
            memory_dicts = [
                {"id": m.id, "text": m.text, "confidence": m.confidence}
                for m in memories
            ]
            clusters = self.quality_monitor.detect_duplicate_clusters(
                memories=memory_dicts, similarity_matrix=similarity_matrix
            )

            logger.info(f"Found {len(clusters)} duplicate clusters for user {user_id}")

            return clusters

        except Exception as e:
            logger.error(f"Failed to detect duplicates for user {user_id}: {e}")
            return []

    async def analyze_topic_coverage(
        self, user_id: str, topics: Optional[list[str]] = None
    ) -> list[TopicCoverage]:
        """
        Analyze memory coverage across topics.

        Args:
            user_id: User identifier
            topics: Optional predefined topics (extracts from tags if not provided)

        Returns:
            List of topic coverage analysis
        """
        try:
            # Get all user memories
            memories = await self.get_memories(user_id=user_id, limit=10000)

            if not memories:
                return []

            # Convert to dicts for analyzer
            memory_dicts = [
                {
                    "id": m.id,
                    "text": m.text,
                    "tags": m.tags,
                    "confidence": m.confidence,
                    "updated_at": m.updated_at,
                }
                for m in memories
            ]

            # Analyze coverage
            coverage = self.quality_monitor.analyze_topic_coverage(
                memories=memory_dicts, topics=topics
            )

            logger.info(f"Analyzed coverage for {len(coverage)} topics for user {user_id}")

            return coverage

        except Exception as e:
            logger.error(f"Failed to analyze topic coverage for user {user_id}: {e}")
            return []

    async def assess_memory_store_health(
        self, user_id: str
    ) -> Optional[MemoryStoreHealth]:
        """
        Assess overall health of user's memory store.

        Args:
            user_id: User identifier

        Returns:
            MemoryStoreHealth with comprehensive assessment
        """
        try:
            # Get all user memories
            memories = await self.get_memories(user_id=user_id, limit=10000)

            if not memories:
                return MemoryStoreHealth(
                    user_id=user_id,
                    overall_health_score=0.0,
                    health_status="critical",
                    recommendations=["No memories found in store."],
                )

            # Assess each memory
            health_scores: list[MemoryHealthScore] = []
            for memory in memories:
                health = self.quality_monitor.assess_memory_health(
                    memory_id=memory.id,
                    text=memory.text,
                    confidence=memory.confidence,
                    importance=memory.importance,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at,
                    tags=memory.tags,
                    metadata=memory.metadata,
                )
                health_scores.append(health)

            # Detect duplicates
            duplicate_clusters = await self.detect_duplicates(user_id)

            # Analyze topic coverage
            topic_coverage = await self.analyze_topic_coverage(user_id)

            # Assess overall store health
            store_health = self.quality_monitor.assess_memory_store_health(
                user_id=user_id,
                memory_health_scores=health_scores,
                duplicate_clusters=duplicate_clusters,
                topic_coverage=topic_coverage,
            )

            logger.info(
                f"Memory store health for user {user_id}: "
                f"{store_health.health_status.value} ({store_health.overall_health_score:.1f}/100)"
            )

            return store_health

        except Exception as e:
            logger.error(f"Failed to assess memory store health for user {user_id}: {e}")
            return None

    async def get_stale_memories(
        self, user_id: str, threshold_days: Optional[int] = None
    ) -> list[Memory]:
        """
        Get memories that are stale (not updated recently).

        Args:
            user_id: User identifier
            threshold_days: Custom threshold (uses default if not provided)

        Returns:
            List of stale memories
        """
        try:
            memories = await self.get_memories(user_id=user_id, limit=10000)
            threshold = threshold_days or self.quality_monitor.stale_threshold_days

            stale_memories = []
            now = datetime.now(timezone.utc)

            for memory in memories:
                updated_at = memory.updated_at
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)

                days_since_update = (now - updated_at).total_seconds() / 86400

                if days_since_update > threshold:
                    stale_memories.append(memory)

            logger.info(
                f"Found {len(stale_memories)} stale memories "
                f"(>{threshold} days old) for user {user_id}"
            )

            return stale_memories

        except Exception as e:
            logger.error(f"Failed to get stale memories for user {user_id}: {e}")
            return []

    # Conversation-Aware Memory Methods

    async def add_conversation_turn(
        self,
        conversation_id: str,
        session_id: str,
        user_id: str,
        speaker: str,
        text: str,
        role: SpeakerRole = SpeakerRole.USER,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Add a turn to a conversation and track it.

        Args:
            conversation_id: Unique conversation identifier
            session_id: Session identifier
            user_id: User identifier
            speaker: Name of the speaker
            text: Content of the turn
            role: Speaker role (USER, ASSISTANT, SYSTEM, PARTICIPANT)
            metadata: Optional metadata

        Returns:
            Parsed conversation turn
        """
        try:
            # Parse the turn
            turn = self.conversation_manager.parse_turn(
                text=text,
                speaker=speaker,
                role=role,
                conversation_id=conversation_id,
                metadata=metadata,
            )

            # Add to conversation context
            self.conversation_manager.add_turn(
                conversation_id=conversation_id,
                session_id=session_id,
                user_id=user_id,
                turn=turn,
            )

            logger.info(
                f"Added turn {turn.turn_id} to conversation {conversation_id} "
                f"by {speaker} ({role})"
            )

            return turn

        except Exception as e:
            logger.error(
                f"Failed to add conversation turn to {conversation_id}: {e}"
            )
            raise

    async def get_conversation_summary(
        self, conversation_id: str, use_llm: bool = True
    ) -> Optional[ConversationSummary]:
        """
        Get a multi-level summary of a conversation.

        Args:
            conversation_id: Conversation identifier
            use_llm: Whether to use LLM for better summarization

        Returns:
            Conversation summary with key points, decisions, action items, etc.
        """
        try:
            summary = self.conversation_manager.summarize_conversation(
                conversation_id=conversation_id, use_llm=use_llm
            )

            if summary:
                logger.info(
                    f"Generated summary for conversation {conversation_id}: "
                    f"{summary.total_turns} turns, {len(summary.key_points)} key points, "
                    f"{len(summary.decisions)} decisions, {len(summary.action_items)} actions"
                )

            return summary

        except Exception as e:
            logger.error(
                f"Failed to get conversation summary for {conversation_id}: {e}"
            )
            return None

    async def extract_conversation_memories(
        self,
        conversation_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        auto_tag: bool = True,
    ) -> list[Memory]:
        """
        Extract and store memories from a conversation.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            session_id: Optional session identifier
            auto_tag: Whether to auto-tag with conversation topics

        Returns:
            List of created memories
        """
        try:
            summary = await self.get_conversation_summary(
                conversation_id=conversation_id, use_llm=True
            )

            if not summary:
                logger.warning(
                    f"No summary available for conversation {conversation_id}"
                )
                return []

            created_memories = []

            # Create memory from key points
            for idx, point in enumerate(summary.key_points):
                tags = summary.topics_discussed if auto_tag else []
                tags.append(f"conversation:{conversation_id}")

                memory = await self.create_memory(
                    text=point,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="fact",
                    importance=7.0,
                    tags=tags,
                    metadata={
                        "source": "conversation",
                        "conversation_id": conversation_id,
                        "key_point_index": idx,
                    },
                )
                created_memories.append(memory)

            # Create memories from decisions
            for decision in summary.decisions:
                tags = summary.topics_discussed if auto_tag else []
                tags.extend(["decision", f"conversation:{conversation_id}"])

                memory = await self.create_memory(
                    text=f"Decision: {decision.decision}. Rationale: {decision.rationale}",
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="fact",
                    importance=9.0,
                    tags=tags,
                    metadata={
                        "source": "conversation_decision",
                        "conversation_id": conversation_id,
                        "decision_maker": decision.decision_maker,
                        "topic": decision.topic,
                    },
                )
                created_memories.append(memory)

            # Create memories from action items
            for action in summary.action_items:
                tags = summary.topics_discussed if auto_tag else []
                tags.extend(["action_item", f"conversation:{conversation_id}"])

                memory = await self.create_memory(
                    text=f"Action: {action.action} (Priority: {action.priority})",
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="task",
                    importance=8.0,
                    tags=tags,
                    metadata={
                        "source": "conversation_action",
                        "conversation_id": conversation_id,
                        "assignee": action.assignee,
                        "deadline": action.deadline.isoformat()
                        if action.deadline
                        else None,
                        "status": action.status,
                    },
                )
                created_memories.append(memory)

            logger.info(
                f"Extracted {len(created_memories)} memories from conversation {conversation_id}"
            )

            return created_memories

        except Exception as e:
            logger.error(
                f"Failed to extract memories from conversation {conversation_id}: {e}"
            )
            return []

    # Memory Merge Methods

    async def suggest_memory_merges(
        self, user_id: str, limit: int = 100
    ) -> list[MergeCandidate]:
        """
        Suggest memory merges based on similarity.

        Args:
            user_id: User identifier
            limit: Maximum number of memories to analyze

        Returns:
            List of merge candidates with similarity scores and conflicts
        """
        try:
            # Get recent memories
            memories = await self.get_memories(user_id=user_id, limit=limit)

            if len(memories) < 2:
                logger.info(f"Not enough memories for user {user_id} to suggest merges")
                return []

            # Convert to dict format for merge engine
            memory_dicts = []
            for mem in memories:
                memory_dicts.append(
                    {
                        "id": mem.id,
                        "text": mem.text,
                        "type": mem.type.value,
                        "importance": mem.importance,
                        "confidence": mem.confidence,
                        "tags": mem.tags,
                        "metadata": mem.metadata,
                        "created_at": mem.created_at,
                        "updated_at": mem.updated_at,
                    }
                )

            # Calculate similarity matrix
            similarity_matrix = {}
            for i, mem1 in enumerate(memory_dicts):
                for j, mem2 in enumerate(memory_dicts):
                    if i < j:
                        # Get embeddings and calculate similarity
                        result1 = await self.retriever.search(
                            query=mem1["text"],
                            user_id=user_id,
                            k=1,
                            filters={"id": mem1["id"]},
                        )
                        result2 = await self.retriever.search(
                            query=mem2["text"],
                            user_id=user_id,
                            k=1,
                            filters={"id": mem2["id"]},
                        )

                        if result1 and result2:
                            # Simple cosine similarity between embeddings
                            emb1 = result1[0].metadata.get("embedding", [])
                            emb2 = result2[0].metadata.get("embedding", [])
                            if emb1 and emb2:
                                similarity = float(
                                    np.dot(emb1, emb2)
                                    / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                                )
                                similarity_matrix[(mem1["id"], mem2["id"])] = (
                                    similarity
                                )

            # Get merge suggestions
            candidates = self.merge_engine.suggest_merges(
                memories=memory_dicts, similarity_matrix=similarity_matrix
            )

            logger.info(
                f"Found {len(candidates)} merge candidates for user {user_id}"
            )

            return candidates

        except Exception as e:
            logger.error(f"Failed to suggest merges for user {user_id}: {e}")
            return []

    async def merge_memories(
        self,
        memory_ids: list[str],
        user_id: str,
        strategy: Optional[MergeStrategy] = None,
        manual_resolutions: Optional[dict[str, Any]] = None,
    ) -> Optional[MergeResult]:
        """
        Merge multiple memories into one.

        Args:
            memory_ids: List of memory IDs to merge
            user_id: User identifier
            strategy: Merge strategy (NEWEST, HIGHEST_CONFIDENCE, LONGEST, COMBINE, MANUAL)
            manual_resolutions: Manual conflict resolutions

        Returns:
            Merge result with merged memory ID and metrics
        """
        try:
            if len(memory_ids) < 2:
                logger.warning("Need at least 2 memories to merge")
                return None

            # Fetch memories
            memories = []
            for mem_id in memory_ids:
                mem = await self.get_memory(memory_id=mem_id)
                if mem and mem.user_id == user_id:
                    memories.append(
                        {
                            "id": mem.id,
                            "text": mem.text,
                            "type": mem.type.value,
                            "importance": mem.importance,
                            "confidence": mem.confidence,
                            "tags": mem.tags,
                            "metadata": mem.metadata,
                            "created_at": mem.created_at,
                            "updated_at": mem.updated_at,
                        }
                    )

            if len(memories) != len(memory_ids):
                logger.error("Some memories not found or don't belong to user")
                return None

            # Execute merge
            result = self.merge_engine.merge_memories(
                memories=memories,
                strategy=strategy,
                user_id=user_id,
                manual_resolutions=manual_resolutions,
            )

            if result.success:
                # Create the merged memory in the store
                merged_data = result.rollback_data.get("merged_memory", {})

                merged_memory = await self.create_memory(
                    text=merged_data["text"],
                    user_id=user_id,
                    memory_type=merged_data["type"],
                    importance=merged_data["importance"],
                    tags=merged_data["tags"],
                    metadata=merged_data.get("metadata", {}),
                    check_duplicate=False,  # Already verified
                )

                # Delete original memories
                for mem_id in memory_ids:
                    await self.delete_memory(memory_id=mem_id)

                # Update result with actual merged memory ID
                result.merged_memory_id = merged_memory.id

                logger.info(
                    f"Successfully merged {len(memory_ids)} memories into {merged_memory.id} "
                    f"using {result.strategy_used} strategy"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to merge memories: {e}")
            return None

    async def preview_memory_merge(
        self,
        memory_ids: list[str],
        user_id: str,
        strategy: Optional[MergeStrategy] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Preview what a memory merge would look like without executing it.

        Args:
            memory_ids: List of memory IDs to preview merge
            user_id: User identifier
            strategy: Optional merge strategy

        Returns:
            Preview of merged memory with conflicts and quality metrics
        """
        try:
            if len(memory_ids) < 2:
                return None

            # Fetch memories
            memories = []
            for mem_id in memory_ids:
                mem = await self.get_memory(memory_id=mem_id)
                if mem and mem.user_id == user_id:
                    memories.append(
                        {
                            "id": mem.id,
                            "text": mem.text,
                            "type": mem.type.value,
                            "importance": mem.importance,
                            "confidence": mem.confidence,
                            "tags": mem.tags,
                            "metadata": mem.metadata,
                            "created_at": mem.created_at,
                            "updated_at": mem.updated_at,
                        }
                    )

            if len(memories) != len(memory_ids):
                return None

            # Get preview
            preview = self.merge_engine.preview_merge(
                memories=memories, strategy=strategy
            )

            logger.info(f"Generated merge preview for {len(memory_ids)} memories")

            return preview

        except Exception as e:
            logger.error(f"Failed to preview merge: {e}")
            return None
