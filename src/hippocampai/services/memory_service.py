"""Comprehensive memory management service with CRUD, batch, dedup, and consolidation."""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.dedup import MemoryDeduplicator
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
        """
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.redis = redis_store
        self.llm = llm

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
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
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
    ) -> Memory:
        """
        Create a new memory with optional deduplication.

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

        logger.info(f"Created memory {memory.id} for user {user_id}")
        return memory

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory object or None if not found
        """
        # Try Redis cache first
        cached = await self.redis.get_memory(memory_id)
        if cached:
            return Memory(**cached)

        # Fall back to vector store
        # Try both collections
        for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
            try:
                points = self.qdrant.retrieve(collection_name=collection, ids=[memory_id])
                if points:
                    payload = points[0]["payload"]
                    # Cache in Redis
                    await self.redis.set_memory(memory_id, payload)
                    return Memory(**payload)
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
            filtered_memories = []
            filtered_vectors = []
            for memory, vector in zip(memory_objects, vectors):
                action, _ = self.deduplicator.check_duplicate(memory, memory.user_id)
                if action == "add":
                    filtered_memories.append(memory)
                    filtered_vectors.append(vector)
                else:
                    logger.debug(f"Skipping duplicate memory for user {memory.user_id}")
            memory_objects = filtered_memories
            vectors = filtered_vectors

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
