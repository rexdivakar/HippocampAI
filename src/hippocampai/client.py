"""Main MemoryClient - public API."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import Config, get_config
from hippocampai.embed.embedder import get_embedder
from hippocampai.graph import MemoryGraph, RelationType
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.extractor import MemoryExtractor
from hippocampai.pipeline.importance import ImportanceScorer
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.storage import MemoryKVStore
from hippocampai.telemetry import OperationType, get_telemetry
from hippocampai.utils.context_injection import ContextInjector
from hippocampai.vector.qdrant_store import QdrantStore
from hippocampai.versioning import AuditEntry, ChangeType, MemoryVersionControl

logger = logging.getLogger(__name__)

PresetType = Literal["local", "cloud", "production", "development"]


class MemoryClient:
    """Production-ready memory client with hybrid retrieval."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        collection_facts: Optional[str] = None,
        collection_prefs: Optional[str] = None,
        embed_model: Optional[str] = None,
        embed_quantized: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        bm25_backend: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        hnsw_M: Optional[int] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        half_lives: Optional[Dict[str, int]] = None,
        allow_cloud: Optional[bool] = None,
        config: Optional[Config] = None,
        enable_telemetry: bool = True,
    ):
        """Initialize memory client."""
        self.config = config or get_config()
        self.telemetry = get_telemetry(enabled=enable_telemetry)

        # Override config with params
        if qdrant_url:
            self.config.qdrant_url = qdrant_url
        if collection_facts:
            self.config.collection_facts = collection_facts
        if collection_prefs:
            self.config.collection_prefs = collection_prefs
        if embed_model:
            self.config.embed_model = embed_model
        if embed_quantized is not None:
            self.config.embed_quantized = embed_quantized
        if reranker_model:
            self.config.reranker_model = reranker_model
        if llm_provider:
            self.config.llm_provider = llm_provider
        if llm_model:
            self.config.llm_model = llm_model
        if hnsw_M:
            self.config.hnsw_m = hnsw_M
        if ef_construction:
            self.config.ef_construction = ef_construction
        if ef_search:
            self.config.ef_search = ef_search
        if weights:
            self.config.weight_sim = weights.get("sim", self.config.weight_sim)
            self.config.weight_rerank = weights.get("rerank", self.config.weight_rerank)
            self.config.weight_recency = weights.get("recency", self.config.weight_recency)
            self.config.weight_importance = weights.get("importance", self.config.weight_importance)
        if allow_cloud is not None:
            self.config.allow_cloud = allow_cloud

        # Initialize components
        self.qdrant = QdrantStore(
            url=self.config.qdrant_url,
            collection_facts=self.config.collection_facts,
            collection_prefs=self.config.collection_prefs,
            dimension=self.config.embed_dimension,
            hnsw_m=self.config.hnsw_m,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
        )

        self.embedder = get_embedder(
            model_name=self.config.embed_model,
            batch_size=self.config.embed_batch_size,
            quantized=self.config.embed_quantized,
            dimension=self.config.embed_dimension,
        )

        self.reranker = Reranker(
            model_name=self.config.reranker_model, cache_ttl=self.config.rerank_cache_ttl
        )

        self.retriever = HybridRetriever(
            qdrant_store=self.qdrant,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k_qdrant=self.config.top_k_qdrant,
            top_k_final=self.config.top_k_final,
            rrf_k=self.config.rrf_k,
            weights=self.config.get_weights(),
            half_lives=self.config.get_half_lives(),
        )

        # LLM (optional)
        self.llm = None
        if self.config.llm_provider == "ollama":
            self.llm = OllamaLLM(model=self.config.llm_model, base_url=self.config.llm_base_url)
        elif self.config.llm_provider == "openai" and self.config.allow_cloud:
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = OpenAILLM(api_key=api_key, model=self.config.llm_model)

        # Pipeline
        self.extractor = MemoryExtractor(llm=self.llm, mode="hybrid")
        self.deduplicator = MemoryDeduplicator(
            qdrant_store=self.qdrant, embedder=self.embedder, reranker=self.reranker
        )
        self.consolidator = MemoryConsolidator(llm=self.llm)
        self.scorer = ImportanceScorer(llm=self.llm)

        # Advanced Features
        self.graph = MemoryGraph()
        self.version_control = MemoryVersionControl()
        self.kv_store = MemoryKVStore(cache_ttl=300)  # 5 min cache
        self.context_injector = ContextInjector()

        # Scheduler (optional)
        self.scheduler = None
        if enable_telemetry and self.config.enable_scheduler:
            from hippocampai.scheduler import MemoryScheduler

            self.scheduler = MemoryScheduler(client=self, config=self.config)

        logger.info("MemoryClient initialized with advanced features")

    @classmethod
    def from_preset(cls, preset: PresetType, **overrides) -> "MemoryClient":
        """Create MemoryClient from preset configuration.

        Available presets:
        - "local": Ollama + local Qdrant (fully self-hosted)
        - "cloud": OpenAI + local Qdrant (cloud LLM, local vector DB)
        - "production": Optimized settings for production workloads
        - "development": Fast settings for development/testing
        """
        if preset == "local":
            config = Config(
                qdrant_url="http://localhost:6333",
                llm_provider="ollama",
                llm_model="qwen2.5:7b-instruct",
                embed_model="BAAI/bge-small-en-v1.5",
                allow_cloud=False,
            )
        elif preset == "cloud":
            config = Config(
                qdrant_url="http://localhost:6333",
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                embed_model="BAAI/bge-small-en-v1.5",
                allow_cloud=True,
            )
        elif preset == "production":
            config = Config(
                qdrant_url="http://localhost:6333",
                hnsw_m=64,  # Higher quality
                ef_construction=512,
                ef_search=256,
                top_k_qdrant=500,  # More candidates
                top_k_final=50,
                weight_sim=0.50,
                weight_rerank=0.30,  # More emphasis on reranking
                weight_recency=0.10,
                weight_importance=0.10,
            )
        elif preset == "development":
            config = Config(
                qdrant_url="http://localhost:6333",
                hnsw_m=16,  # Faster
                ef_construction=100,
                ef_search=50,
                top_k_qdrant=50,  # Fewer candidates
                top_k_final=10,
                embed_quantized=True,  # Faster embeddings
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return cls(config=config)

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        ttl_days: Optional[int] = None,
    ) -> Memory:
        """Store a memory."""
        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.REMEMBER,
            user_id=user_id,
            session_id=session_id,
            memory_type=type,
            text_length=len(text),
        )

        try:
            # Calculate expiration if TTL is set
            expires_at = None
            if ttl_days is not None:
                expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)

            memory = Memory(
                text=text,
                user_id=user_id,
                session_id=session_id,
                type=MemoryType(type),
                importance=importance or self.scorer.score(text, type),
                tags=tags or [],
                expires_at=expires_at,
            )

            # Calculate size metrics
            memory.calculate_size_metrics()

            # Track size in telemetry
            self.telemetry.track_memory_size(memory.text_length, memory.token_count)

            # Check duplicates
            self.telemetry.add_event(trace_id, "deduplication_check", status="in_progress")
            action, dup_ids = self.deduplicator.check_duplicate(memory, user_id)
            self.telemetry.add_event(
                trace_id,
                "deduplication_check",
                status="success",
                action=action,
                duplicates=len(dup_ids),
            )

            if action == "skip":
                logger.info(f"Skipping duplicate memory: {memory.id}")
                self.telemetry.end_trace(trace_id, status="skipped", result={"duplicate": True})
                return memory

            # Store
            collection = memory.collection_name(
                self.config.collection_facts, self.config.collection_prefs
            )

            self.telemetry.add_event(trace_id, "embedding", status="in_progress")
            vector = self.embedder.encode_single(memory.text)
            self.telemetry.add_event(trace_id, "embedding", status="success")

            self.telemetry.add_event(trace_id, "vector_store", status="in_progress")
            self.qdrant.upsert(
                collection_name=collection,
                id=memory.id,
                vector=vector,
                payload=memory.model_dump(mode="json"),
            )
            self.telemetry.add_event(
                trace_id, "vector_store", status="success", collection=collection
            )

            logger.info(f"Stored memory: {memory.id}")
            self.telemetry.end_trace(
                trace_id,
                status="success",
                result={"memory_id": memory.id, "collection": collection},
            )
            return memory
        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories."""
        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.RECALL,
            user_id=user_id,
            session_id=session_id,
            query=query,
            k=k,
        )

        try:
            # Rebuild BM25 if needed
            if not self.retriever.bm25_facts:
                self.telemetry.add_event(trace_id, "bm25_rebuild", status="in_progress")
                self.retriever.rebuild_bm25(user_id)
                self.telemetry.add_event(trace_id, "bm25_rebuild", status="success")

            self.telemetry.add_event(trace_id, "hybrid_retrieval", status="in_progress")
            results = self.retriever.retrieve(
                query=query, user_id=user_id, session_id=session_id, k=k, filters=filters
            )
            self.telemetry.add_event(
                trace_id, "hybrid_retrieval", status="success", results_count=len(results)
            )

            logger.info(f"Retrieved {len(results)} memories for user {user_id}")
            self.telemetry.end_trace(
                trace_id, status="success", result={"count": len(results), "k": k}
            )
            return results
        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    def extract_from_conversation(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> List[Memory]:
        """Extract memories from conversation."""
        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.EXTRACT,
            user_id=user_id,
            session_id=session_id,
            conversation_length=len(conversation),
        )

        try:
            self.telemetry.add_event(trace_id, "extraction", status="in_progress")
            memories = self.extractor.extract(conversation, user_id, session_id)
            self.telemetry.add_event(
                trace_id, "extraction", status="success", extracted_count=len(memories)
            )

            # Store each
            stored_count = 0
            for mem in memories:
                self.remember(
                    text=mem.text,
                    user_id=mem.user_id,
                    session_id=mem.session_id,
                    type=mem.type.value,
                    importance=mem.importance,
                    tags=mem.tags,
                )
                stored_count += 1

            logger.info(f"Extracted and stored {stored_count} memories from conversation")
            self.telemetry.end_trace(
                trace_id,
                status="success",
                result={"extracted": len(memories), "stored": stored_count},
            )
            return memories
        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    # Telemetry Access Methods
    def get_telemetry_metrics(self) -> Dict[str, any]:
        """Get telemetry metrics summary."""
        return self.telemetry.get_metrics_summary()

    def get_recent_operations(self, limit: int = 10, operation: Optional[str] = None):
        """Get recent memory operations with traces."""
        from hippocampai.telemetry import OperationType

        op_type = OperationType(operation) if operation else None
        return self.telemetry.get_recent_traces(limit=limit, operation=op_type)

    def export_telemetry(self, trace_ids: Optional[List[str]] = None) -> List[Dict]:
        """Export telemetry data for external analysis."""
        return self.telemetry.export_traces(trace_ids=trace_ids)

    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get memory size and usage statistics for a user.

        Args:
            user_id: User ID to get statistics for

        Returns:
            Dictionary with memory statistics including size metrics
        """
        memories = self.get_memories(user_id, limit=10000)

        if not memories:
            return {
                "total_memories": 0,
                "total_characters": 0,
                "total_tokens": 0,
                "avg_memory_size_chars": 0,
                "avg_memory_size_tokens": 0,
                "largest_memory_chars": 0,
                "smallest_memory_chars": 0,
            }

        total_chars = sum(m.text_length for m in memories)
        total_tokens = sum(m.token_count for m in memories)
        char_sizes = [m.text_length for m in memories]

        return {
            "total_memories": len(memories),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_memory_size_chars": total_chars / len(memories),
            "avg_memory_size_tokens": total_tokens / len(memories),
            "largest_memory_chars": max(char_sizes),
            "smallest_memory_chars": min(char_sizes),
            "by_type": self._get_size_by_type(memories),
        }

    def _get_size_by_type(self, memories: List[Memory]) -> Dict[str, Dict[str, Any]]:
        """Get size statistics grouped by memory type."""
        by_type = {}
        for mem in memories:
            type_name = mem.type.value
            if type_name not in by_type:
                by_type[type_name] = {
                    "count": 0,
                    "total_chars": 0,
                    "total_tokens": 0,
                }
            by_type[type_name]["count"] += 1
            by_type[type_name]["total_chars"] += mem.text_length
            by_type[type_name]["total_tokens"] += mem.token_count

        # Calculate averages
        for type_name in by_type:
            count = by_type[type_name]["count"]
            by_type[type_name]["avg_chars"] = by_type[type_name]["total_chars"] / count
            by_type[type_name]["avg_tokens"] = by_type[type_name]["total_tokens"] / count

        return by_type

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update an existing memory."""

        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.UPDATE,
            user_id="system",  # Will be updated with actual user_id
            memory_id=memory_id,
        )

        try:
            # First, find which collection the memory is in
            memory_data = None
            collection = None

            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                self.telemetry.add_event(trace_id, f"fetch_from_{coll}", status="in_progress")
                memory_data = self.qdrant.get(coll, memory_id)
                if memory_data:
                    collection = coll
                    self.telemetry.add_event(trace_id, f"fetch_from_{coll}", status="success")
                    break
                self.telemetry.add_event(trace_id, f"fetch_from_{coll}", status="not_found")

            if not memory_data:
                logger.warning(f"Memory {memory_id} not found")
                self.telemetry.end_trace(trace_id, status="error", result={"error": "not_found"})
                return None

            # Parse existing memory
            payload = memory_data["payload"]
            memory = Memory(**payload)

            # Update fields
            if text is not None:
                memory.text = text
                # Recalculate size metrics
                memory.calculate_size_metrics()

                # Track updated size in telemetry
                self.telemetry.track_memory_size(memory.text_length, memory.token_count)

                # Re-embed if text changed
                self.telemetry.add_event(trace_id, "re_embedding", status="in_progress")
                vector = self.embedder.encode_single(text)
                self.telemetry.add_event(trace_id, "re_embedding", status="success")
            else:
                vector = None

            if importance is not None:
                memory.importance = importance
            if tags is not None:
                memory.tags = tags
            if metadata is not None:
                memory.metadata.update(metadata)
            if expires_at is not None:
                memory.expires_at = expires_at

            memory.updated_at = datetime.now(timezone.utc)

            # Update in Qdrant
            self.telemetry.add_event(trace_id, "update_vector_store", status="in_progress")
            if vector is not None:
                # Full upsert with new vector
                self.qdrant.upsert(
                    collection_name=collection,
                    id=memory_id,
                    vector=vector,
                    payload=memory.model_dump(mode="json"),
                )
            else:
                # Payload update only
                self.qdrant.update(
                    collection_name=collection,
                    id=memory_id,
                    payload=memory.model_dump(mode="json"),
                )
            self.telemetry.add_event(trace_id, "update_vector_store", status="success")

            logger.info(f"Updated memory: {memory_id}")
            self.telemetry.end_trace(
                trace_id,
                status="success",
                result={
                    "memory_id": memory_id,
                    "updated_fields": len(
                        [x for x in [text, importance, tags, metadata, expires_at] if x is not None]
                    ),
                },
            )
            return memory

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return None

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a memory by ID."""
        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.DELETE,
            user_id=user_id or "system",
            memory_id=memory_id,
        )

        try:
            # Try both collections
            deleted = False
            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                self.telemetry.add_event(trace_id, f"delete_from_{coll}", status="in_progress")
                # Check if exists first
                memory_data = self.qdrant.get(coll, memory_id)
                if memory_data:
                    # Verify user_id if provided
                    if user_id and memory_data["payload"].get("user_id") != user_id:
                        logger.warning(
                            f"User {user_id} attempted to delete memory {memory_id} owned by {memory_data['payload'].get('user_id')}"
                        )
                        self.telemetry.end_trace(
                            trace_id, status="error", result={"error": "unauthorized"}
                        )
                        return False

                    self.qdrant.delete(collection_name=coll, ids=[memory_id])
                    deleted = True
                    self.telemetry.add_event(trace_id, f"delete_from_{coll}", status="success")
                    break
                self.telemetry.add_event(trace_id, f"delete_from_{coll}", status="not_found")

            if deleted:
                logger.info(f"Deleted memory: {memory_id}")
                self.telemetry.end_trace(
                    trace_id, status="success", result={"memory_id": memory_id}
                )
                return True
            else:
                logger.warning(f"Memory {memory_id} not found")
                self.telemetry.end_trace(trace_id, status="error", result={"error": "not_found"})
                return False

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return False

    def get_memories(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Get memories with advanced filtering (without semantic search).

        Args:
            user_id: User ID to filter by
            filters: Optional filters:
                - type: Memory type (str or list)
                - tags: Tags to filter by (str or list)
                - session_id: Session ID
                - min_importance: Minimum importance score
                - max_importance: Maximum importance score
                - include_expired: Include expired memories (default: False)
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects
        """

        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.GET,
            user_id=user_id,
            filters=filters,
            limit=limit,
        )

        try:
            filters = filters or {}
            include_expired = filters.pop("include_expired", False)
            min_importance = filters.pop("min_importance", None)
            max_importance = filters.pop("max_importance", None)

            # Build Qdrant filters
            qdrant_filters = {"user_id": user_id}
            if "type" in filters:
                qdrant_filters["type"] = filters["type"]
            if "tags" in filters:
                qdrant_filters["tags"] = filters["tags"]

            # Fetch from both collections
            all_memories = []
            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                self.telemetry.add_event(trace_id, f"fetch_from_{coll}", status="in_progress")
                results = self.qdrant.scroll(
                    collection_name=coll,
                    filters=qdrant_filters,
                    limit=limit,
                )
                all_memories.extend(results)
                self.telemetry.add_event(
                    trace_id, f"fetch_from_{coll}", status="success", count=len(results)
                )

            # Parse into Memory objects
            memories = []
            for data in all_memories:
                memory = Memory(**data["payload"])

                # Apply additional filters
                if not include_expired and memory.is_expired():
                    continue
                if min_importance is not None and memory.importance < min_importance:
                    continue
                if max_importance is not None and memory.importance > max_importance:
                    continue
                if "session_id" in filters and memory.session_id != filters["session_id"]:
                    continue

                memories.append(memory)

            # Sort by creation date (most recent first)
            memories.sort(key=lambda m: m.created_at, reverse=True)
            memories = memories[:limit]

            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            self.telemetry.end_trace(trace_id, status="success", result={"count": len(memories)})
            return memories

        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return []

    def expire_memories(self, user_id: Optional[str] = None) -> int:
        """Remove expired memories.

        Args:
            user_id: Optional user ID to filter by (if None, expires for all users)

        Returns:
            Number of memories expired
        """

        # Start telemetry trace
        trace_id = self.telemetry.start_trace(
            operation=OperationType.EXPIRE,
            user_id=user_id or "all",
        )

        try:
            expired_count = 0
            filters = {"user_id": user_id} if user_id else {}

            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                self.telemetry.add_event(trace_id, f"scan_{coll}", status="in_progress")
                results = self.qdrant.scroll(
                    collection_name=coll,
                    filters=filters,
                    limit=10000,
                )

                expired_ids = []
                for data in results:
                    memory = Memory(**data["payload"])
                    if memory.is_expired():
                        expired_ids.append(data["id"])

                if expired_ids:
                    self.qdrant.delete(collection_name=coll, ids=expired_ids)
                    expired_count += len(expired_ids)

                self.telemetry.add_event(
                    trace_id, f"scan_{coll}", status="success", expired=len(expired_ids)
                )

            logger.info(f"Expired {expired_count} memories")
            self.telemetry.end_trace(
                trace_id, status="success", result={"expired_count": expired_count}
            )
            return expired_count

        except Exception as e:
            logger.error(f"Failed to expire memories: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return 0

    # === BATCH OPERATIONS ===

    def add_memories(
        self,
        memories: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        """Batch add multiple memories at once.

        Args:
            memories: List of memory dictionaries with keys: text, type, importance, tags, ttl_days
            user_id: User ID
            session_id: Optional session ID

        Returns:
            List of created Memory objects
        """
        trace_id = self.telemetry.start_trace(
            operation=OperationType.REMEMBER,
            user_id=user_id,
            session_id=session_id,
            batch_size=len(memories),
        )

        created_memories = []

        try:
            for mem_data in memories:
                memory = self.remember(
                    text=mem_data.get("text", ""),
                    user_id=user_id,
                    session_id=session_id,
                    type=mem_data.get("type", "fact"),
                    importance=mem_data.get("importance"),
                    tags=mem_data.get("tags", []),
                    ttl_days=mem_data.get("ttl_days"),
                )
                created_memories.append(memory)

            self.telemetry.end_trace(
                trace_id, status="success", result={"created": len(created_memories)}
            )
            return created_memories

        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    def delete_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> int:
        """Batch delete multiple memories.

        Args:
            memory_ids: List of memory IDs to delete
            user_id: Optional user ID for authorization

        Returns:
            Number of memories successfully deleted
        """
        trace_id = self.telemetry.start_trace(
            operation=OperationType.DELETE,
            user_id=user_id or "system",
            batch_size=len(memory_ids),
        )

        deleted_count = 0

        try:
            for memory_id in memory_ids:
                if self.delete_memory(memory_id, user_id):
                    deleted_count += 1

            self.telemetry.end_trace(trace_id, status="success", result={"deleted": deleted_count})
            return deleted_count

        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    # === GRAPH OPERATIONS ===

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
    ) -> bool:
        """Add a relationship between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relationship
            weight: Relationship weight/strength

        Returns:
            True if successful
        """
        # Add audit entry
        if self.version_control:
            self.version_control.add_audit_entry(
                memory_id=source_id,
                change_type=ChangeType.RELATIONSHIP_ADDED,
                metadata={"target_id": target_id, "relation_type": relation_type.value},
            )

        return self.graph.add_relationship(source_id, target_id, relation_type, weight)

    def get_related_memories(
        self,
        memory_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1,
    ) -> List[tuple]:
        """Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_types: Filter by specific relation types
            max_depth: How many hops to traverse

        Returns:
            List of (memory_id, relation_type, weight) tuples
        """
        return self.graph.get_related_memories(memory_id, relation_types, max_depth)

    def get_memory_clusters(self, user_id: str) -> List[set]:
        """Find clusters of related memories.

        Args:
            user_id: User ID to filter by

        Returns:
            List of memory ID sets (clusters)
        """
        return self.graph.get_clusters(user_id)

    def export_graph_to_json(
        self, file_path: str, user_id: Optional[str] = None, indent: int = 2
    ) -> str:
        """Export memory graph to a JSON file for persistence.

        Args:
            file_path: Path where the JSON file will be saved
            user_id: Optional user ID to export only a specific user's graph
            indent: JSON indentation level (default: 2)

        Returns:
            The file path where the graph was saved

        Example:
            >>> client.export_graph_to_json("memory_graph.json")
            >>> client.export_graph_to_json("alice_graph.json", user_id="alice")
        """
        return self.graph.export_to_json(file_path, user_id, indent)

    def import_graph_from_json(self, file_path: str, merge: bool = True) -> Dict:
        """Import memory graph from a JSON file.

        Args:
            file_path: Path to the JSON file to import
            merge: If True, merge with existing graph; if False, replace existing graph

        Returns:
            Dictionary with import statistics including:
            - file_path: Path that was imported
            - nodes_before/after: Node counts before and after import
            - edges_before/after: Edge counts before and after import
            - nodes_imported: Number of nodes in the imported file
            - edges_imported: Number of edges in the imported file
            - merged: Whether the graph was merged or replaced

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid graph format

        Example:
            >>> stats = client.import_graph_from_json("memory_graph.json")
            >>> print(f"Imported {stats['nodes_imported']} nodes")
            >>> stats = client.import_graph_from_json("backup.json", merge=False)
        """
        return self.graph.import_from_json(file_path, merge)

    # === VERSION CONTROL ===

    def get_memory_history(self, memory_id: str) -> List:
        """Get version history for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of MemoryVersion objects
        """
        return self.version_control.get_version_history(memory_id)

    def rollback_memory(self, memory_id: str, version_number: int) -> Optional[Memory]:
        """Rollback memory to a previous version.

        Args:
            memory_id: Memory ID
            version_number: Version to rollback to

        Returns:
            Updated Memory object or None
        """
        data = self.version_control.rollback(memory_id, version_number)
        if data:
            # Update the memory
            return self.update_memory(
                memory_id=memory_id,
                text=data.get("text"),
                importance=data.get("importance"),
                tags=data.get("tags"),
            )

        return None

    # === AUDIT TRAIL ===

    def get_audit_trail(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit trail entries.

        Args:
            memory_id: Filter by memory ID
            user_id: Filter by user ID
            change_type: Filter by change type
            limit: Maximum entries to return

        Returns:
            List of AuditEntry objects
        """
        return self.version_control.get_audit_trail(memory_id, user_id, change_type, limit)

    # === MEMORY ACCESS TRACKING ===

    def track_memory_access(self, memory_id: str, user_id: str):
        """Track that a memory was accessed (updates access_count).

        Args:
            memory_id: Memory ID
            user_id: User who accessed it
        """
        # Find and update the memory
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                payload = memory_data["payload"]
                payload["access_count"] = payload.get("access_count", 0) + 1
                payload["last_accessed_at"] = datetime.now(timezone.utc).isoformat()

                self.qdrant.update(coll, memory_id, payload)

                # Add audit entry
                if self.version_control:
                    self.version_control.add_audit_entry(
                        memory_id=memory_id,
                        change_type=ChangeType.ACCESSED,
                        user_id=user_id,
                    )
                break

    # === SNAPSHOT MANAGEMENT ===

    def create_snapshot(self, collection: str = "facts") -> str:
        """Create a snapshot of a memory collection.

        Args:
            collection: "facts" or "prefs"

        Returns:
            Snapshot name/ID
        """
        coll_name = (
            self.config.collection_facts if collection == "facts" else self.config.collection_prefs
        )

        snapshot_name = self.qdrant.create_snapshot(coll_name)
        logger.info(f"Created snapshot: {snapshot_name} for collection {coll_name}")
        return snapshot_name

    # === CONTEXT INJECTION ===

    def inject_context(
        self,
        prompt: str,
        query: str,
        user_id: str,
        k: int = 5,
        template: str = "default",
    ) -> str:
        """Inject relevant memories into a prompt.

        Args:
            prompt: Original prompt/query
            query: Query to find relevant memories
            user_id: User ID
            k: Number of memories to inject
            template: Formatting template

        Returns:
            Prompt with injected context
        """
        # Retrieve relevant memories
        results = self.recall(query, user_id, k=k)

        # Track access
        for result in results:
            self.track_memory_access(result.memory.id, user_id)

        # Inject into prompt
        self.context_injector.template = template
        return self.context_injector.inject_retrieval_results(prompt, results)

    # === ADVANCED FILTERS ===

    def get_memories_advanced(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "created_at",  # created_at, importance, access_count
        sort_order: str = "desc",
        limit: int = 100,
    ) -> List[Memory]:
        """Get memories with advanced filtering and sorting.

        Args:
            user_id: User ID
            filters: Advanced filters (supports all get_memories filters plus metadata filters)
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            limit: Maximum results

        Returns:
            List of Memory objects
        """
        # Get base memories
        memories = self.get_memories(user_id, filters, limit=limit * 2)  # Get more for sorting

        # Apply metadata filters if provided
        if filters and "metadata" in filters:
            metadata_filters = filters["metadata"]
            filtered_memories = []

            for mem in memories:
                match = True
                for key, value in metadata_filters.items():
                    if key not in mem.metadata or mem.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_memories.append(mem)

            memories = filtered_memories

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "importance":
            memories.sort(key=lambda m: m.importance, reverse=reverse)
        elif sort_by == "access_count":
            memories.sort(key=lambda m: m.access_count, reverse=reverse)
        else:  # created_at
            memories.sort(key=lambda m: m.created_at, reverse=reverse)

        return memories[:limit]

    # === SCHEDULER METHODS ===

    def start_scheduler(self):
        """Start the background scheduler for memory maintenance tasks."""
        if self.scheduler:
            self.scheduler.start()
            logger.info("Background scheduler started")
        else:
            logger.warning(
                "Scheduler not initialized (enable_scheduler=False or enable_telemetry=False)"
            )

    def stop_scheduler(self):
        """Stop the background scheduler."""
        if self.scheduler:
            self.scheduler.stop()
            logger.info("Background scheduler stopped")

    def get_scheduler_status(self) -> dict:
        """Get scheduler status and job information."""
        if self.scheduler:
            return self.scheduler.get_job_status()
        return {"status": "not_initialized", "jobs": []}

    def consolidate_all_memories(self, similarity_threshold: float = 0.85) -> int:
        """Consolidate similar memories across all users.

        This method is called by the scheduler for periodic consolidation.

        Args:
            similarity_threshold: Minimum similarity score to consider memories for consolidation

        Returns:
            Number of memory clusters consolidated
        """
        consolidated_count = 0

        try:
            # Get all memories from both collections
            all_memories = []
            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                results = self.qdrant.scroll(
                    collection_name=coll,
                    filters={},
                    limit=10000,
                )
                all_memories.extend(results)

            if not all_memories:
                logger.info("No memories to consolidate")
                return 0

            # Group by user
            user_memories = {}
            for data in all_memories:
                memory = Memory(**data["payload"])
                user_id = memory.user_id
                if user_id not in user_memories:
                    user_memories[user_id] = []
                user_memories[user_id].append(memory)

            # Consolidate for each user
            for user_id, memories in user_memories.items():
                if len(memories) < 2:
                    continue

                # Find similar memory clusters
                clusters = self._find_similar_clusters(memories, similarity_threshold)

                for cluster in clusters:
                    if len(cluster) < 2:
                        continue

                    # Consolidate cluster
                    consolidated = self.consolidator.consolidate(cluster)
                    if consolidated:
                        # Delete old memories
                        old_ids = [m.id for m in cluster]
                        self.delete_memories(old_ids, user_id)

                        # Store consolidated memory
                        self.remember(
                            text=consolidated.text,
                            user_id=consolidated.user_id,
                            session_id=consolidated.session_id,
                            type=consolidated.type.value,
                            importance=consolidated.importance,
                            tags=consolidated.tags,
                        )

                        consolidated_count += 1
                        logger.info(f"Consolidated {len(cluster)} memories for user {user_id}")

            logger.info(f"Total consolidation completed: {consolidated_count} clusters")
            return consolidated_count

        except Exception as e:
            logger.error(f"Consolidation failed: {e}", exc_info=True)
            return consolidated_count

    def _find_similar_clusters(
        self, memories: List[Memory], threshold: float = 0.85
    ) -> List[List[Memory]]:
        """Find clusters of similar memories.

        Args:
            memories: List of memories to cluster
            threshold: Similarity threshold

        Returns:
            List of memory clusters
        """
        if len(memories) < 2:
            return []

        clusters = []
        processed = set()

        for i, mem1 in enumerate(memories):
            if mem1.id in processed:
                continue

            # Start new cluster
            cluster = [mem1]
            processed.add(mem1.id)

            # Find similar memories
            for mem2 in memories[i + 1 :]:
                if mem2.id in processed:
                    continue

                # Check if same type
                if mem1.type != mem2.type:
                    continue

                # Calculate similarity using reranker
                try:
                    score = self.reranker.rerank_single(mem1.text, mem2.text)
                    if score >= threshold:
                        cluster.append(mem2)
                        processed.add(mem2.id)
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    continue

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def apply_importance_decay(self) -> int:
        """Apply importance decay to all memories based on age.

        This method is called by the scheduler for periodic decay.

        Returns:
            Number of memories updated
        """
        updated_count = 0

        try:
            now = datetime.now(timezone.utc)

            # Get half-lives from config
            half_lives = self.config.get_half_lives()

            # Process all memories
            for coll in [self.config.collection_facts, self.config.collection_prefs]:
                results = self.qdrant.scroll(
                    collection_name=coll,
                    filters={},
                    limit=10000,
                )

                for data in results:
                    memory = Memory(**data["payload"])

                    # Calculate age in days
                    age_days = (now - memory.created_at).total_seconds() / 86400

                    # Get half-life for memory type
                    half_life = half_lives.get(memory.type.value, 30)

                    # Calculate decay factor: importance * (0.5 ^ (age / half_life))
                    decay_factor = 0.5 ** (age_days / half_life)
                    new_importance = memory.importance * decay_factor

                    # Only update if decay is significant (> 0.1 change)
                    if abs(memory.importance - new_importance) > 0.1:
                        self.update_memory(
                            memory_id=memory.id,
                            importance=new_importance,
                        )
                        updated_count += 1

            logger.info(f"Importance decay completed: {updated_count} memories updated")
            return updated_count

        except Exception as e:
            logger.error(f"Importance decay failed: {e}", exc_info=True)
            return updated_count
