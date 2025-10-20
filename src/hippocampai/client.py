"""Main MemoryClient - public API."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import Config, get_config
from hippocampai.embed.embedder import get_embedder
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.extractor import MemoryExtractor
from hippocampai.pipeline.importance import ImportanceScorer
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.telemetry import OperationType, get_telemetry
from hippocampai.vector.qdrant_store import QdrantStore

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

        logger.info("MemoryClient initialized")

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
                from datetime import timedelta
                expires_at = datetime.utcnow() + timedelta(days=ttl_days)

            memory = Memory(
                text=text,
                user_id=user_id,
                session_id=session_id,
                type=MemoryType(type),
                importance=importance or self.scorer.score(text, type),
                tags=tags or [],
                expires_at=expires_at,
            )

            # Check duplicates
            self.telemetry.add_event(trace_id, "deduplication_check", status="in_progress")
            action, dup_ids = self.deduplicator.check_duplicate(memory, user_id)
            self.telemetry.add_event(
                trace_id, "deduplication_check", status="success", action=action, duplicates=len(dup_ids)
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
            self.telemetry.add_event(trace_id, "vector_store", status="success", collection=collection)

            logger.info(f"Stored memory: {memory.id}")
            self.telemetry.end_trace(
                trace_id, status="success", result={"memory_id": memory.id, "collection": collection}
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
                trace_id, status="success", result={"extracted": len(memories), "stored": stored_count}
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

            memory.updated_at = datetime.utcnow()

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
                trace_id, status="success", result={"memory_id": memory_id, "updated_fields": len([x for x in [text, importance, tags, metadata, expires_at] if x is not None])}
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
                        logger.warning(f"User {user_id} attempted to delete memory {memory_id} owned by {memory_data['payload'].get('user_id')}")
                        self.telemetry.end_trace(trace_id, status="error", result={"error": "unauthorized"})
                        return False

                    self.qdrant.delete(collection_name=coll, ids=[memory_id])
                    deleted = True
                    self.telemetry.add_event(trace_id, f"delete_from_{coll}", status="success")
                    break
                self.telemetry.add_event(trace_id, f"delete_from_{coll}", status="not_found")

            if deleted:
                logger.info(f"Deleted memory: {memory_id}")
                self.telemetry.end_trace(trace_id, status="success", result={"memory_id": memory_id})
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
                self.telemetry.add_event(trace_id, f"fetch_from_{coll}", status="success", count=len(results))

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
            self.telemetry.end_trace(
                trace_id, status="success", result={"count": len(memories)}
            )
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

                self.telemetry.add_event(trace_id, f"scan_{coll}", status="success", expired=len(expired_ids))

            logger.info(f"Expired {expired_count} memories")
            self.telemetry.end_trace(
                trace_id, status="success", result={"expired_count": expired_count}
            )
            return expired_count

        except Exception as e:
            logger.error(f"Failed to expire memories: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return 0
