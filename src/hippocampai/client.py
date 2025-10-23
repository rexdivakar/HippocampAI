"""Main MemoryClient - public API."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Set
from uuid import uuid4

from hippocampai.adapters.provider_groq import GroqLLM
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import Config, get_config
from hippocampai.embed.embedder import get_embedder
from hippocampai.graph import MemoryGraph, RelationType
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.models.session import Session, SessionSearchResult, SessionStatus
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.extractor import MemoryExtractor
from hippocampai.pipeline.importance import ImportanceScorer
from hippocampai.pipeline.semantic_clustering import SemanticCategorizer
from hippocampai.pipeline.smart_updater import SmartMemoryUpdater
from hippocampai.pipeline.temporal import TemporalAnalyzer, TimeRange, ScheduledMemory, Timeline
from hippocampai.pipeline.insights import InsightAnalyzer, Pattern, BehaviorChange, PreferenceDrift, HabitScore, Trend
from hippocampai.multiagent import MultiAgentManager
from hippocampai.models.agent import Agent, AgentRole, Run, AgentPermission, PermissionType, MemoryVisibility
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.session import SessionManager
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
        elif self.config.llm_provider == "groq" and self.config.allow_cloud:
            import os

            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.llm = GroqLLM(api_key=api_key, model=self.config.llm_model)

        # Pipeline
        self.extractor = MemoryExtractor(llm=self.llm, mode="hybrid")
        self.deduplicator = MemoryDeduplicator(
            qdrant_store=self.qdrant, embedder=self.embedder, reranker=self.reranker
        )
        self.consolidator = MemoryConsolidator(llm=self.llm)
        self.scorer = ImportanceScorer(llm=self.llm)
        self.smart_updater = SmartMemoryUpdater(llm=self.llm, similarity_threshold=0.85)
        self.categorizer = SemanticCategorizer(llm=self.llm)
        self.temporal_analyzer = TemporalAnalyzer(llm=self.llm)  # Temporal reasoning
        self.insight_analyzer = InsightAnalyzer(llm=self.llm)  # Cross-session insights

        # Advanced Features
        self.graph = MemoryGraph()
        self.version_control = MemoryVersionControl()
        self.kv_store = MemoryKVStore(cache_ttl=300)  # 5 min cache
        self.context_injector = ContextInjector()
        self.multiagent = MultiAgentManager()  # Multi-agent support

        # Session Management
        self.session_manager = SessionManager(
            qdrant_store=self.qdrant,
            embedder=self.embedder,
            llm=self.llm,
            collection_name="hippocampai_sessions",
        )

        # Scheduler (optional)
        self.scheduler = None
        if enable_telemetry and self.config.enable_scheduler:
            from hippocampai.scheduler import MemoryScheduler

            self.scheduler = MemoryScheduler(client=self, config=self.config)

        logger.info("MemoryClient initialized with advanced features and session management")

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
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        visibility: Optional[str] = None,
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
                agent_id=agent_id,
                run_id=run_id,
                visibility=visibility or MemoryVisibility.PRIVATE.value,
            )

            # Auto-enrich with semantic categorization
            self.telemetry.add_event(trace_id, "semantic_enrichment", status="in_progress")
            original_type = memory.type
            memory = self.categorizer.enrich_memory_with_categories(memory)
            logger.info(f"Enrichment: {original_type} -> {memory.type} for text '{memory.text[:50]}'")
            self.telemetry.add_event(trace_id, "semantic_enrichment", status="success")

            # Calculate size metrics
            memory.calculate_size_metrics()

            # Track size in telemetry
            self.telemetry.track_memory_size(memory.text_length, memory.token_count)

            # Check for similar memories and decide on smart update
            self.telemetry.add_event(trace_id, "smart_update_check", status="in_progress")
            existing_memories = self.get_memories(user_id, limit=100)
            similar = self.categorizer.find_similar_memories(memory, existing_memories, similarity_threshold=0.85)

            if similar:
                # Found similar memory, use smart updater to decide action
                existing_memory = similar[0][0]  # Most similar memory
                decision = self.smart_updater.should_update_memory(existing_memory, text)

                self.telemetry.add_event(
                    trace_id,
                    "smart_update_check",
                    status="success",
                    action=decision.action,
                    reason=decision.reason,
                )

                if decision.action == "skip":
                    # Update confidence of existing memory
                    updated_existing = self.smart_updater.update_confidence(existing_memory, "reinforcement")
                    self.update_memory(
                        memory_id=existing_memory.id,
                        importance=updated_existing.confidence * 10,  # Scale confidence to importance
                    )
                    logger.info(f"Skipping similar memory: {decision.reason}")
                    self.telemetry.end_trace(trace_id, status="skipped", result={"reason": decision.reason})
                    return existing_memory

                elif decision.action == "update":
                    # Update existing memory
                    if decision.merged_memory:
                        self.update_memory(
                            memory_id=existing_memory.id,
                            text=decision.merged_memory.text,
                            importance=decision.merged_memory.importance,
                            tags=decision.merged_memory.tags,
                        )
                        logger.info(f"Updated existing memory: {decision.reason}")
                        self.telemetry.end_trace(trace_id, status="updated", result={"reason": decision.reason})
                        return decision.merged_memory

                elif decision.action == "merge":
                    # Delete old, store merged
                    if decision.merged_memory:
                        self.delete_memory(existing_memory.id, user_id)
                        memory = decision.merged_memory
                        logger.info(f"Merged memories: {decision.reason}")
                # If action is "keep_both", continue with normal storage
            else:
                self.telemetry.add_event(trace_id, "smart_update_check", status="success", action="new")

            # Check duplicates (basic dedup as fallback)
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
                logger.info(f"Skipping duplicate memory: {memory.id}, type={memory.type}")
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

            logger.info(f"Stored memory: {memory.id}, type={memory.type}")
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

    # === SESSION MANAGEMENT ===

    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Session:
        """Create a new conversation session.

        Args:
            user_id: User ID
            title: Optional session title
            parent_session_id: Optional parent session for hierarchical sessions
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            Created Session object
        """
        return self.session_manager.create_session(
            user_id=user_id,
            title=title,
            parent_session_id=parent_session_id,
            metadata=metadata,
            tags=tags,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object or None if not found
        """
        return self.session_manager.get_session(session_id)

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Session]:
        """Update session fields.

        Args:
            session_id: Session ID
            title: Optional new title
            summary: Optional summary
            status: Optional status
            metadata: Optional metadata to merge
            tags: Optional tags

        Returns:
            Updated Session or None if not found
        """
        return self.session_manager.update_session(
            session_id=session_id,
            title=title,
            summary=summary,
            status=status,
            metadata=metadata,
            tags=tags,
        )

    def track_session_message(
        self,
        session_id: str,
        text: str,
        user_id: str,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        auto_boundary_detect: bool = True,
    ) -> Optional[Session]:
        """Track a message in a session and optionally detect boundaries.

        Args:
            session_id: Session ID
            text: Message text
            user_id: User ID
            type: Memory type
            importance: Optional importance score
            tags: Optional tags
            auto_boundary_detect: Whether to auto-detect session boundaries

        Returns:
            Updated Session or new Session if boundary detected
        """
        # Check for session boundary
        if auto_boundary_detect:
            boundary_detected, reason = self.session_manager.detect_session_boundary(
                session_id, text
            )
            if boundary_detected:
                logger.info(f"Session boundary detected: {reason}. Creating new session.")
                # Complete old session
                self.complete_session(session_id)
                # Create new session
                old_session = self.session_manager.get_session(session_id)
                new_session = self.session_manager.create_session(
                    user_id=user_id,
                    title=f"Session continued from {session_id[:8]}",
                    parent_session_id=session_id,
                    metadata={"boundary_reason": reason, "previous_session": session_id},
                )
                session_id = new_session.id

        # Store memory
        memory = self.remember(
            text=text,
            user_id=user_id,
            session_id=session_id,
            type=type,
            importance=importance,
            tags=tags,
        )

        # Track in session
        session = self.session_manager.track_message(session_id, memory, auto_extract=True)

        return session

    def complete_session(
        self, session_id: str, generate_summary: bool = True
    ) -> Optional[Session]:
        """Complete a session and generate summary.

        Args:
            session_id: Session ID
            generate_summary: Whether to generate summary

        Returns:
            Completed Session or None
        """
        return self.session_manager.complete_session(session_id, generate_summary)

    def search_sessions(
        self,
        query: str,
        user_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SessionSearchResult]:
        """Search sessions by semantic similarity.

        Args:
            query: Search query
            user_id: Optional user ID filter
            k: Number of results
            filters: Optional filters (status, tags, etc.)

        Returns:
            List of SessionSearchResult objects
        """
        return self.session_manager.search_sessions(query, user_id, k, filters)

    def get_user_sessions(
        self,
        user_id: str,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[Session]:
        """Get sessions for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of Session objects sorted by date
        """
        return self.session_manager.get_user_sessions(user_id, status, limit)

    def get_session_memories(self, session_id: str, limit: int = 100) -> List[Memory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            limit: Maximum number of memories

        Returns:
            List of Memory objects
        """
        return self.get_memories(user_id="*", filters={"session_id": session_id}, limit=limit)

    def get_child_sessions(self, parent_session_id: str) -> List[Session]:
        """Get child sessions of a parent session.

        Args:
            parent_session_id: Parent session ID

        Returns:
            List of child Session objects
        """
        return self.session_manager.get_child_sessions(parent_session_id)

    def summarize_session(self, session_id: str, force: bool = False) -> Optional[str]:
        """Generate summary for a session.

        Args:
            session_id: Session ID
            force: Force re-summarization

        Returns:
            Generated summary or None
        """
        return self.session_manager.summarize_session(session_id, force)

    def extract_session_facts(
        self, session_id: str, force: bool = False
    ) -> List:
        """Extract key facts from session.

        Args:
            session_id: Session ID
            force: Force re-extraction

        Returns:
            List of SessionFact objects
        """
        return self.session_manager.extract_session_facts(session_id, force)

    def extract_session_entities(
        self, session_id: str, force: bool = False
    ) -> Dict[str, Any]:
        """Extract entities from session.

        Args:
            session_id: Session ID
            force: Force re-extraction

        Returns:
            Dictionary of entity_name -> Entity
        """
        return self.session_manager.extract_session_entities(session_id, force)

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session statistics
        """
        return self.session_manager.get_session_statistics(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted successfully
        """
        return self.session_manager.delete_session(session_id)

    # === SMART MEMORY UPDATES & CLUSTERING ===

    def reconcile_user_memories(self, user_id: str) -> List[Memory]:
        """Reconcile and resolve conflicts in user's memories.

        Args:
            user_id: User ID

        Returns:
            List of reconciled memories
        """
        memories = self.get_memories(user_id, limit=1000)
        if not memories:
            return []

        reconciled = self.smart_updater.reconcile_memories(memories, user_id)

        logger.info(f"Reconciled {len(memories)} memories into {len(reconciled)} for user {user_id}")
        return reconciled

    def cluster_user_memories(self, user_id: str, max_clusters: int = 10):
        """Cluster user's memories by topics.

        Args:
            user_id: User ID
            max_clusters: Maximum number of clusters

        Returns:
            List of MemoryCluster objects
        """
        memories = self.get_memories(user_id, limit=1000)
        if not memories:
            return []

        clusters = self.categorizer.cluster_memories(memories, max_clusters=max_clusters)

        logger.info(f"Clustered {len(memories)} memories into {len(clusters)} topics for user {user_id}")
        return clusters

    def suggest_memory_tags(self, memory: Memory, max_tags: int = 5) -> List[str]:
        """Suggest tags for a memory.

        Args:
            memory: Memory to suggest tags for
            max_tags: Maximum number of tags

        Returns:
            List of suggested tags
        """
        return self.categorizer.suggest_tags(memory, max_tags=max_tags)

    def refine_memory_quality(self, memory_id: str, context: Optional[str] = None) -> Optional[Memory]:
        """Refine a memory's text quality using LLM.

        Args:
            memory_id: Memory ID to refine
            context: Optional context for refinement

        Returns:
            Refined memory or None if not found
        """
        # Fetch memory
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])

                # Refine
                refined = self.smart_updater.refine_memory(memory, context)

                # Update if changed
                if refined.text != memory.text:
                    return self.update_memory(
                        memory_id=memory_id,
                        text=refined.text,
                        importance=refined.importance,
                    )

                return memory

        return None

    def detect_topic_shift(self, user_id: str, window_size: int = 10) -> Optional[str]:
        """Detect if there's been a shift in conversation topics.

        Args:
            user_id: User ID
            window_size: Number of recent memories to analyze

        Returns:
            New dominant topic if shift detected, None otherwise
        """
        recent_memories = self.get_memories(user_id, limit=window_size * 2)
        if not recent_memories:
            return None

        # Sort by creation time
        recent_memories.sort(key=lambda m: m.created_at)

        return self.categorizer.detect_topic_shift(recent_memories, window_size=window_size)

    # === MULTI-AGENT SUPPORT ===

    def create_agent(
        self,
        name: str,
        user_id: str,
        role: AgentRole = AgentRole.ASSISTANT,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create a new agent with its own memory space.

        Args:
            name: Agent name
            user_id: User ID owning the agent
            role: Agent role (assistant, specialist, coordinator, observer)
            description: Optional description
            metadata: Optional metadata

        Returns:
            Created Agent object

        Example:
            >>> agent = client.create_agent("Research Assistant", "alice", role=AgentRole.SPECIALIST)
        """
        return self.multiagent.create_agent(name, user_id, role, description, metadata)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.multiagent.get_agent(agent_id)

    def list_agents(self, user_id: Optional[str] = None) -> List[Agent]:
        """List all agents, optionally filtered by user."""
        return self.multiagent.list_agents(user_id)

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[AgentRole] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Agent]:
        """Update agent properties."""
        return self.multiagent.update_agent(agent_id, name, description, role, metadata, is_active)

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and its associated data."""
        return self.multiagent.delete_agent(agent_id)

    def create_run(
        self,
        agent_id: str,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Create a new run for an agent.

        Runs organize memories in a User → Agent → Run hierarchy.

        Args:
            agent_id: Agent ID
            user_id: User ID
            name: Optional run name
            metadata: Optional metadata

        Returns:
            Created Run object

        Example:
            >>> run = client.create_run(agent.id, "alice", name="Research Session 1")
            >>> memory = client.remember("Key finding", "alice", agent_id=agent.id, run_id=run.id)
        """
        return self.multiagent.create_run(agent_id, user_id, name, metadata)

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run by ID."""
        return self.multiagent.get_run(run_id)

    def list_runs(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Run]:
        """List runs, optionally filtered by agent or user."""
        return self.multiagent.list_runs(agent_id, user_id)

    def complete_run(self, run_id: str, status: str = "completed") -> Optional[Run]:
        """Mark a run as completed."""
        return self.multiagent.complete_run(run_id, status)

    def grant_agent_permission(
        self,
        granter_agent_id: str,
        grantee_agent_id: str,
        permissions: Set[PermissionType],
        memory_filters: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> AgentPermission:
        """Grant permission for one agent to access another's memories.

        Args:
            granter_agent_id: Agent granting permission
            grantee_agent_id: Agent receiving permission
            permissions: Set of permissions (READ, WRITE, SHARE, DELETE)
            memory_filters: Optional filters for which memories to share
            expires_at: Optional expiration datetime

        Returns:
            Created AgentPermission object

        Example:
            >>> # Allow agent2 to read agent1's memories
            >>> perm = client.grant_agent_permission(
            ...     agent1.id,
            ...     agent2.id,
            ...     {PermissionType.READ}
            ... )
        """
        return self.multiagent.grant_permission(
            granter_agent_id, grantee_agent_id, permissions, memory_filters, expires_at
        )

    def revoke_agent_permission(self, permission_id: str) -> bool:
        """Revoke an agent permission."""
        return self.multiagent.revoke_permission(permission_id)

    def check_agent_permission(
        self, agent_id: str, target_agent_id: str, permission: PermissionType
    ) -> bool:
        """Check if an agent has permission to access another agent's memories."""
        return self.multiagent.check_permission(agent_id, target_agent_id, permission)

    def list_agent_permissions(
        self,
        granter_agent_id: Optional[str] = None,
        grantee_agent_id: Optional[str] = None,
    ) -> List[AgentPermission]:
        """List permissions, optionally filtered."""
        return self.multiagent.list_permissions(granter_agent_id, grantee_agent_id)

    def transfer_memory(
        self,
        memory_id: str,
        source_agent_id: str,
        target_agent_id: str,
        transfer_type: str = "copy",
    ) -> Optional[Any]:
        """Transfer a memory from one agent to another.

        Args:
            memory_id: Memory ID to transfer
            source_agent_id: Source agent ID
            target_agent_id: Target agent ID
            transfer_type: "copy", "move", or "share"

        Returns:
            MemoryTransfer record or None if not allowed

        Example:
            >>> # Copy a memory from agent1 to agent2
            >>> transfer = client.transfer_memory(memory.id, agent1.id, agent2.id, "copy")
        """
        # Get the memory
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])

                # Transfer
                transfer = self.multiagent.transfer_memory(
                    memory, source_agent_id, target_agent_id, transfer_type
                )

                if transfer and transfer_type in ["copy", "share"]:
                    # Create copy for target agent
                    copied = memory.model_copy(deep=True)
                    copied.id = str(uuid4())
                    copied.agent_id = target_agent_id
                    copied.metadata["transferred_from"] = source_agent_id
                    copied.metadata["transfer_type"] = transfer_type

                    # Store copied memory
                    collection = memory.collection_name(
                        self.config.collection_facts, self.config.collection_prefs
                    )
                    vector = self.embedder.encode_single(copied.text)
                    self.qdrant.upsert(
                        collection_name=collection,
                        id=copied.id,
                        vector=vector,
                        payload=copied.model_dump(mode="json"),
                    )

                    # If move, delete original
                    if transfer_type == "move":
                        self.delete_memory(memory_id, memory.user_id)

                return transfer

        return None

    def get_agent_memories(
        self,
        agent_id: str,
        requesting_agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Get memories for an agent, respecting permissions.

        Args:
            agent_id: Agent whose memories to retrieve
            requesting_agent_id: Agent requesting access (for permission check)
            filters: Optional additional filters
            limit: Maximum memories to return

        Returns:
            List of accessible Memory objects

        Example:
            >>> # Get agent1's own memories
            >>> memories = client.get_agent_memories(agent1.id)
            >>>
            >>> # Get agent1's memories from agent2's perspective (filtered by permissions)
            >>> memories = client.get_agent_memories(agent1.id, requesting_agent_id=agent2.id)
        """
        # Get agent to verify user_id
        agent = self.multiagent.get_agent(agent_id)
        if not agent:
            return []

        # Get all memories for the user
        filters = filters or {}
        filters["agent_id"] = agent_id
        memories = self.get_memories(agent.user_id, filters=filters, limit=limit)

        # Filter by permissions if requesting from another agent
        if requesting_agent_id and requesting_agent_id != agent_id:
            memories = self.multiagent.filter_accessible_memories(
                requesting_agent_id, memories, PermissionType.READ
            )

        return memories

    def get_agent_stats(self, agent_id: str) -> Optional[Any]:
        """Get memory statistics for an agent.

        Returns:
            AgentMemoryStats object with detailed statistics
        """
        agent = self.multiagent.get_agent(agent_id)
        if not agent:
            return None

        memories = self.get_memories(agent.user_id, limit=10000)
        return self.multiagent.get_agent_stats(agent_id, memories)

    # === TEMPORAL REASONING ===

    def get_memories_by_time_range(
        self,
        user_id: str,
        time_range: Optional[TimeRange] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Get memories within a specific time range.

        Args:
            user_id: User ID
            time_range: Predefined time range (e.g., LAST_WEEK, THIS_MONTH)
            start_time: Custom start time (if time_range not specified)
            end_time: Custom end time (if time_range not specified)
            filters: Additional filters
            limit: Maximum memories to return

        Returns:
            List of memories within time range

        Example:
            >>> # Get memories from last week
            >>> memories = client.get_memories_by_time_range("alice", time_range=TimeRange.LAST_WEEK)
            >>>
            >>> # Get memories from custom date range
            >>> from datetime import datetime, timedelta, timezone
            >>> start = datetime.now(timezone.utc) - timedelta(days=7)
            >>> end = datetime.now(timezone.utc)
            >>> memories = client.get_memories_by_time_range("alice", start_time=start, end_time=end)
        """
        # Get all memories
        all_memories = self.get_memories(user_id, filters=filters, limit=limit * 2)

        # Filter by time range
        filtered = self.temporal_analyzer.filter_by_time_range(
            all_memories, time_range, start_time, end_time
        )

        return filtered[:limit]

    def build_memory_narrative(
        self, user_id: str, time_range: Optional[TimeRange] = None, title: Optional[str] = None
    ) -> str:
        """Build a chronological narrative from user's memories.

        Args:
            user_id: User ID
            time_range: Optional time range filter
            title: Optional narrative title

        Returns:
            Formatted chronological narrative

        Example:
            >>> narrative = client.build_memory_narrative("alice", TimeRange.LAST_MONTH, "My Month")
            >>> print(narrative)
        """
        memories = self.get_memories(user_id, limit=1000)

        if time_range:
            memories = self.temporal_analyzer.filter_by_time_range(memories, time_range)

        return self.temporal_analyzer.build_chronological_narrative(memories, title)

    def create_memory_timeline(
        self,
        user_id: str,
        title: str = "Memory Timeline",
        time_range: Optional[TimeRange] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Timeline:
        """Create a timeline of user's memories.

        Args:
            user_id: User ID
            title: Timeline title
            time_range: Optional predefined time range
            start_time: Optional custom start time
            end_time: Optional custom end time

        Returns:
            Timeline object with temporal events

        Example:
            >>> timeline = client.create_memory_timeline("alice", "Last Week", TimeRange.LAST_WEEK)
            >>> print(f"Timeline has {len(timeline.events)} events")
            >>> print(f"Duration: {timeline.get_duration()}")
        """
        memories = self.get_memories(user_id, limit=1000)

        if time_range:
            memories = self.temporal_analyzer.filter_by_time_range(memories, time_range)
        elif start_time and end_time:
            memories = self.temporal_analyzer.filter_by_time_range(
                memories, None, start_time, end_time
            )

        return self.temporal_analyzer.create_timeline(memories, user_id, title, start_time, end_time)

    def analyze_event_sequences(
        self, user_id: str, max_gap_hours: int = 24
    ) -> List[List[Memory]]:
        """Identify sequences of related events in memories.

        Args:
            user_id: User ID
            max_gap_hours: Maximum time gap to consider events related

        Returns:
            List of event sequences (each is a list of memories)

        Example:
            >>> sequences = client.analyze_event_sequences("alice", max_gap_hours=6)
            >>> for i, seq in enumerate(sequences):
            >>>     print(f"Sequence {i+1}: {len(seq)} related events")
        """
        memories = self.get_memories(user_id, limit=1000)
        return self.temporal_analyzer.analyze_event_sequences(memories, max_gap_hours)

    def schedule_memory(
        self,
        text: str,
        user_id: str,
        scheduled_for: datetime,
        type: str = "fact",
        tags: Optional[List[str]] = None,
        recurrence: Optional[str] = None,
        reminder_offset: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledMemory:
        """Schedule a memory for future creation.

        Args:
            text: Memory text
            user_id: User ID
            scheduled_for: When to create the memory
            type: Memory type
            tags: Optional tags
            recurrence: Optional recurrence ("daily", "weekly", "monthly")
            reminder_offset: Minutes before scheduled_for to trigger
            metadata: Optional metadata

        Returns:
            ScheduledMemory object

        Example:
            >>> from datetime import datetime, timedelta, timezone
            >>> tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            >>> scheduled = client.schedule_memory(
            >>>     "Follow up on project",
            >>>     "alice",
            >>>     tomorrow,
            >>>     recurrence="daily"
            >>> )
        """
        return self.temporal_analyzer.schedule_memory(
            text=text,
            user_id=user_id,
            scheduled_for=scheduled_for,
            type=MemoryType(type),
            tags=tags,
            recurrence=recurrence,
            reminder_offset=reminder_offset,
            metadata=metadata,
        )

    def get_due_scheduled_memories(self) -> List[ScheduledMemory]:
        """Get scheduled memories that are due for creation.

        Returns:
            List of due scheduled memories

        Example:
            >>> due = client.get_due_scheduled_memories()
            >>> for scheduled in due:
            >>>     memory = client.remember(scheduled.text, scheduled.user_id)
            >>>     client.trigger_scheduled_memory(scheduled.id)
        """
        return self.temporal_analyzer.get_due_scheduled_memories()

    def trigger_scheduled_memory(self, scheduled_id: str) -> bool:
        """Mark a scheduled memory as triggered.

        Args:
            scheduled_id: Scheduled memory ID

        Returns:
            True if triggered successfully
        """
        return self.temporal_analyzer.trigger_scheduled_memory(scheduled_id)

    def get_temporal_summary(self, user_id: str) -> Dict[str, Any]:
        """Get temporal statistics for user's memories.

        Args:
            user_id: User ID

        Returns:
            Dictionary with temporal statistics

        Example:
            >>> stats = client.get_temporal_summary("alice")
            >>> print(f"Time span: {stats['time_span_days']} days")
            >>> print(f"Peak activity: {stats['peak_activity_hour']}")
        """
        memories = self.get_memories(user_id, limit=10000)
        return self.temporal_analyzer.get_temporal_summary(memories, user_id)

    # === CROSS-SESSION INSIGHTS ===

    def detect_patterns(
        self, user_id: str, session_ids: Optional[List[str]] = None
    ) -> List[Pattern]:
        """Detect behavioral patterns across memories and sessions.

        Args:
            user_id: User ID
            session_ids: Optional list of session IDs to analyze

        Returns:
            List of detected patterns

        Example:
            >>> patterns = client.detect_patterns("alice")
            >>> for pattern in patterns[:5]:
            >>>     print(f"{pattern.pattern_type}: {pattern.description}")
            >>>     print(f"  Confidence: {pattern.confidence:.2f}")
            >>>     print(f"  Occurrences: {pattern.occurrences}")
        """
        memories = self.get_memories(user_id, limit=10000)

        # Get sessions if IDs provided
        sessions = None
        if session_ids:
            sessions = [self.session_manager.get_session(sid) for sid in session_ids]
            sessions = [s for s in sessions if s is not None]

        return self.insight_analyzer.detect_patterns(memories, user_id, sessions)

    def track_behavior_changes(
        self,
        user_id: str,
        comparison_days: int = 30,
    ) -> List[BehaviorChange]:
        """Track changes in user behavior between time periods.

        Args:
            user_id: User ID
            comparison_days: Days to use for comparison (compares recent vs older)

        Returns:
            List of detected behavior changes

        Example:
            >>> changes = client.track_behavior_changes("alice", comparison_days=30)
            >>> for change in changes:
            >>>     print(f"{change.change_type.value}: {change.description}")
            >>>     print(f"  Confidence: {change.confidence:.2f}")
        """
        all_memories = self.get_memories(user_id, limit=10000)

        # Split into old and new periods
        cutoff = datetime.now(timezone.utc) - timedelta(days=comparison_days)
        old_memories = [m for m in all_memories if m.created_at < cutoff]
        new_memories = [m for m in all_memories if m.created_at >= cutoff]

        return self.insight_analyzer.track_behavior_changes(old_memories, new_memories, user_id)

    def analyze_preference_drift(
        self, user_id: str, category: Optional[str] = None
    ) -> List[PreferenceDrift]:
        """Analyze how user preferences have changed over time.

        Args:
            user_id: User ID
            category: Optional category to filter

        Returns:
            List of preference drift analyses

        Example:
            >>> drifts = client.analyze_preference_drift("alice")
            >>> for drift in drifts:
            >>>     print(f"Category: {drift.category}")
            >>>     print(f"  Original: {drift.original_preference}")
            >>>     print(f"  Current: {drift.current_preference}")
            >>>     print(f"  Drift score: {drift.drift_score:.2f}")
        """
        memories = self.get_memories(user_id, limit=10000)
        return self.insight_analyzer.analyze_preference_drift(memories, user_id, category)

    def detect_habits(self, user_id: str, min_occurrences: int = 5) -> List[HabitScore]:
        """Detect and score potential habits from user's memories.

        Args:
            user_id: User ID
            min_occurrences: Minimum occurrences to consider as habit

        Returns:
            List of habit scores (sorted by score)

        Example:
            >>> habits = client.detect_habits("alice", min_occurrences=5)
            >>> for habit in habits[:3]:
            >>>     print(f"Behavior: {habit.behavior}")
            >>>     print(f"  Habit score: {habit.habit_score:.2f}")
            >>>     print(f"  Status: {habit.status}")
            >>>     print(f"  Frequency: {habit.frequency} times")
            >>>     print(f"  Consistency: {habit.consistency:.2f}")
        """
        memories = self.get_memories(user_id, limit=10000)
        return self.insight_analyzer.detect_habit_formation(memories, user_id, min_occurrences)

    def analyze_trends(self, user_id: str, window_days: int = 30) -> List[Trend]:
        """Analyze long-term trends in user behavior.

        Args:
            user_id: User ID
            window_days: Analysis window in days

        Returns:
            List of trends

        Example:
            >>> trends = client.analyze_trends("alice", window_days=30)
            >>> for trend in trends:
            >>>     print(f"Category: {trend.category}")
            >>>     print(f"  Trend: {trend.trend_type} ({trend.direction})")
            >>>     print(f"  Strength: {trend.strength:.2f}")
        """
        memories = self.get_memories(user_id, limit=10000)
        return self.insight_analyzer.analyze_trends(memories, user_id, window_days)
