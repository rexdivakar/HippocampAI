"""Main MemoryClient - public API."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, Union, cast
from uuid import uuid4

from hippocampai.adapters.provider_anthropic import AnthropicLLM
from hippocampai.adapters.provider_groq import GroqLLM
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import Config, get_config
from hippocampai.embed.embedder import get_embedder
from hippocampai.graph import RelationType
from hippocampai.graph.knowledge_graph import KnowledgeGraph
from hippocampai.models.agent import (
    Agent,
    AgentPermission,
    AgentRole,
    MemoryVisibility,
    PermissionType,
    Run,
)
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.models.session import Session, SessionSearchResult, SessionStatus
from hippocampai.multiagent import MultiAgentManager
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.conversation_memory import ConversationSummary, ConversationTurn
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.entity_recognition import (
    Entity,
    EntityRecognizer,
    EntityRelationship,
    EntityType,
)
from hippocampai.pipeline.extractor import MemoryExtractor
from hippocampai.pipeline.fact_extraction import ExtractedFact, FactExtractionPipeline
from hippocampai.pipeline.importance import ImportanceScorer
from hippocampai.pipeline.insights import (
    BehaviorChange,
    HabitScore,
    InsightAnalyzer,
    Pattern,
    PreferenceDrift,
    Trend,
)
from hippocampai.pipeline.memory_merge import MergeCandidate, MergeResult
from hippocampai.pipeline.semantic_clustering import MemoryCluster, SemanticCategorizer
from hippocampai.pipeline.smart_updater import SmartMemoryUpdater
from hippocampai.pipeline.summarization import SessionSummary, Summarizer, SummaryStyle
from hippocampai.pipeline.temporal import ScheduledMemory, TemporalAnalyzer, Timeline, TimeRange
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.session import SessionManager
from hippocampai.storage import MemoryKVStore
from hippocampai.telemetry import MemoryTrace, OperationType, get_telemetry
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
        weights: Optional[dict[str, float]] = None,
        half_lives: Optional[dict[str, int]] = None,
        allow_cloud: Optional[bool] = None,
        config: Optional[Config] = None,
        enable_telemetry: bool = True,
        user_auth: Optional[bool] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize memory client.

        Args:
            user_auth: Whether to use authentication (None = auto-detect from env, False = local mode, True = remote mode)
            api_key: API key for authentication (if None, will check HIPPOCAMPAI_API_KEY env var)
        """
        self.config = config or get_config()
        self.telemetry = get_telemetry(enabled=enable_telemetry)

        # Authentication setup
        import os

        self.user_auth = (
            user_auth
            if user_auth is not None
            else os.getenv("HIPPOCAMPAI_USER_AUTH", "false").lower() == "true"
        )
        self.api_key = api_key or os.getenv("HIPPOCAMPAI_API_KEY")

        if self.user_auth and not self.api_key:
            logger.warning(
                "user_auth=True but no API key provided. Set api_key parameter or HIPPOCAMPAI_API_KEY environment variable."
            )

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
        self.llm: Optional[Union[OllamaLLM, OpenAILLM, GroqLLM, AnthropicLLM]] = None
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
        elif self.config.llm_provider == "anthropic" and self.config.allow_cloud:
            import os

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm = AnthropicLLM(api_key=api_key, model=self.config.llm_model)

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

        # Intelligence modules
        self.fact_extractor = FactExtractionPipeline(llm=self.llm)
        self.entity_recognizer = EntityRecognizer(llm=self.llm)
        self.summarizer = Summarizer(llm=self.llm)

        # Advanced Features
        self.graph = KnowledgeGraph()  # Enhanced with entity and fact support
        self.version_control = MemoryVersionControl()
        self.kv_store = MemoryKVStore(cache_ttl=300)  # 5 min cache
        self.context_injector = ContextInjector()
        self.multiagent = MultiAgentManager()  # Multi-agent support

        # NEW: Conflict Resolution & Provenance Tracking
        from hippocampai.pipeline.conflict_resolution import (
            ConflictResolutionStrategy,
            MemoryConflictResolver,
        )
        from hippocampai.pipeline.provenance_tracker import ProvenanceTracker

        self.conflict_resolver = MemoryConflictResolver(
            embedder=self.embedder,
            llm=self.llm,
            default_strategy=ConflictResolutionStrategy.TEMPORAL,
            similarity_threshold=0.75,
            contradiction_threshold=0.85,
        )
        self.provenance_tracker = ProvenanceTracker(llm=self.llm)

        # NEW: Memory Lifecycle Management
        from hippocampai.pipeline.memory_lifecycle import LifecycleConfig, MemoryLifecycleManager

        self.lifecycle_manager = MemoryLifecycleManager(config=LifecycleConfig())

        # NEW: Memory Health & Quality Monitoring
        from hippocampai.pipeline.memory_health import MemoryHealthMonitor
        from hippocampai.pipeline.memory_observability import MemoryObservabilityMonitor
        from hippocampai.pipeline.temporal_enhancement import EnhancedTemporalAnalyzer

        self.health_monitor = MemoryHealthMonitor(
            qdrant_store=self.qdrant,
            embedder=self.embedder,
            stale_threshold_days=180,
            near_duplicate_threshold=0.85,
            exact_duplicate_threshold=0.95,
        )
        self.enhanced_temporal = EnhancedTemporalAnalyzer(
            default_half_life_days=90,
            freshness_window_days=30,
        )
        self.observability = MemoryObservabilityMonitor(
            enable_profiling=enable_telemetry,
            slow_query_threshold_ms=1000.0,
            track_access_patterns=True,
        )

        # NEW: Conversation-Aware Memory & Merge Engine
        from hippocampai.pipeline.conversation_memory import ConversationMemoryManager
        from hippocampai.pipeline.memory_merge import MemoryMergeEngine

        self.conversation_manager = ConversationMemoryManager(
            llm=self.llm,
        )
        self.merge_engine = MemoryMergeEngine(
            similarity_threshold=0.85,
            auto_merge_threshold=0.7,
        )

        # NEW: Memory Operations
        from hippocampai.pipeline.memory_operations import MemoryOperations

        self.memory_ops = MemoryOperations()

        # NEW: Adaptive Learning Engine
        from hippocampai.pipeline.adaptive_learning import AdaptiveLearningEngine

        self.adaptive_learning = AdaptiveLearningEngine(
            access_history_limit=10000,
            pattern_analysis_window_days=30,
        )

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

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Dict with authentication headers
        """
        headers = {}

        if self.user_auth:
            headers["X-User-Auth"] = "true"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["X-User-Auth"] = "false"

        return headers

    @classmethod
    def from_preset(cls, preset: PresetType, **overrides: Any) -> "MemoryClient":
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
        tags: Optional[list[str]] = None,
        ttl_days: Optional[int] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        visibility: Optional[str] = None,
        auto_resolve_conflicts: bool = False,
        resolution_strategy: str = "temporal",
    ) -> Memory:
        """
        Store a memory with optional automatic conflict resolution.

        Args:
            text: Memory text content
            user_id: User identifier
            session_id: Optional session identifier
            type: Memory type (fact, preference, goal, habit, event, context)
            importance: Importance score (0-10), auto-calculated if None
            tags: Optional tags for categorization
            ttl_days: Time-to-live in days, None for no expiration
            agent_id: Optional agent identifier for multi-agent systems
            run_id: Optional run identifier
            visibility: Memory visibility (private, shared, public)
            auto_resolve_conflicts: If True, automatically detect and resolve conflicts (default: False)
            resolution_strategy: Strategy for auto-resolution - "temporal" (latest wins),
                                "confidence" (higher confidence wins), "importance" (higher importance wins),
                                "auto_merge" (LLM merges both), "keep_both" (flag both) (default: "temporal")

        Returns:
            Memory object (may be merged/updated if conflicts were auto-resolved)

        Example:
            >>> # Basic usage (no auto-resolve)
            >>> memory = client.remember("I love coffee", user_id="alice")

            >>> # With auto-resolve (Mem0-style)
            >>> memory = client.remember(
            ...     "I hate coffee now",
            ...     user_id="alice",
            ...     auto_resolve_conflicts=True,
            ...     resolution_strategy="temporal"  # Latest wins
            ... )

            >>> # Auto-merge to preserve history
            >>> memory = client.remember(
            ...     "I work at Facebook",
            ...     user_id="alice",
            ...     auto_resolve_conflicts=True,
            ...     resolution_strategy="auto_merge"  # Merges with existing
            ... )
        """
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
            logger.info(
                f"Enrichment: {original_type} -> {memory.type} for text '{memory.text[:50]}'"
            )
            self.telemetry.add_event(trace_id, "semantic_enrichment", status="success")

            # Calculate size metrics
            memory.calculate_size_metrics()

            # Track size in telemetry
            self.telemetry.track_memory_size(memory.text_length, memory.token_count)

            # Check for similar memories and decide on smart update
            self.telemetry.add_event(trace_id, "smart_update_check", status="in_progress")
            existing_memories = self.get_memories(user_id, limit=100)
            similar = self.categorizer.find_similar_memories(
                memory, existing_memories, similarity_threshold=0.85
            )

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
                    updated_existing = self.smart_updater.update_confidence(
                        existing_memory, "reinforcement"
                    )
                    self.update_memory(
                        memory_id=existing_memory.id,
                        importance=updated_existing.confidence
                        * 10,  # Scale confidence to importance
                    )
                    logger.info(f"Skipping similar memory: {decision.reason}")
                    self.telemetry.end_trace(
                        trace_id, status="skipped", result={"reason": decision.reason}
                    )
                    return existing_memory

                if decision.action == "update":
                    # Update existing memory
                    if decision.merged_memory:
                        self.update_memory(
                            memory_id=existing_memory.id,
                            text=decision.merged_memory.text,
                            importance=decision.merged_memory.importance,
                            tags=decision.merged_memory.tags,
                        )
                        logger.info(f"Updated existing memory: {decision.reason}")
                        self.telemetry.end_trace(
                            trace_id, status="updated", result={"reason": decision.reason}
                        )
                        return decision.merged_memory

                elif decision.action == "merge":
                    # Delete old, store merged
                    if decision.merged_memory:
                        self.delete_memory(existing_memory.id, user_id)
                        memory = decision.merged_memory
                        logger.info(f"Merged memories: {decision.reason}")
                # If action is "keep_both", continue with normal storage
            else:
                self.telemetry.add_event(
                    trace_id, "smart_update_check", status="success", action="new"
                )

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

            # AUTO-RESOLVE CONFLICTS (Mem0-style)
            if auto_resolve_conflicts:
                self.telemetry.add_event(trace_id, "auto_conflict_resolution", status="in_progress")

                # Refetch memories to include the newly stored one
                all_memories = self.get_memories(user_id, limit=100)
                # Remove the newly stored memory from the list to compare against others
                other_memories = [m for m in all_memories if m.id != memory.id]

                # Detect conflicts with the newly stored memory
                # Use LLM if available for better conflict detection
                conflicts = self.conflict_resolver.detect_conflicts(
                    memory, other_memories, check_llm=(self.llm is not None)
                )

                if conflicts:
                    logger.info(
                        f"Auto-resolve: Found {len(conflicts)} conflict(s) for memory {memory.id}, "
                        f"using strategy '{resolution_strategy}'"
                    )

                    from hippocampai.pipeline.conflict_resolution import ConflictResolutionStrategy

                    # Resolve each conflict
                    for conflict in conflicts:
                        try:
                            resolution = self.conflict_resolver.resolve_conflict(
                                conflict, strategy=ConflictResolutionStrategy(resolution_strategy)
                            )

                            # Apply resolution
                            if resolution.action in ["keep_first", "keep_second"]:
                                # Delete the loser memory
                                for mem_id in resolution.deleted_memory_ids:
                                    self.delete_memory(mem_id, user_id)
                                    logger.info(
                                        f"Auto-resolve: Deleted conflicting memory {mem_id}"
                                    )

                                # If the new memory was deleted (keep_first), return the existing one
                                if memory.id in resolution.deleted_memory_ids and resolution.updated_memory:
                                    memory = resolution.updated_memory
                                    logger.info(
                                        f"Auto-resolve: Keeping existing memory {memory.id} "
                                        f"(strategy: {resolution_strategy})"
                                    )

                            elif resolution.action == "merge":
                                # Delete originals, store merged
                                if resolution.updated_memory:
                                    for mem_id in resolution.deleted_memory_ids:
                                        self.delete_memory(mem_id, user_id)

                                    # Store the merged memory
                                    merged_collection = resolution.updated_memory.collection_name(
                                        self.config.collection_facts, self.config.collection_prefs
                                    )
                                    merged_vector = self.embedder.encode_single(
                                        resolution.updated_memory.text
                                    )
                                    self.qdrant.upsert(
                                        collection_name=merged_collection,
                                        id=resolution.updated_memory.id,
                                        vector=merged_vector,
                                        payload=resolution.updated_memory.model_dump(mode="json"),
                                    )

                                    memory = resolution.updated_memory
                                    logger.info(
                                        f"Auto-resolve: Merged into memory {memory.id}, "
                                        f"deleted {len(resolution.deleted_memory_ids)} memories"
                                    )

                                    # Track provenance for merged memory
                                    self.provenance_tracker.track_merge(
                                        memory,
                                        [conflict.memory_1, conflict.memory_2],
                                        merge_strategy=resolution_strategy,
                                    )

                            elif resolution.action in ["flag", "keep_both"]:
                                # Update both memories with conflict metadata
                                if conflict.memory_1.metadata.get("has_conflict"):
                                    self.update_memory(
                                        conflict.memory_1.id, metadata=conflict.memory_1.metadata
                                    )
                                if conflict.memory_2.metadata.get("has_conflict"):
                                    self.update_memory(
                                        conflict.memory_2.id, metadata=conflict.memory_2.metadata
                                    )
                                logger.info(
                                    f"Auto-resolve: Flagged memories for review "
                                    f"({conflict.memory_1.id}, {conflict.memory_2.id})"
                                )

                        except Exception as e:
                            logger.error(f"Auto-resolve failed for conflict: {e}")
                            # Continue with other conflicts

                    self.telemetry.add_event(
                        trace_id,
                        "auto_conflict_resolution",
                        status="success",
                        conflicts_found=len(conflicts),
                        strategy=resolution_strategy,
                    )
                else:
                    self.telemetry.add_event(
                        trace_id, "auto_conflict_resolution", status="success", conflicts_found=0
                    )

            self.telemetry.end_trace(
                trace_id,
                status="success",
                result={
                    "memory_id": memory.id,
                    "collection": collection,
                    "auto_resolved": auto_resolve_conflicts,
                },
            )

            # Track memory creation event
            try:
                from hippocampai.monitoring.memory_tracker import (
                    MemoryEventSeverity,
                    MemoryEventType,
                    get_tracker,
                )

                tracker = get_tracker()
                tracker.track_event(
                    memory_id=memory.id,
                    user_id=user_id,
                    event_type=MemoryEventType.CREATED,
                    severity=MemoryEventSeverity.INFO,
                    metadata={
                        "type": memory.type.value,
                        "importance": memory.importance,
                        "session_id": session_id,
                        "tags": tags or [],
                        "auto_resolved": auto_resolve_conflicts,
                    },
                    success=True,
                )
            except Exception as track_err:
                logger.warning(f"Failed to track memory creation: {track_err}")

            # Track Prometheus metrics
            try:
                from hippocampai.monitoring.prometheus_metrics import (
                    memories_created_total,
                    memory_operations_total,
                    memory_size_bytes,
                )

                memory_operations_total.labels(operation="create", status="success").inc()
                memories_created_total.labels(memory_type=memory.type.value).inc()
                memory_size_bytes.labels(memory_type=memory.type.value).observe(
                    len(text.encode("utf-8"))
                )
            except Exception as metrics_err:
                logger.warning(f"Failed to track Prometheus metrics: {metrics_err}")

            return memory
        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})

            # Track failed memory creation
            try:
                from hippocampai.monitoring.memory_tracker import (
                    MemoryEventSeverity,
                    MemoryEventType,
                    get_tracker,
                )

                tracker = get_tracker()
                tracker.track_event(
                    memory_id="unknown",
                    user_id=user_id,
                    event_type=MemoryEventType.CREATED,
                    severity=MemoryEventSeverity.ERROR,
                    metadata={"text_length": len(text), "type": type},
                    success=False,
                    error_message=str(e),
                )
            except Exception as track_err:
                logger.warning(f"Failed to track memory creation error: {track_err}")

            raise

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
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
            results = cast(
                list[Any],
                self.retriever.retrieve(
                    query=query, user_id=user_id, session_id=session_id, k=k, filters=filters
                ),
            )
            self.telemetry.add_event(
                trace_id, "hybrid_retrieval", status="success", results_count=len(results)
            )

            logger.info(f"Retrieved {len(results)} memories for user {user_id}")
            self.telemetry.end_trace(
                trace_id, status="success", result={"count": len(results), "k": k}
            )

            # Track search event for each retrieved memory
            try:
                from hippocampai.monitoring.memory_tracker import (
                    MemoryEventSeverity,
                    MemoryEventType,
                    get_tracker,
                )

                tracker = get_tracker()
                for result in results:
                    tracker.track_event(
                        memory_id=result.memory_id
                        if hasattr(result, "memory_id")
                        else str(result.id)
                        if hasattr(result, "id")
                        else "unknown",
                        user_id=user_id,
                        event_type=MemoryEventType.SEARCHED,
                        severity=MemoryEventSeverity.DEBUG,
                        metadata={
                            "query": query,
                            "score": result.score if hasattr(result, "score") else None,
                            "rank": results.index(result) + 1,
                        },
                        success=True,
                    )
            except Exception as track_err:
                logger.warning(f"Failed to track search events: {track_err}")

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
        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    def extract_from_conversation(
        self, conversation: str, user_id: str, session_id: Optional[str] = None
    ) -> list[Memory]:
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
    def get_telemetry_metrics(self) -> dict[str, Any]:
        """Get telemetry metrics summary."""
        return self.telemetry.get_metrics_summary()

    def get_recent_operations(self, limit: int = 10, operation: Optional[str] = None) -> list[MemoryTrace]:
        """Get recent memory operations with traces."""
        from hippocampai.telemetry import OperationType

        op_type = OperationType(operation) if operation else None
        return self.telemetry.get_recent_traces(limit=limit, operation=op_type)

    def export_telemetry(self, trace_ids: Optional[list[str]] = None) -> list[dict]:
        """Export telemetry data for external analysis."""
        return cast(list[dict], self.telemetry.export_traces(trace_ids=trace_ids))

    def get_memory_statistics(self, user_id: str) -> dict[str, Any]:
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

    def _get_size_by_type(self, memories: list[Memory]) -> dict[str, dict[str, Any]]:
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
            by_type[type_name]["avg_chars"] = int(by_type[type_name]["total_chars"] / count)
            by_type[type_name]["avg_tokens"] = int(by_type[type_name]["total_tokens"] / count)

        return by_type

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
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

            if not memory_data or collection is None:
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

            # Track memory update event
            try:
                from hippocampai.monitoring.memory_tracker import (
                    MemoryEventSeverity,
                    MemoryEventType,
                    get_tracker,
                )

                tracker = get_tracker()
                updated_fields = []
                if text is not None:
                    updated_fields.append("text")
                if importance is not None:
                    updated_fields.append("importance")
                if tags is not None:
                    updated_fields.append("tags")
                if metadata is not None:
                    updated_fields.append("metadata")
                if expires_at is not None:
                    updated_fields.append("expires_at")

                tracker.track_event(
                    memory_id=memory_id,
                    user_id=memory.user_id,
                    event_type=MemoryEventType.UPDATED,
                    severity=MemoryEventSeverity.INFO,
                    metadata={"updated_fields": updated_fields},
                    success=True,
                )
            except Exception as track_err:
                logger.warning(f"Failed to track memory update: {track_err}")

            # Track Prometheus metrics
            try:
                from hippocampai.monitoring.prometheus_metrics import memory_operations_total

                memory_operations_total.labels(operation="update", status="success").inc()
            except Exception as metrics_err:
                logger.warning(f"Failed to track Prometheus update metrics: {metrics_err}")

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

                # Track memory deletion event
                try:
                    from hippocampai.monitoring.memory_tracker import (
                        MemoryEventSeverity,
                        MemoryEventType,
                        get_tracker,
                    )

                    tracker = get_tracker()
                    tracker.track_event(
                        memory_id=memory_id,
                        user_id=user_id or "system",
                        event_type=MemoryEventType.DELETED,
                        severity=MemoryEventSeverity.INFO,
                        metadata={},
                        success=True,
                    )
                except Exception as track_err:
                    logger.warning(f"Failed to track memory deletion: {track_err}")

                # Track Prometheus metrics
                try:
                    from hippocampai.monitoring.prometheus_metrics import (
                        memory_operations_total,
                    )

                    memory_operations_total.labels(operation="delete", status="success").inc()
                    # We don't have memory_type here, so we'll skip memories_deleted_total for now
                except Exception as metrics_err:
                    logger.warning(f"Failed to track Prometheus delete metrics: {metrics_err}")

                return True
            logger.warning(f"Memory {memory_id} not found")
            self.telemetry.end_trace(trace_id, status="error", result={"error": "not_found"})
            return False

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            return False

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a single memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory object if found, None otherwise

        Example:
            >>> memory = client.get_memory("abc123")
            >>> if memory:
            ...     print(f"Found: {memory.text}")
        """
        try:
            # Try facts collection first
            result = self.qdrant.get(
                collection_name=self.config.collection_facts,
                id=memory_id,
            )
            if result:
                return Memory(**result["payload"])

            # Try prefs collection
            result = self.qdrant.get(
                collection_name=self.config.collection_prefs,
                id=memory_id,
            )
            if result:
                return Memory(**result["payload"])

            return None
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None

    def get_memories(
        self,
        user_id: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Memory]:
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
        memories: list[dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> list[Memory]:
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

    def delete_memories(self, memory_ids: list[str], user_id: Optional[str] = None) -> int:
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
        relation_types: Optional[list[RelationType]] = None,
        max_depth: int = 1,
    ) -> list[tuple]:
        """Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_types: Filter by specific relation types
            max_depth: How many hops to traverse

        Returns:
            List of (memory_id, relation_type, weight) tuples
        """
        return cast(list[tuple], self.graph.get_related_memories(memory_id, relation_types, max_depth))

    def get_memory_clusters(self, user_id: str) -> list[set]:
        """Find clusters of related memories.

        Args:
            user_id: User ID to filter by

        Returns:
            List of memory ID sets (clusters)
        """
        return cast(list[set], self.graph.get_clusters(user_id))

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

    def import_graph_from_json(self, file_path: str, merge: bool = True) -> dict:
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

    def get_memory_history(self, memory_id: str) -> list:
        """Get version history for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of MemoryVersion objects
        """
        return cast(list, self.version_control.get_version_history(memory_id))

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
    ) -> list[AuditEntry]:
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

    def track_memory_access(self, memory_id: str, user_id: str) -> None:
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
        filters: Optional[dict[str, Any]] = None,
        sort_by: str = "created_at",  # created_at, importance, access_count
        sort_order: str = "desc",
        limit: int = 100,
    ) -> list[Memory]:
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

    def start_scheduler(self) -> None:
        """Start the background scheduler for memory maintenance tasks."""
        if self.scheduler:
            self.scheduler.start()
            logger.info("Background scheduler started")
        else:
            logger.warning(
                "Scheduler not initialized (enable_scheduler=False or enable_telemetry=False)"
            )

    def stop_scheduler(self) -> None:
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
            user_memories: dict[str, list[Memory]] = {}
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
        self, memories: list[Memory], threshold: float = 0.85
    ) -> list[list[Memory]]:
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

        except Exception as e:
            logger.error("Importance decay failed: %s", e, exc_info=True)

        # Single exit point satisfies S3516
        return updated_count

    # === SESSION MANAGEMENT ===

    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
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
        tags: Optional[list[str]] = None,
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

    def complete_session(self, session_id: str, generate_summary: bool = True) -> Optional[Session]:
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
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SessionSearchResult]:
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
    ) -> list[Session]:
        """Get sessions for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of Session objects sorted by date
        """
        return self.session_manager.get_user_sessions(user_id, status, limit)

    def get_session_memories(self, session_id: str, limit: int = 100) -> list[Memory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            limit: Maximum number of memories

        Returns:
            List of Memory objects
        """
        return self.get_memories(user_id="*", filters={"session_id": session_id}, limit=limit)

    def get_child_sessions(self, parent_session_id: str) -> list[Session]:
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

    def extract_session_facts(self, session_id: str, force: bool = False) -> list:
        """Extract key facts from session.

        Args:
            session_id: Session ID
            force: Force re-extraction

        Returns:
            List of SessionFact objects
        """
        return cast(list, self.session_manager.extract_session_facts(session_id, force))

    def extract_session_entities(self, session_id: str, force: bool = False) -> dict[str, Any]:
        """Extract entities from session.

        Args:
            session_id: Session ID
            force: Force re-extraction

        Returns:
            Dictionary of entity_name -> Entity
        """
        return cast(dict[str, Any], self.session_manager.extract_session_entities(session_id, force))

    def get_session_statistics(self, session_id: str) -> dict[str, Any]:
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

    def reconcile_user_memories(self, user_id: str) -> list[Memory]:
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

        logger.info(
            f"Reconciled {len(memories)} memories into {len(reconciled)} for user {user_id}"
        )
        return reconciled

    def cluster_user_memories(self, user_id: str, max_clusters: int = 10) -> list[MemoryCluster]:
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

        logger.info(
            f"Clustered {len(memories)} memories into {len(clusters)} topics for user {user_id}"
        )
        return clusters

    def suggest_memory_tags(self, memory: Memory, max_tags: int = 5) -> list[str]:
        """Suggest tags for a memory.

        Args:
            memory: Memory to suggest tags for
            max_tags: Maximum number of tags

        Returns:
            List of suggested tags
        """
        return self.categorizer.suggest_tags(memory, max_tags=max_tags)

    def refine_memory_quality(
        self, memory_id: str, context: Optional[str] = None
    ) -> Optional[Memory]:
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
        metadata: Optional[dict[str, Any]] = None,
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

    def list_agents(self, user_id: Optional[str] = None) -> list[Agent]:
        """List all agents, optionally filtered by user."""
        return self.multiagent.list_agents(user_id)

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[AgentRole] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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

    def list_runs(self, agent_id: Optional[str] = None, user_id: Optional[str] = None) -> list[Run]:
        """List runs, optionally filtered by agent or user."""
        return self.multiagent.list_runs(agent_id, user_id)

    def complete_run(self, run_id: str, status: str = "completed") -> Optional[Run]:
        """Mark a run as completed."""
        return self.multiagent.complete_run(run_id, status)

    def grant_agent_permission(
        self,
        granter_agent_id: str,
        grantee_agent_id: str,
        permissions: set[PermissionType],
        memory_filters: Optional[dict[str, Any]] = None,
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
    ) -> list[AgentPermission]:
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
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Memory]:
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
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Memory]:
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

        return self.temporal_analyzer.create_timeline(
            memories, user_id, title, start_time, end_time
        )

    def analyze_event_sequences(self, user_id: str, max_gap_hours: int = 24) -> list[list[Memory]]:
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
        tags: Optional[list[str]] = None,
        recurrence: Optional[str] = None,
        reminder_offset: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
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

    def get_due_scheduled_memories(self) -> list[ScheduledMemory]:
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

    def get_temporal_summary(self, user_id: str) -> dict[str, Any]:
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
        self, user_id: str, session_ids: Optional[list[str]] = None
    ) -> list[Pattern]:
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
        sessions: Optional[list[Session]] = None
        if session_ids:
            sessions_list = [self.session_manager.get_session(sid) for sid in session_ids]
            sessions = [s for s in sessions_list if s is not None]

        return self.insight_analyzer.detect_patterns(memories, user_id, sessions)

    def track_behavior_changes(
        self,
        user_id: str,
        comparison_days: int = 30,
    ) -> list[BehaviorChange]:
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
    ) -> list[PreferenceDrift]:
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

    def detect_habits(self, user_id: str, min_occurrences: int = 5) -> list[HabitScore]:
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

    def analyze_trends(self, user_id: str, window_days: int = 30) -> list[Trend]:
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

    # === INTELLIGENCE FEATURES ===

    def extract_facts(
        self,
        text: str,
        source: str = "text",
        user_id: Optional[str] = None,
    ) -> list[ExtractedFact]:
        """Extract structured facts from text.

        Args:
            text: Input text to extract facts from
            source: Source identifier (e.g., "conversation", "document")
            user_id: Optional user ID for context

        Returns:
            List of ExtractedFact objects

        Example:
            >>> facts = client.extract_facts(
            ...     "John works at Google in San Francisco. He studied Computer Science at MIT.",
            ...     source="profile"
            ... )
            >>> for fact in facts:
            ...     print(f"{fact.category.value}: {fact.fact}")
        """
        return self.fact_extractor.extract_facts(text, source, user_id)

    def extract_facts_from_conversation(
        self,
        conversation: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from a multi-turn conversation.

        Args:
            conversation: Full conversation text
            user_id: User ID
            session_id: Optional session ID

        Returns:
            List of ExtractedFact objects

        Example:
            >>> conversation = '''
            ... User: I'm a software engineer at Tesla
            ... Assistant: That's great! How long have you been there?
            ... User: About 2 years now
            ... '''
            >>> facts = client.extract_facts_from_conversation(conversation, "alice")
        """
        return self.fact_extractor.extract_from_conversation(conversation, user_id, session_id)

    def extract_entities(
        self,
        text: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[Entity]:
        """Extract named entities from text.

        Args:
            text: Input text
            context: Optional context (memory_id, user_id, etc.)

        Returns:
            List of Entity objects

        Example:
            >>> entities = client.extract_entities(
            ...     "Elon Musk founded SpaceX in California in 2002"
            ... )
            >>> for entity in entities:
            ...     print(f"{entity.type.value}: {entity.text}")
        """
        return self.entity_recognizer.extract_entities(text, context)

    def extract_relationships(
        self,
        text: str,
        entities: Optional[list[Entity]] = None,
    ) -> list[EntityRelationship]:
        """Extract relationships between entities.

        Args:
            text: Input text
            entities: Previously extracted entities (optional)

        Returns:
            List of EntityRelationship objects

        Example:
            >>> relationships = client.extract_relationships(
            ...     "Steve Jobs worked at Apple and lived in California"
            ... )
            >>> for rel in relationships:
            ...     print(f"{rel.relation_type.value}: confidence {rel.confidence}")
        """
        return self.entity_recognizer.extract_relationships(text, entities)

    def get_entity_profile(self, entity_id: str) -> Optional[Any]:
        """Get complete profile for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            EntityProfile object or None

        Example:
            >>> profile = client.get_entity_profile("person_john_doe")
            >>> if profile:
            ...     print(f"Name: {profile.canonical_name}")
            ...     print(f"Mentions: {profile.mention_count}")
            ...     print(f"Aliases: {profile.aliases}")
        """
        return self.entity_recognizer.get_entity_profile(entity_id)

    def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        min_mentions: int = 1,
    ) -> list[Any]:
        """Search for entities by name.

        Args:
            query: Search query
            entity_type: Optional entity type filter
            min_mentions: Minimum number of mentions

        Returns:
            List of EntityProfile objects

        Example:
            >>> # Search for all people named "John"
            >>> results = client.search_entities("john", entity_type=EntityType.PERSON)
            >>>
            >>> # Search for frequently mentioned entities
            >>> results = client.search_entities("tesla", min_mentions=5)
        """
        return cast(list[Any], self.entity_recognizer.search_entities(query, entity_type, min_mentions))

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        session_id: str = "unknown",
        style: SummaryStyle = SummaryStyle.CONCISE,
        entities: Optional[list[str]] = None,
    ) -> SessionSummary:
        """Generate summary for a conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            session_id: Session identifier
            style: Summary style (concise, detailed, bullet_points, etc.)
            entities: Pre-extracted entities (optional)

        Returns:
            SessionSummary object

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "I need help with Python"},
            ...     {"role": "assistant", "content": "I'd be happy to help!"},
            ... ]
            >>> summary = client.summarize_conversation(
            ...     messages,
            ...     session_id="session_123",
            ...     style=SummaryStyle.BULLET_POINTS
            ... )
            >>> print(summary.summary)
            >>> print(f"Topics: {summary.topics}")
            >>> print(f"Action items: {summary.action_items}")
        """
        return self.summarizer.summarize_session(messages, session_id, style, entities)

    def create_rolling_summary(
        self,
        messages: list[dict[str, Any]],
        window_size: int = 10,
        style: SummaryStyle = SummaryStyle.CONCISE,
    ) -> str:
        """Create rolling summary of recent messages.

        Args:
            messages: All messages
            window_size: Number of recent messages to summarize
            style: Summary style

        Returns:
            Summary text

        Example:
            >>> summary = client.create_rolling_summary(messages, window_size=5)
        """
        return self.summarizer.create_rolling_summary(messages, window_size, style)

    def extract_conversation_insights(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
    ) -> dict[str, Any]:
        """Extract insights from conversation.

        Args:
            messages: Conversation messages
            user_id: User identifier

        Returns:
            Dictionary with insights including decisions, learning points, patterns

        Example:
            >>> insights = client.extract_conversation_insights(messages, "alice")
            >>> print(f"Topics: {insights['topics']}")
            >>> print(f"Sentiment: {insights['sentiment']}")
            >>> print(f"Key decisions: {insights['key_decisions']}")
        """
        return self.summarizer.extract_insights(messages, user_id)

    # === KNOWLEDGE GRAPH OPERATIONS ===

    def add_entity_to_graph(
        self,
        entity: Entity,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an entity node to the knowledge graph.

        Args:
            entity: Entity object
            metadata: Additional metadata

        Returns:
            Node ID for the entity

        Example:
            >>> entities = client.extract_entities("Steve Jobs founded Apple")
            >>> for entity in entities:
            ...     node_id = client.add_entity_to_graph(entity)
        """
        return self.graph.add_entity(entity, metadata)

    def add_fact_to_graph(
        self,
        fact: ExtractedFact,
        fact_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a fact node to the knowledge graph.

        Args:
            fact: ExtractedFact object
            fact_id: Optional custom fact ID
            metadata: Additional metadata

        Returns:
            Node ID for the fact

        Example:
            >>> facts = client.extract_facts("John works at Google")
            >>> for fact in facts:
            ...     node_id = client.add_fact_to_graph(fact)
        """
        return self.graph.add_fact(fact, fact_id, metadata)

    def link_memory_to_entity(
        self,
        memory_id: str,
        entity_id: str,
        relation_type: RelationType = RelationType.RELATED_TO,
        confidence: float = 0.9,
    ) -> bool:
        """Link a memory to an entity in the knowledge graph.

        Args:
            memory_id: Memory node ID
            entity_id: Entity ID
            relation_type: Type of relationship
            confidence: Confidence score

        Returns:
            True if successful

        Example:
            >>> # Extract entities and link to memory
            >>> entities = client.extract_entities(memory.text)
            >>> for entity in entities:
            ...     client.add_entity_to_graph(entity)
            ...     client.link_memory_to_entity(memory.id, entity.entity_id)
        """
        return self.graph.link_memory_to_entity(memory_id, entity_id, relation_type, confidence)

    def link_memory_to_fact(
        self,
        memory_id: str,
        fact_id: str,
        confidence: float = 0.9,
    ) -> bool:
        """Link a memory to a fact extracted from it.

        Args:
            memory_id: Memory node ID
            fact_id: Fact ID
            confidence: Confidence score

        Returns:
            True if successful
        """
        return self.graph.link_memory_to_fact(memory_id, fact_id, confidence)

    def get_entity_memories(self, entity_id: str) -> list[str]:
        """Get all memories mentioning an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of memory IDs

        Example:
            >>> memory_ids = client.get_entity_memories("person_john_doe")
            >>> for mem_id in memory_ids:
            ...     # Fetch and process memories
            ...     pass
        """
        return self.graph.get_entity_memories(entity_id)

    def get_entity_facts(self, entity_id: str) -> list[str]:
        """Get all facts about an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of fact IDs

        Example:
            >>> fact_ids = client.get_entity_facts("organization_google")
        """
        return self.graph.get_entity_facts(entity_id)

    def get_entity_connections(
        self,
        entity_id: str,
        max_distance: int = 2,
    ) -> dict[str, list[tuple]]:
        """Find all entities connected to a given entity.

        Args:
            entity_id: Source entity ID
            max_distance: Maximum distance (hops) to search

        Returns:
            Dictionary mapping relation types to lists of (entity_id, distance) tuples

        Example:
            >>> connections = client.get_entity_connections("person_john_doe", max_distance=2)
            >>> for relation_type, entities in connections.items():
            ...     print(f"{relation_type}: {len(entities)} connected entities")
        """
        return self.graph.find_entity_connections(entity_id, max_distance)

    def get_knowledge_subgraph(
        self,
        center_id: str,
        radius: int = 2,
        include_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get a subgraph around a central node.

        Args:
            center_id: Central node ID (memory, entity, or fact)
            radius: How many hops to include
            include_types: Filter by node types (e.g., ["entity", "fact"])

        Returns:
            Subgraph data with nodes and edges

        Example:
            >>> subgraph = client.get_knowledge_subgraph("person_john_doe", radius=2)
            >>> print(f"Nodes: {len(subgraph['nodes'])}")
            >>> print(f"Edges: {len(subgraph['edges'])}")
        """
        from hippocampai.graph.knowledge_graph import NodeType

        node_types = None
        if include_types:
            node_types = [NodeType(t) for t in include_types]

        return self.graph.get_knowledge_subgraph(center_id, radius, node_types)

    def get_entity_timeline(self, entity_id: str) -> list[dict[str, Any]]:
        """Get chronological timeline of facts and memories about an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of timeline events sorted by time

        Example:
            >>> timeline = client.get_entity_timeline("person_john_doe")
            >>> for event in timeline:
            ...     print(f"{event['timestamp']}: {event['type']} - {event.get('text', '')}")
        """
        return self.graph.get_entity_timeline(entity_id)

    def infer_knowledge(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Infer new facts from existing knowledge graph patterns.

        Args:
            user_id: Optional user ID to limit inference

        Returns:
            List of inferred facts with confidence scores

        Example:
            >>> inferred = client.infer_knowledge(user_id="alice")
            >>> for fact in inferred:
            ...     print(f"{fact['fact']} (confidence: {fact['confidence']:.2f})")
            ...     print(f"  Rule: {fact['rule']}")
        """
        return self.graph.infer_new_facts(user_id)

    def enrich_memory_with_intelligence(
        self,
        memory: Memory,
        add_to_graph: bool = True,
    ) -> dict[str, Any]:
        """Enrich a memory with facts, entities, and add to knowledge graph.

        This is a convenience method that extracts facts and entities from a memory
        and optionally adds them to the knowledge graph.

        Args:
            memory: Memory object to enrich
            add_to_graph: Whether to add extracted information to knowledge graph

        Returns:
            Dictionary with extracted facts, entities, and relationships

        Example:
            >>> memory = client.remember("Steve Jobs founded Apple in California", "alice")
            >>> enrichment = client.enrich_memory_with_intelligence(memory)
            >>> print(f"Facts: {len(enrichment['facts'])}")
            >>> print(f"Entities: {len(enrichment['entities'])}")
            >>> print(f"Relationships: {len(enrichment['relationships'])}")
        """
        # Extract facts
        facts = self.fact_extractor.extract_facts(
            memory.text,
            source=f"memory_{memory.id}",
            user_id=memory.user_id,
        )

        # Extract entities
        entities = self.entity_recognizer.extract_entities(
            memory.text,
            context={"memory_id": memory.id, "user_id": memory.user_id},
        )

        # Extract relationships
        relationships = self.entity_recognizer.extract_relationships(memory.text, entities)

        # Add to knowledge graph if requested
        if add_to_graph:
            # Add entities
            for entity in entities:
                try:
                    self.graph.add_entity(entity)
                    # Link memory to entity
                    self.graph.link_memory_to_entity(memory.id, entity.entity_id)
                except Exception as e:
                    logger.warning(f"Failed to add entity to graph: {e}")

            # Add facts
            for fact in facts:
                try:
                    fact_id = f"fact_{hash(fact.fact) % 10**10}"
                    self.graph.add_fact(fact, fact_id)
                    # Link memory to fact
                    self.graph.link_memory_to_fact(memory.id, fact_id)

                    # Link facts to entities they mention
                    for entity_name in fact.entities:
                        # Try to find matching entity
                        for entity in entities:
                            if entity.text.lower() == entity_name.lower():
                                self.graph.link_fact_to_entity(fact_id, entity.entity_id)
                                break
                except Exception as e:
                    logger.warning(f"Failed to add fact to graph: {e}")

            # Add entity relationships
            for relationship in relationships:
                try:
                    self.graph.link_entities(relationship)
                except Exception as e:
                    logger.warning(f"Failed to add relationship to graph: {e}")

        return {
            "facts": facts,
            "entities": entities,
            "relationships": relationships,
            "graph_updated": add_to_graph,
        }

    # === CONFLICT RESOLUTION ===

    def detect_memory_conflicts(
        self,
        user_id: str,
        check_llm: bool = True,
        memory_type: Optional[str] = None,
    ) -> list:
        """
        Detect conflicts in user's memories.

        Args:
            user_id: User ID to check memories for
            check_llm: Whether to use LLM for deep contradiction analysis (slower but more accurate)
            memory_type: Optional memory type filter

        Returns:
            List of MemoryConflict objects

        Example:
            >>> conflicts = client.detect_memory_conflicts("alice", check_llm=True)
            >>> print(f"Found {len(conflicts)} conflicts")
            >>> for conflict in conflicts:
            ...     print(f"{conflict.conflict_type}: {conflict.memory_1.text} vs {conflict.memory_2.text}")
        """
        # Get user memories
        memories = self.get_memories(user_id, limit=1000)

        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.type == memory_type]

        if not memories:
            return []

        # Detect conflicts
        conflicts = self.conflict_resolver.batch_detect_conflicts(memories, check_llm=check_llm)

        logger.info(f"Detected {len(conflicts)} conflicts for user {user_id}")
        return cast(list[Any], conflicts)

    def resolve_memory_conflict(
        self,
        conflict_id: str,
        strategy: Optional[str] = None,
        apply_resolution: bool = True,
    ) -> dict[str, Any]:
        """
        Resolve a specific memory conflict.

        Args:
            conflict_id: Conflict ID to resolve
            strategy: Resolution strategy ("temporal", "confidence", "importance", "user_review", "auto_merge", "keep_both")
            apply_resolution: Whether to apply the resolution (update/delete memories)

        Returns:
            Dictionary with resolution details

        Example:
            >>> conflicts = client.detect_memory_conflicts("alice")
            >>> resolution = client.resolve_memory_conflict(
            ...     conflicts[0].id,
            ...     strategy="temporal",
            ...     apply_resolution=True
            ... )
            >>> print(resolution["action"])  # "keep_second", "keep_first", "merge", etc.
        """
        # Note: In a real implementation, you'd store conflicts in a database
        # For now, this is a placeholder showing the API structure
        logger.warning("resolve_memory_conflict requires conflict storage - implement as needed")

        return {
            "conflict_id": conflict_id,
            "strategy": strategy,
            "applied": apply_resolution,
            "note": "Conflict storage not yet implemented - conflicts are detected on-demand",
        }

    def auto_resolve_conflicts(
        self,
        user_id: str,
        strategy: str = "temporal",
        memory_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Automatically detect and resolve all conflicts for a user.

        Args:
            user_id: User ID
            strategy: Default resolution strategy
            memory_type: Optional memory type filter

        Returns:
            Dictionary with resolution summary

        Example:
            >>> result = client.auto_resolve_conflicts("alice", strategy="confidence")
            >>> print(f"Resolved {result['resolved_count']} conflicts")
            >>> print(f"Deleted {result['deleted_count']} memories")
        """
        from hippocampai.pipeline.conflict_resolution import ConflictResolutionStrategy

        # Get conflicts
        conflicts = self.detect_memory_conflicts(user_id, check_llm=True, memory_type=memory_type)

        if not conflicts:
            return {
                "user_id": user_id,
                "conflicts_found": 0,
                "resolved_count": 0,
                "deleted_count": 0,
                "merged_count": 0,
            }

        # Resolve each conflict
        resolved_count = 0
        deleted_ids = []
        merged_count = 0

        for conflict in conflicts:
            try:
                resolution = self.conflict_resolver.resolve_conflict(
                    conflict, strategy=ConflictResolutionStrategy(strategy)
                )

                if resolution.action in ["keep_first", "keep_second"]:
                    # Delete the loser
                    for mem_id in resolution.deleted_memory_ids:
                        self.delete_memory(mem_id)
                        deleted_ids.append(mem_id)
                    resolved_count += 1

                elif resolution.action == "merge":
                    # Create merged memory, delete originals
                    if resolution.updated_memory:
                        self.remember(
                            text=resolution.updated_memory.text,
                            user_id=user_id,
                            type=resolution.updated_memory.type.value,
                            importance=resolution.updated_memory.importance,
                        )
                        for mem_id in resolution.deleted_memory_ids:
                            self.delete_memory(mem_id)
                            deleted_ids.append(mem_id)
                        merged_count += 1
                        resolved_count += 1

                elif resolution.action == "flag":
                    # Update memories with conflict flags
                    if conflict.memory_1.metadata.get("has_conflict"):
                        self.update_memory(
                            conflict.memory_1.id, metadata=conflict.memory_1.metadata
                        )
                    if conflict.memory_2.metadata.get("has_conflict"):
                        self.update_memory(
                            conflict.memory_2.id, metadata=conflict.memory_2.metadata
                        )

            except Exception as e:
                logger.error(f"Failed to resolve conflict: {e}")

        return {
            "user_id": user_id,
            "conflicts_found": len(conflicts),
            "resolved_count": resolved_count,
            "deleted_count": len(deleted_ids),
            "merged_count": merged_count,
            "deleted_memory_ids": deleted_ids,
        }

    # === PROVENANCE & LINEAGE ===

    def track_memory_provenance(
        self,
        memory: Memory,
        source: str = "conversation",
        parent_ids: Optional[list[str]] = None,
        citations: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Initialize or update provenance tracking for a memory.

        Args:
            memory: Memory to track
            source: Source of memory ("conversation", "api_direct", "inference", "merge", "import")
            parent_ids: Parent memory IDs if derived
            citations: List of citation dicts

        Returns:
            Dictionary with lineage information

        Example:
            >>> memory = client.remember("John works at Google", user_id="alice")
            >>> lineage = client.track_memory_provenance(
            ...     memory,
            ...     source="conversation",
            ...     citations=[{"source_type": "message", "source_text": "User mentioned..."}]
            ... )
        """
        from hippocampai.models.provenance import Citation, MemorySource

        # Convert citations
        citation_objects = []
        if citations:
            for cit in citations:
                citation_objects.append(
                    Citation(
                        source_type=cit.get("source_type", "external"),
                        source_id=cit.get("source_id"),
                        source_url=cit.get("source_url"),
                        source_text=cit.get("source_text"),
                        confidence=cit.get("confidence", 1.0),
                    )
                )

        # Initialize lineage
        lineage = self.provenance_tracker.init_lineage(
            memory=memory,
            source=MemorySource(source),
            created_by=memory.user_id,
            parent_ids=parent_ids,
            citations=citation_objects,
        )

        # Update memory in storage
        self.update_memory(memory.id, metadata=memory.metadata)

        return lineage.model_dump()

    def get_memory_lineage(self, memory_id: str) -> Optional[dict[str, Any]]:
        """
        Get complete lineage information for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            Lineage dict or None if not found

        Example:
            >>> lineage = client.get_memory_lineage(memory_id)
            >>> print(f"Source: {lineage['source']}")
            >>> print(f"Parents: {lineage['parent_memory_ids']}")
            >>> print(f"Transformations: {len(lineage['transformations'])}")
        """
        # Try to get memory from both collections
        memory_data = None
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                break

        if not memory_data:
            return None

        memory = Memory(**memory_data["payload"])

        if "lineage" in memory.metadata:
            return cast(Optional[dict[str, Any]], memory.metadata["lineage"])

        return None

    def get_memory_provenance_chain(self, memory_id: str) -> Optional[dict[str, Any]]:
        """
        Build complete provenance chain for a memory (including ancestors).

        Args:
            memory_id: Memory ID

        Returns:
            Provenance chain dict or None

        Example:
            >>> chain = client.get_memory_provenance_chain(memory_id)
            >>> print(f"Chain length: {chain['total_generations']}")
            >>> print(f"Root memories: {chain['root_memory_ids']}")
            >>> for link in chain['chain']:
            ...     print(f"  - {link['memory_id']}: {link['source']}")
        """
        # Get all memories for user (needed to build full chain)
        memory_data = None
        memory = None

        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])
                break

        if not memory:
            return None

        # Get all user memories for chain building
        all_memories_list = self.get_memories(memory.user_id, limit=1000)
        all_memories = {m.id: m for m in all_memories_list}

        # Build chain
        chain = self.provenance_tracker.build_provenance_chain(memory, all_memories)

        return cast(Optional[dict[str, Any]], chain.model_dump())

    def assess_memory_quality(
        self, memory_id: str, use_llm: bool = True
    ) -> Optional[dict[str, Any]]:
        """
        Assess quality of a memory.

        Args:
            memory_id: Memory ID
            use_llm: Whether to use LLM for assessment (more accurate)

        Returns:
            Quality metrics dict or None

        Example:
            >>> quality = client.assess_memory_quality(memory_id, use_llm=True)
            >>> print(f"Overall score: {quality['overall_score']}")
            >>> print(f"Specificity: {quality['specificity']}")
            >>> print(f"Verifiability: {quality['verifiability']}")
        """
        # Get memory
        memory = None
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])
                break

        if not memory:
            return None

        # Assess quality
        quality_metrics = self.provenance_tracker.assess_quality(memory, use_llm=use_llm)

        # Store in memory metadata
        memory.metadata["quality_metrics"] = quality_metrics.model_dump()
        memory.metadata["quality_assessed_at"] = datetime.now(timezone.utc).isoformat()
        self.update_memory(memory_id, metadata=memory.metadata)

        return cast(Optional[dict[str, Any]], quality_metrics.model_dump())

    def add_memory_citation(
        self,
        memory_id: str,
        source_type: str,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        source_text: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Optional[dict[str, Any]]:
        """
        Add a citation to a memory.

        Args:
            memory_id: Memory ID
            source_type: Type of source ("conversation", "document", "url", "memory", "external")
            source_id: Source identifier
            source_url: Source URL
            source_text: Excerpt from source
            confidence: Confidence in citation (0-1)

        Returns:
            Updated lineage dict or None

        Example:
            >>> lineage = client.add_memory_citation(
            ...     memory_id,
            ...     source_type="url",
            ...     source_url="https://example.com/article",
            ...     source_text="According to the article...",
            ...     confidence=0.95
            ... )
        """
        # Get memory
        memory = None
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])
                break

        if not memory:
            return None

        # Add citation
        lineage = self.provenance_tracker.add_citation(
            memory=memory,
            source_type=source_type,
            source_id=source_id,
            source_url=source_url,
            source_text=source_text,
            confidence=confidence,
        )

        # Update memory
        self.update_memory(memory_id, metadata=memory.metadata)

        return cast(Optional[dict[str, Any]], lineage.model_dump())

    def extract_memory_citations(
        self, memory_id: str, context: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Extract potential citations from memory text using LLM.

        Args:
            memory_id: Memory ID
            context: Optional context for citation extraction

        Returns:
            List of citation dicts

        Example:
            >>> citations = client.extract_memory_citations(memory_id, context="User said...")
            >>> for cit in citations:
            ...     print(f"{cit['source_type']}: {cit['source_text']}")
        """
        # Get memory
        memory = None
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                memory = Memory(**memory_data["payload"])
                break

        if not memory:
            return []

        # Extract citations
        citations = self.provenance_tracker.extract_citations(memory, context=context)

        return [cit.model_dump() for cit in citations]

    def get_derived_memories(self, memory_id: str) -> list[Memory]:
        """
        Get all memories derived from a specific memory.

        Args:
            memory_id: Parent memory ID

        Returns:
            List of derived memories

        Example:
            >>> derived = client.get_derived_memories(parent_memory_id)
            >>> print(f"Found {len(derived)} derived memories")
        """
        # Get the parent memory to find user_id
        parent_memory = None
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                parent_memory = Memory(**memory_data["payload"])
                break

        if not parent_memory:
            return []

        # Search through user's memories
        all_memories = self.get_memories(parent_memory.user_id, limit=1000)

        # Filter to only those with this parent
        derived = []
        for memory in all_memories:
            lineage_data = memory.metadata.get("lineage", {})
            parent_ids = lineage_data.get("parent_memory_ids", [])
            if memory_id in parent_ids:
                derived.append(memory)

        return derived

    # ============================================================================
    # MEMORY HEALTH & QUALITY MONITORING
    # ============================================================================

    def get_memory_health_score(
        self,
        user_id: str,
        include_stale_detection: bool = True,
        include_duplicate_detection: bool = True,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive health score for user's memory store.

        Args:
            user_id: User identifier
            include_stale_detection: Run stale memory detection
            include_duplicate_detection: Run duplicate clustering

        Returns:
            Health score dict with overall and component scores

        Example:
            >>> health = client.get_memory_health_score("user123")
            >>> print(f"Overall health: {health['overall_score']}/100")
            >>> print(f"Recommendations: {health['recommendations']}")
        """
        health_score = self.health_monitor.calculate_health_score(
            user_id=user_id,
            include_stale_detection=include_stale_detection,
            include_duplicate_detection=include_duplicate_detection,
        )
        return health_score.model_dump()

    def detect_stale_memories(self, user_id: str) -> list[dict[str, Any]]:
        """
        Detect potentially stale/outdated memories.

        Args:
            user_id: User identifier

        Returns:
            List of stale memory dicts with reasons

        Example:
            >>> stale = client.detect_stale_memories("user123")
            >>> for mem in stale[:5]:
            ...     print(f"{mem['text'][:50]}... - {mem['staleness_reason']}")
        """
        stale_memories = self.health_monitor.detect_stale_memories(user_id)
        return [m.model_dump() for m in stale_memories]

    def detect_duplicate_clusters(
        self,
        user_id: str,
        min_cluster_size: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Detect clusters of duplicate/similar memories.

        Args:
            user_id: User identifier
            min_cluster_size: Minimum cluster size to report

        Returns:
            List of duplicate cluster dicts

        Example:
            >>> clusters = client.detect_duplicate_clusters("user123")
            >>> for cluster in clusters:
            ...     print(f"Cluster: {len(cluster['memory_ids'])} duplicates")
            ...     print(f"Representative: {cluster['representative_text']}")
        """
        clusters = self.health_monitor.detect_duplicate_clusters(
            user_id=user_id,
            min_cluster_size=min_cluster_size,
        )
        return [c.model_dump() for c in clusters]

    def detect_near_duplicates(
        self,
        user_id: str,
        suggest_merge: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect near-duplicate pairs with merge suggestions.

        Args:
            user_id: User identifier
            suggest_merge: Generate merge suggestions

        Returns:
            List of near-duplicate warning dicts

        Example:
            >>> warnings = client.detect_near_duplicates("user123")
            >>> for warning in warnings[:3]:
            ...     print(f"Similarity: {warning['similarity_score']:.2f}")
            ...     print(f"Suggestion: {warning['merge_suggestion']}")
        """
        warnings = self.health_monitor.detect_near_duplicates(
            user_id=user_id,
            suggest_merge=suggest_merge,
        )
        return [w.model_dump() for w in warnings]

    def analyze_memory_coverage(self, user_id: str) -> dict[str, Any]:
        """
        Analyze memory coverage across topics and types.

        Args:
            user_id: User identifier

        Returns:
            Coverage report dict

        Example:
            >>> coverage = client.analyze_memory_coverage("user123")
            >>> print(f"Topics: {coverage['topic_distribution']}")
            >>> print(f"Well covered: {coverage['well_covered_topics']}")
            >>> print(f"Gaps: {coverage['coverage_gaps']}")
        """
        report = self.health_monitor.analyze_coverage(user_id)
        return report.model_dump()

    # ============================================================================
    # ENHANCED TEMPORAL FEATURES
    # ============================================================================

    def calculate_memory_freshness(
        self,
        memory: Memory,
        reference_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive freshness score for a memory.

        Args:
            memory: Memory to score
            reference_date: Reference date (defaults to now)

        Returns:
            Freshness score dict with breakdown

        Example:
            >>> memory = client.get_memories("user123", limit=1)[0]
            >>> freshness = client.calculate_memory_freshness(memory)
            >>> print(f"Freshness: {freshness['freshness_score']:.2f}")
            >>> print(f"Age: {freshness['age_days']} days")
        """
        score = self.enhanced_temporal.calculate_freshness_score(
            memory=memory,
            reference_date=reference_date,
        )
        return score.model_dump()

    def apply_time_decay(
        self,
        memory: Memory,
        decay_type: str = "exponential",
        reference_date: Optional[datetime] = None,
    ) -> float:
        """
        Apply time decay function to memory importance.

        Args:
            memory: Memory to decay
            decay_type: Type of decay (exponential, linear, logarithmic, step)
            reference_date: Reference date

        Returns:
            Decayed importance score

        Example:
            >>> memory = client.get_memories("user123", limit=1)[0]
            >>> decayed = client.apply_time_decay(memory, decay_type="exponential")
            >>> print(f"Original: {memory.importance}, Decayed: {decayed:.2f}")
        """
        decay_function = self.enhanced_temporal.decay_functions.get(decay_type)
        if not decay_function:
            logger.warning(f"Unknown decay type: {decay_type}, using default")
            decay_function = None

        return self.enhanced_temporal.apply_time_decay(
            memory=memory,
            decay_function=decay_function,
            reference_date=reference_date,
        )

    def get_adaptive_time_window(
        self,
        query: str,
        user_id: str,
        context_type: str = "relevant",
    ) -> dict[str, Any]:
        """
        Get auto-adjusted temporal context window for query.

        Args:
            query: Query text
            user_id: User identifier
            context_type: Type of context (recent, relevant, seasonal)

        Returns:
            Temporal window dict

        Example:
            >>> window = client.get_adaptive_time_window("recent updates", "user123")
            >>> print(f"Window: {window['window_size_days']} days")
            >>> print(f"From: {window['start_date']} to: {window['end_date']}")
        """
        memories = self.get_memories(user_id, limit=1000)
        window = self.enhanced_temporal.get_adaptive_context_window(
            query=query,
            memories=memories,
            context_type=context_type,
        )
        return window.model_dump()

    def forecast_memory_patterns(
        self,
        user_id: str,
        forecast_days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Predict future memory patterns based on historical data.

        Args:
            user_id: User identifier
            forecast_days: Days to forecast ahead

        Returns:
            List of forecast dicts

        Example:
            >>> forecasts = client.forecast_memory_patterns("user123", forecast_days=30)
            >>> for forecast in forecasts:
            ...     print(f"Type: {forecast['forecast_type']}")
            ...     print(f"Predictions: {forecast['predictions']}")
        """
        memories = self.get_memories(user_id, limit=1000)
        forecasts = self.enhanced_temporal.forecast_memory_patterns(
            memories=memories,
            forecast_days=forecast_days,
        )
        return [f.model_dump() for f in forecasts]

    def predict_future_patterns(
        self,
        user_id: str,
        pattern_type: str = "recurring",
    ) -> list[dict[str, Any]]:
        """
        Predict when patterns might recur.

        Args:
            user_id: User identifier
            pattern_type: Type of pattern (recurring, seasonal)

        Returns:
            List of pattern prediction dicts

        Example:
            >>> predictions = client.predict_future_patterns("user123", "recurring")
            >>> for pred in predictions:
            ...     print(f"{pred['description']}")
            ...     print(f"Next occurrence: {pred['predicted_date']}")
        """
        memories = self.get_memories(user_id, limit=1000)
        predictions = self.enhanced_temporal.predict_future_patterns(
            memories=memories,
            pattern_type=pattern_type,
        )
        return [p.model_dump() for p in predictions]

    # ============================================================================
    # DEBUGGING & OBSERVABILITY
    # ============================================================================

    def explain_retrieval(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        """
        Explain why each memory was retrieved and ranked.

        Args:
            query: Original query
            results: Retrieved results

        Returns:
            List of explanation dicts

        Example:
            >>> results = client.search_memories("user123", "python tips")
            >>> explanations = client.explain_retrieval("python tips", results)
            >>> for exp in explanations:
            ...     print(f"Rank {exp['rank']}: {exp['explanation']}")
        """
        explanations = self.observability.explain_retrieval(
            query=query,
            results=results,
        )
        return [e.model_dump() for e in explanations]

    def visualize_similarity_scores(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Create visualization data for similarity scores.

        Args:
            query: Query text
            results: Retrieval results
            top_k: Number of top results to visualize

        Returns:
            Visualization data dict

        Example:
            >>> results = client.search_memories("user123", "python")
            >>> viz = client.visualize_similarity_scores("python", results)
            >>> print(f"Score distribution: {viz['score_distribution']}")
            >>> print(f"Avg score: {viz['avg_score']:.2f}")
        """
        viz = self.observability.visualize_similarity_scores(
            query=query,
            results=results,
            top_k=top_k,
        )
        return viz.model_dump()

    def generate_access_heatmap(
        self,
        user_id: str,
        time_period_days: int = 30,
    ) -> dict[str, Any]:
        """
        Generate heatmap of memory access patterns.

        Args:
            user_id: User identifier
            time_period_days: Time period to analyze

        Returns:
            Access heatmap dict

        Example:
            >>> heatmap = client.generate_access_heatmap("user123", time_period_days=30)
            >>> print(f"Peak hours: {heatmap['peak_hours']}")
            >>> print(f"Hot memories: {heatmap['hot_memories']}")
            >>> print(f"Access by type: {heatmap['access_by_type']}")
        """
        memories = self.get_memories(user_id, limit=1000)
        heatmap = self.observability.generate_access_heatmap(
            user_id=user_id,
            memories=memories,
            time_period_days=time_period_days,
        )
        return heatmap.model_dump()

    def profile_query_performance(
        self,
        query: str,
        user_id: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Profile query performance with detailed timing breakdown.

        Args:
            query: Query text
            user_id: User identifier
            k: Number of results to retrieve

        Returns:
            Performance profile dict with timings and bottlenecks

        Example:
            >>> profile = client.profile_query_performance(
            ...     "recent work",
            ...     user_id="alice",
            ...     k=10
            ... )
            >>> print(f"Total time: {profile['total_time_ms']:.2f}ms")
            >>> print(f"Bottlenecks: {profile['bottlenecks']}")
            >>> print(f"Recommendations: {profile['recommendations']}")
        """
        import time

        start_time = time.time()
        stage_timings = {}

        # Vector search
        vec_start = time.time()
        results = self.recall(query, user_id=user_id, k=k)
        stage_timings["vector_search"] = (time.time() - vec_start) * 1000

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000

        # Identify bottlenecks
        bottlenecks = []
        if stage_timings.get("vector_search", 0) > 500:
            bottlenecks.append("vector_search_slow")

        # Generate recommendations
        recommendations = []
        if total_time_ms > 1000:
            recommendations.append("Consider using Redis caching")
        if stage_timings.get("vector_search", 0) > 500:
            recommendations.append("Optimize vector index or reduce search space")

        return {
            "query": query,
            "total_time_ms": total_time_ms,
            "stage_timings": stage_timings,
            "memory_count": len(results),
            "vector_search_ms": stage_timings.get("vector_search"),
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
        }

    def get_adaptive_context_window(
        self,
        query: str,
        user_id: str,
        context_type: str = "relevant",
    ) -> dict[str, Any]:
        """
        Get adaptive temporal context window based on query.

        Args:
            query: Query text
            user_id: User identifier
            context_type: Type of context (recent, relevant, seasonal)

        Returns:
            Temporal context window dict

        Example:
            >>> window = client.get_adaptive_context_window(
            ...     "recent meetings",
            ...     user_id="alice",
            ...     context_type="recent"
            ... )
            >>> print(f"Window size: {window['window_size_days']} days")
            >>> print(f"Start: {window['start_date']}")
            >>> print(f"Confidence: {window['confidence']}")
        """
        # Get user memories for analysis
        memories = self.get_memories(user_id, limit=1000)

        # Use enhanced temporal analyzer
        window = self.enhanced_temporal.get_adaptive_context_window(
            query=query,
            memories=memories,
            context_type=context_type,
        )

        return window.model_dump()

    def get_performance_snapshot(self) -> dict[str, Any]:
        """
        Get current performance snapshot.

        Returns:
            Performance snapshot dict

        Example:
            >>> snapshot = client.get_performance_snapshot()
            >>> print(f"Avg query time: {snapshot['avg_query_time_ms']:.2f}ms")
            >>> print(f"Performance score: {snapshot['performance_score']:.1f}/100")
        """
        snapshot = self.observability.get_performance_snapshot()
        return snapshot.model_dump()

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report dict

        Example:
            >>> report = client.get_performance_report()
            >>> print(f"Slowest queries: {report['slowest_queries']}")
            >>> print(f"Bottlenecks: {report['common_bottlenecks']}")
            >>> print(f"Recommendations: {report['recommendations']}")
        """
        return self.observability.generate_performance_report()

    def identify_slow_queries(
        self,
        threshold_ms: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Identify queries that exceeded performance threshold.

        Args:
            threshold_ms: Custom threshold (uses default if None)

        Returns:
            List of slow query profile dicts

        Example:
            >>> slow = client.identify_slow_queries(threshold_ms=500)
            >>> for query in slow[:5]:
            ...     print(f"Query: {query['query']}")
            ...     print(f"Time: {query['total_time_ms']:.2f}ms")
            ...     print(f"Bottlenecks: {query['bottlenecks']}")
        """
        slow_queries = self.observability.identify_slow_queries(threshold_ms)
        return [q.model_dump() for q in slow_queries]

    # ========================================================================
    # Memory Lifecycle & Tiering
    # ========================================================================

    def get_memory_temperature(self, memory_id: str) -> Optional[dict[str, Any]]:
        """
        Get temperature metrics for a memory.

        Returns lifecycle information including:
        - Current tier (hot/warm/cold/archived/hibernated)
        - Temperature score (0-100)
        - Access frequency and recency
        - Recommended tier based on access patterns

        Args:
            memory_id: Memory identifier

        Returns:
            Dictionary with temperature metrics or None if not found

        Example:
            >>> temp = client.get_memory_temperature("mem_123")
            >>> print(f"Tier: {temp['tier']}")
            >>> print(f"Temperature: {temp['temperature_score']:.1f}")
            >>> print(f"Access frequency: {temp['access_frequency']:.2f}/day")
        """
        try:
            # Get memory from storage
            memory = self.get_memory(memory_id)
            if not memory:
                return None

            # Calculate temperature
            temperature = self.lifecycle_manager.calculate_temperature(
                memory_id=memory.id,
                created_at=memory.created_at,
                access_count=memory.access_count,
                last_access=memory.updated_at,
                importance=memory.importance,
            )

            # Return as dictionary
            return {
                "memory_id": temperature.memory_id,
                "tier": temperature.tier.value,
                "temperature_score": temperature.temperature_score,
                "access_frequency": temperature.access_frequency,
                "recency_score": temperature.recency_score,
                "importance_weight": temperature.importance_weight,
                "access_count": temperature.access_count,
                "days_since_creation": temperature.days_since_creation,
                "days_since_last_access": temperature.days_since_last_access,
                "last_access": temperature.last_access.isoformat()
                if temperature.last_access
                else None,
                "created_at": temperature.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get memory temperature: {e}")
            return None

    def migrate_memory_tier(self, memory_id: str, target_tier: str) -> dict[str, Any]:
        """
        Manually migrate a memory to a specific storage tier.

        Tiers:
        - hot: Frequently accessed, cached in Redis + Vector DB
        - warm: Occasionally accessed, in Vector DB
        - cold: Rarely accessed, in Vector DB
        - archived: Very old, compressed in Vector DB
        - hibernated: Extremely old, highly compressed

        Args:
            memory_id: Memory identifier
            target_tier: Target tier (hot, warm, cold, archived, hibernated)

        Returns:
            Migration result dictionary

        Raises:
            ValueError: If tier is invalid or memory not found

        Example:
            >>> result = client.migrate_memory_tier("mem_123", "archived")
            >>> print(f"Migrated to: {result['target_tier']}")
        """
        from hippocampai.pipeline.memory_lifecycle import MemoryTier

        try:
            # Validate tier
            target_tier_enum = MemoryTier(target_tier.lower())

            # Get memory from storage
            memory = self.get_memory(memory_id)
            if not memory:
                raise ValueError(f"Memory {memory_id} not found")

            # Update metadata with new tier
            memory_data = memory.model_dump()
            if "metadata" not in memory_data:
                memory_data["metadata"] = {}
            if "lifecycle" not in memory_data["metadata"]:
                memory_data["metadata"]["lifecycle"] = {}

            memory_data["metadata"]["lifecycle"]["tier"] = target_tier_enum.value
            memory_data["metadata"]["lifecycle"]["last_tier_update"] = datetime.now(
                timezone.utc
            ).isoformat()

            # Update in vector store
            collection = (
                self.qdrant.collection_facts
                if memory.type == MemoryType.FACT
                else self.qdrant.collection_prefs
            )
            self.qdrant.update(
                collection_name=collection,
                id=memory.id,
                payload=memory_data,
            )

            logger.info(f"Migrated memory {memory_id} to tier {target_tier_enum.value}")

            return {
                "memory_id": memory_id,
                "target_tier": target_tier_enum.value,
                "migrated": True,
            }
        except Exception as e:
            logger.error(f"Failed to migrate memory tier: {e}")
            raise

    def get_tier_statistics(self, user_id: str) -> dict[str, Any]:
        """
        Get statistics about memory tiers for a user.

        Returns:
        - Total memories and size
        - Distribution across tiers
        - Average temperature per tier
        - Lifecycle configuration

        Args:
            user_id: User identifier

        Returns:
            Dictionary with tier statistics

        Example:
            >>> stats = client.get_tier_statistics("user123")
            >>> print(f"Total memories: {stats['total_memories']}")
            >>> for tier, count in stats['tier_counts'].items():
            ...     print(f"{tier}: {count} memories")
        """
        try:
            # Get all memories for user
            memories = self.get_memories(user_id=user_id, limit=10000)

            # Initialize statistics
            tier_counts: dict[str, int] = {}
            tier_temperatures: dict[str, list[float]] = {}
            total_size = 0
            total_accesses = 0

            # Process each memory
            for memory in memories:
                # Calculate temperature
                temperature = self.lifecycle_manager.calculate_temperature(
                    memory_id=memory.id,
                    created_at=memory.created_at,
                    access_count=memory.access_count,
                    last_access=memory.updated_at,
                    importance=memory.importance,
                )

                # Update tier counts
                tier = temperature.tier.value
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Track temperature scores
                if tier not in tier_temperatures:
                    tier_temperatures[tier] = []
                tier_temperatures[tier].append(temperature.temperature_score)

                # Update totals
                total_accesses += memory.access_count
                # Estimate size (text length)
                total_size += len(memory.text.encode("utf-8"))

            # Calculate average temperatures per tier
            tier_average_temperatures = {
                tier: sum(temps) / len(temps) if temps else 0.0
                for tier, temps in tier_temperatures.items()
            }

            return {
                "user_id": user_id,
                "total_memories": len(memories),
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "tier_counts": tier_counts,
                "tier_average_temperatures": tier_average_temperatures,
            }
        except Exception as e:
            logger.error(f"Failed to get tier statistics: {e}")
            return {}

    # Conversation-Aware Memory Methods

    def add_conversation_turn(
        self,
        conversation_id: str,
        speaker: str,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: str = "user",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Add a turn to a conversation and track it.

        Args:
            conversation_id: Unique conversation identifier
            speaker: Name of the speaker
            text: Content of the turn
            user_id: Optional user identifier (defaults to conversation_id)
            session_id: Optional session identifier
            role: Speaker role (user, assistant, system, participant)
            metadata: Optional metadata

        Returns:
            Parsed conversation turn

        Example:
            ```python
            client = MemoryClient()

            # Add user turn
            turn = client.add_conversation_turn(
                conversation_id="conv_123",
                speaker="Alice",
                text="What's the weather like today?",
                role="user"
            )

            # Add assistant turn
            turn = client.add_conversation_turn(
                conversation_id="conv_123",
                speaker="Bot",
                text="It's sunny and 72°F",
                role="assistant"
            )
            ```
        """
        from hippocampai.pipeline.conversation_memory import SpeakerRole

        try:
            role_enum = SpeakerRole(role.upper())
        except ValueError:
            role_enum = SpeakerRole.USER

        user_id = user_id or conversation_id
        session_id = session_id or conversation_id

        turn = self.conversation_manager.parse_turn(
            text=text,
            speaker=speaker,
            role=role_enum,
            conversation_id=conversation_id,
            metadata=metadata,
        )

        self.conversation_manager.add_turn(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            turn=turn,
        )

        logger.info(f"Added conversation turn {turn.turn_id} to {conversation_id} by {speaker}")

        return turn

    def get_conversation_summary(self, conversation_id: str, use_llm: bool = True) -> Optional[ConversationSummary]:
        """
        Get a multi-level summary of a conversation.

        Args:
            conversation_id: Conversation identifier
            use_llm: Whether to use LLM for better summarization

        Returns:
            Conversation summary with key points, decisions, action items, etc.

        Example:
            ```python
            client = MemoryClient()

            # Get conversation summary
            summary = client.get_conversation_summary("conv_123")

            print(f"Total turns: {summary.total_turns}")
            print(f"Key points: {summary.key_points}")
            print(f"Decisions: {summary.decisions}")
            print(f"Action items: {summary.action_items}")
            print(f"Unresolved questions: {summary.unresolved_questions}")
            ```
        """
        summary = self.conversation_manager.summarize_conversation(
            conversation_id=conversation_id, use_llm=use_llm
        )

        if summary:
            logger.info(
                f"Generated summary for conversation {conversation_id}: "
                f"{summary.total_turns} turns, {len(summary.key_points)} key points"
            )

        return summary

    def extract_conversation_memories(
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

        Example:
            ```python
            client = MemoryClient()

            # Add conversation turns
            client.add_conversation_turn("conv_123", "Alice", "I love pizza", role="user")
            client.add_conversation_turn("conv_123", "Bot", "Great! What toppings?", role="assistant")
            client.add_conversation_turn("conv_123", "Alice", "Pepperoni and mushrooms", role="user")

            # Extract memories from conversation
            memories = client.extract_conversation_memories(
                conversation_id="conv_123",
                user_id="alice_123"
            )

            print(f"Extracted {len(memories)} memories")
            ```
        """
        try:
            summary = self.get_conversation_summary(conversation_id=conversation_id, use_llm=True)

            if not summary:
                logger.warning(f"No summary available for conversation {conversation_id}")
                return []

            created_memories = []

            # Create memory from key points
            for idx, point in enumerate(summary.key_points):
                tags = list(summary.topics_discussed) if auto_tag else []
                tags.append(f"conversation:{conversation_id}")

                memory = self.add(
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
                tags = list(summary.topics_discussed) if auto_tag else []
                tags.extend(["decision", f"conversation:{conversation_id}"])

                memory = self.add(
                    text=f"Decision: {decision.text}. Rationale: {decision.rationale}",
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="fact",
                    importance=9.0,
                    tags=tags,
                    metadata={
                        "source": "conversation_decision",
                        "conversation_id": conversation_id,
                        "decision_maker": decision.made_by,
                        "topic": decision.related_topic,
                    },
                )
                created_memories.append(memory)

            # Create memories from action items
            for action in summary.action_items:
                tags = list(summary.topics_discussed) if auto_tag else []
                tags.extend(["action_item", f"conversation:{conversation_id}"])

                memory = self.add(
                    text=f"Action: {action.text} (Priority: {action.priority})",
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="task",
                    importance=8.0,
                    tags=tags,
                    metadata={
                        "source": "conversation_action",
                        "conversation_id": conversation_id,
                        "assignee": action.assignee,
                        "deadline": action.deadline.isoformat() if action.deadline else None,
                        "status": action.status,
                    },
                )
                created_memories.append(memory)

            logger.info(
                f"Extracted {len(created_memories)} memories from conversation {conversation_id}"
            )

            return created_memories

        except Exception as e:
            logger.error(f"Failed to extract memories from conversation {conversation_id}: {e}")
            return []

    # Memory Merge Methods

    def suggest_memory_merges(self, user_id: str, limit: int = 100) -> list[MergeCandidate]:
        """
        Suggest memory merges based on similarity.

        Args:
            user_id: User identifier
            limit: Maximum number of memories to analyze

        Returns:
            List of merge candidates with similarity scores and conflicts

        Example:
            ```python
            client = MemoryClient()

            # Get merge suggestions
            candidates = client.suggest_memory_merges(user_id="user_123", limit=50)

            for candidate in candidates:
                print(f"Merge cluster: {candidate.memory_ids}")
                print(f"Similarity: {candidate.similarity_score}")
                print(f"Conflicts: {len(candidate.conflicts)}")
                print(f"Quality gain: {candidate.estimated_quality_gain}")
            ```
        """
        try:
            # Get recent memories
            memories = self.search(query="", user_id=user_id, k=limit, filters={})

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

            # Calculate similarity matrix using embeddings
            import numpy as np

            similarity_matrix: dict[tuple[str, str], float] = {}
            for i, mem1 in enumerate(memory_dicts):
                mem1_text = str(mem1["text"])
                mem1_id = str(mem1["id"])
                emb1 = self.embedder.encode_single(mem1_text)
                for j, mem2 in enumerate(memory_dicts):
                    if i < j:
                        mem2_text = str(mem2["text"])
                        mem2_id = str(mem2["id"])
                        emb2 = self.embedder.encode_single(mem2_text)
                        similarity = float(
                            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        )
                        similarity_matrix[(mem1_id, mem2_id)] = similarity

            # Get merge suggestions
            candidates = self.merge_engine.suggest_merges(
                memories=memory_dicts, similarity_matrix=similarity_matrix
            )

            logger.info(f"Found {len(candidates)} merge candidates for user {user_id}")

            return candidates

        except Exception as e:
            logger.error(f"Failed to suggest merges for user {user_id}: {e}")
            return []

    def merge_memories(
        self,
        memory_ids: list[str],
        user_id: str,
        strategy: str = "combine",
        manual_resolutions: Optional[dict[str, Any]] = None,
    ) -> Optional[MergeResult]:
        """
        Merge multiple memories into one.

        Args:
            memory_ids: List of memory IDs to merge
            user_id: User identifier
            strategy: Merge strategy (newest, highest_confidence, longest, combine, manual)
            manual_resolutions: Manual conflict resolutions

        Returns:
            Merge result with merged memory ID and metrics

        Example:
            ```python
            client = MemoryClient()

            # Merge memories
            result = client.merge_memories(
                memory_ids=["mem_1", "mem_2"],
                user_id="user_123",
                strategy="combine"
            )

            print(f"Merged into: {result.merged_memory_id}")
            print(f"Conflicts resolved: {result.conflicts_resolved}")
            print(f"Quality before: {result.quality_before}")
            print(f"Quality after: {result.quality_after}")
            ```
        """
        from hippocampai.pipeline.memory_merge import MergeStrategy

        try:
            strategy_enum = MergeStrategy(strategy.lower())
        except ValueError:
            strategy_enum = MergeStrategy.COMBINE

        try:
            if len(memory_ids) < 2:
                logger.warning("Need at least 2 memories to merge")
                return None

            # Fetch memories
            memories = []
            for mem_id in memory_ids:
                mem = self.get(memory_id=mem_id)
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
                strategy=strategy_enum,
                user_id=user_id,
                manual_resolutions=manual_resolutions,
            )

            if result.success:
                # Create the merged memory
                merged_data = result.rollback_data.get("merged_memory", {})

                merged_memory = self.add(
                    text=merged_data["text"],
                    user_id=user_id,
                    memory_type=merged_data["type"],
                    importance=merged_data["importance"],
                    tags=merged_data["tags"],
                    metadata=merged_data.get("metadata", {}),
                )

                # Delete original memories
                for mem_id in memory_ids:
                    self.delete(memory_id=mem_id)

                # Update result with actual merged memory ID
                result.merged_memory_id = merged_memory.id

                logger.info(
                    f"Successfully merged {len(memory_ids)} memories into {merged_memory.id}"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to merge memories: {e}")
            return None

    def preview_memory_merge(
        self,
        memory_ids: list[str],
        user_id: str,
        strategy: str = "combine",
    ) -> Optional[dict[str, Any]]:
        """
        Preview what a memory merge would look like without executing it.

        Args:
            memory_ids: List of memory IDs to preview merge
            user_id: User identifier
            strategy: Optional merge strategy

        Returns:
            Preview of merged memory with conflicts and quality metrics

        Example:
            ```python
            client = MemoryClient()

            # Preview merge
            preview = client.preview_memory_merge(
                memory_ids=["mem_1", "mem_2"],
                user_id="user_123"
            )

            print(preview["preview_text"])
            print(f"Conflicts: {preview['conflicts']}")
            print(f"Estimated quality: {preview['estimated_quality']}")
            ```
        """
        from hippocampai.pipeline.memory_merge import MergeStrategy

        try:
            strategy_enum = MergeStrategy(strategy.lower())
        except ValueError:
            strategy_enum = MergeStrategy.COMBINE

        try:
            if len(memory_ids) < 2:
                return None

            # Fetch memories
            memories = []
            for mem_id in memory_ids:
                mem = self.get(memory_id=mem_id)
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
            preview = self.merge_engine.preview_merge(memories=memories, strategy=strategy_enum)

            logger.info(f"Generated merge preview for {len(memory_ids)} memories")

            return preview

        except Exception as e:
            logger.error(f"Failed to preview merge: {e}")
            return None

    # Memory Operations Methods

    def clone_memory(
        self,
        memory_id: str,
        new_user_id: Optional[str] = None,
        preserve_timestamps: bool = False,
        tag_additions: Optional[list[str]] = None,
        metadata_overrides: Optional[dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """
        Clone a memory with optional modifications.

        Args:
            memory_id: ID of memory to clone
            new_user_id: Optional new user ID for clone
            preserve_timestamps: Keep original created/updated timestamps
            tag_additions: Additional tags to add to clone
            metadata_overrides: Metadata fields to override

        Returns:
            Cloned memory

        Example:
            ```python
            client = MemoryClient()

            # Clone memory to another user
            cloned = client.clone_memory(
                memory_id="mem_123",
                new_user_id="user_456",
                tag_additions=["cloned"],
                metadata_overrides={"source": "cloned_from_user_123"}
            )

            print(f"Cloned to: {cloned.id}")
            ```
        """
        from hippocampai.pipeline.memory_operations import CloneOptions

        try:
            # Get original memory
            original = self.get(memory_id=memory_id)
            if not original:
                logger.error(f"Memory {memory_id} not found")
                return None

            # Convert to dict
            memory_dict = {
                "id": original.id,
                "text": original.text,
                "user_id": original.user_id,
                "session_id": original.session_id,
                "type": original.type.value,
                "importance": original.importance,
                "confidence": original.confidence,
                "tags": original.tags,
                "metadata": original.metadata,
                "created_at": original.created_at,
                "updated_at": original.updated_at,
            }

            # Create clone options
            options = CloneOptions(
                preserve_timestamps=preserve_timestamps,
                new_user_id=new_user_id,
                tag_additions=tag_additions or [],
                metadata_overrides=metadata_overrides or {},
            )

            # Clone memory
            cloned_dict = self.memory_ops.clone_memory(memory_dict, options)

            # Create in store
            cloned_memory = self.add(
                text=cloned_dict["text"],
                user_id=cloned_dict["user_id"],
                session_id=cloned_dict.get("session_id"),
                memory_type=cloned_dict["type"],
                importance=cloned_dict["importance"],
                tags=cloned_dict["tags"],
                metadata=cloned_dict["metadata"],
            )

            logger.info(f"Cloned memory {memory_id} to {cloned_memory.id}")

            return cloned_memory

        except Exception as e:
            logger.error(f"Failed to clone memory {memory_id}: {e}")
            return None

    def create_memory_template(
        self,
        template_name: str,
        text_template: str,
        default_importance: float = 5.0,
        default_type: str = "fact",
        default_tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Create a reusable memory template.

        Args:
            template_name: Name of the template
            text_template: Text with {variable} placeholders
            default_importance: Default importance score
            default_type: Default memory type
            default_tags: Default tags

        Returns:
            Template dict

        Example:
            ```python
            client = MemoryClient()

            # Create template
            template = client.create_memory_template(
                template_name="greeting",
                text_template="User {name} prefers to be greeted as {greeting}",
                default_importance=6.0,
                default_tags=["preference", "greeting"]
            )

            # Use template
            memory = client.instantiate_template(
                template=template,
                variables={"name": "Alice", "greeting": "Dr. Smith"},
                user_id="user_123"
            )
            ```
        """
        template = self.memory_ops.create_template_memory(
            template_name=template_name,
            text_template=text_template,
            default_importance=default_importance,
            default_type=default_type,
            default_tags=default_tags or [],
        )

        logger.info(f"Created memory template: {template_name}")

        return template

    def instantiate_template(
        self,
        template: dict[str, Any],
        variables: dict[str, str],
        user_id: str,
        session_id: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> Memory:
        """
        Create a memory from a template.

        Args:
            template: Template dict from create_memory_template
            variables: Variables to fill in template
            user_id: User ID for new memory
            session_id: Optional session ID
            overrides: Override default template values

        Returns:
            Created memory
        """
        memory_dict = self.memory_ops.instantiate_template(
            template=template,
            variables=variables,
            user_id=user_id,
            session_id=session_id,
            overrides=overrides,
        )

        # Create in store
        memory = self.add(
            text=memory_dict["text"],
            user_id=memory_dict["user_id"],
            session_id=memory_dict.get("session_id"),
            memory_type=memory_dict["type"],
            importance=memory_dict["importance"],
            tags=memory_dict["tags"],
            metadata=memory_dict["metadata"],
        )

        logger.info(f"Instantiated template {template['template_name']}: {memory.id}")

        return memory

    def batch_update_memories(
        self,
        user_id: str,
        filters: Optional[dict[str, Any]] = None,
        set_importance: Optional[float] = None,
        adjust_importance: Optional[float] = None,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        set_metadata: Optional[dict[str, Any]] = None,
        archive: bool = False,
    ) -> Any:
        """
        Batch update memories matching filter criteria.

        Args:
            user_id: User ID
            filters: Filter criteria dict
            set_importance: Set importance to this value
            adjust_importance: Add/subtract from importance
            add_tags: Tags to add
            remove_tags: Tags to remove
            set_metadata: Metadata to set/update
            archive: Whether to archive matched memories

        Returns:
            Batch update result

        Example:
            ```python
            client = MemoryClient()

            # Boost importance of all work-related memories
            result = client.batch_update_memories(
                user_id="user_123",
                filters={"tags_include": ["work"]},
                adjust_importance=1.0,
                add_tags=["prioritized"]
            )

            print(f"Updated {result.total_updated} memories")
            ```
        """
        from hippocampai.pipeline.memory_operations import (
            ArchivalReason,
            BatchUpdateFilter,
            BatchUpdateOperation,
        )

        try:
            # Get memories
            all_memories = self.search(query="", user_id=user_id, k=10000, filters={})

            # Convert to dicts
            memory_dicts = []
            for mem in all_memories:
                memory_dicts.append(
                    {
                        "id": mem.id,
                        "text": mem.text,
                        "user_id": mem.user_id,
                        "session_id": mem.session_id,
                        "type": mem.type.value,
                        "importance": mem.importance,
                        "confidence": mem.confidence,
                        "tags": mem.tags,
                        "metadata": mem.metadata,
                        "created_at": mem.created_at,
                        "updated_at": mem.updated_at,
                    }
                )

            # Build filter
            filter_criteria = BatchUpdateFilter(user_ids=[user_id])
            if filters:
                for key, value in filters.items():
                    setattr(filter_criteria, key, value)

            # Build operation
            operation = BatchUpdateOperation(
                set_importance=set_importance,
                adjust_importance=adjust_importance,
                add_tags=add_tags or [],
                remove_tags=remove_tags or [],
                set_metadata=set_metadata or {},
                archive=archive,
                archive_reason=ArchivalReason.MANUAL if archive else None,
            )

            # Execute batch update
            result = self.memory_ops.batch_update(
                memories=memory_dicts,
                filter_criteria=filter_criteria,
                operation=operation,
            )

            # Update in store
            for memory_dict in memory_dicts:
                memory_id = str(memory_dict["id"])
                if memory_id in result.updated_memory_ids:
                    # Update the memory
                    self.update(
                        memory_id=memory_id,
                        text=memory_dict["text"],
                        importance=memory_dict["importance"],
                        tags=memory_dict["tags"],
                        metadata=memory_dict["metadata"],
                    )

            logger.info(
                f"Batch update: {result.total_updated} memories updated in {result.execution_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to batch update memories: {e}")
            return None

    def schedule_memory_maintenance(
        self,
        action: str,
        interval_hours: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Schedule automatic memory maintenance.

        Args:
            action: Type of maintenance (refresh_embeddings, update_metadata, etc.)
            interval_hours: Hours between executions
            filters: Optional filter criteria

        Returns:
            Maintenance schedule

        Example:
            ```python
            client = MemoryClient()

            # Schedule daily importance recalculation
            schedule = client.schedule_memory_maintenance(
                action="recalculate_importance",
                interval_hours=24
            )

            print(f"Scheduled {schedule.action} every {schedule.interval_hours}h")
            ```
        """
        from hippocampai.pipeline.memory_operations import (
            BatchUpdateFilter,
            MaintenanceAction,
            MaintenanceSchedule,
        )

        try:
            action_enum = MaintenanceAction(action.lower())
        except ValueError:
            logger.error(f"Invalid maintenance action: {action}")
            return None

        filter_criteria = None
        if filters:
            filter_criteria = BatchUpdateFilter(**filters)

        schedule = MaintenanceSchedule(
            action=action_enum,
            interval_hours=interval_hours,
            next_run=datetime.now(timezone.utc),
            filter_criteria=filter_criteria,
        )

        scheduled = self.memory_ops.schedule_maintenance(schedule)

        logger.info(f"Scheduled {action} maintenance every {interval_hours} hours")

        return scheduled

    def archive_memories(
        self,
        user_id: str,
        stale_threshold_days: Optional[int] = None,
        low_importance_threshold: Optional[float] = None,
        dry_run: bool = True,
    ) -> Any:
        """
        Archive old or low-value memories.

        Args:
            user_id: User ID
            stale_threshold_days: Archive if not updated in N days
            low_importance_threshold: Archive if importance below this
            dry_run: Preview without actually archiving

        Returns:
            Batch update result with archival stats

        Example:
            ```python
            client = MemoryClient()

            # Preview archival
            result = client.archive_memories(
                user_id="user_123",
                stale_threshold_days=180,
                low_importance_threshold=2.0,
                dry_run=True
            )

            print(f"Would archive {result.total_matched} memories")

            # Execute archival
            result = client.archive_memories(
                user_id="user_123",
                stale_threshold_days=180,
                low_importance_threshold=2.0,
                dry_run=False
            )
            ```
        """
        from hippocampai.pipeline.memory_operations import ArchivalPolicy, ArchivalReason

        try:
            # Get memories
            all_memories = self.search(query="", user_id=user_id, k=10000, filters={})

            # Convert to dicts
            memory_dicts = []
            for mem in all_memories:
                memory_dicts.append(
                    {
                        "id": mem.id,
                        "text": mem.text,
                        "user_id": mem.user_id,
                        "importance": mem.importance,
                        "confidence": mem.confidence,
                        "tags": mem.tags,
                        "metadata": mem.metadata,
                        "created_at": mem.created_at,
                        "updated_at": mem.updated_at,
                        "access_count": getattr(mem, "access_count", 0),
                    }
                )

            # Create policy
            policy = ArchivalPolicy(
                name="manual_archival",
                stale_threshold_days=stale_threshold_days,
                low_importance_threshold=low_importance_threshold,
                archive_reason=ArchivalReason.RETENTION_POLICY,
                dry_run=dry_run,
            )

            # Apply policy
            result = self.memory_ops.apply_archival_policies(
                memories=memory_dicts, policies=[policy]
            )

            if not dry_run:
                # Update archived memories in store
                for memory_dict in memory_dicts:
                    if (
                        memory_dict.get("archived")
                        and memory_dict["id"] in result.archived_memory_ids
                    ):
                        # Ensure metadata is a dict
                        if not isinstance(memory_dict.get("metadata"), dict):
                            memory_dict["metadata"] = {}
                        metadata = memory_dict["metadata"]
                        if isinstance(metadata, dict):
                            metadata["archived"] = True
                            archived_at = memory_dict.get("archived_at", datetime.now(timezone.utc))
                            if isinstance(archived_at, datetime):
                                metadata["archived_at"] = archived_at.isoformat()
                            else:
                                metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
                            self.update(
                                memory_id=str(memory_dict["id"]),
                                metadata=metadata,
                            )

            mode = "DRY RUN: " if dry_run else ""
            logger.info(f"{mode}Archived {result.total_archived} memories for user {user_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to archive memories: {e}")
            return None

    def garbage_collect_memories(
        self,
        user_id: str,
        policy: str = "moderate",
        min_age_days: int = 30,
        dry_run: bool = True,
    ) -> Any:
        """
        Garbage collect low-value memories.

        Args:
            user_id: User ID
            policy: GC policy (aggressive, moderate, conservative)
            min_age_days: Minimum age before considering for GC
            dry_run: Preview without actually deleting

        Returns:
            Garbage collection result

        Example:
            ```python
            client = MemoryClient()

            # Preview GC
            result = client.garbage_collect_memories(
                user_id="user_123",
                policy="moderate",
                min_age_days=90,
                dry_run=True
            )

            print(f"Would collect {result.total_collected} memories")
            print(f"Space reclaimed: {result.space_reclaimed_bytes} bytes")

            # Execute GC
            result = client.garbage_collect_memories(
                user_id="user_123",
                policy="moderate",
                min_age_days=90,
                dry_run=False
            )
            ```
        """
        from hippocampai.pipeline.memory_operations import (
            GarbageCollectionConfig,
            GarbageCollectionPolicy,
        )

        try:
            policy_enum = GarbageCollectionPolicy(policy.lower())
        except ValueError:
            logger.error(f"Invalid GC policy: {policy}")
            return None

        try:
            # Get memories
            all_memories = self.search(query="", user_id=user_id, k=10000, filters={})

            # Convert to dicts
            memory_dicts = []
            for mem in all_memories:
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
                        "access_count": getattr(mem, "access_count", 0),
                    }
                )

            # Create GC config
            config = GarbageCollectionConfig(
                policy=policy_enum, min_age_days=min_age_days, dry_run=dry_run
            )

            # Execute GC
            result = self.memory_ops.garbage_collect(memories=memory_dicts, config=config)

            if not dry_run:
                # Delete collected memories
                for mem_id in result.collected_memory_ids:
                    self.delete(memory_id=mem_id)

            mode = "DRY RUN: " if dry_run else ""
            logger.info(
                f"{mode}Garbage collected {result.total_collected} memories "
                f"({result.space_reclaimed_bytes} bytes)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to garbage collect memories: {e}")
            return None

    # Adaptive Learning Methods

    def record_memory_access(
        self,
        memory_id: str,
        access_type: str = "read",
        context: Optional[str] = None,
        co_accessed_ids: Optional[list[str]] = None,
        relevance_score: Optional[float] = None,
    ) -> None:
        """
        Record a memory access event for learning patterns.

        Args:
            memory_id: Memory that was accessed
            access_type: Type of access (read, update, search, etc.)
            context: Optional context or query that led to access
            co_accessed_ids: Other memories accessed in same request
            relevance_score: Relevance score if from search

        Example:
            ```python
            client = MemoryClient()

            # Record access when searching
            memories = client.search("pizza toppings", user_id="user_123")
            for mem in memories:
                client.record_memory_access(
                    memory_id=mem.id,
                    access_type="search",
                    context="pizza toppings",
                    relevance_score=mem.score
                )
            ```
        """
        from hippocampai.pipeline.adaptive_learning import AccessEvent

        event = AccessEvent(
            memory_id=memory_id,
            timestamp=datetime.now(timezone.utc),
            access_type=access_type,
            context=context,
            co_accessed_ids=co_accessed_ids or [],
            relevance_score=relevance_score,
        )

        self.adaptive_learning.record_access(event)

        logger.debug(f"Recorded {access_type} access for memory {memory_id}")

    def analyze_memory_access_pattern(self, memory_id: str, force_refresh: bool = False) -> Any:
        """
        Analyze access pattern for a memory.

        Args:
            memory_id: Memory to analyze
            force_refresh: Force new analysis even if cached

        Returns:
            Access pattern analysis

        Example:
            ```python
            client = MemoryClient()

            analysis = client.analyze_memory_access_pattern("mem_123")

            print(f"Pattern: {analysis.pattern_type}")
            print(f"Frequency: {analysis.access_frequency} times/day")
            print(f"Trend: {analysis.trend}")
            print(f"Co-occurring memories: {analysis.co_occurring_memories}")
            ```
        """
        analysis = self.adaptive_learning.analyze_access_pattern(
            memory_id, force_refresh=force_refresh
        )

        if analysis:
            logger.info(
                f"Access pattern for {memory_id}: {analysis.pattern_type.value} "
                f"({analysis.access_frequency:.2f} accesses/day, {analysis.trend} trend)"
            )

        return analysis

    def get_refresh_recommendations(self, user_id: str, limit: int = 20) -> list:
        """
        Get recommendations for memories that should be refreshed.

        Args:
            user_id: User identifier
            limit: Maximum recommendations to return

        Returns:
            List of refresh recommendations, sorted by priority

        Example:
            ```python
            client = MemoryClient()

            recommendations = client.get_refresh_recommendations("user_123", limit=10)

            for rec in recommendations:
                print(f"Memory {rec.memory_id}:")
                print(f"  Priority: {rec.priority}")
                print(f"  Reason: {rec.reason}")
                print(f"  Staleness: {rec.staleness_score:.2f}")
                print(f"  Suggested sources: {rec.suggested_sources}")
            ```
        """
        try:
            # Get user memories
            memories = self.search(query="", user_id=user_id, k=1000, filters={})

            recommendations = []

            for mem in memories:
                # Get or analyze access pattern
                pattern = self.adaptive_learning.analyze_access_pattern(mem.id)

                # Get refresh recommendation
                rec = self.adaptive_learning.recommend_refresh(
                    memory_id=mem.id,
                    current_importance=mem.importance,
                    last_updated=mem.updated_at,
                    access_pattern=pattern,
                )

                if rec:
                    recommendations.append(rec)

            # Sort by priority
            priority_order = {
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3,
            }
            recommendations.sort(
                key=lambda r: (
                    priority_order.get(r.priority.value, 99),
                    -r.staleness_score,
                )
            )

            logger.info(
                f"Generated {len(recommendations)} refresh recommendations for user {user_id}"
            )

            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Failed to get refresh recommendations: {e}")
            return []

    def get_compression_recommendations(self, user_id: str, limit: int = 20) -> list:
        """
        Get recommendations for adaptive memory compression.

        Args:
            user_id: User identifier
            limit: Maximum recommendations to return

        Returns:
            List of compression recommendations, sorted by space savings

        Example:
            ```python
            client = MemoryClient()

            recommendations = client.get_compression_recommendations("user_123")

            total_savings = sum(r.estimated_space_savings for r in recommendations)
            print(f"Potential space savings: {total_savings} bytes")

            for rec in recommendations:
                print(f"Memory {rec.memory_id}:")
                print(f"  {rec.current_level} -> {rec.recommended_level}")
                print(f"  Reason: {rec.reason}")
                print(f"  Savings: {rec.estimated_space_savings} bytes")
            ```
        """
        try:
            # Get user memories
            memories = self.search(query="", user_id=user_id, k=1000, filters={})

            recommendations = []

            for mem in memories:
                # Calculate age
                age_days = (datetime.now(timezone.utc) - mem.created_at).total_seconds() / 86400

                # Get current compression level from metadata
                from hippocampai.pipeline.adaptive_learning import CompressionLevel

                current_compression = CompressionLevel(
                    mem.metadata.get("compression_level", "none")
                )

                # Get or analyze access pattern
                pattern = self.adaptive_learning.analyze_access_pattern(mem.id)

                # Get compression recommendation
                rec = self.adaptive_learning.recommend_compression(
                    memory_id=mem.id,
                    current_importance=mem.importance,
                    age_days=int(age_days),
                    text_length=len(mem.text),
                    current_compression=current_compression,
                    access_pattern=pattern,
                )

                if rec:
                    recommendations.append(rec)

            # Sort by space savings
            recommendations.sort(key=lambda r: r.estimated_space_savings, reverse=True)

            total_savings = sum(r.estimated_space_savings for r in recommendations)

            logger.info(
                f"Generated {len(recommendations)} compression recommendations "
                f"for user {user_id} (potential {total_savings} bytes savings)"
            )

            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Failed to get compression recommendations: {e}")
            return []

    def apply_compression_recommendation(self, memory_id: str, compression_level: str) -> Optional[Memory]:
        """
        Apply compression to a memory.

        Args:
            memory_id: Memory to compress
            compression_level: Compression level (none, light, moderate, aggressive, archived)

        Returns:
            Updated memory

        Example:
            ```python
            client = MemoryClient()

            # Get recommendations
            recs = client.get_compression_recommendations("user_123")

            # Apply first recommendation
            if recs:
                compressed = client.apply_compression_recommendation(
                    recs[0].memory_id,
                    recs[0].recommended_level.value
                )
            ```
        """
        from hippocampai.pipeline.adaptive_learning import CompressionLevel

        try:
            compression_enum = CompressionLevel(compression_level.lower())
        except ValueError:
            logger.error(f"Invalid compression level: {compression_level}")
            return None

        try:
            # Get memory
            memory = self.get(memory_id=memory_id)
            if not memory:
                return None

            # Apply compression based on level
            compressed_text = memory.text

            if compression_enum == CompressionLevel.LIGHT:
                # Light: Remove redundant whitespace, minor cleanup
                compressed_text = " ".join(memory.text.split())

            elif compression_enum == CompressionLevel.MODERATE:
                # Moderate: Summarize if LLM available
                if self.llm:
                    target_length = len(memory.text) // 2
                    prompt = f"Summarize the following text in approximately {target_length} characters:\n\n{memory.text}"
                    compressed_text = self.llm.generate(prompt, max_tokens=target_length // 4)
                else:
                    # Truncate to 50%
                    compressed_text = memory.text[: len(memory.text) // 2] + "..."

            elif compression_enum == CompressionLevel.AGGRESSIVE:
                # Aggressive: Heavy summarization
                if self.llm:
                    target_length = len(memory.text) // 3
                    prompt = f"Provide a brief summary of the following text in approximately {target_length} characters:\n\n{memory.text}"
                    compressed_text = self.llm.generate(prompt, max_tokens=target_length // 4)
                else:
                    # Truncate to 30%
                    compressed_text = memory.text[: len(memory.text) // 3] + "..."

            elif compression_enum == CompressionLevel.ARCHIVED:
                # Archived: Minimal representation
                compressed_text = memory.text[:100] + "... [archived]"

            # Update memory
            memory.metadata["compression_level"] = compression_level
            memory.metadata["original_length"] = len(memory.text)
            memory.metadata["compressed_at"] = datetime.now(timezone.utc).isoformat()

            updated = self.update(
                memory_id=memory_id,
                text=compressed_text,
                metadata=memory.metadata,
            )

            logger.info(
                f"Compressed memory {memory_id} to {compression_level} level "
                f"({len(memory.text)} -> {len(compressed_text)} bytes)"
            )

            return updated

        except Exception as e:
            logger.error(f"Failed to apply compression: {e}")
            return None

    def get_access_statistics(self) -> dict[str, Any]:
        """
        Get overall access statistics and patterns.

        Returns:
            Dictionary with access statistics

        Example:
            ```python
            client = MemoryClient()

            stats = client.get_access_statistics()

            print(f"Total accesses: {stats['total_access_events']}")
            print(f"Unique memories: {stats['unique_memories_accessed']}")
            print(f"Top accessed: {stats['top_accessed_memories'][:5]}")
            print(f"Top contexts: {stats['top_access_contexts'][:5]}")
            ```
        """
        stats = self.adaptive_learning.get_access_statistics()

        logger.info(
            f"Access statistics: {stats['total_access_events']} events, "
            f"{stats['unique_memories_accessed']} unique memories"
        )

        return stats

    # Convenience method aliases for common operations
    def add(self, **kwargs: Any) -> Memory:
        """Alias for remember() method."""
        return self.remember(**kwargs)

    def search(self, **kwargs: Any) -> list[Memory]:
        """Alias for recall() method."""
        results = self.recall(**kwargs)
        return [r.memory for r in results]

    def get(self, memory_id: str, **kwargs: Any) -> Optional[Memory]:
        """Alias for get_memory() method."""
        return self.get_memory(memory_id=memory_id, **kwargs)

    def delete(self, memory_id: str, **kwargs: Any) -> bool:
        """Alias for delete_memory() method."""
        return self.delete_memory(memory_id=memory_id, **kwargs)

    def update(self, memory_id: str, **kwargs: Any) -> Optional[Memory]:
        """Alias for update_memory() method."""
        return self.update_memory(memory_id=memory_id, **kwargs)
