"""Main MemoryClient - public API."""

import logging
from typing import Dict, List, Literal, Optional

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
            memory = Memory(
                text=text,
                user_id=user_id,
                session_id=session_id,
                type=MemoryType(type),
                importance=importance or self.scorer.score(text, type),
                tags=tags or [],
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
        self, query: str, user_id: str, session_id: Optional[str] = None, k: int = 5
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
            results = self.retriever.retrieve(query=query, user_id=user_id, session_id=session_id, k=k)
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
