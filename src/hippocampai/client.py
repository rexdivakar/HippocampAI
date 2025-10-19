"""Main MemoryClient - public API."""

import logging
from typing import Dict, List, Optional
from hippocampai.config import Config, get_config
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult
from hippocampai.vector.qdrant_store import QdrantStore
from hippocampai.embed.embedder import get_embedder
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.retriever import HybridRetriever
from hippocampai.pipeline.extractor import MemoryExtractor
from hippocampai.pipeline.dedup import MemoryDeduplicator
from hippocampai.pipeline.consolidate import MemoryConsolidator
from hippocampai.pipeline.importance import ImportanceScorer
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM

logger = logging.getLogger(__name__)


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
        config: Optional[Config] = None
    ):
        """Initialize memory client."""
        self.config = config or get_config()

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
            ef_search=self.config.ef_search
        )

        self.embedder = get_embedder(
            model_name=self.config.embed_model,
            batch_size=self.config.embed_batch_size,
            quantized=self.config.embed_quantized,
            dimension=self.config.embed_dimension
        )

        self.reranker = Reranker(
            model_name=self.config.reranker_model,
            cache_ttl=self.config.rerank_cache_ttl
        )

        self.retriever = HybridRetriever(
            qdrant_store=self.qdrant,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k_qdrant=self.config.top_k_qdrant,
            top_k_final=self.config.top_k_final,
            rrf_k=self.config.rrf_k,
            weights=self.config.get_weights(),
            half_lives=self.config.get_half_lives()
        )

        # LLM (optional)
        self.llm = None
        if self.config.llm_provider == "ollama":
            self.llm = OllamaLLM(
                model=self.config.llm_model,
                base_url=self.config.llm_base_url
            )
        elif self.config.llm_provider == "openai" and self.config.allow_cloud:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = OpenAILLM(api_key=api_key, model=self.config.llm_model)

        # Pipeline
        self.extractor = MemoryExtractor(llm=self.llm, mode="hybrid")
        self.deduplicator = MemoryDeduplicator(
            qdrant_store=self.qdrant,
            embedder=self.embedder,
            reranker=self.reranker
        )
        self.consolidator = MemoryConsolidator(llm=self.llm)
        self.scorer = ImportanceScorer(llm=self.llm)

        logger.info("MemoryClient initialized")

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Store a memory."""
        memory = Memory(
            text=text,
            user_id=user_id,
            session_id=session_id,
            type=MemoryType(type),
            importance=importance or self.scorer.score(text, type),
            tags=tags or []
        )

        # Check duplicates
        action, dup_ids = self.deduplicator.check_duplicate(memory, user_id)
        if action == "skip":
            logger.info(f"Skipping duplicate memory: {memory.id}")
            return memory

        # Store
        collection = memory.collection_name(
            self.config.collection_facts,
            self.config.collection_prefs
        )
        vector = self.embedder.encode_single(memory.text)

        self.qdrant.upsert(
            collection_name=collection,
            id=memory.id,
            vector=vector,
            payload=memory.model_dump(mode="json")
        )

        logger.info(f"Stored memory: {memory.id}")
        return memory

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve memories."""
        # Rebuild BM25 if needed
        if not self.retriever.bm25_facts:
            self.retriever.rebuild_bm25(user_id)

        results = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            session_id=session_id,
            k=k
        )

        logger.info(f"Retrieved {len(results)} memories for user {user_id}")
        return results

    def extract_from_conversation(
        self,
        conversation: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[Memory]:
        """Extract memories from conversation."""
        memories = self.extractor.extract(conversation, user_id, session_id)

        # Store each
        for mem in memories:
            self.remember(
                text=mem.text,
                user_id=mem.user_id,
                session_id=mem.session_id,
                type=mem.type.value,
                importance=mem.importance,
                tags=mem.tags
            )

        return memories
