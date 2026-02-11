"""Hybrid retriever with two-stage ranking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, RetrievalResult
from hippocampai.models.search import SearchMode
from hippocampai.retrieval.bm25 import BM25Retriever
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.router import QueryRouter
from hippocampai.retrieval.rrf import reciprocal_rank_fusion
from hippocampai.utils.scoring import fuse_scores, recency_score
from hippocampai.utils.time import parse_iso_datetime
from hippocampai.vector.qdrant_store import QdrantStore

if TYPE_CHECKING:
    from hippocampai.feedback.feedback_manager import FeedbackManager
    from hippocampai.retrieval.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever: BM25 + Embeddings → RRF → Qdrant topK → CrossEncoder → Final."""

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker,
        top_k_qdrant: int = 200,
        top_k_final: int = 20,
        rrf_k: int = 60,
        weights: Optional[dict[str, float]] = None,
        half_lives: Optional[dict[str, int]] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        feedback_manager: Optional[FeedbackManager] = None,
    ):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.reranker = reranker
        self.router = QueryRouter()
        self.top_k_qdrant = top_k_qdrant
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.graph_retriever = graph_retriever
        self.feedback_manager = feedback_manager

        self.weights = weights or {"sim": 0.55, "rerank": 0.20, "recency": 0.15, "importance": 0.10}

        self.half_lives = half_lives or {
            "preference": 90,
            "goal": 90,
            "habit": 90,
            "fact": 30,
            "event": 14,
            "context": 30,
        }

        # BM25 indices (in-memory, rebuilt periodically)
        self.bm25_facts: Optional[BM25Retriever] = None
        self.bm25_prefs: Optional[BM25Retriever] = None
        self.corpus_facts: list[tuple[str, str]] = []  # [(id, text), ...]
        self.corpus_prefs: list[tuple[str, str]] = []

    def rebuild_bm25(self, user_id: str) -> None:
        """Rebuild BM25 indices for a user."""
        # Fetch all memories for user
        facts_data = self.qdrant.scroll(
            collection_name=self.qdrant.collection_facts, filters={"user_id": user_id}, limit=10000
        )
        prefs_data = self.qdrant.scroll(
            collection_name=self.qdrant.collection_prefs, filters={"user_id": user_id}, limit=10000
        )

        self.corpus_facts = [(d["id"], d["payload"]["text"]) for d in facts_data]
        self.corpus_prefs = [(d["id"], d["payload"]["text"]) for d in prefs_data]

        if self.corpus_facts:
            self.bm25_facts = BM25Retriever([text for _, text in self.corpus_facts])
        if self.corpus_prefs:
            self.bm25_prefs = BM25Retriever([text for _, text in self.corpus_prefs])

        logger.info(f"Rebuilt BM25: facts={len(self.corpus_facts)}, prefs={len(self.corpus_prefs)}")

    def _route_to_collections(self, query: str) -> list[str]:
        """Determine which collections to search based on query routing."""
        route = self.router.route(query)
        if route == "both":
            return [self.qdrant.collection_facts, self.qdrant.collection_prefs]
        if route == "facts":
            return [self.qdrant.collection_facts]
        return [self.qdrant.collection_prefs]

    def _perform_bm25_search(
        self, collection: str, query: str
    ) -> list[tuple[str, float]]:
        """Perform BM25 search for a collection."""
        if collection == self.qdrant.collection_facts and self.bm25_facts:
            bm25_results = self.bm25_facts.search(query, top_k=self.top_k_qdrant)
            return [(self.corpus_facts[idx][0], score) for idx, score in bm25_results]
        if collection == self.qdrant.collection_prefs and self.bm25_prefs:
            bm25_results = self.bm25_prefs.search(query, top_k=self.top_k_qdrant)
            return [(self.corpus_prefs[idx][0], score) for idx, score in bm25_results]
        return []

    def _fuse_rankings(
        self,
        vector_ranking: list[tuple[str, float]],
        bm25_ranking: list[tuple[str, float]],
        search_mode: SearchMode,
        graph_ranking: Optional[list[tuple[str, float]]] = None,
    ) -> list[tuple[str, float]]:
        """Fuse rankings based on search mode."""
        if search_mode == SearchMode.GRAPH_HYBRID:
            rankings: list[list[tuple[str, float]]] = []
            if vector_ranking:
                rankings.append(vector_ranking)
            if bm25_ranking:
                rankings.append(bm25_ranking)
            if graph_ranking:
                rankings.append(graph_ranking)
            if rankings:
                return cast(
                    list[tuple[str, float]],
                    reciprocal_rank_fusion(rankings, k=self.rrf_k),
                )
            return []
        if search_mode == SearchMode.HYBRID and vector_ranking and bm25_ranking:
            return cast(
                list[tuple[str, float]],
                reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=self.rrf_k)
            )
        if search_mode == SearchMode.VECTOR_ONLY and vector_ranking:
            return vector_ranking
        if search_mode == SearchMode.KEYWORD_ONLY and bm25_ranking:
            return bm25_ranking
        return vector_ranking if vector_ranking else bm25_ranking

    def _compute_final_score(
        self,
        payload: dict,
        sim_score: float,
        rerank_score: float,
        enable_reranking: bool,
        doc_id: Optional[str] = None,
        graph_score: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Compute final score and component scores."""
        importance_norm = payload.get("importance", 5.0) / 10.0
        created_at = parse_iso_datetime(payload["created_at"])
        mem_type = payload.get("type", "fact")
        half_life = self.half_lives.get(mem_type, 30)
        recency = recency_score(created_at, half_life)

        rerank_norm = (rerank_score + 10) / 20
        rerank_norm = max(0, min(1, rerank_norm))

        # Feedback score (neutral 0.5 if no feedback manager)
        feedback_score = 0.5
        if self.feedback_manager is not None and doc_id is not None:
            feedback_score = self.feedback_manager.get_memory_feedback_score(doc_id)

        final_score = fuse_scores(
            sim=sim_score, rerank=rerank_norm, recency=recency,
            importance=importance_norm, weights=self.weights,
            graph=graph_score, feedback=feedback_score,
        )
        return final_score, rerank_norm, recency, importance_norm

    def _search_collection(
        self,
        collection: str,
        query: str,
        query_vector: Optional[list[float]],
        final_filters: dict,
        search_mode: SearchMode,
    ) -> list[tuple[str, dict]]:
        """Search a single collection and return candidates."""
        # Perform vector search
        vector_results: list[dict] = []
        vector_ranking: list[tuple[str, float]] = []
        if search_mode in [SearchMode.HYBRID, SearchMode.VECTOR_ONLY, SearchMode.GRAPH_HYBRID] and query_vector is not None:
            vector_arr = np.array(query_vector) if isinstance(query_vector, list) else query_vector
            vector_results = self.qdrant.search(
                collection_name=collection, vector=vector_arr,
                limit=self.top_k_qdrant, filters=final_filters,
            )
            vector_ranking = [(r["id"], r["score"]) for r in vector_results]

        # Perform BM25 search
        bm25_ranking: list[tuple[str, float]] = []
        if search_mode in [SearchMode.HYBRID, SearchMode.KEYWORD_ONLY, SearchMode.GRAPH_HYBRID]:
            bm25_ranking = self._perform_bm25_search(collection, query)

        # Fuse results
        fused = self._fuse_rankings(vector_ranking, bm25_ranking, search_mode)

        # Build ID to result mapping
        if search_mode == SearchMode.KEYWORD_ONLY and not vector_results:
            id_to_result = {}
            for doc_id, score in fused:
                result = self.qdrant.get(collection_name=collection, id=doc_id)
                if result:
                    id_to_result[doc_id] = {"id": doc_id, "score": score, "payload": result["payload"]}
        else:
            id_to_result = {r["id"]: r for r in vector_results}

        # Collect candidates
        candidates = []
        for doc_id, _ in fused[: self.top_k_qdrant]:
            if doc_id in id_to_result:
                candidates.append((doc_id, id_to_result[doc_id]))
        return candidates

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[dict] = None,
        search_mode: SearchMode = SearchMode.HYBRID,
        enable_reranking: bool = True,
        enable_score_breakdown: bool = True,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval with two-stage ranking.

        Flow:
        1. Route query to collections
        2. BM25 + Vector → RRF → top 200 (or vector-only/keyword-only based on search_mode)
        3. CrossEncoder rerank → top 20 (if enable_reranking=True)
        4. Score fusion → final K

        Args:
            query: Search query
            user_id: User ID
            session_id: Optional session ID
            k: Number of results to return
            filters: Optional filters
            search_mode: Search mode (hybrid, vector_only, keyword_only)
            enable_reranking: Enable cross-encoder reranking
            enable_score_breakdown: Include score breakdown in results
        """
        collections = self._route_to_collections(query)

        # Embed query (only if needed for vector search)
        query_vector = None
        if search_mode in [SearchMode.HYBRID, SearchMode.VECTOR_ONLY, SearchMode.GRAPH_HYBRID]:
            query_vector = self.embedder.encode_single(query)

        # Graph retrieval scores (used for GRAPH_HYBRID score fusion)
        graph_scores: dict[str, float] = {}
        if search_mode == SearchMode.GRAPH_HYBRID and self.graph_retriever is not None:
            graph_results = self.graph_retriever.search(
                query=query, user_id=user_id, top_k=self.top_k_qdrant,
            )
            graph_scores = dict(graph_results)

        # Construct filters
        base_filters = {"user_id": user_id}
        if session_id:
            base_filters["session_id"] = session_id
        final_filters = {**filters, **base_filters} if filters else base_filters

        # Search all collections
        all_candidates = []
        for collection in collections:
            vector_list: Optional[list[float]] = (
                query_vector.tolist() if query_vector is not None else None
            )
            candidates = self._search_collection(
                collection, query, vector_list, final_filters, search_mode
            )
            all_candidates.extend(candidates)

        if not all_candidates:
            return []

        # Stage 2: Cross-encoder rerank (if enabled)
        rerank_input = [
            (cand[0], cand[1]["payload"]["text"], cand[1]["score"]) for cand in all_candidates
        ]

        if enable_reranking:
            reranked = self.reranker.rerank(query, rerank_input, top_k=self.top_k_qdrant)
        else:
            # Skip reranking, use original scores
            reranked = [(doc_id, text, score, 0.0) for doc_id, text, score in rerank_input]

        # Score fusion
        results = []
        for doc_id, text, sim_score, rerank_score in reranked[:50]:
            payload = next(c[1]["payload"] for c in all_candidates if c[0] == doc_id)
            final_score, rerank_norm, recency, importance_norm = self._compute_final_score(
                payload, sim_score, rerank_score, enable_reranking,
                doc_id=doc_id, graph_score=graph_scores.get(doc_id, 0.0),
            )

            mem_type = payload.get("type", "fact")
            created_at = parse_iso_datetime(payload["created_at"])
            memory = Memory(
                id=doc_id, text=text, user_id=payload["user_id"],
                session_id=payload.get("session_id"), type=mem_type,
                importance=payload.get("importance", 5.0),
                confidence=payload.get("confidence", 0.9),
                tags=payload.get("tags", []), created_at=created_at,
                updated_at=parse_iso_datetime(payload["updated_at"]),
                access_count=payload.get("access_count", 0),
                metadata=payload.get("metadata", {}),
            )

            breakdown = {}
            if enable_score_breakdown:
                breakdown = {
                    "sim": sim_score, "rerank": rerank_norm if enable_reranking else 0.0,
                    "recency": recency, "importance": importance_norm, "final": final_score,
                    "search_mode": search_mode.value, "reranking_enabled": enable_reranking,
                }

            if session_id is None or memory.session_id == session_id:
                results.append(RetrievalResult(memory=memory, score=final_score, breakdown=breakdown))

        # Sort and return top k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
