"""Hybrid retriever with two-stage ranking."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, RetrievalResult
from hippocampai.retrieval.bm25 import BM25Retriever
from hippocampai.retrieval.rerank import Reranker
from hippocampai.retrieval.router import QueryRouter
from hippocampai.retrieval.rrf import reciprocal_rank_fusion
from hippocampai.utils.scoring import fuse_scores, recency_score
from hippocampai.utils.time import parse_iso_datetime
from hippocampai.vector.qdrant_store import QdrantStore

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
        weights: Optional[Dict[str, float]] = None,
        half_lives: Optional[Dict[str, int]] = None,
    ):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.reranker = reranker
        self.router = QueryRouter()
        self.top_k_qdrant = top_k_qdrant
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

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
        self.corpus_facts: List[Tuple[str, str]] = []  # [(id, text), ...]
        self.corpus_prefs: List[Tuple[str, str]] = []

    def rebuild_bm25(self, user_id: str):
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

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval with two-stage ranking.

        Flow:
        1. Route query to collections
        2. BM25 + Vector → RRF → top 200
        3. CrossEncoder rerank → top 20
        4. Score fusion → final K
        """
        # Route query
        route = self.router.route(query)
        collections = []
        if route == "both":
            collections = [self.qdrant.collection_facts, self.qdrant.collection_prefs]
        elif route == "facts":
            collections = [self.qdrant.collection_facts]
        else:
            collections = [self.qdrant.collection_prefs]

        # Embed query
        query_vector = self.embedder.encode_single(query)

        all_candidates = []

        for collection in collections:
            # Vector search
            vector_results = self.qdrant.search(
                collection_name=collection,
                vector=query_vector,
                limit=self.top_k_qdrant,
                filters={"user_id": user_id} if not filters else {**filters, "user_id": user_id},
            )

            vector_ranking = [(r["id"], r["score"]) for r in vector_results]

            # BM25 search (if available)
            bm25_ranking = []
            if collection == self.qdrant.collection_facts and self.bm25_facts:
                bm25_results = self.bm25_facts.search(query, top_k=self.top_k_qdrant)
                bm25_ranking = [(self.corpus_facts[idx][0], score) for idx, score in bm25_results]
            elif collection == self.qdrant.collection_prefs and self.bm25_prefs:
                bm25_results = self.bm25_prefs.search(query, top_k=self.top_k_qdrant)
                bm25_ranking = [(self.corpus_prefs[idx][0], score) for idx, score in bm25_results]

            # RRF fusion
            if bm25_ranking:
                fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=self.rrf_k)
            else:
                fused = vector_ranking

            # Collect candidates
            id_to_result = {r["id"]: r for r in vector_results}
            for doc_id, rrf_score in fused[: self.top_k_qdrant]:
                if doc_id in id_to_result:
                    all_candidates.append((doc_id, id_to_result[doc_id]))

        if not all_candidates:
            return []

        # Stage 2: Cross-encoder rerank
        rerank_input = [
            (cand[0], cand[1]["payload"]["text"], cand[1]["score"]) for cand in all_candidates
        ]
        reranked = self.reranker.rerank(query, rerank_input, top_k=self.top_k_qdrant)

        # Score fusion
        results = []
        for doc_id, text, sim_score, rerank_score in reranked[:50]:  # Top 50 for fusion
            payload = next(c[1]["payload"] for c in all_candidates if c[0] == doc_id)

            # Compute component scores
            importance_norm = payload.get("importance", 5.0) / 10.0
            created_at = parse_iso_datetime(payload["created_at"])
            mem_type = payload.get("type", "fact")
            half_life = self.half_lives.get(mem_type, 30)
            recency = recency_score(created_at, half_life)

            # Normalize rerank score
            rerank_norm = (rerank_score + 10) / 20  # typical range [-10, 10]
            rerank_norm = max(0, min(1, rerank_norm))

            # Fuse
            final_score = fuse_scores(
                sim=sim_score,
                rerank=rerank_norm,
                recency=recency,
                importance=importance_norm,
                weights=self.weights,
            )

            memory = Memory(
                id=doc_id,
                text=text,
                user_id=payload["user_id"],
                session_id=payload.get("session_id"),
                type=mem_type,
                importance=payload.get("importance", 5.0),
                confidence=payload.get("confidence", 0.9),
                tags=payload.get("tags", []),
                created_at=created_at,
                updated_at=parse_iso_datetime(payload["updated_at"]),
                access_count=payload.get("access_count", 0),
                metadata=payload.get("metadata", {}),
            )

            results.append(
                RetrievalResult(
                    memory=memory,
                    score=final_score,
                    breakdown={
                        "sim": sim_score,
                        "rerank": rerank_norm,
                        "recency": recency,
                        "importance": importance_norm,
                        "final": final_score,
                    },
                )
            )

        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
