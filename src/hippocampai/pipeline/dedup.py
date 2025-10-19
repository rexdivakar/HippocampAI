"""Memory deduplicator using embeddings + reranker."""

import logging
from typing import List, Literal

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory
from hippocampai.retrieval.rerank import Reranker
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class MemoryDeduplicator:
    """Deduplicate memories using semantic similarity."""

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker,
        similarity_threshold: float = 0.88,
    ):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.reranker = reranker
        self.threshold = similarity_threshold

    def check_duplicate(
        self, new_memory: Memory, user_id: str
    ) -> tuple[Literal["store", "skip", "update"], List[str]]:
        """
        Check if memory is duplicate.

        Returns:
            (action, duplicate_ids)
            action: "store", "skip", or "update"
        """
        collection = new_memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )

        # Vector search for candidates
        vector = self.embedder.encode_single(new_memory.text)
        candidates = self.qdrant.search(
            collection_name=collection, vector=vector, limit=10, filters={"user_id": user_id}
        )

        if not candidates:
            return ("store", [])

        # Check similarity
        duplicates = []
        for cand in candidates:
            if cand["score"] > self.threshold:
                duplicates.append(cand["id"])

        if not duplicates:
            return ("store", [])

        # If exact duplicate, skip
        if len(duplicates) == 1 and candidates[0]["score"] > 0.95:
            return ("skip", duplicates)

        # If similar, update existing
        return ("update", duplicates)
