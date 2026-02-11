"""Graph-based retriever: entity extraction from query -> graph traversal -> scored memory list."""

import logging

from hippocampai.graph.knowledge_graph import KnowledgeGraph
from hippocampai.pipeline.entity_recognition import EntityRecognizer

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves memories by traversing the knowledge graph.

    Flow:
    1. Extract entities from query text
    2. Look up entity nodes in the knowledge graph
    3. Find memories linked to those entities (direct + expansion)
    4. Score memories by entity confidence, edge weight, and hop distance
    5. Filter to user, normalize, and return top_k
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        entity_recognizer: EntityRecognizer,
        max_depth: int = 2,
    ) -> None:
        self.graph = graph
        self.entity_recognizer = entity_recognizer
        self.max_depth = max_depth

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Search the knowledge graph for memories related to the query.

        Args:
            query: Search query text.
            user_id: Filter results to this user's memories.
            top_k: Maximum number of results to return.

        Returns:
            List of (memory_id, score) tuples sorted by score descending.
        """
        # 1. Extract entities from the query
        entities = self.entity_recognizer.extract_entities(query)
        if not entities:
            return []

        # Accumulate scores per memory_id
        memory_scores: dict[str, float] = {}

        for entity in entities:
            # 2. Direct links: memories connected to this entity
            direct_memories = self.graph.get_entity_memories(entity.entity_id)
            for mem_id in direct_memories:
                score = entity.confidence * 1.0  # hop_distance = 0
                self._accumulate_score(memory_scores, mem_id, score)

            # 3. Expansion: memories reachable via graph traversal
            related = self.graph.get_related_memories(
                self.graph._entity_index.get(entity.entity_id, ""),
                max_depth=self.max_depth,
            )
            for related_id, _relation, weight in related:
                # Estimate hop distance from weight decay
                hop_distance = 1  # conservative default
                score = entity.confidence * weight / (1 + hop_distance)
                self._accumulate_score(memory_scores, related_id, score)

        # 4. Filter to user_id memories only
        user_memories = self.graph._user_graphs.get(user_id, set())
        filtered: dict[str, float] = {
            mid: sc for mid, sc in memory_scores.items() if mid in user_memories
        }

        if not filtered:
            return []

        # 5. Normalize scores to [0, 1]
        max_score = max(filtered.values())
        if max_score > 0:
            normalized = {mid: sc / max_score for mid, sc in filtered.items()}
        else:
            normalized = filtered

        # Sort by score descending and return top_k
        ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _accumulate_score(
        scores: dict[str, float], memory_id: str, score: float
    ) -> None:
        """Accumulate the best score for a memory_id."""
        if memory_id in scores:
            scores[memory_id] = max(scores[memory_id], score)
        else:
            scores[memory_id] = score
