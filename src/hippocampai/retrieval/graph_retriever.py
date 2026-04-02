"""Graph-based retriever: entity extraction from query -> graph traversal -> scored memory list."""

import logging

import networkx as nx

from hippocampai.graph.knowledge_graph import KnowledgeGraph, NodeType
from hippocampai.pipeline.entity_recognition import EntityRecognizer

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves memories by traversing the knowledge graph.

    Flow:
    1. Extract entities from query text
    2. Look up entity nodes in the knowledge graph
    3. BFS from each entity node up to max_depth hops
    4. Score memory nodes by entity confidence, edge weight, and actual hop distance
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
        g: nx.DiGraph = self.graph.graph

        for entity in entities:
            entity_node_id = self.graph._entity_index.get(entity.entity_id, "")
            if not entity_node_id or entity_node_id not in g:
                continue

            # 2. Direct links: memories at hop distance 0 from this entity node
            direct_memories = self.graph.get_entity_memories(entity.entity_id)
            for mem_id in direct_memories:
                score = entity.confidence * 1.0  # hop_distance = 0, no decay
                self._accumulate_score(memory_scores, mem_id, score)

            # 3. BFS expansion: compute actual hop distances from entity_node_id
            # single_source_shortest_path_length gives {node: hop_distance}
            hop_distances: dict[str, int] = nx.single_source_shortest_path_length(
                g, entity_node_id, cutoff=self.max_depth
            )

            for neighbor_id, hop_distance in hop_distances.items():
                if hop_distance == 0:
                    continue  # already handled as direct above

                node_data = g.nodes.get(neighbor_id, {})
                node_type = node_data.get("node_type")

                # Only score MEMORY nodes found during expansion
                if node_type != NodeType.MEMORY:
                    continue

                # Aggregate edge weight along the shortest path
                path = nx.shortest_path(g, entity_node_id, neighbor_id)
                path_weight = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = g.edges.get((u, v), {})
                    path_weight *= edge_data.get("weight", 1.0)

                score = entity.confidence * path_weight / (1 + hop_distance)
                self._accumulate_score(memory_scores, neighbor_id, score)

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
