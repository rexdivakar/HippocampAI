"""Graph index for memory relationships using NetworkX."""

import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Types of relationships between memories."""

    RELATED_TO = "related_to"  # General semantic relation
    CAUSED_BY = "caused_by"  # Causal relationship
    LEADS_TO = "leads_to"  # Temporal/consequence
    CONTRADICTS = "contradicts"  # Conflicting information
    SUPPORTS = "supports"  # Supporting evidence
    PART_OF = "part_of"  # Hierarchical relationship
    SIMILAR_TO = "similar_to"  # Semantic similarity
    SUPERSEDES = "supersedes"  # Newer version/update


class MemoryGraph:
    """Graph-based index for tracking relationships between memories."""

    def __init__(self) -> None:
        """Initialize memory graph."""
        self.graph: nx.DiGraph = nx.DiGraph()  # Directed graph for relationships
        self._memory_index: dict[str, dict] = {}  # memory_id -> memory_data
        self._user_graphs: dict[str, set[str]] = defaultdict(set)  # user_id -> memory_ids
        self._dirty_callback: Optional[Callable[[], None]] = None

    def register_dirty_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to invoke whenever the graph is mutated.

        The callback is invoked after every successful add_memory, add_relationship,
        or remove_memory call. The implementation of the callback (e.g. rate-limiting
        via a timestamp guard) is the caller's responsibility.

        Args:
            callback: Zero-argument callable to invoke on mutation.
        """
        self._dirty_callback = callback

    def _on_dirty(self) -> None:
        """Fire the dirty callback if one is registered."""
        if self._dirty_callback is not None:
            self._dirty_callback()

    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        metadata: Optional[dict] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add a memory node to the graph.

        Args:
            memory_id: Unique memory identifier.
            user_id: Owner user identifier.
            metadata: Optional extra node attributes.
            embedding: Optional pre-computed embedding vector stored on the node
                for use by embedding-based relationship suggestion.
        """
        node_attrs: dict[str, Any] = {"user_id": user_id, **(metadata or {})}
        if embedding is not None:
            node_attrs["embedding"] = embedding
        self.graph.add_node(memory_id, **node_attrs)
        self._memory_index[memory_id] = {"user_id": user_id, "metadata": metadata or {}}
        self._user_graphs[user_id].add(memory_id)
        logger.debug(f"Added memory {memory_id} to graph for user {user_id}")
        self._on_dirty()

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a relationship between two memories."""
        if source_id not in self.graph or target_id not in self.graph:
            logger.warning("Cannot add relationship: one or both memories not in graph")
            return False

        self.graph.add_edge(
            source_id, target_id, relation=relation_type.value, weight=weight, **(metadata or {})
        )
        logger.debug(f"Added {relation_type.value} relationship: {source_id} -> {target_id}")
        self._on_dirty()
        return True

    def get_related_memories(
        self,
        memory_id: str,
        relation_types: Optional[list[RelationType]] = None,
        max_depth: int = 1,
        direction: str = "both",  # 'outgoing', 'incoming', 'both'
    ) -> list[tuple[str, str, float]]:
        """
        Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_types: Filter by specific relation types
            max_depth: How many hops to traverse (1 = direct neighbors only)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (memory_id, relation_type, weight) tuples
        """
        if memory_id not in self.graph:
            return []

        related = []

        if direction in ("outgoing", "both"):
            # Get outgoing relationships
            for successor in nx.single_source_shortest_path_length(
                self.graph, memory_id, cutoff=max_depth
            ).keys():
                if successor == memory_id:
                    continue

                # Get edge data
                if self.graph.has_edge(memory_id, successor):
                    edge_data = self.graph[memory_id][successor]
                    relation = edge_data.get("relation", "related_to")
                    weight = edge_data.get("weight", 1.0)

                    if relation_types is None or RelationType(relation) in relation_types:
                        related.append((successor, relation, weight))

        if direction in ("incoming", "both"):
            # Get incoming relationships
            reverse_graph = self.graph.reverse()
            for predecessor in nx.single_source_shortest_path_length(
                reverse_graph, memory_id, cutoff=max_depth
            ).keys():
                if predecessor == memory_id:
                    continue

                # Get edge data
                if self.graph.has_edge(predecessor, memory_id):
                    edge_data = self.graph[predecessor][memory_id]
                    relation = edge_data.get("relation", "related_to")
                    weight = edge_data.get("weight", 1.0)

                    if relation_types is None or RelationType(relation) in relation_types:
                        related.append((predecessor, relation, weight))

        return related

    def find_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """Find shortest path between two memories."""
        try:
            path: list[str] = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_clusters(self, user_id: Optional[str] = None) -> list[set[str]]:
        """
        Find clusters of related memories using community detection.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of memory ID sets (clusters)
        """
        if user_id:
            # Create subgraph for user
            user_nodes = self._user_graphs.get(user_id, set())
            subgraph = self.graph.subgraph(user_nodes)
        else:
            subgraph = self.graph

        # Convert to undirected for community detection
        undirected = subgraph.to_undirected()

        # Find weakly connected components as clusters
        clusters = list(nx.connected_components(undirected))
        return clusters

    def get_most_connected(
        self, user_id: Optional[str] = None, top_k: int = 10
    ) -> list[tuple[str, int]]:
        """
        Get most connected memories (highest degree).

        Args:
            user_id: Optional user ID to filter by
            top_k: Number of results to return

        Returns:
            List of (memory_id, degree) tuples
        """
        if user_id:
            user_nodes = self._user_graphs.get(user_id, set())
            degrees = [
                (
                    node,
                    len(list(self.graph.predecessors(node)))
                    + len(list(self.graph.successors(node))),
                )
                for node in user_nodes
            ]
        else:
            degrees = [
                (
                    node,
                    len(list(self.graph.predecessors(node)))
                    + len(list(self.graph.successors(node))),
                )
                for node in self.graph.nodes()
            ]

        # Sort by degree descending
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_k]

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory and all its relationships from the graph."""
        if memory_id in self.graph:
            user_id = self._memory_index[memory_id]["user_id"]
            self.graph.remove_node(memory_id)
            del self._memory_index[memory_id]
            self._user_graphs[user_id].discard(memory_id)
            logger.debug(f"Removed memory {memory_id} from graph")
            self._on_dirty()

    def get_graph_stats(self, user_id: Optional[str] = None) -> dict:
        """Get statistics about the graph."""
        if user_id:
            user_nodes = self._user_graphs.get(user_id, set())
            subgraph = self.graph.subgraph(user_nodes)
        else:
            subgraph = self.graph

        num_nodes = subgraph.number_of_nodes()
        # Sum of (in-degree + out-degree) for all nodes = 2 * number_of_edges
        degree_sum = 2 * subgraph.number_of_edges()

        return {
            "num_nodes": num_nodes,
            "num_edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph) if num_nodes > 0 else 0.0,
            "num_clusters": len(self.get_clusters(user_id)),
            "avg_degree": degree_sum / max(num_nodes, 1),
        }

    def suggest_relationships(
        self,
        memory_id: str,
        candidates: list[str],
        threshold: float = 0.7,
        source_embedding: Optional[np.ndarray] = None,
    ) -> list[tuple[str, RelationType, float]]:
        """Suggest potential relationships between memories.

        Uses cosine similarity over stored embeddings when ``source_embedding``
        is provided and candidate nodes carry an ``embedding`` attribute.
        Falls back to Jaccard similarity over common graph neighbours when
        embeddings are unavailable.

        Args:
            memory_id: Source memory whose relationships to suggest.
            candidates: Candidate memory IDs to evaluate.
            threshold: Minimum similarity score to include a suggestion.
            source_embedding: Optional embedding vector for ``memory_id``.
                When supplied, embedding-based cosine similarity is preferred
                over the neighbour-set heuristic.

        Returns:
            List of (candidate_id, RelationType, confidence) tuples sorted by
            confidence descending.
        """
        suggestions: list[tuple[str, RelationType, float]] = []

        if memory_id not in self.graph or not candidates:
            return suggestions

        # --- Embedding-based path ---
        if source_embedding is not None:
            src_norm = float(np.linalg.norm(source_embedding))
            if src_norm == 0.0:
                # Zero-norm vector: fall through to Jaccard
                source_embedding = None
            else:
                for candidate in candidates:
                    if candidate not in self.graph or candidate == memory_id:
                        continue

                    cand_embedding = self.graph.nodes[candidate].get("embedding")
                    if cand_embedding is None:
                        continue

                    cand_arr = np.asarray(cand_embedding, dtype=np.float32)
                    cand_norm = float(np.linalg.norm(cand_arr))
                    if cand_norm == 0.0:
                        continue

                    cosine = float(np.dot(source_embedding, cand_arr) / (src_norm * cand_norm))
                    if cosine >= threshold:
                        relation = (
                            RelationType.SIMILAR_TO if cosine >= 0.85 else RelationType.RELATED_TO
                        )
                        suggestions.append((candidate, relation, cosine))

                suggestions.sort(key=lambda x: x[2], reverse=True)
                return suggestions

        # --- Jaccard fallback (neighbour-set heuristic) ---
        memory_neighbors = set(self.graph.neighbors(memory_id))
        for candidate in candidates:
            if candidate not in self.graph or candidate == memory_id:
                continue

            candidate_neighbors = set(self.graph.neighbors(candidate))
            common_neighbors = memory_neighbors & candidate_neighbors
            if not common_neighbors:
                continue

            all_neighbors = memory_neighbors | candidate_neighbors
            similarity = len(common_neighbors) / len(all_neighbors)
            if similarity >= threshold:
                suggestions.append((candidate, RelationType.RELATED_TO, similarity))

        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions

    def export_to_dict(self, user_id: Optional[str] = None) -> dict:
        """Export graph to dictionary format for serialization."""
        import numpy as np

        if user_id:
            user_nodes = self._user_graphs.get(user_id, set())
            subgraph = self.graph.subgraph(user_nodes)
        else:
            subgraph = self.graph

        # Convert graph to a mutable copy so we can sanitize ndarray attributes
        graph_copy = subgraph.copy()
        for _, attrs in graph_copy.nodes(data=True):
            for key, val in list(attrs.items()):
                if isinstance(val, np.ndarray):
                    attrs[key] = val.tolist()

        return cast(dict[Any, Any], nx.node_link_data(graph_copy, edges="links"))

    def import_from_dict(self, data: dict) -> None:
        """Import graph from dictionary format."""
        imported_graph = nx.node_link_graph(data, edges="links")
        self.graph = nx.compose(self.graph, imported_graph)

        # Rebuild indices
        for node, attrs in imported_graph.nodes(data=True):
            user_id = attrs.get("user_id")
            if user_id:
                self._memory_index[node] = {"user_id": user_id, "metadata": attrs}
                self._user_graphs[user_id].add(node)

    def export_to_json(
        self, file_path: Union[str, Path], user_id: Optional[str] = None, indent: int = 2
    ) -> str:
        """
        Export graph to a JSON file.

        Args:
            file_path: Path where the JSON file will be saved
            user_id: Optional user ID to export only a specific user's graph
            indent: JSON indentation level (default: 2)

        Returns:
            The file path where the graph was saved

        Example:
            >>> graph.export_to_json("memory_graph.json")
            >>> graph.export_to_json("alice_graph.json", user_id="alice")
        """
        file_path = Path(file_path)

        # Get graph data
        graph_data = self.export_to_dict(user_id)

        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=indent, ensure_ascii=False)

        logger.info(
            f"Exported graph to {file_path} (nodes: {len(graph_data['nodes'])}, edges: {len(graph_data['links'])})"
        )
        return str(file_path)

    def import_from_json(self, file_path: Union[str, Path], merge: bool = True) -> dict:
        """
        Import graph from a JSON file.

        Args:
            file_path: Path to the JSON file to import
            merge: If True, merge with existing graph; if False, replace existing graph

        Returns:
            Dictionary with import statistics

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON

        Example:
            >>> graph.import_from_json("memory_graph.json")
            >>> graph.import_from_json("backup.json", merge=False)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        # Load from file
        with open(file_path, encoding="utf-8") as f:
            graph_data = json.load(f)

        # Validate structure
        if (
            not isinstance(graph_data, dict)
            or "nodes" not in graph_data
            or "links" not in graph_data
        ):
            raise ValueError(
                f"Invalid graph format in {file_path}. Expected 'nodes' and 'links' keys."
            )

        # Store old counts for stats
        old_node_count = self.graph.number_of_nodes()
        old_edge_count = self.graph.number_of_edges()

        # Clear existing graph if not merging
        if not merge:
            self.graph.clear()
            self._memory_index.clear()
            self._user_graphs.clear()
            logger.info("Cleared existing graph before import (merge=False)")

        # Import data
        self.import_from_dict(graph_data)

        # Calculate stats
        new_node_count = self.graph.number_of_nodes()
        new_edge_count = self.graph.number_of_edges()

        stats = {
            "file_path": str(file_path),
            "nodes_before": old_node_count,
            "edges_before": old_edge_count,
            "nodes_after": new_node_count,
            "edges_after": new_edge_count,
            "nodes_imported": len(graph_data["nodes"]),
            "edges_imported": len(graph_data["links"]),
            "merged": merge,
        }

        logger.info(
            f"Imported graph from {file_path} "
            f"(nodes: {old_node_count} -> {new_node_count}, edges: {old_edge_count} -> {new_edge_count})"
        )

        return stats
