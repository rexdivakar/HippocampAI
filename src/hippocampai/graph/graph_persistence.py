"""Save/load KnowledgeGraph to/from JSON."""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from hippocampai.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def save(graph: KnowledgeGraph, file_path: str) -> None:
    """Serialize KnowledgeGraph to JSON file.

    Persists the NetworkX graph data along with the auxiliary indices
    (_entity_index, _fact_index, _topic_index, _user_graphs, _memory_index).

    Args:
        graph: KnowledgeGraph instance to save.
        file_path: Destination JSON file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    graph_data: dict[str, Any] = nx.node_link_data(graph.graph, edges="links")

    # Serialize auxiliary indices
    data: dict[str, Any] = {
        "graph": graph_data,
        "entity_index": graph._entity_index,
        "fact_index": graph._fact_index,
        "topic_index": graph._topic_index,
        "user_graphs": {uid: list(mids) for uid, mids in graph._user_graphs.items()},
        "memory_index": graph._memory_index,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        f"Saved knowledge graph to {path} "
        f"(nodes={graph.graph.number_of_nodes()}, edges={graph.graph.number_of_edges()})"
    )


def load(file_path: str) -> KnowledgeGraph:
    """Deserialize KnowledgeGraph from JSON file.

    Args:
        file_path: Source JSON file path.

    Returns:
        Restored KnowledgeGraph instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    kg = KnowledgeGraph()

    # Restore NetworkX graph
    graph_data = data.get("graph", {})
    if graph_data:
        kg.graph = nx.node_link_graph(graph_data, edges="links")

    # Restore auxiliary indices
    kg._entity_index = data.get("entity_index", {})
    kg._fact_index = data.get("fact_index", {})
    kg._topic_index = data.get("topic_index", {})

    user_graphs_raw = data.get("user_graphs", {})
    for uid, mids in user_graphs_raw.items():
        kg._user_graphs[uid] = set(mids)

    kg._memory_index = data.get("memory_index", {})

    logger.info(
        f"Loaded knowledge graph from {path} "
        f"(nodes={kg.graph.number_of_nodes()}, edges={kg.graph.number_of_edges()})"
    )
    return kg
