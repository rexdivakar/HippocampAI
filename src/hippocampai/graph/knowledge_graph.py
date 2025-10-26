"""Enhanced knowledge graph with entity and fact integration.

This module extends the memory graph with:
- Entity nodes and relationships
- Fact tracking
- Semantic connections between memories, entities, and facts
- Temporal edges
- Knowledge inference
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from hippocampai.graph.memory_graph import MemoryGraph, RelationType
from hippocampai.pipeline.entity_recognition import Entity, EntityRelationship
from hippocampai.pipeline.fact_extraction import ExtractedFact

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    MEMORY = "memory"
    ENTITY = "entity"
    FACT = "fact"
    TOPIC = "topic"


class KnowledgeGraph(MemoryGraph):
    """Enhanced memory graph with entity and fact support."""

    def __init__(self):
        """Initialize knowledge graph."""
        super().__init__()

        # Additional indices for entities and facts
        self._entity_index: Dict[str, str] = {}  # entity_id -> node_id mapping
        self._fact_index: Dict[str, str] = {}  # fact_id -> node_id mapping
        self._topic_index: Dict[str, str] = {}  # topic -> node_id mapping

    def add_entity(self, entity: Entity, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add an entity node to the graph.

        Args:
            entity: Entity object
            metadata: Additional metadata

        Returns:
            Node ID for the entity
        """
        node_id = f"entity_{entity.entity_id}"

        # Add node to graph
        self.graph.add_node(
            node_id,
            node_type=NodeType.ENTITY.value,
            entity_id=entity.entity_id,
            text=entity.text,
            entity_type=entity.type.value,
            confidence=entity.confidence,
            canonical_name=entity.canonical_name,
            first_seen=entity.first_seen.isoformat(),
            last_seen=entity.last_seen.isoformat(),
            mention_count=entity.mention_count,
            **(metadata or {}),
        )

        self._entity_index[entity.entity_id] = node_id
        logger.debug(f"Added entity node: {entity.text} ({entity.type.value})")

        return node_id

    def add_fact(
        self,
        fact: ExtractedFact,
        fact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a fact node to the graph.

        Args:
            fact: ExtractedFact object
            fact_id: Optional custom fact ID
            metadata: Additional metadata

        Returns:
            Node ID for the fact
        """
        if fact_id is None:
            fact_id = f"fact_{hash(fact.fact) % 10**10}"

        node_id = f"fact_{fact_id}"

        # Add node to graph
        self.graph.add_node(
            node_id,
            node_type=NodeType.FACT.value,
            fact_id=fact_id,
            fact_text=fact.fact,
            category=fact.category.value,
            confidence=fact.confidence,
            temporal=fact.temporal,
            temporal_type=fact.temporal_type.value,
            source=fact.source,
            extracted_at=fact.extracted_at.isoformat(),
            **(metadata or {}),
        )

        self._fact_index[fact_id] = node_id
        logger.debug(f"Added fact node: {fact.fact[:50]}...")

        return node_id

    def add_topic(self, topic: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a topic node to the graph.

        Args:
            topic: Topic name
            metadata: Additional metadata

        Returns:
            Node ID for the topic
        """
        if topic in self._topic_index:
            return self._topic_index[topic]

        node_id = f"topic_{topic.lower().replace(' ', '_')}"

        self.graph.add_node(
            node_id, node_type=NodeType.TOPIC.value, topic=topic, **(metadata or {})
        )

        self._topic_index[topic] = node_id
        logger.debug(f"Added topic node: {topic}")

        return node_id

    def link_memory_to_entity(
        self,
        memory_id: str,
        entity_id: str,
        relation_type: RelationType = RelationType.RELATED_TO,
        confidence: float = 0.9,
    ) -> bool:
        """Link a memory to an entity.

        Args:
            memory_id: Memory node ID
            entity_id: Entity ID
            relation_type: Type of relationship
            confidence: Confidence score

        Returns:
            True if successful
        """
        entity_node_id = self._entity_index.get(entity_id)
        if not entity_node_id:
            logger.warning(f"Entity {entity_id} not found in graph")
            return False

        return self.add_relationship(
            memory_id,
            entity_node_id,
            relation_type,
            weight=confidence,
            metadata={"edge_type": "memory_entity"},
        )

    def link_memory_to_fact(self, memory_id: str, fact_id: str, confidence: float = 0.9) -> bool:
        """Link a memory to a fact extracted from it.

        Args:
            memory_id: Memory node ID
            fact_id: Fact ID
            confidence: Confidence score

        Returns:
            True if successful
        """
        fact_node_id = self._fact_index.get(fact_id)
        if not fact_node_id:
            logger.warning(f"Fact {fact_id} not found in graph")
            return False

        return self.add_relationship(
            memory_id,
            fact_node_id,
            RelationType.SUPPORTS,
            weight=confidence,
            metadata={"edge_type": "memory_fact"},
        )

    def link_fact_to_entity(self, fact_id: str, entity_id: str, confidence: float = 0.9) -> bool:
        """Link a fact to an entity it mentions.

        Args:
            fact_id: Fact ID
            entity_id: Entity ID
            confidence: Confidence score

        Returns:
            True if successful
        """
        fact_node_id = self._fact_index.get(fact_id)
        entity_node_id = self._entity_index.get(entity_id)

        if not fact_node_id or not entity_node_id:
            logger.warning("Fact or entity not found in graph")
            return False

        return self.add_relationship(
            fact_node_id,
            entity_node_id,
            RelationType.RELATED_TO,
            weight=confidence,
            metadata={"edge_type": "fact_entity"},
        )

    def link_entities(self, relationship: EntityRelationship) -> bool:
        """Link two entities with a relationship.

        Args:
            relationship: EntityRelationship object

        Returns:
            True if successful
        """
        from_node_id = self._entity_index.get(relationship.from_entity_id)
        to_node_id = self._entity_index.get(relationship.to_entity_id)

        if not from_node_id or not to_node_id:
            logger.warning("One or both entities not found in graph")
            return False

        # Map entity relation types to graph relation types
        relation_mapping = {
            "works_at": RelationType.RELATED_TO,
            "located_in": RelationType.PART_OF,
            "studied_at": RelationType.RELATED_TO,
            "knows": RelationType.RELATED_TO,
            "manages": RelationType.RELATED_TO,
        }

        relation_type = relation_mapping.get(relationship.relation_type, RelationType.RELATED_TO)

        return self.add_relationship(
            from_node_id,
            to_node_id,
            relation_type,
            weight=relationship.confidence,
            metadata={
                "edge_type": "entity_entity",
                "original_relation": relationship.relation_type,
                "context": relationship.context,
            },
        )

    def link_memory_to_topic(self, memory_id: str, topic: str, confidence: float = 0.8) -> bool:
        """Link a memory to a topic.

        Args:
            memory_id: Memory node ID
            topic: Topic name
            confidence: Confidence score

        Returns:
            True if successful
        """
        topic_node_id = self._topic_index.get(topic)
        if not topic_node_id:
            # Create topic node if it doesn't exist
            topic_node_id = self.add_topic(topic)

        return self.add_relationship(
            memory_id,
            topic_node_id,
            RelationType.PART_OF,
            weight=confidence,
            metadata={"edge_type": "memory_topic"},
        )

    def get_entity_memories(self, entity_id: str) -> List[str]:
        """Get all memories mentioning an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of memory IDs
        """
        entity_node_id = self._entity_index.get(entity_id)
        if not entity_node_id:
            return []

        # Get all nodes connected to this entity
        memories = []
        for neighbor in self.graph.neighbors(entity_node_id):
            node_data = self.graph.nodes[neighbor]
            if node_data.get("node_type") == NodeType.MEMORY.value:
                memories.append(neighbor)

        return memories

    def get_entity_facts(self, entity_id: str) -> List[str]:
        """Get all facts about an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of fact IDs
        """
        entity_node_id = self._entity_index.get(entity_id)
        if not entity_node_id:
            return []

        facts = []
        for neighbor in self.graph.predecessors(entity_node_id):
            node_data = self.graph.nodes[neighbor]
            if node_data.get("node_type") == NodeType.FACT.value:
                facts.append(node_data.get("fact_id"))

        return facts

    def get_topic_memories(self, topic: str) -> List[str]:
        """Get all memories about a topic.

        Args:
            topic: Topic name

        Returns:
            List of memory IDs
        """
        topic_node_id = self._topic_index.get(topic)
        if not topic_node_id:
            return []

        memories = []
        for neighbor in self.graph.predecessors(topic_node_id):
            node_data = self.graph.nodes[neighbor]
            if node_data.get("node_type") == NodeType.MEMORY.value:
                memories.append(neighbor)

        return memories

    def find_entity_connections(
        self, entity_id: str, max_distance: int = 2
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Find all entities connected to a given entity.

        Args:
            entity_id: Source entity ID
            max_distance: Maximum distance (hops) to search

        Returns:
            Dictionary mapping relation types to lists of (entity_id, distance) tuples
        """
        entity_node_id = self._entity_index.get(entity_id)
        if not entity_node_id:
            return {}

        connections = {}

        # Use BFS to find connected entities
        import networkx as nx

        try:
            # Get all paths within max_distance
            paths = nx.single_source_shortest_path_length(
                self.graph, entity_node_id, cutoff=max_distance
            )

            for target_node, distance in paths.items():
                if target_node == entity_node_id:
                    continue

                node_data = self.graph.nodes[target_node]
                if node_data.get("node_type") == NodeType.ENTITY.value:
                    target_entity_id = node_data.get("entity_id")

                    # Get relation type from edge
                    if self.graph.has_edge(entity_node_id, target_node):
                        edge_data = self.graph[entity_node_id][target_node]
                        relation = edge_data.get("original_relation", "related_to")
                    else:
                        relation = "related_to"

                    if relation not in connections:
                        connections[relation] = []
                    connections[relation].append((target_entity_id, distance))

        except Exception as e:
            logger.warning(f"Error finding entity connections: {e}")

        return connections

    def get_knowledge_subgraph(
        self, center_id: str, radius: int = 2, include_types: Optional[List[NodeType]] = None
    ) -> Dict[str, Any]:
        """Get a subgraph around a central node.

        Args:
            center_id: Central node ID (memory, entity, or fact)
            radius: How many hops to include
            include_types: Filter by node types

        Returns:
            Subgraph data with nodes and edges
        """
        if center_id not in self.graph:
            return {"nodes": [], "edges": []}

        # Get nodes within radius
        import networkx as nx

        try:
            nodes_in_radius = nx.single_source_shortest_path_length(
                self.graph, center_id, cutoff=radius
            )

            # Filter by type if specified
            if include_types:
                type_values = [t.value for t in include_types]
                filtered_nodes = []
                for node_id in nodes_in_radius.keys():
                    node_data = self.graph.nodes[node_id]
                    if node_data.get("node_type") in type_values:
                        filtered_nodes.append(node_id)
                nodes_in_radius = {n: 0 for n in filtered_nodes}

            # Build subgraph
            subgraph = self.graph.subgraph(nodes_in_radius.keys())

            # Convert to dict format
            nodes = []
            for node_id in subgraph.nodes():
                node_data = dict(self.graph.nodes[node_id])
                node_data["id"] = node_id
                nodes.append(node_data)

            edges = []
            for source, target in subgraph.edges():
                edge_data = dict(self.graph[source][target])
                edge_data["source"] = source
                edge_data["target"] = target
                edges.append(edge_data)

            return {"nodes": nodes, "edges": edges, "center": center_id, "radius": radius}

        except Exception as e:
            logger.warning(f"Error building subgraph: {e}")
            return {"nodes": [], "edges": []}

    def get_entity_timeline(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of facts and memories about an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of timeline events sorted by time
        """
        entity_node_id = self._entity_index.get(entity_id)
        if not entity_node_id:
            return []

        timeline = []

        # Get all connected facts and memories
        for neighbor in list(self.graph.predecessors(entity_node_id)) + list(
            self.graph.successors(entity_node_id)
        ):
            node_data = self.graph.nodes[neighbor]
            node_type = node_data.get("node_type")

            if node_type == NodeType.FACT.value:
                timeline.append(
                    {
                        "type": "fact",
                        "id": node_data.get("fact_id"),
                        "text": node_data.get("fact_text"),
                        "timestamp": node_data.get("extracted_at"),
                        "temporal": node_data.get("temporal"),
                        "temporal_type": node_data.get("temporal_type"),
                    }
                )
            elif node_type == NodeType.MEMORY.value:
                timeline.append(
                    {
                        "type": "memory",
                        "id": neighbor,
                        "timestamp": node_data.get("created_at"),
                    }
                )

        # Sort by timestamp
        timeline.sort(key=lambda x: x.get("timestamp", ""))

        return timeline

    def infer_new_facts(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Infer new facts from existing knowledge graph patterns.

        Args:
            user_id: Optional user ID to limit inference

        Returns:
            List of inferred facts with confidence scores
        """
        inferred = []

        # Simple inference rules
        # Rule 1: If A works_at B and B located_in C, then A likely located_in C
        # Rule 2: If multiple facts about same entity, aggregate patterns

        # Get all entity relationships
        for entity_id, entity_node_id in self._entity_index.items():
            # Get entity's relationships
            relationships = list(self.graph.edges(entity_node_id, data=True))

            for source, target, edge_data in relationships:
                relation = edge_data.get("original_relation")

                # Apply inference rules
                if relation == "works_at":
                    # Check if organization has location
                    org_relationships = list(self.graph.edges(target, data=True))
                    for org_source, org_target, org_edge_data in org_relationships:
                        if org_edge_data.get("original_relation") == "located_in":
                            # Infer person is located in same place
                            target_data = self.graph.nodes[org_target]
                            if target_data.get("node_type") == NodeType.ENTITY.value:
                                location = target_data.get("text")
                                inferred.append(
                                    {
                                        "entity_id": entity_id,
                                        "fact": f"{self.graph.nodes[source].get('text', '')} likely located in {location}",
                                        "confidence": 0.7,
                                        "rule": "works_at_location_inference",
                                        "supporting_facts": [source, target, org_target],
                                    }
                                )

        return inferred
