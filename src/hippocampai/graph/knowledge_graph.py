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
from typing import Any, Optional

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

    def __init__(self) -> None:
        """Initialize knowledge graph."""
        super().__init__()

        # Additional indices for entities and facts
        self._entity_index: dict[str, str] = {}  # entity_id -> node_id mapping
        self._fact_index: dict[str, str] = {}  # fact_id -> node_id mapping
        self._topic_index: dict[str, str] = {}  # topic -> node_id mapping

    def add_entity(self, entity: Entity, metadata: Optional[dict[str, Any]] = None) -> str:
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
        metadata: Optional[dict[str, Any]] = None,
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

    def add_topic(self, topic: str, metadata: Optional[dict[str, Any]] = None) -> str:
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

        result: bool = self.add_relationship(
            memory_id,
            entity_node_id,
            relation_type,
            weight=confidence,
            metadata={"edge_type": "memory_entity"},
        )
        return result

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

        result: bool = self.add_relationship(
            memory_id,
            fact_node_id,
            RelationType.SUPPORTS,
            weight=confidence,
            metadata={"edge_type": "memory_fact"},
        )
        return result

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

        result: bool = self.add_relationship(
            fact_node_id,
            entity_node_id,
            RelationType.RELATED_TO,
            weight=confidence,
            metadata={"edge_type": "fact_entity"},
        )
        return result

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

        result: bool = self.add_relationship(
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
        return result

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

        result: bool = self.add_relationship(
            memory_id,
            topic_node_id,
            RelationType.PART_OF,
            weight=confidence,
            metadata={"edge_type": "memory_topic"},
        )
        return result

    def get_entity_memories(self, entity_id: str) -> list[str]:
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

    def get_entity_facts(self, entity_id: str) -> list[str]:
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
                fact_id = node_data.get("fact_id")
                if fact_id is not None:
                    facts.append(fact_id)

        return facts

    def get_topic_memories(self, topic: str) -> list[str]:
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
    ) -> dict[str, list[tuple[str, int]]]:
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

        connections: dict[str, list[tuple[str, int]]] = {}

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

                    # Skip if entity_id is None
                    if target_entity_id is None:
                        continue

                    # Ensure it's a string
                    target_entity_id = str(target_entity_id)

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
        self, center_id: str, radius: int = 2, include_types: Optional[list[NodeType]] = None
    ) -> dict[str, Any]:
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
                nodes_in_radius = dict.fromkeys(filtered_nodes, 0)

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

    def get_entity_timeline(self, entity_id: str) -> list[dict[str, Any]]:
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
        timeline.sort(key=lambda x: str(x.get("timestamp", "")))

        return timeline

    def infer_new_facts(
        self, user_id: Optional[str] = None, llm: Optional[Any] = None
    ) -> list[dict[str, Any]]:
        """Infer new facts from existing knowledge graph patterns.

        Applies pattern-based rules first.  When *llm* is provided, an
        additional LLM-backed inference pass is run over each entity's
        local neighbourhood to surface open-domain facts that the rule set
        cannot capture.

        Args:
            user_id: Optional user ID to limit inference scope.
            llm: Optional LLM adapter (must expose ``generate(prompt, max_tokens)``).
                 When ``None`` only pattern rules are applied.

        Returns:
            Deduplicated list of inferred facts, each a dict with keys:
            ``entity_id``, ``fact``, ``confidence``, ``rule``, ``supporting_facts``.
        """
        inferred: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()  # (entity_id, fact_text) dedup key

        def _add(item: dict[str, Any]) -> None:
            key = (item["entity_id"], item["fact"])
            if key not in seen:
                seen.add(key)
                inferred.append(item)

        for entity_id, entity_node_id in self._entity_index.items():
            relationships = list(self.graph.edges(entity_node_id, data=True))
            _ = self.graph.nodes[entity_node_id].get("text", "")  # reserved for future use

            for source, target, edge_data in relationships:
                relation = edge_data.get("original_relation")
                source_text = self.graph.nodes[source].get("text", "")
                target_text = self.graph.nodes[target].get("text", "")

                # Rule 1: works_at -> located_in
                if relation == "works_at":
                    for _, loc_target, loc_edge in self.graph.edges(target, data=True):
                        if loc_edge.get("original_relation") == "located_in":
                            loc_data = self.graph.nodes[loc_target]
                            if loc_data.get("node_type") == NodeType.ENTITY.value:
                                location = loc_data.get("text", "")
                                _add({
                                    "entity_id": entity_id,
                                    "fact": f"{source_text} likely located in {location}",
                                    "confidence": 0.7,
                                    "rule": "works_at_location_inference",
                                    "supporting_facts": [source, target, loc_target],
                                })

                # Rule 2: studied_at -> located_in
                elif relation == "studied_at":
                    for _, loc_target, loc_edge in self.graph.edges(target, data=True):
                        if loc_edge.get("original_relation") == "located_in":
                            loc_data = self.graph.nodes[loc_target]
                            if loc_data.get("node_type") == NodeType.ENTITY.value:
                                location = loc_data.get("text", "")
                                _add({
                                    "entity_id": entity_id,
                                    "fact": f"{source_text} likely located in {location} during studies",
                                    "confidence": 0.7,
                                    "rule": "studied_at_location_inference",
                                    "supporting_facts": [source, target, loc_target],
                                })

                # Rule 3: manages B; B works_at C -> A works_at C
                elif relation == "manages":
                    for _, org_target, org_edge in self.graph.edges(target, data=True):
                        if org_edge.get("original_relation") == "works_at":
                            org_data = self.graph.nodes[org_target]
                            if org_data.get("node_type") == NodeType.ENTITY.value:
                                org_name = org_data.get("text", "")
                                _add({
                                    "entity_id": entity_id,
                                    "fact": f"{source_text} likely works at {org_name}",
                                    "confidence": 0.7,
                                    "rule": "manages_works_at_inference",
                                    "supporting_facts": [source, target, org_target],
                                })

                # Rule 4: knows B; B works_at C -> A has connection to C
                elif relation == "knows":
                    for _, org_target, org_edge in self.graph.edges(target, data=True):
                        if org_edge.get("original_relation") == "works_at":
                            org_data = self.graph.nodes[org_target]
                            if org_data.get("node_type") == NodeType.ENTITY.value:
                                org_name = org_data.get("text", "")
                                _add({
                                    "entity_id": entity_id,
                                    "fact": f"{source_text} has a connection to {org_name} via {target_text}",
                                    "confidence": 0.5,
                                    "rule": "knows_organization_inference",
                                    "supporting_facts": [source, target, org_target],
                                })

                # Rule 5: founded_by -> B works_at A (founder relationship)
                elif relation == "founded_by":
                    founder_data = self.graph.nodes[target]
                    if founder_data.get("node_type") == NodeType.ENTITY.value:
                        founder_name = founder_data.get("text", "")
                        _add({
                            "entity_id": entity_id,
                            "fact": f"{founder_name} likely works at {source_text} as a founder",
                            "confidence": 0.8,
                            "rule": "founded_by_inverse",
                            "supporting_facts": [source, target],
                        })

                # Rule 7: part_of transitive (A part_of B; B part_of C -> A part_of C)
                elif relation == "part_of":
                    for _, grand_target, grand_edge in self.graph.edges(target, data=True):
                        if grand_edge.get("original_relation") == "part_of":
                            grand_data = self.graph.nodes[grand_target]
                            if grand_data.get("node_type") == NodeType.ENTITY.value:
                                grand_name = grand_data.get("text", "")
                                _add({
                                    "entity_id": entity_id,
                                    "fact": f"{source_text} is transitively part of {grand_name}",
                                    "confidence": 0.65,
                                    "rule": "part_of_transitive",
                                    "supporting_facts": [source, target, grand_target],
                                })

        # Rule 6: co-location — entities that share a located_in target are co-located
        # Group entities by their location target (cap group size to avoid O(n^2) explosion)
        _MAX_CO_LOCATION_GROUP = 50
        location_to_entities: dict[str, list[tuple[str, str]]] = {}
        for entity_id, entity_node_id in self._entity_index.items():
            for _, loc_target, edge_data in self.graph.edges(entity_node_id, data=True):
                if edge_data.get("original_relation") == "located_in":
                    loc_data = self.graph.nodes.get(loc_target, {})
                    if loc_data.get("node_type") == NodeType.ENTITY.value:
                        if loc_target not in location_to_entities:
                            location_to_entities[loc_target] = []
                        if len(location_to_entities[loc_target]) < _MAX_CO_LOCATION_GROUP:
                            location_to_entities[loc_target].append(
                                (entity_id, entity_node_id)
                            )

        for loc_node_id, group in location_to_entities.items():
            loc_name = self.graph.nodes[loc_node_id].get("text", "")
            for i, (eid_a, nid_a) in enumerate(group):
                for _, nid_b in group[i + 1:]:
                    name_a = self.graph.nodes[nid_a].get("text", "")
                    name_b = self.graph.nodes[nid_b].get("text", "")
                    _add({
                        "entity_id": eid_a,
                        "fact": f"{name_a} and {name_b} are co-located in {loc_name}",
                        "confidence": 0.4,
                        "rule": "co_location_inference",
                        "supporting_facts": [nid_a, nid_b, loc_node_id],
                    })

        # LLM-backed inference (only when an LLM adapter is provided)
        if llm is not None:
            for fact in self._infer_facts_llm(llm):
                _add(fact)

        logger.info(
            f"infer_new_facts produced {len(inferred)} facts "
            f"({'with' if llm else 'without'} LLM pass)"
        )
        return inferred

    def _infer_facts_llm(self, llm: Any) -> list[dict[str, Any]]:
        """Use the LLM to infer new facts from each entity's local neighbourhood.

        This method summarises an entity's immediate graph neighbourhood into a
        short text prompt and asks the LLM to state additional facts that can be
        confidently inferred.  The entity text is sanitised (newlines stripped,
        length capped) before being inserted into the prompt to reduce prompt
        injection risk.

        Args:
            llm: LLM adapter exposing ``generate(prompt: str, max_tokens: int) -> str``.

        Returns:
            List of inferred fact dicts (same schema as ``infer_new_facts``).
        """
        results: list[dict[str, Any]] = []

        for entity_id, entity_node_id in self._entity_index.items():
            entity_data = self.graph.nodes.get(entity_node_id, {})
            raw_text = entity_data.get("text", "")
            # Sanitise: strip newlines and cap length to bound token usage
            entity_name = raw_text.replace("\n", " ").replace("\r", " ")[:200]

            # Collect 1-hop neighbourhood summary
            neighbour_facts: list[str] = []
            for _, neighbor, edge_data in self.graph.edges(entity_node_id, data=True):
                neighbor_data = self.graph.nodes.get(neighbor, {})
                neighbor_text = neighbor_data.get("text", "")[:100].replace("\n", " ")
                relation = edge_data.get("original_relation", edge_data.get("relation", "related_to"))
                neighbour_facts.append(f"{entity_name} {relation} {neighbor_text}")

            if not neighbour_facts:
                continue

            facts_str = "; ".join(neighbour_facts[:10])  # Cap at 10 to keep prompt bounded
            prompt = (
                f"Given these known facts about {entity_name}: {facts_str}. "
                "What additional facts can be confidently inferred? "
                "Return each fact on its own line in exactly this format: "
                "FACT: [text] | CONFIDENCE: [0.0-1.0]"
            )

            try:
                response = llm.generate(prompt, max_tokens=256)
                for line in response.splitlines():
                    line = line.strip()
                    if not line.startswith("FACT:"):
                        continue
                    try:
                        fact_part, conf_part = line.split("|", 1)
                        fact_text = fact_part.replace("FACT:", "").strip()
                        conf_str = conf_part.replace("CONFIDENCE:", "").strip()
                        confidence = float(conf_str)
                        confidence = max(0.0, min(1.0, confidence))
                        results.append({
                            "entity_id": entity_id,
                            "fact": fact_text,
                            "confidence": confidence,
                            "rule": "llm_inference",
                            "supporting_facts": [entity_node_id],
                        })
                    except (ValueError, AttributeError):
                        continue
            except Exception as exc:
                logger.warning(f"LLM inference failed for entity {entity_id}: {exc}")

        return results
