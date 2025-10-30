"""Relationship mapping and analysis pipeline.

This module provides:
- Advanced relationship extraction between entities and memories
- Relationship strength scoring
- Relationship network analysis
- Relationship visualization data
- Co-occurrence analysis
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .entity_recognition import EntityRelationship, RelationType

logger = logging.getLogger(__name__)


class RelationshipStrength(str, Enum):
    """Strength levels for relationships."""

    VERY_WEAK = "very_weak"  # 0.0-0.2
    WEAK = "weak"  # 0.2-0.4
    MODERATE = "moderate"  # 0.4-0.6
    STRONG = "strong"  # 0.6-0.8
    VERY_STRONG = "very_strong"  # 0.8-1.0


class ScoredRelationship(BaseModel):
    """Relationship with computed strength score."""

    from_entity_id: str
    to_entity_id: str
    relation_type: RelationType
    confidence: float = Field(..., ge=0.0, le=1.0)
    strength_score: float = Field(..., ge=0.0, le=1.0, description="Computed relationship strength")
    strength_level: RelationshipStrength
    co_occurrence_count: int = Field(default=0, description="Number of times entities co-occur")
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipCluster(BaseModel):
    """A cluster of related entities."""

    cluster_id: str
    entities: list[str] = Field(default_factory=list)
    relationships: list[ScoredRelationship] = Field(default_factory=list)
    cluster_type: str = Field(default="general")
    cohesion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipPath(BaseModel):
    """A path between two entities through relationships."""

    from_entity: str
    to_entity: str
    path: list[tuple[str, RelationType, str]] = Field(
        default_factory=list, description="List of (entity, relation_type, entity) tuples"
    )
    path_length: int
    path_strength: float = Field(..., ge=0.0, le=1.0)


class RelationshipNetwork(BaseModel):
    """Complete relationship network analysis."""

    entities: list[str] = Field(default_factory=list)
    relationships: list[ScoredRelationship] = Field(default_factory=list)
    clusters: list[RelationshipCluster] = Field(default_factory=list)
    central_entities: list[tuple[str, float]] = Field(
        default_factory=list, description="(entity_id, centrality_score)"
    )
    network_density: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipMapper:
    """Advanced relationship mapping and analysis."""

    def __init__(self):
        """Initialize relationship mapper."""
        # Relationship storage: (from_entity, to_entity, relation_type) -> ScoredRelationship
        self.relationships: dict[tuple[str, str, RelationType], ScoredRelationship] = {}

        # Co-occurrence matrix: (entity1, entity2) -> count
        self.co_occurrences: dict[tuple[str, str], int] = defaultdict(int)

        # Entity to relationships index
        self.entity_relationships: dict[str, list[ScoredRelationship]] = defaultdict(list)

    def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: RelationType,
        confidence: float,
        context: Optional[str] = None,
    ) -> ScoredRelationship:
        """Add or update a relationship.

        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relation_type: Type of relationship
            confidence: Confidence score
            context: Context where relationship was found

        Returns:
            The scored relationship
        """
        key = (from_entity_id, to_entity_id, relation_type)

        if key in self.relationships:
            # Update existing relationship
            rel = self.relationships[key]
            rel.confidence = max(rel.confidence, confidence)
            rel.co_occurrence_count += 1
            rel.last_seen = datetime.now(timezone.utc)
            if context and context not in rel.contexts:
                rel.contexts.append(context)

            # Recompute strength
            rel.strength_score = self._compute_strength_score(rel)
            rel.strength_level = self._get_strength_level(rel.strength_score)

        else:
            # Create new relationship
            strength_score = confidence  # Initial strength is confidence
            rel = ScoredRelationship(
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                relation_type=relation_type,
                confidence=confidence,
                strength_score=strength_score,
                strength_level=self._get_strength_level(strength_score),
                co_occurrence_count=1,
                contexts=[context] if context else [],
            )
            self.relationships[key] = rel

            # Add to entity index
            self.entity_relationships[from_entity_id].append(rel)
            self.entity_relationships[to_entity_id].append(rel)

        # Update co-occurrence
        co_key = tuple(sorted([from_entity_id, to_entity_id]))
        self.co_occurrences[co_key] += 1

        return rel

    def add_relationship_from_entity_relationship(
        self, entity_rel: EntityRelationship
    ) -> ScoredRelationship:
        """Add relationship from EntityRelationship object.

        Args:
            entity_rel: EntityRelationship object

        Returns:
            The scored relationship
        """
        return self.add_relationship(
            from_entity_id=entity_rel.from_entity_id,
            to_entity_id=entity_rel.to_entity_id,
            relation_type=entity_rel.relation_type,
            confidence=entity_rel.confidence,
            context=entity_rel.context,
        )

    def _compute_strength_score(self, rel: ScoredRelationship) -> float:
        """Compute relationship strength based on multiple factors.

        Args:
            rel: The relationship to score

        Returns:
            Strength score (0.0-1.0)
        """
        # Base score from confidence
        score = rel.confidence * 0.4

        # Frequency boost (more co-occurrences = stronger)
        frequency_score = min(1.0, rel.co_occurrence_count / 10.0) * 0.3
        score += frequency_score

        # Recency boost (more recent = stronger)
        time_diff = (datetime.now(timezone.utc) - rel.last_seen).days
        recency_score = max(0.0, 1.0 - (time_diff / 365.0)) * 0.2  # Decay over 1 year
        score += recency_score

        # Context diversity boost (more contexts = stronger)
        context_score = min(1.0, len(rel.contexts) / 5.0) * 0.1
        score += context_score

        return min(1.0, score)

    def _get_strength_level(self, strength_score: float) -> RelationshipStrength:
        """Convert strength score to categorical level.

        Args:
            strength_score: Numerical strength score

        Returns:
            Categorical strength level
        """
        if strength_score >= 0.8:
            return RelationshipStrength.VERY_STRONG
        if strength_score >= 0.6:
            return RelationshipStrength.STRONG
        if strength_score >= 0.4:
            return RelationshipStrength.MODERATE
        if strength_score >= 0.2:
            return RelationshipStrength.WEAK
        return RelationshipStrength.VERY_WEAK

    def get_entity_relationships(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        min_strength: float = 0.0,
    ) -> list[ScoredRelationship]:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity ID
            relation_type: Filter by relation type
            min_strength: Minimum strength threshold

        Returns:
            List of scored relationships
        """
        relationships = self.entity_relationships.get(entity_id, [])

        # Apply filters
        if relation_type:
            relationships = [r for r in relationships if r.relation_type == relation_type]

        if min_strength > 0.0:
            relationships = [r for r in relationships if r.strength_score >= min_strength]

        # Sort by strength (strongest first)
        relationships.sort(key=lambda r: r.strength_score, reverse=True)

        return relationships

    def find_relationship_path(
        self, from_entity: str, to_entity: str, max_depth: int = 3
    ) -> Optional[RelationshipPath]:
        """Find shortest path between two entities.

        Args:
            from_entity: Source entity ID
            to_entity: Target entity ID
            max_depth: Maximum path length to search

        Returns:
            RelationshipPath if path exists, None otherwise
        """
        if from_entity == to_entity:
            return RelationshipPath(
                from_entity=from_entity,
                to_entity=to_entity,
                path=[],
                path_length=0,
                path_strength=1.0,
            )

        # BFS to find shortest path
        queue = [(from_entity, [], 1.0)]  # (current_entity, path, strength)
        visited = {from_entity}

        while queue:
            current, path, strength = queue.pop(0)

            if len(path) >= max_depth:
                continue

            # Get relationships for current entity
            rels = self.entity_relationships.get(current, [])

            for rel in rels:
                # Determine next entity
                if rel.from_entity_id == current:
                    next_entity = rel.to_entity_id
                else:
                    next_entity = rel.from_entity_id

                if next_entity in visited:
                    continue

                # Compute cumulative strength
                new_strength = strength * rel.strength_score

                # Build path
                new_path = path + [(current, rel.relation_type, next_entity)]

                # Check if we reached target
                if next_entity == to_entity:
                    return RelationshipPath(
                        from_entity=from_entity,
                        to_entity=to_entity,
                        path=new_path,
                        path_length=len(new_path),
                        path_strength=new_strength,
                    )

                # Continue search
                visited.add(next_entity)
                queue.append((next_entity, new_path, new_strength))

        return None

    def compute_entity_centrality(self, entity_id: str) -> float:
        """Compute centrality score for an entity.

        Centrality measures how "connected" an entity is in the network.

        Args:
            entity_id: Entity ID

        Returns:
            Centrality score (0.0-1.0)
        """
        rels = self.entity_relationships.get(entity_id, [])

        if not rels:
            return 0.0

        # Weighted by relationship strength (degree centrality with weights)
        weighted_degree = sum(r.strength_score for r in rels)

        # Normalize by max possible connections
        max_connections = len(self.entity_relationships)
        if max_connections == 0:
            return 0.0

        centrality = min(1.0, (weighted_degree / (max_connections * 0.5)))

        return centrality

    def detect_relationship_clusters(self, min_cluster_size: int = 2) -> list[RelationshipCluster]:
        """Detect clusters of strongly related entities.

        Args:
            min_cluster_size: Minimum number of entities in a cluster

        Returns:
            List of relationship clusters
        """
        # Build adjacency list for strong relationships
        adjacency: dict[str, set[str]] = defaultdict(set)

        for rel in self.relationships.values():
            if rel.strength_score >= 0.5:  # Only consider moderate+ relationships
                adjacency[rel.from_entity_id].add(rel.to_entity_id)
                adjacency[rel.to_entity_id].add(rel.from_entity_id)

        # Find connected components using DFS
        visited = set()
        clusters = []

        def dfs(node: str, cluster: set[str]):
            visited.add(node)
            cluster.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for entity in adjacency:
            if entity not in visited:
                cluster_entities: set[str] = set()
                dfs(entity, cluster_entities)

                if len(cluster_entities) >= min_cluster_size:
                    # Get relationships within cluster
                    cluster_rels = []
                    for rel in self.relationships.values():
                        if (
                            rel.from_entity_id in cluster_entities
                            and rel.to_entity_id in cluster_entities
                        ):
                            cluster_rels.append(rel)

                    # Compute cohesion (average relationship strength)
                    cohesion = (
                        sum(r.strength_score for r in cluster_rels) / len(cluster_rels)
                        if cluster_rels
                        else 0.0
                    )

                    cluster = RelationshipCluster(
                        cluster_id=f"cluster_{len(clusters)}",
                        entities=list(cluster_entities),
                        relationships=cluster_rels,
                        cohesion_score=cohesion,
                        metadata={"size": len(cluster_entities), "edge_count": len(cluster_rels)},
                    )
                    clusters.append(cluster)

        # Sort clusters by size and cohesion
        clusters.sort(key=lambda c: (len(c.entities), c.cohesion_score), reverse=True)

        return clusters

    def analyze_network(self) -> RelationshipNetwork:
        """Perform comprehensive network analysis.

        Returns:
            Complete network analysis
        """
        # Get all unique entities
        entities = list(self.entity_relationships.keys())

        # Get all relationships
        relationships = list(self.relationships.values())

        # Detect clusters
        clusters = self.detect_relationship_clusters()

        # Compute centrality for all entities
        central_entities = [(entity, self.compute_entity_centrality(entity)) for entity in entities]
        central_entities.sort(key=lambda x: x[1], reverse=True)

        # Compute network density
        num_entities = len(entities)
        num_relationships = len(relationships)
        max_relationships = (num_entities * (num_entities - 1)) / 2
        density = num_relationships / max_relationships if max_relationships > 0 else 0.0

        return RelationshipNetwork(
            entities=entities,
            relationships=relationships,
            clusters=clusters,
            central_entities=central_entities[:20],  # Top 20
            network_density=density,
            metadata={
                "num_entities": num_entities,
                "num_relationships": num_relationships,
                "num_clusters": len(clusters),
                "avg_relationship_strength": (
                    sum(r.strength_score for r in relationships) / num_relationships
                    if num_relationships > 0
                    else 0.0
                ),
            },
        )

    def get_co_occurring_entities(
        self, entity_id: str, min_occurrences: int = 2
    ) -> list[tuple[str, int]]:
        """Get entities that frequently co-occur with the given entity.

        Args:
            entity_id: Entity ID
            min_occurrences: Minimum co-occurrence count

        Returns:
            List of (entity_id, co_occurrence_count) tuples
        """
        co_occurring = []

        for (e1, e2), count in self.co_occurrences.items():
            if count < min_occurrences:
                continue

            if e1 == entity_id:
                co_occurring.append((e2, count))
            elif e2 == entity_id:
                co_occurring.append((e1, count))

        # Sort by co-occurrence count (highest first)
        co_occurring.sort(key=lambda x: x[1], reverse=True)

        return co_occurring

    def get_relationship_statistics(self) -> dict[str, Any]:
        """Get statistics about relationships.

        Returns:
            Dictionary of statistics
        """
        relationships = list(self.relationships.values())

        stats = {
            "total_relationships": len(relationships),
            "total_entities": len(self.entity_relationships),
            "by_type": {},
            "by_strength": {level.value: 0 for level in RelationshipStrength},
            "avg_strength": 0.0,
            "avg_co_occurrence": 0.0,
        }

        # Count by type and strength
        for rel in relationships:
            rel_type = rel.relation_type.value
            stats["by_type"][rel_type] = stats["by_type"].get(rel_type, 0) + 1
            stats["by_strength"][rel.strength_level.value] += 1

        # Compute averages
        if relationships:
            stats["avg_strength"] = sum(r.strength_score for r in relationships) / len(
                relationships
            )
            stats["avg_co_occurrence"] = sum(r.co_occurrence_count for r in relationships) / len(
                relationships
            )

        return stats

    def export_for_visualization(self) -> dict[str, Any]:
        """Export network data in a format suitable for visualization.

        Returns:
            Dictionary with nodes and edges for visualization tools (e.g., D3.js, Cytoscape)
        """
        nodes = []
        edges = []

        # Create nodes
        entity_centrality = {
            entity: self.compute_entity_centrality(entity)
            for entity in self.entity_relationships.keys()
        }

        for entity_id, centrality in entity_centrality.items():
            nodes.append(
                {
                    "id": entity_id,
                    "centrality": centrality,
                    "relationship_count": len(self.entity_relationships[entity_id]),
                }
            )

        # Create edges
        for rel in self.relationships.values():
            edges.append(
                {
                    "source": rel.from_entity_id,
                    "target": rel.to_entity_id,
                    "relation_type": rel.relation_type.value,
                    "strength": rel.strength_score,
                    "strength_level": rel.strength_level.value,
                    "co_occurrence_count": rel.co_occurrence_count,
                }
            )

        return {"nodes": nodes, "edges": edges, "metadata": self.get_relationship_statistics()}
