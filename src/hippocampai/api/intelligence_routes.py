"""API routes for Advanced Intelligence features.

This module provides REST API endpoints for:
- Fact extraction with quality scoring
- Entity recognition and profiling
- Relationship mapping and analysis
- Semantic clustering
- Temporal analytics
- Summarization
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from hippocampai.config import get_config
from hippocampai.models.memory import Memory
from hippocampai.pipeline.entity_recognition import EntityRecognizer, EntityType, RelationType
from hippocampai.pipeline.fact_extraction import FactExtractionPipeline
from hippocampai.pipeline.relationship_mapping import RelationshipMapper
from hippocampai.pipeline.semantic_clustering import SemanticCategorizer
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/intelligence", tags=["intelligence"])

# Initialize services
config = get_config()
fact_extractor = FactExtractionPipeline()
entity_recognizer = EntityRecognizer()
relationship_mapper = RelationshipMapper()
semantic_categorizer = SemanticCategorizer()
temporal_analytics = TemporalAnalytics()


# ==================== Request/Response Models ====================


class FactExtractionRequest(BaseModel):
    """Request for fact extraction."""

    text: str = Field(..., description="Text to extract facts from")
    source: str = Field(default="api", description="Source identifier")
    user_id: Optional[str] = Field(None, description="User ID for context")
    with_quality: bool = Field(default=True, description="Include quality metrics")


class FactExtractionResponse(BaseModel):
    """Response for fact extraction."""

    facts: list[dict[str, Any]]
    count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction."""

    text: str = Field(..., description="Text to extract entities from")
    context: Optional[dict[str, Any]] = Field(None, description="Context metadata")


class EntityExtractionResponse(BaseModel):
    """Response for entity extraction."""

    entities: list[dict[str, Any]]
    count: int
    statistics: dict[str, Any] = Field(default_factory=dict)


class EntitySearchRequest(BaseModel):
    """Request for entity search."""

    query: str = Field(..., description="Search query")
    entity_type: Optional[str] = Field(None, description="Filter by entity type")
    min_mentions: int = Field(default=1, description="Minimum mention count")


class RelationshipAnalysisRequest(BaseModel):
    """Request for relationship analysis."""

    text: str = Field(..., description="Text to extract relationships from")
    entity_ids: Optional[list[str]] = Field(None, description="Specific entities to analyze")


class RelationshipAnalysisResponse(BaseModel):
    """Response for relationship analysis."""

    relationships: list[dict[str, Any]]
    network: dict[str, Any]
    visualization_data: dict[str, Any]


class SemanticClusteringRequest(BaseModel):
    """Request for semantic clustering."""

    memories: list[dict[str, Any]] = Field(..., description="Memories to cluster")
    max_clusters: int = Field(default=10, description="Maximum number of clusters")
    hierarchical: bool = Field(default=False, description="Use hierarchical clustering")


class SemanticClusteringResponse(BaseModel):
    """Response for semantic clustering."""

    clusters: list[dict[str, Any]]
    count: int
    quality_metrics: dict[str, Any] = Field(default_factory=dict)


class TemporalAnalyticsRequest(BaseModel):
    """Request for temporal analytics."""

    memories: list[dict[str, Any]] = Field(..., description="Memories to analyze")
    analysis_type: str = Field(..., description="Type: peak_activity, patterns, trends, clusters")
    time_window_days: int = Field(default=30, description="Analysis time window")
    timezone_offset: int = Field(default=0, description="Timezone offset in hours")


class TemporalAnalyticsResponse(BaseModel):
    """Response for temporal analytics."""

    analysis: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== Fact Extraction Endpoints ====================


@router.post("/facts:extract", response_model=FactExtractionResponse)
async def extract_facts(request: FactExtractionRequest):
    """Extract structured facts from text with confidence scores.

    This endpoint performs intelligent fact extraction using both pattern matching
    and LLM-based analysis. Each fact includes quality metrics and confidence scores.
    """
    try:
        if request.with_quality:
            facts = fact_extractor.extract_facts_with_quality(
                text=request.text, source=request.source, user_id=request.user_id
            )
        else:
            facts = fact_extractor.extract_facts(
                text=request.text, source=request.source, user_id=request.user_id
            )

        return FactExtractionResponse(
            facts=[fact.model_dump() for fact in facts],
            count=len(facts),
            metadata={"source": request.source, "with_quality": request.with_quality},
        )

    except Exception as e:
        logger.error(f"Fact extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact extraction failed: {str(e)}",
        )


# ==================== Entity Recognition Endpoints ====================


@router.post("/entities:extract", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """Extract and recognize named entities from text.

    Supports recognition of people, organizations, locations, dates, skills,
    products, emails, URLs, frameworks, certifications, and more.
    """
    try:
        entities = entity_recognizer.extract_entities(text=request.text, context=request.context)

        # Get statistics
        stats = entity_recognizer.get_entity_statistics()

        return EntityExtractionResponse(
            entities=[entity.model_dump() for entity in entities],
            count=len(entities),
            statistics=stats,
        )

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}",
        )


@router.post("/entities:search", response_model=dict[str, Any])
async def search_entities(request: EntitySearchRequest):
    """Search for entities by query string.

    Searches across entity canonical names and aliases. Supports filtering
    by entity type and minimum mention count.
    """
    try:
        entity_type = EntityType(request.entity_type) if request.entity_type else None

        results = entity_recognizer.search_entities(
            query=request.query, entity_type=entity_type, min_mentions=request.min_mentions
        )

        return {
            "entities": [profile.model_dump() for profile in results],
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity search failed: {str(e)}",
        )


@router.get("/entities/{entity_id}", response_model=dict[str, Any])
async def get_entity_profile(entity_id: str):
    """Get complete profile for an entity.

    Returns the entity's canonical name, aliases, relationships, mentions,
    and timeline of activity.
    """
    try:
        profile = entity_recognizer.get_entity_profile(entity_id)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Entity {entity_id} not found"
            )

        return {
            "entity": profile.model_dump(),
            "timeline": entity_recognizer.get_entity_timeline(entity_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity profile: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==================== Relationship Mapping Endpoints ====================


@router.post("/relationships:analyze", response_model=RelationshipAnalysisResponse)
async def analyze_relationships(request: RelationshipAnalysisRequest):
    """Analyze relationships between entities.

    Extracts relationships from text and computes relationship strength scores,
    network structure, and visualization data.
    """
    try:
        # Extract entities first
        entities = entity_recognizer.extract_entities(text=request.text)

        # Extract relationships
        relationships = entity_recognizer.extract_relationships(
            text=request.text, entities=entities
        )

        # Add to relationship mapper
        for rel in relationships:
            relationship_mapper.add_relationship_from_entity_relationship(rel)

        # Analyze network
        network = relationship_mapper.analyze_network()

        # Get visualization data
        viz_data = relationship_mapper.export_for_visualization()

        return RelationshipAnalysisResponse(
            relationships=[rel.model_dump() for rel in relationships],
            network=network.model_dump(),
            visualization_data=viz_data,
        )

    except Exception as e:
        logger.error(f"Relationship analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Relationship analysis failed: {str(e)}",
        )


@router.get("/relationships/{entity_id}", response_model=dict[str, Any])
async def get_entity_relationships(
    entity_id: str,
    relation_type: Optional[str] = None,
    min_strength: float = 0.0,
):
    """Get all relationships for an entity.

    Returns relationships filtered by type and minimum strength, sorted by
    relationship strength score.
    """
    try:
        rel_type = RelationType(relation_type) if relation_type else None

        relationships = relationship_mapper.get_entity_relationships(
            entity_id=entity_id, relation_type=rel_type, min_strength=min_strength
        )

        # Get co-occurring entities
        co_occurring = relationship_mapper.get_co_occurring_entities(entity_id)

        return {
            "entity_id": entity_id,
            "relationships": [rel.model_dump() for rel in relationships],
            "count": len(relationships),
            "co_occurring_entities": [
                {"entity_id": eid, "count": count} for eid, count in co_occurring
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get entity relationships: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/relationships:network", response_model=dict[str, Any])
async def get_relationship_network():
    """Get complete relationship network analysis.

    Returns comprehensive network analysis including entities, relationships,
    clusters, central entities, and network density metrics.
    """
    try:
        network = relationship_mapper.analyze_network()
        viz_data = relationship_mapper.export_for_visualization()
        stats = relationship_mapper.get_relationship_statistics()

        return {
            "network": network.model_dump(),
            "visualization": viz_data,
            "statistics": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get relationship network: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==================== Semantic Clustering Endpoints ====================


@router.post("/clustering:analyze", response_model=SemanticClusteringResponse)
async def cluster_memories(request: SemanticClusteringRequest):
    """Cluster memories by semantic similarity.

    Groups memories into topical clusters using either standard or hierarchical
    clustering. Computes quality metrics for each cluster.
    """
    try:
        # Convert dict memories to Memory objects
        memories = [Memory(**mem_dict) for mem_dict in request.memories]

        if request.hierarchical:
            result = semantic_categorizer.hierarchical_cluster_memories(
                memories, min_cluster_size=2
            )
            clusters = result["clusters"]
            hierarchy = result.get("hierarchy", [])

            # Compute quality metrics for each cluster
            quality_metrics = {}
            for i, cluster_data in enumerate(clusters):
                cluster_memories = cluster_data["memories"]
                # Create MemoryCluster object
                from hippocampai.pipeline.semantic_clustering import MemoryCluster

                cluster = MemoryCluster(topic=cluster_data["topic"], memories=cluster_memories)
                metrics = semantic_categorizer.compute_cluster_quality_metrics(cluster)
                quality_metrics[f"cluster_{i}"] = metrics

            return SemanticClusteringResponse(
                clusters=clusters,
                count=len(clusters),
                quality_metrics={
                    "per_cluster": quality_metrics,
                    "hierarchy": [{"size": len(h[0]), "similarity": h[1]} for h in hierarchy],
                },
            )
        cluster_objects = semantic_categorizer.cluster_memories(
            memories, max_clusters=request.max_clusters
        )

        # Convert to dicts and compute metrics
        clusters = []
        quality_metrics = {}

        for i, cluster in enumerate(cluster_objects):
            cluster_dict = {
                "topic": cluster.topic,
                "memories": [mem.model_dump() for mem in cluster.memories],
                "tags": cluster.tags,
                "size": len(cluster.memories),
            }
            clusters.append(cluster_dict)

            # Compute quality metrics
            metrics = semantic_categorizer.compute_cluster_quality_metrics(cluster)
            quality_metrics[f"cluster_{i}"] = metrics

        return SemanticClusteringResponse(
            clusters=clusters,
            count=len(clusters),
            quality_metrics={"per_cluster": quality_metrics},
        )

    except Exception as e:
        logger.error(f"Semantic clustering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic clustering failed: {str(e)}",
        )


@router.post("/clustering:optimize", response_model=dict[str, Any])
async def optimize_cluster_count(request: dict[str, Any]):
    """Determine optimal number of clusters using elbow method.

    Analyzes different cluster counts to find the optimal balance between
    cluster cohesion and cluster count.
    """
    try:
        memories = [Memory(**mem_dict) for mem_dict in request["memories"]]
        min_k = request.get("min_k", 2)
        max_k = request.get("max_k", 15)

        optimal_k = semantic_categorizer.optimize_cluster_count(memories, min_k=min_k, max_k=max_k)

        return {
            "optimal_cluster_count": optimal_k,
            "min_k": min_k,
            "max_k": max_k,
        }

    except Exception as e:
        logger.error(f"Cluster optimization failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==================== Temporal Analytics Endpoints ====================


@router.post("/temporal:analyze", response_model=TemporalAnalyticsResponse)
async def analyze_temporal_patterns(request: TemporalAnalyticsRequest):
    """Perform temporal analysis on memories.

    Supports multiple analysis types:
    - peak_activity: Find peak activity times
    - patterns: Detect recurring temporal patterns
    - trends: Analyze trends over time
    - clusters: Cluster memories by temporal proximity
    """
    try:
        memories = [Memory(**mem_dict) for mem_dict in request.memories]

        if request.analysis_type == "peak_activity":
            analysis = temporal_analytics.analyze_peak_activity(
                memories, timezone_offset=request.timezone_offset
            )
            return TemporalAnalyticsResponse(
                analysis=analysis.model_dump(),
                metadata={"analysis_type": "peak_activity"},
            )

        if request.analysis_type == "patterns":
            patterns = temporal_analytics.detect_temporal_patterns(memories)
            return TemporalAnalyticsResponse(
                analysis={
                    "patterns": [p.model_dump() for p in patterns],
                    "count": len(patterns),
                },
                metadata={"analysis_type": "patterns"},
            )

        if request.analysis_type == "trends":
            activity_trend = temporal_analytics.analyze_trends(
                memories, time_window_days=request.time_window_days, metric="activity"
            )
            importance_trend = temporal_analytics.analyze_trends(
                memories, time_window_days=request.time_window_days, metric="importance"
            )

            return TemporalAnalyticsResponse(
                analysis={
                    "activity_trend": activity_trend.model_dump(),
                    "importance_trend": importance_trend.model_dump(),
                },
                metadata={"analysis_type": "trends", "time_window_days": request.time_window_days},
            )

        if request.analysis_type == "clusters":
            clusters = temporal_analytics.cluster_by_time(memories)
            return TemporalAnalyticsResponse(
                analysis={
                    "clusters": [c.model_dump() for c in clusters],
                    "count": len(clusters),
                },
                metadata={"analysis_type": "clusters"},
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown analysis type: {request.analysis_type}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal analysis failed: {str(e)}",
        )


@router.post("/temporal:peak-times", response_model=dict[str, Any])
async def get_peak_times(request: dict[str, Any]):
    """Get peak activity times for memories.

    Returns detailed breakdown of activity by hour, day of week, and time period.
    """
    try:
        memories = [Memory(**mem_dict) for mem_dict in request["memories"]]
        timezone_offset = request.get("timezone_offset", 0)

        peak_analysis = temporal_analytics.analyze_peak_activity(
            memories, timezone_offset=timezone_offset
        )

        return {
            "peak_analysis": peak_analysis.model_dump(),
            "metadata": {"total_memories": len(memories), "timezone_offset": timezone_offset},
        }

    except Exception as e:
        logger.error(f"Peak time analysis failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==================== Health Check ====================


@router.get("/health", response_model=dict[str, str])
async def health_check():
    """Health check endpoint for intelligence services."""
    return {
        "status": "healthy",
        "services": "Advanced Intelligence APIs",
        "version": "0.2.0",
    }
