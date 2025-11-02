# Advanced Intelligence Features Documentation

This document provides comprehensive documentation for the new Advanced Intelligence APIs and Temporal Intelligence features added to HippocampAI.

## Table of Contents

1. [Overview](#overview)
2. [Fact Extraction Service](#fact-extraction-service)
3. [Entity Recognition API](#entity-recognition-api)
4. [Relationship Mapping](#relationship-mapping)
5. [Semantic Clustering](#semantic-clustering)
6. [Temporal Intelligence](#temporal-intelligence)
7. [API Endpoints](#api-endpoints)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)

---

## Overview

HippocampAI now includes advanced intelligence capabilities that go beyond simple memory storage and retrieval:

### Advanced Intelligence APIs
- **Fact Extraction**: Extract structured facts from text with confidence scores and quality metrics
- **Entity Recognition**: Identify and track entities (people, organizations, skills, etc.) with canonical naming
- **Relationship Mapping**: Analyze relationships between entities with strength scoring
- **Semantic Clustering**: Group memories by semantic similarity with hierarchical analysis
- **Summarization**: Generate summaries in multiple formats (bullet points, narratives, action items)
- **Insight Generation**: Detect patterns, behavior changes, and preference drifts

### Temporal Intelligence
- **Time-based Queries**: Search memories using natural language time expressions
- **Event Timelines**: Build chronological narratives of events
- **Memory Scheduling**: Schedule future reminders with recurrence support
- **Temporal Analytics**: Analyze peak activity times, detect patterns, and forecast trends

---

## Fact Extraction Service

### Features

The Fact Extraction Service extracts structured, verifiable facts from unstructured text with enhanced quality scoring.

#### Supported Fact Categories
- Employment
- Occupation
- Location
- Education
- Relationship
- Preference
- Skill
- Experience
- Contact
- Event
- Goal
- Habit
- Opinion
- Attribute
- Possession

#### Quality Metrics

Each extracted fact includes quality metrics:

```python
{
    "specificity": 0.85,      # How specific the fact is (0.0-1.0)
    "verifiability": 0.90,    # How verifiable the fact is (0.0-1.0)
    "completeness": 0.88,     # How complete the fact is (0.0-1.0)
    "clarity": 0.92,          # How clear the fact is (0.0-1.0)
    "relevance": 0.87,        # How relevant the fact is (0.0-1.0)
    "overall_quality": 0.88   # Overall quality score (0.0-1.0)
}
```

### Usage

#### Python API

```python
from hippocampai.pipeline.fact_extraction import FactExtractionPipeline

extractor = FactExtractionPipeline()

# Extract facts with quality scoring
facts = extractor.extract_facts_with_quality(
    text="I work at Google as a Senior Engineer in Mountain View.",
    source="conversation",
    user_id="user123"
)

for fact in facts:
    print(f"Fact: {fact.fact}")
    print(f"Category: {fact.category}")
    print(f"Confidence: {fact.confidence}")
    print(f"Quality Score: {fact.quality_score}")
    print(f"Entities: {fact.entities}")
```

#### REST API

```bash
POST /v1/intelligence/facts:extract
Content-Type: application/json

{
    "text": "I work at Google as a Senior Engineer in Mountain View.",
    "source": "api",
    "user_id": "user123",
    "with_quality": true
}
```

---

## Entity Recognition API

### Features

The Entity Recognition API identifies and tracks entities across memories with advanced features:

#### Supported Entity Types

**Standard Types:**
- Person
- Organization
- Location
- Date/Time
- Money
- Product
- Event

**Extended Types:**
- Email
- Phone
- URL
- Language (programming languages)
- Framework (software frameworks)
- Tool (dev tools)
- Industry
- Degree (academic degrees)
- Certification

#### Entity Profiles

Each entity maintains a profile with:
- Canonical name (normalized)
- Aliases (all variations seen)
- Relationships to other entities
- Mention count and timestamps
- Timeline of appearances
- Attributes and metadata

### Usage

#### Python API

```python
from hippocampai.pipeline.entity_recognition import EntityRecognizer

recognizer = EntityRecognizer()

# Extract entities
entities = recognizer.extract_entities(
    text="John Smith works at Microsoft in Seattle.",
    context={"source": "conversation"}
)

for entity in entities:
    print(f"Entity: {entity.text}")
    print(f"Type: {entity.type}")
    print(f"Canonical: {entity.canonical_name}")
    print(f"Confidence: {entity.confidence}")

# Get entity profile
profile = recognizer.get_entity_profile("person_john_smith")
print(f"Canonical Name: {profile.canonical_name}")
print(f"Aliases: {profile.aliases}")
print(f"Mentions: {profile.mention_count}")
print(f"Relationships: {len(profile.relationships)}")

# Search entities
results = recognizer.search_entities(
    query="john",
    entity_type=EntityType.PERSON,
    min_mentions=2
)
```

#### REST API

```bash
POST /v1/intelligence/entities:extract
Content-Type: application/json

{
    "text": "John Smith works at Microsoft in Seattle.",
    "context": {"source": "conversation"}
}
```

---

## Relationship Mapping

### Features

The Relationship Mapping system analyzes connections between entities with:

#### Relationship Types
- WORKS_AT
- LOCATED_IN
- FOUNDED_BY
- MANAGES
- KNOWS
- STUDIED_AT
- PART_OF
- SIMILAR_TO
- RELATED_TO

#### Strength Scoring

Relationships are scored based on multiple factors:

```python
{
    "strength_score": 0.85,          # Overall strength (0.0-1.0)
    "strength_level": "very_strong",  # Categorical level
    "confidence": 0.90,               # Extraction confidence
    "co_occurrence_count": 15,        # Times entities appeared together
    "contexts": ["...", "..."]        # Contexts where relationship was found
}
```

#### Network Analysis

- **Centrality**: Identify most connected entities
- **Clustering**: Detect groups of strongly related entities
- **Path Finding**: Find shortest paths between entities
- **Density**: Measure network connectivity

### Usage

#### Python API

```python
from hippocampai.pipeline.relationship_mapping import RelationshipMapper

mapper = RelationshipMapper()

# Add relationship
mapper.add_relationship(
    from_entity_id="person_alice",
    to_entity_id="organization_microsoft",
    relation_type=RelationType.WORKS_AT,
    confidence=0.95,
    context="Alice works at Microsoft"
)

# Get entity relationships
rels = mapper.get_entity_relationships(
    entity_id="person_alice",
    min_strength=0.5
)

# Analyze network
network = mapper.analyze_network()
print(f"Network Density: {network.network_density}")
print(f"Central Entities: {network.central_entities[:5]}")

# Find path between entities
path = mapper.find_relationship_path(
    from_entity="person_alice",
    to_entity="person_bob",
    max_depth=3
)

# Export for visualization
viz_data = mapper.export_for_visualization()
# Use with D3.js, Cytoscape, etc.
```

#### REST API

```bash
POST /v1/intelligence/relationships:analyze
Content-Type: application/json

{
    "text": "Alice works at Microsoft in Seattle.",
    "entity_ids": ["person_alice", "organization_microsoft"]
}
```

---

## Semantic Clustering

### Features

Semantic Clustering groups memories by topic and similarity with:

#### Clustering Methods
- **Standard Clustering**: Topic-based clustering
- **Hierarchical Clustering**: Multi-level cluster hierarchy
- **Optimal K Selection**: Automatic cluster count optimization

#### Quality Metrics

```python
{
    "cohesion": 0.85,           # Intra-cluster similarity
    "diversity": 0.70,          # Type and tag diversity
    "temporal_density": 0.65,   # Time clustering
    "size": 12,                 # Number of memories
    "tag_count": 8              # Unique tags
}
```

### Usage

#### Python API

```python
from hippocampai.pipeline.semantic_clustering import SemanticCategorizer

categorizer = SemanticCategorizer()

# Standard clustering
clusters = categorizer.cluster_memories(
    memories=memories,
    max_clusters=10
)

for cluster in clusters:
    print(f"Topic: {cluster.topic}")
    print(f"Size: {len(cluster.memories)}")
    print(f"Tags: {cluster.tags}")

# Hierarchical clustering
result = categorizer.hierarchical_cluster_memories(
    memories=memories,
    min_cluster_size=2
)

# Compute quality metrics
metrics = categorizer.compute_cluster_quality_metrics(cluster)

# Optimize cluster count
optimal_k = categorizer.optimize_cluster_count(
    memories=memories,
    min_k=2,
    max_k=15
)
```

#### REST API

```bash
POST /v1/intelligence/clustering:analyze
Content-Type: application/json

{
    "memories": [...],
    "max_clusters": 10,
    "hierarchical": true
}
```

---

## Temporal Intelligence

### Features

#### 1. Peak Activity Analysis

Identify when users are most active:

```python
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

analytics = TemporalAnalytics()

peak_analysis = analytics.analyze_peak_activity(
    memories=memories,
    timezone_offset=-8  # PST
)

print(f"Peak Hour: {peak_analysis.peak_hour}")
print(f"Peak Day: {peak_analysis.peak_day}")
print(f"Peak Period: {peak_analysis.peak_time_period}")
```

#### 2. Pattern Detection

Detect recurring temporal patterns:

```python
patterns = analytics.detect_temporal_patterns(
    memories=memories,
    min_occurrences=3
)

for pattern in patterns:
    print(f"Pattern: {pattern.description}")
    print(f"Type: {pattern.pattern_type}")  # daily, weekly, interval
    print(f"Frequency: {pattern.frequency}")
    print(f"Regularity: {pattern.regularity_score}")
    print(f"Next Predicted: {pattern.next_predicted}")
```

#### 3. Trend Analysis

Analyze trends over time:

```python
# Activity trend
activity_trend = analytics.analyze_trends(
    memories=memories,
    time_window_days=30,
    metric="activity"
)

print(f"Direction: {activity_trend.direction}")  # increasing/decreasing/stable
print(f"Strength: {activity_trend.strength}")
print(f"Change Rate: {activity_trend.change_rate}")
print(f"Forecast: {activity_trend.forecast}")

# Importance trend
importance_trend = analytics.analyze_trends(
    memories=memories,
    time_window_days=30,
    metric="importance"
)
```

#### 4. Temporal Clustering

Group memories by temporal proximity:

```python
clusters = analytics.cluster_by_time(
    memories=memories,
    max_gap_hours=24
)

for cluster in clusters:
    print(f"Duration: {cluster.duration_hours} hours")
    print(f"Density: {cluster.density} memories/hour")
    print(f"Time Range: {cluster.start_time} to {cluster.end_time}")
```

### Usage

#### REST API

```bash
POST /v1/intelligence/temporal:analyze
Content-Type: application/json

{
    "memories": [...],
    "analysis_type": "peak_activity",  # or "patterns", "trends", "clusters"
    "time_window_days": 30,
    "timezone_offset": -8
}
```

---

## API Endpoints

### Fact Extraction

- `POST /v1/intelligence/facts:extract` - Extract facts from text

### Entity Recognition

- `POST /v1/intelligence/entities:extract` - Extract entities
- `POST /v1/intelligence/entities:search` - Search entities
- `GET /v1/intelligence/entities/{entity_id}` - Get entity profile

### Relationship Mapping

- `POST /v1/intelligence/relationships:analyze` - Analyze relationships
- `GET /v1/intelligence/relationships/{entity_id}` - Get entity relationships
- `GET /v1/intelligence/relationships:network` - Get network analysis

### Semantic Clustering

- `POST /v1/intelligence/clustering:analyze` - Cluster memories
- `POST /v1/intelligence/clustering:optimize` - Optimize cluster count

### Temporal Analytics

- `POST /v1/intelligence/temporal:analyze` - General temporal analysis
- `POST /v1/intelligence/temporal:peak-times` - Peak activity analysis

### Health Check

- `GET /v1/intelligence/health` - Service health check

---

## Usage Examples

### Complete Workflow Example

```python
from hippocampai.client import MemoryClient
from hippocampai.pipeline.fact_extraction import FactExtractionPipeline
from hippocampai.pipeline.entity_recognition import EntityRecognizer
from hippocampai.pipeline.relationship_mapping import RelationshipMapper

# Initialize
client = MemoryClient()
fact_extractor = FactExtractionPipeline()
entity_recognizer = EntityRecognizer()
relationship_mapper = RelationshipMapper()

# Store conversation
conversation = """
User: I work at Google as a Senior Engineer.
User: My colleague John Smith and I are working on a new AI project.
User: We're based in Mountain View, California.
"""

# Extract and store memories
memories = client.extract_from_conversation(conversation, user_id="user123")

# Extract facts
for memory in memories:
    facts = fact_extractor.extract_facts_with_quality(
        text=memory.text,
        source="conversation",
        user_id="user123"
    )
    for fact in facts:
        print(f"Fact: {fact.fact} (Quality: {fact.quality_score:.2f})")

# Extract entities and relationships
for memory in memories:
    entities = entity_recognizer.extract_entities(memory.text)
    relationships = entity_recognizer.extract_relationships(memory.text, entities)

    for rel in relationships:
        relationship_mapper.add_relationship_from_entity_relationship(rel)

# Analyze network
network = relationship_mapper.analyze_network()
print(f"Network has {len(network.entities)} entities and {len(network.relationships)} relationships")

# Get user's memories and analyze temporally
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

all_memories = client.recall(query="", user_id="user123", k=100)
analytics = TemporalAnalytics()

peak_analysis = analytics.analyze_peak_activity(all_memories.memories)
print(f"User is most active at {peak_analysis.peak_hour}:00 on {peak_analysis.peak_day.value}s")

patterns = analytics.detect_temporal_patterns(all_memories.memories)
for pattern in patterns:
    print(f"Detected pattern: {pattern.description}")
```

---

## Best Practices

### 1. Fact Extraction

- **Use Quality Scores**: Filter facts by `quality_score >= 0.7` for high-quality data
- **Verify High-Value Facts**: Facts about employment, education require higher confidence
- **Temporal Context**: Pay attention to temporal information for time-sensitive facts

### 2. Entity Recognition

- **Canonical Names**: Always use canonical names for entity matching
- **Entity Merging**: Regularly check for similar entities and merge when appropriate
- **Context Metadata**: Provide rich context to improve extraction accuracy

### 3. Relationship Mapping

- **Strength Thresholds**: Use `min_strength >= 0.5` for significant relationships
- **Regular Analysis**: Periodically analyze the network to detect new clusters
- **Co-occurrence Tracking**: Monitor co-occurrence counts to identify strong associations

### 4. Semantic Clustering

- **Optimize K**: Use `optimize_cluster_count()` to find the right number of clusters
- **Quality Metrics**: Monitor cohesion scores; aim for `cohesion >= 0.6`
- **Hierarchical for Large Sets**: Use hierarchical clustering for > 50 memories

### 5. Temporal Analytics

- **Timezone Awareness**: Always provide correct timezone offsets
- **Time Windows**: Use appropriate time windows (7 days for patterns, 30 days for trends)
- **Pattern Confidence**: Only trust patterns with `confidence >= 0.7`

---

## Performance Considerations

### Scalability

- **Batch Processing**: Process memories in batches of 50-100 for optimal performance
- **Caching**: Entity profiles and relationship networks are cached in memory
- **Async Operations**: Use async APIs for large-scale processing

### Resource Usage

- **Memory**: Entity and relationship storage scales with O(n) where n = unique entities
- **Computation**: Hierarchical clustering is O(n²), use for < 500 memories
- **Network Analysis**: Full network analysis is expensive; run periodically, not per-request

---

## Troubleshooting

### Common Issues

**Issue**: Low fact extraction quality scores
- **Solution**: Ensure input text is well-formed and specific; avoid ambiguous statements

**Issue**: Entity duplicates despite canonical naming
- **Solution**: Use `merge_entities()` to consolidate duplicates; check similarity threshold

**Issue**: No temporal patterns detected
- **Solution**: Ensure sufficient data (min 10-20 memories); check time spans cover patterns

**Issue**: Poor clustering cohesion
- **Solution**: Reduce cluster count; ensure memories have semantic diversity

---

## Future Enhancements

Planned features for upcoming releases:

1. **LLM Integration**: Optional LLM-powered extraction for improved accuracy
2. **Knowledge Base Linking**: Link entities to external knowledge bases
3. **Anomaly Detection**: Identify unusual patterns and outliers
4. **Forecasting**: Predict future activity patterns
5. **Multi-Language Support**: Entity extraction in multiple languages
6. **Advanced Visualization**: Interactive network visualization tools

---

## Contributing

To contribute new intelligence features:

1. Add pipeline module in `src/hippocampai/pipeline/`
2. Add API routes in `src/hippocampai/api/intelligence_routes.py`
3. Update client methods in `src/hippocampai/client.py`
4. Add comprehensive tests
5. Update documentation

---

## License

Copyright © 2025 HippocampAI. All rights reserved.

See LICENSE file for complete license terms.
