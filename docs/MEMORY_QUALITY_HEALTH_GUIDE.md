# Memory Quality & Health Monitoring - Implementation Guide

## Overview

The Memory Quality & Health Monitoring system provides comprehensive assessment and analysis of memory store quality, including health scoring, duplicate detection, staleness tracking, and topic coverage analysis.

## Features Implemented

### 1. **Memory Health Scoring** ✅
Individual memory quality assessment with detailed metrics:

- **Overall Health Score** (0-100): Weighted combination of multiple factors
- **Component Scores**:
  - Completeness (25%): Text length, tags, metadata richness
  - Clarity (20%): Text quality, sentence structure, uniqueness
  - Freshness (20%): How recently updated (exponential decay)
  - Confidence (20%): Confidence level (0-1 scale)
  - Importance (15%): Importance weight (0-10 scale)

- **Issue Detection**:
  - `DUPLICATE`: Exact or near-exact duplicates
  - `NEAR_DUPLICATE`: Similar memories (70-85% similarity)
  - `STALE`: Not updated in > 90 days
  - `LOW_CONFIDENCE`: Confidence < 0.5
  - `LOW_IMPORTANCE`: Importance < 3.0
  - `INCOMPLETE`: Very short text (< 10 chars)
  - `CONTRADICTORY`: Conflicting information
  - `ORPHANED`: No tags or metadata

- **Automated Recommendations**: Context-specific suggestions for each memory

### 2. **Soft Duplicate Clustering** ✅
Advanced duplicate detection with similarity-based clustering:

- **Similarity Thresholds**:
  - Duplicates: ≥ 85% similarity
  - Near-duplicates: 70-85% similarity
  - Graph-based clustering algorithm

- **Cluster Actions**:
  - `merge`: High similarity duplicates that should be consolidated
  - `review`: Near-duplicates requiring human review
  - `keep_all`: Low similarity, keep separate

- **Canonical Memory Selection**: Identifies the best representative in each cluster

### 3. **Stale Memory Detection** ✅
Temporal tracking with configurable staleness thresholds:

- **Default Threshold**: 90 days since last update
- **Customizable per User**: Adjustable threshold
- **Staleness Impact on Health**: Exponential decay with 30-day half-life
- **Freshness Score**: 100 at day 0 → ~50 at 30 days → ~25 at 60 days

### 4. **Memory Coverage Analysis** ✅
Topic representation and diversity assessment:

- **Topic Extraction**: Automatic from tags or custom topic list
- **Coverage Metrics**:
  - Memory count per topic
  - Average quality per topic
  - Average recency (days since last update)
  - Top keywords extracted from content
  - Coverage score (0-100):
    - Count score (0-50 points): Based on # of memories
    - Quality score (0-30 points): Average confidence
    - Recency score (0-20 points): Recent updates

- **Insights**:
  - Well-represented topics
  - Under-represented topics
  - Stale topics needing refresh

### 5. **Memory Store Health Assessment** ✅
Overall health dashboard for entire memory store:

- **Health Status Levels**:
  - `EXCELLENT`: Score ≥ 90
  - `GOOD`: Score ≥ 75
  - `FAIR`: Score ≥ 60
  - `POOR`: Score ≥ 40
  - `CRITICAL`: Score < 40

- **Aggregate Metrics**:
  - Total memories
  - Healthy vs. problematic count
  - Stale memory count
  - Duplicate cluster count
  - Average memory quality
  - Issues breakdown by type

- **Health Score Calculation**:
  - Base: Average memory quality (50%)
  - Penalties:
    - Duplicate ratio (up to -20%)
    - Stale ratio (up to -20%)
  - Bonus: Topic diversity (+10%)

- **Automated Recommendations**:
  - High stale count warnings
  - Duplicate consolidation suggestions
  - Quality improvement advice
  - Topic diversity recommendations

## Architecture

### Core Module: `memory_quality.py`

```
src/hippocampai/pipeline/memory_quality.py
```

**Classes**:
- `MemoryHealthScore`: Individual memory assessment
- `DuplicateCluster`: Cluster of similar memories
- `TopicCoverage`: Topic representation analysis
- `MemoryStoreHealth`: Overall store assessment
- `MemoryQualityMonitor`: Main monitoring engine

### Integration: `memory_service.py`

**New Methods**:
```python
async def assess_memory_health(memory_id: str) -> MemoryHealthScore
async def detect_duplicates(user_id: str) -> list[DuplicateCluster]
async def analyze_topic_coverage(user_id: str) -> list[TopicCoverage]
async def assess_memory_store_health(user_id: str) -> MemoryStoreHealth
async def get_stale_memories(user_id: str) -> list[Memory]
```

## Usage Examples

### Python Library Usage

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Assess individual memory health
health = client.assess_memory_health("memory_123")
print(f"Health Score: {health.overall_score}/100")
print(f"Issues: {health.issues}")
print(f"Recommendations: {health.recommendations}")

# Detect duplicates
clusters = client.detect_duplicates("user_123")
for cluster in clusters:
    print(f"Cluster: {cluster.similarity_score:.2f} similarity")
    print(f"  Memories: {cluster.memory_ids}")
    print(f"  Action: {cluster.suggested_action}")

# Analyze topic coverage
coverage = client.analyze_topic_coverage("user_123")
for topic in coverage:
    print(f"{topic.topic}: {topic.memory_count} memories")
    print(f"  Quality: {topic.avg_quality:.1f}/100")
    print(f"  Coverage: {topic.coverage_score:.1f}/100")

# Get overall store health
store_health = client.assess_memory_store_health("user_123")
print(f"Status: {store_health.health_status}")
print(f"Score: {store_health.overall_health_score:.1f}/100")
print(f"Healthy: {store_health.healthy_memories}/{store_health.total_memories}")
print(f"Duplicates: {store_health.duplicate_clusters} clusters")
print("Recommendations:")
for rec in store_health.recommendations:
    print(f"  - {rec}")

# Get stale memories
stale = client.get_stale_memories("user_123", threshold_days=60)
print(f"Found {len(stale)} stale memories")
```

## Health Scoring Algorithm

### Individual Memory Score
```
Overall Score =
  0.25 × Completeness +
  0.20 × Clarity +
  0.20 × Freshness +
  0.20 × Confidence +
  0.15 × Importance
```

### Store Health Score
```
Overall Health =
  0.50 × Average Memory Quality -
  (min(duplicate_ratio × 100, 20)) -
  (min(stale_ratio × 100, 20)) +
  (min(topic_count / 10 × 10, 10))
```

### Freshness Decay
```
Freshness = 100 × 0.5^(days_since_update / 30)
```

## Configuration

### Quality Monitor Parameters

```python
MemoryQualityMonitor(
    stale_threshold_days=90,           # Days until memory is stale
    duplicate_threshold=0.85,           # Similarity for duplicates
    near_duplicate_threshold=0.70,      # Similarity for near-duplicates
    min_text_length=10,                 # Minimum acceptable text length
    min_confidence=0.5                  # Minimum acceptable confidence
)
```

## Performance Considerations

### Duplicate Detection
- **Complexity**: O(n²) for n memories
- **Optimization**: Uses precomputed embeddings when available
- **Recommendation**: Run periodically (daily/weekly) for large stores
- **Batching**: Process in chunks for > 1000 memories

### Store Health Assessment
- **Memory Limit**: 10,000 memories per assessment
- **Caching**: Results can be cached for 1-6 hours
- **Background Processing**: Can run asynchronously

## Best Practices

### 1. Regular Health Checks
```python
# Daily health check
health = await service.assess_memory_store_health(user_id)
if health.health_status in ["poor", "critical"]:
    # Trigger cleanup workflow
    await cleanup_low_quality_memories(user_id)
```

### 2. Proactive Duplicate Management
```python
# Weekly duplicate detection
clusters = await service.detect_duplicates(user_id)
for cluster in clusters:
    if cluster.suggested_action == "merge":
        # Auto-merge high-confidence duplicates
        await merge_memories(cluster.memory_ids, cluster.canonical_memory_id)
```

### 3. Stale Memory Maintenance
```python
# Monthly stale memory review
stale = await service.get_stale_memories(user_id, threshold_days=180)
for memory in stale:
    # Flag for review or automatic refresh
    await flag_for_review(memory.id, reason="stale")
```

### 4. Topic Coverage Monitoring
```python
# Track coverage trends
coverage = await service.analyze_topic_coverage(user_id)
under_represented = [t for t in coverage if t.coverage_score < 40]
# Prompt user to add memories for these topics
```

## Metrics & Monitoring

### Key Performance Indicators (KPIs)

1. **Overall Health Trend**
   - Target: > 75 (GOOD status)
   - Alert: < 60 (FAIR or below)

2. **Duplicate Ratio**
   - Target: < 5% of memories in clusters
   - Alert: > 10%

3. **Stale Ratio**
   - Target: < 20% older than threshold
   - Alert: > 30%

4. **Average Memory Quality**
   - Target: > 70/100
   - Alert: < 60/100

5. **Topic Diversity**
   - Target: > 5 distinct topics
   - Alert: < 3 topics

## Troubleshooting

### Low Health Scores

**Problem**: Store health consistently < 60

**Solutions**:
1. Review and clean up low-confidence memories
2. Add tags and metadata to orphaned memories
3. Consolidate duplicate clusters
4. Refresh stale memories with current information
5. Encourage users to update old memories

### High Duplicate Count

**Problem**: Many duplicate clusters detected

**Solutions**:
1. Review deduplication threshold (may be too low)
2. Implement automatic merging for high-similarity duplicates
3. Add user confirmation workflow for near-duplicates
4. Improve memory creation to check for existing similar content

### Poor Topic Coverage

**Problem**: Few topics or uneven distribution

**Solutions**:
1. Encourage diverse memory creation
2. Suggest topics with low representation
3. Add topic tags to existing memories
4. Implement guided memory creation

## Future Enhancements

### Planned Features
- [ ] Contradiction detection using LLM analysis
- [ ] Memory quality trends over time
- [ ] Automated cleanup recommendations
- [ ] User-specific quality thresholds
- [ ] Integration with notification system
- [ ] Quality-based search ranking
- [ ] Health score history tracking
- [ ] Comparative analysis across users

## API Reference

### Service Methods

#### `assess_memory_health(memory_id: str) -> MemoryHealthScore`
Assess health of a single memory.

**Returns**:
- `overall_score`: Overall quality (0-100)
- `completeness_score`: Completeness metric
- `clarity_score`: Clarity metric
- `freshness_score`: Freshness metric
- `confidence_score`: Confidence metric
- `importance_score`: Importance metric
- `issues`: List of quality issues
- `recommendations`: Actionable suggestions
- `is_stale`: Boolean staleness flag
- `duplicate_candidates`: List of potential duplicate IDs

#### `detect_duplicates(user_id: str, threshold: float = None) -> list[DuplicateCluster]`
Detect duplicate memory clusters.

**Parameters**:
- `user_id`: User identifier
- `threshold`: Optional custom similarity threshold

**Returns**: List of DuplicateCluster objects with:
- `cluster_id`: Unique cluster identifier
- `memory_ids`: List of memory IDs in cluster
- `similarity_score`: Average similarity
- `suggested_action`: "merge", "review", or "keep_all"
- `canonical_memory_id`: Best representative

#### `analyze_topic_coverage(user_id: str, topics: list[str] = None) -> list[TopicCoverage]`
Analyze memory coverage across topics.

**Parameters**:
- `user_id`: User identifier
- `topics`: Optional predefined topics (auto-extracts from tags if not provided)

**Returns**: List of TopicCoverage objects with:
- `topic`: Topic name
- `memory_count`: Number of memories
- `avg_quality`: Average quality score
- `avg_recency`: Average days since update
- `keywords`: Top 5 keywords
- `coverage_score`: Coverage quality (0-100)

#### `assess_memory_store_health(user_id: str) -> MemoryStoreHealth`
Assess overall health of memory store.

**Returns**: MemoryStoreHealth object with:
- `overall_health_score`: Overall score (0-100)
- `health_status`: Status level (excellent/good/fair/poor/critical)
- `total_memories`: Total count
- `healthy_memories`: Count with score ≥ 70
- `problematic_memories`: Count with score < 70
- `stale_memories`: Count older than threshold
- `duplicate_clusters`: Number of duplicate clusters
- `avg_memory_quality`: Average quality score
- `issues_by_type`: Dict of issue counts by type
- `topic_coverage`: List of TopicCoverage objects
- `recommendations`: Actionable suggestions

#### `get_stale_memories(user_id: str, threshold_days: int = None) -> list[Memory]`
Get stale memories.

**Parameters**:
- `user_id`: User identifier
- `threshold_days`: Custom threshold (default: 90)

**Returns**: List of Memory objects older than threshold

## Summary

The Memory Quality & Health Monitoring system provides:

✅ **Comprehensive Health Assessment** - Multi-factor scoring with detailed metrics
✅ **Smart Duplicate Detection** - Graph-based clustering with action recommendations
✅ **Staleness Tracking** - Temporal decay with configurable thresholds
✅ **Topic Coverage Analysis** - Identify representation gaps and strengths
✅ **Store-Wide Health Dashboard** - Aggregate metrics and recommendations
✅ **Automated Issue Detection** - Proactive identification of quality problems
✅ **Actionable Recommendations** - Context-specific suggestions for improvement

This system enables proactive memory quality management, helps maintain a healthy knowledge base, and provides insights for continuous improvement.
