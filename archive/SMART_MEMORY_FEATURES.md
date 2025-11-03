# Smart Memory Updates & Semantic Clustering

HippocampAI now includes advanced memory management features inspired by Mem0, providing intelligent memory updates, conflict resolution, and semantic organization.

## Features Overview

### 1. Smart Memory Update Logic

Intelligent decisions on how to handle new memories that are similar to existing ones.

**Capabilities:**
- **Merge vs Update vs Skip Decisions**: Automatically decides whether to merge complementary information, update existing memories, skip duplicates, or keep both
- **Conflict Resolution**: Detects contradictions and resolves them based on recency and confidence
- **Memory Reconciliation**: Reconciles groups of related memories to resolve conflicts
- **Confidence Scoring Evolution**: Confidence scores evolve based on reinforcement, conflicts, and access patterns
- **Memory Refinement**: LLM-based refinement to improve memory quality over time

**Example:**
```python
from hippocampai import EnhancedMemoryClient

client = EnhancedMemoryClient(provider="groq")

# Store initial memory
mem1 = client.remember("I love coffee", user_id="user1", type="preference")

# Similar memory - will be intelligently merged or skipped
mem2 = client.remember("I really enjoy coffee", user_id="user1", type="preference")

# Conflicting memory - will trigger conflict resolution
mem3 = client.remember("I don't like coffee anymore", user_id="user1", type="preference")
# Result: Keeps newer information, lowers confidence of old memory
```

### 2. Semantic Clustering & Auto-Categorization

Automatic organization of memories by topics and semantic meaning.

**Capabilities:**
- **Automatic Memory Clustering**: Groups memories by semantic topics
- **Dynamic Tag Suggestion**: AI-powered tag suggestions based on content
- **Category Auto-Assignment**: Automatically assigns correct MemoryType using pattern matching and LLM
- **Similar Memory Detection**: Finds related memories at write-time to prevent duplicates
- **Topic Modeling**: Identifies dominant topics in conversation
- **Topic Shift Detection**: Detects when conversation topic changes significantly

**Example:**
```python
# Auto-categorization happens automatically
memory = client.remember(
    text="I want to learn Python and machine learning",
    user_id="user1"
)
# Automatically categorized as: type=GOAL
# Auto-tagged with: ["learn", "python", "learning", "goal"]

# Cluster all memories by topic
clusters = client.cluster_user_memories("user1", max_clusters=10)
for cluster in clusters:
    print(f"Topic: {cluster.topic}")
    print(f"Tags: {cluster.tags}")
    print(f"Memories: {len(cluster.memories)}")

# Detect topic shifts in conversation
topic = client.detect_topic_shift("user1", window_size=10)
if topic:
    print(f"Topic shifted to: {topic}")
```

## Implementation Details

### Smart Memory Updater

Located in `src/hippocampai/pipeline/smart_updater.py`

**Key Components:**

1. **UpdateDecision Class**: Represents the decision on how to handle a new memory
   - Actions: `merge`, `update`, `skip`, `keep_both`
   - Includes reasoning and confidence adjustments

2. **SmartMemoryUpdater Class**: Core logic for intelligent updates
   - `should_update_memory()`: Decides action based on similarity and conflicts
   - `resolve_conflict()`: Resolves contradicting memories
   - `reconcile_memories()`: Batch reconciliation of related memories
   - `refine_memory()`: LLM-based quality improvement
   - `update_confidence()`: Evolves confidence scores

**Decision Logic:**
- Similarity > 95%: Skip (reinforce confidence)
- Similarity > 85%: Consider merge/update
- Similarity > 60% + conflict: Resolve conflict
- Similarity > 60% + no conflict: Keep both (complementary)
- Similarity < 60%: Keep both (unrelated)

### Semantic Categorizer

Located in `src/hippocampai/pipeline/semantic_clustering.py`

**Key Components:**

1. **MemoryCluster Class**: Represents a topic cluster
   - Contains: topic name, memories, common tags

2. **SemanticCategorizer Class**: Handles categorization and clustering
   - `suggest_tags()`: Extracts keywords and suggests tags
   - `assign_category()`: Pattern-based + LLM category assignment
   - `find_similar_memories()`: Token-based similarity search
   - `cluster_memories()`: Groups memories by topics
   - `detect_topic_shift()`: Identifies conversation topic changes
   - `enrich_memory_with_categories()`: Auto-enriches with tags and category

**Topic Keywords:**
- Pre-defined keyword maps for common topics: work, personal, health, food, travel, shopping, entertainment, learning, technology, finance

## API Reference

### MemoryClient Methods

All these methods are available in `MemoryClient`, `EnhancedMemoryClient`, and `OptimizedMemoryClient`.

#### `reconcile_user_memories(user_id: str) -> List[Memory]`

Reconcile and resolve conflicts in user's memories.

```python
reconciled = client.reconcile_user_memories("user1")
print(f"Reconciled into {len(reconciled)} memories")
```

#### `cluster_user_memories(user_id: str, max_clusters: int = 10) -> List[MemoryCluster]`

Cluster user's memories by semantic topics.

```python
clusters = client.cluster_user_memories("user1", max_clusters=5)
for cluster in clusters:
    print(f"{cluster.topic}: {len(cluster.memories)} memories")
```

#### `suggest_memory_tags(memory: Memory, max_tags: int = 5) -> List[str]`

Suggest tags for a given memory.

```python
tags = client.suggest_memory_tags(memory, max_tags=5)
print(f"Suggested tags: {tags}")
```

#### `refine_memory_quality(memory_id: str, context: Optional[str] = None) -> Optional[Memory]`

Refine a memory's text quality using LLM.

```python
refined = client.refine_memory_quality(memory_id, context="User prefers concise summaries")
print(f"Refined: {refined.text}")
```

#### `detect_topic_shift(user_id: str, window_size: int = 10) -> Optional[str]`

Detect if there's been a shift in conversation topics.

```python
topic = client.detect_topic_shift("user1", window_size=10)
if topic:
    print(f"New topic: {topic}")
```

## Automatic Features

### Auto-Categorization on Storage

Every memory stored via `remember()` is automatically enriched:

```python
memory = client.remember(
    text="I usually go running every morning",
    user_id="user1",
    type="fact"  # Provided type
)
# Automatically re-categorized to type=HABIT
# Auto-tagged with relevant keywords
```

### Smart Update Detection

When storing a memory, the system automatically:
1. Checks for similar existing memories
2. Decides whether to merge, update, skip, or keep both
3. Resolves conflicts if detected
4. Updates confidence scores

```python
# First storage
mem1 = client.remember("I love coffee", user_id="user1")

# Similar memory - system decides automatically
mem2 = client.remember("I really love coffee", user_id="user1")
# Result: Might skip or merge depending on similarity
```

## Configuration

### Similarity Threshold

Adjust the similarity threshold for smart updates:

```python
from hippocampai import MemoryClient

client = MemoryClient(llm_provider="groq", llm_model="llama-3.3-70b-versatile")

# Access smart updater
client.smart_updater.similarity_threshold = 0.9  # More strict (default: 0.85)
```

### LLM-Based Features

For best results with LLM-based categorization and refinement:
- Use `groq` with `llama-3.3-70b-versatile` for fast, quality results
- Use `openai` with `gpt-4o-mini` for highest accuracy
- LLM is optional; pattern-based categorization works without LLM

## Performance Considerations

### Memory Enrichment Impact

Auto-categorization adds ~50-100ms per memory storage (when LLM is used).
- Pattern-based categorization: ~5-10ms
- LLM tag suggestion: ~50-100ms (optional, triggered when few pattern matches)

### Smart Update Overhead

Smart update detection adds ~100-200ms for the first 100 memories.
- Similarity calculation: O(n) where n = existing memories
- Cached for subsequent similar memories
- Recommend: Use `get_memories(limit=100)` to limit search space

### Clustering Performance

- Clustering 1000 memories: ~1-2 seconds
- Topic detection: ~100-500ms
- Scales linearly with memory count

## Examples

### Complete Integration Example

```python
from hippocampai import EnhancedMemoryClient

client = EnhancedMemoryClient(provider="groq")
user_id = "alice"

# 1. Store memories - automatic enrichment happens
memories_to_store = [
    "I love coffee and tea",
    "I work at Google as an engineer",
    "I want to learn machine learning",
    "I usually exercise in the morning",
    "I visited Paris last summer"
]

for text in memories_to_store:
    mem = client.remember(text=text, user_id=user_id)
    print(f"{text}")
    print(f"  → Type: {mem.type.value}, Tags: {mem.tags}\n")

# 2. Cluster memories by topic
clusters = client.cluster_user_memories(user_id, max_clusters=5)
print(f"\nFound {len(clusters)} clusters:")
for cluster in clusters:
    print(f"  {cluster.topic}: {len(cluster.memories)} memories")

# 3. Detect topic shift
client.remember("I need to finish the project by Friday", user_id)
client.remember("The team meeting is at 2 PM", user_id)
topic = client.detect_topic_shift(user_id, window_size=5)
if topic:
    print(f"\nTopic shifted to: {topic}")

# 4. Reconcile conflicts
reconciled = client.reconcile_user_memories(user_id)
print(f"\nReconciled {len(reconciled)} memories")
```

### Conflict Resolution Example

```python
# Store initial preference
mem1 = client.remember("I prefer working remotely", user_id="bob")

# Store conflicting information
mem2 = client.remember("I don't like working remotely anymore", user_id="bob")

# System automatically resolves:
# - Detects conflict (negation pattern)
# - Favors newer information (recency bias)
# - Lowers confidence of older memory
# - May update or create new memory depending on confidence levels

# Manually reconcile all memories
reconciled = client.reconcile_user_memories("bob")
# Result: One memory remains with highest confidence
```

## Testing

Run the test suite:

```bash
# Run smart memory tests
pytest tests/test_smart_memory.py -v

# Run integration tests
pytest tests/test_smart_memory.py::TestSmartMemoryIntegration -v
```

## Demo Script

See `examples/11_smart_memory_demo.py` for a comprehensive demonstration.

```bash
python examples/11_smart_memory_demo.py
```

## Technical Details

### Memory Confidence Evolution

Confidence scores (0.0-1.0) evolve based on:
- **Reinforcement** (+0.05): When similar memory is stored again
- **Validation** (+0.1): When memory is accessed and used
- **Conflict** (-0.15): When conflicting information appears
- **Access** (+0.01): Small boost for each access
- **Age Decay** (-5% per 90 days): Very old memories lose confidence without access

### Conflict Resolution Strategy

1. Calculate similarity between memories (Jaccard similarity on tokens)
2. Detect negation patterns (not, no, never, don't, no longer, etc.)
3. Compare confidence scores and recency
4. Decision:
   - Recent + High confidence old memory: Keep old, note conflict
   - Old or Low confidence old memory: Replace with new
   - Equal confidence: Keep both, lower confidence of both

### Topic Detection Algorithm

1. Extract keywords from memory text (nouns, important words)
2. Match against topic keyword database
3. Score each topic based on keyword matches
4. Return highest-scoring topic or most common noun
5. For clustering: Group memories with same dominant topic

## Migration Guide

### For Existing Users

The new features are **backward compatible** and activate automatically:

1. **Auto-Categorization**: Already enabled on all `remember()` calls
2. **Smart Updates**: Automatically checks for similar memories
3. **No Breaking Changes**: Existing code continues to work

To opt-out of smart updates (not recommended):
```python
# Access and disable (not recommended)
client.smart_updater = None  # Disables smart update checks
```

### Performance Impact

- **Small datasets** (<1000 memories): Negligible (~50-100ms added)
- **Large datasets** (>10000 memories): Consider setting `limit=100` in smart update checks
- **LLM operations**: Async operations recommended for bulk processing

## Future Enhancements

Planned improvements:
- [ ] Vector-based similarity for better semantic matching
- [ ] Hierarchical topic clustering
- [ ] Temporal topic evolution tracking
- [ ] Confidence decay based on contradicting evidence
- [ ] Batch reconciliation optimization
- [ ] Topic-based memory compression

## References

- Smart Memory Updater: `src/hippocampai/pipeline/smart_updater.py`
- Semantic Categorizer: `src/hippocampai/pipeline/semantic_clustering.py`
- Integration: `src/hippocampai/client.py:1679-1778`
- Tests: `tests/test_smart_memory.py`
- Demo: `examples/11_smart_memory_demo.py`

## Comparison with Mem0

| Feature | HippocampAI | Mem0 |
|---------|-------------|------|
| Smart Updates | ✅ Merge/Update/Skip | ✅ Yes |
| Conflict Resolution | ✅ Recency + Confidence | ✅ Yes |
| Auto-Categorization | ✅ Pattern + LLM | ⚠️ Limited |
| Topic Clustering | ✅ Keyword + Semantic | ❌ No |
| Topic Shift Detection | ✅ Yes | ❌ No |
| Confidence Evolution | ✅ Multi-factor | ✅ Basic |
| Memory Refinement | ✅ LLM-based | ❌ No |
| Open Source | ✅ Apache 2.0 | ✅ Apache 2.0 |

---

**Questions or Issues?** Please open an issue on GitHub!
