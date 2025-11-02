# Search & Retrieval Enhancements Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-28

---

## Overview

HippocampAI now includes powerful search and retrieval enhancements that give you fine-grained control over how memories are searched and retrieved. These features include:

1. **Hybrid Search Modes** - Choose between vector-only, keyword-only, or hybrid search
2. **Reranking Control** - Enable or disable CrossEncoder reranking
3. **Score Breakdowns** - Detailed scoring breakdown for transparency
4. **Saved Searches** - Save frequently used queries for quick retrieval
5. **Search Suggestions** - Auto-suggest queries based on user history

---

## 1. Hybrid Search Modes

### Overview

Control how memories are searched by selecting from three search modes:

- **`SearchMode.HYBRID`** (default): Combines vector and keyword search with RRF fusion
- **`SearchMode.VECTOR_ONLY`**: Uses only semantic vector search (best for conceptual queries)
- **`SearchMode.KEYWORD_ONLY`**: Uses only BM25 keyword search (best for exact matches)

### Usage

```python
from hippocampai.client import MemoryClient
from hippocampai.models.search import SearchMode

client = MemoryClient()

# Hybrid search (default) - best for most use cases
results = client.recall(
    query="What are my work projects?",
    user_id="user_001",
    search_mode=SearchMode.HYBRID,
    k=5
)

# Vector-only search - best for conceptual similarity
results = client.recall(
    query="memories about career growth",
    user_id="user_001",
    search_mode=SearchMode.VECTOR_ONLY,
    k=5
)

# Keyword-only search - best for exact keyword matches
results = client.recall(
    query="San Francisco Google",
    user_id="user_001",
    search_mode=SearchMode.KEYWORD_ONLY,
    k=5
)
```

### When to Use Each Mode

| Mode | Use Case | Example Query |
|------|----------|---------------|
| **HYBRID** | General queries, best overall accuracy | "What did I do last week?" |
| **VECTOR_ONLY** | Conceptual queries, semantic similarity | "memories about happiness" |
| **KEYWORD_ONLY** | Exact keyword search, specific terms | "project alpha" |

---

## 2. Reranking Control

### Overview

Control whether CrossEncoder reranking is applied to search results. Reranking improves accuracy but adds latency.

### Usage

```python
# With reranking (default) - more accurate but slower
results = client.recall(
    query="What are my goals?",
    user_id="user_001",
    enable_reranking=True,  # Default
    k=5
)

# Without reranking - faster but less accurate
results = client.recall(
    query="What are my goals?",
    user_id="user_001",
    enable_reranking=False,
    k=5
)
```

### Performance Comparison

| Mode | Latency | Accuracy | Use Case |
|------|---------|----------|----------|
| **Reranking ON** | ~200-300ms | High | User-facing queries |
| **Reranking OFF** | ~100-150ms | Medium | Background tasks, bulk operations |

---

## 3. Score Breakdowns

### Overview

Get detailed scoring breakdowns for each retrieved memory to understand why it was selected.

### Usage

```python
# Enable score breakdowns (default)
results = client.recall(
    query="recent work",
    user_id="user_001",
    enable_score_breakdown=True,  # Default
    k=5
)

for result in results:
    print(f"Memory: {result.memory.text[:50]}...")
    print(f"Final Score: {result.score:.3f}")

    if result.breakdown:
        print(f"  • Similarity: {result.breakdown['sim']:.3f}")
        print(f"  • Reranking: {result.breakdown['rerank']:.3f}")
        print(f"  • Recency: {result.breakdown['recency']:.3f}")
        print(f"  • Importance: {result.breakdown['importance']:.3f}")
        print(f"  • Mode: {result.breakdown['search_mode']}")
        print(f"  • Reranking: {result.breakdown['reranking_enabled']}")
```

### Breakdown Fields

| Field | Description | Range |
|-------|-------------|-------|
| **sim** | Vector similarity score | 0.0 - 1.0 |
| **rerank** | CrossEncoder reranking score | 0.0 - 1.0 |
| **recency** | Time-based decay score | 0.0 - 1.0 |
| **importance** | User-assigned importance | 0.0 - 1.0 |
| **final** | Weighted combination of all scores | 0.0 - 1.0 |
| **search_mode** | Search mode used | "hybrid" / "vector_only" / "keyword_only" |
| **reranking_enabled** | Whether reranking was used | true / false |

---

## 4. Saved Searches

### Overview

Save frequently used queries for quick retrieval. Saved searches remember all search parameters including mode, filters, and reranking settings.

### Basic Usage

```python
from hippocampai.search import SavedSearchManager
from hippocampai.models.search import SearchMode

# Initialize manager
search_mgr = SavedSearchManager()

# Save a search
saved_search = search_mgr.save_search(
    name="My Weekly Review",
    query="What did I accomplish this week?",
    user_id="user_001",
    search_mode=SearchMode.HYBRID,
    enable_reranking=True,
    filters={"type": "event"},
    k=10,
    tags=["weekly", "review"],
    metadata={"category": "productivity"}
)

print(f"Saved search ID: {saved_search.id}")
```

### Executing Saved Searches

```python
# Execute saved search
search = search_mgr.execute_saved_search(saved_search.id)

# Use the search parameters
results = client.recall(
    query=search.query,
    user_id=search.user_id,
    search_mode=search.search_mode,
    enable_reranking=search.enable_reranking,
    filters=search.filters,
    k=search.k
)

print(f"Use count: {search.use_count}")
print(f"Last used: {search.last_used_at}")
```

### Managing Saved Searches

```python
# Get all saved searches for a user
searches = search_mgr.get_user_searches("user_001")

# Get most frequently used searches
most_used = search_mgr.get_most_used("user_001", limit=5)

# Search by name
found = search_mgr.search_by_name("user_001", "weekly")

# Get searches by tag
tagged = search_mgr.get_user_searches("user_001", tags=["review"])

# Update saved search
search_mgr.update_search(
    search_id=saved_search.id,
    name="Updated Name",
    k=20
)

# Delete saved search
search_mgr.delete_search(saved_search.id, user_id="user_001")
```

### Statistics

```python
stats = search_mgr.get_statistics("user_001")

print(f"Total searches: {stats['total_searches']}")
print(f"Total uses: {stats['total_uses']}")
print(f"Avg uses: {stats['avg_uses_per_search']:.2f}")
print(f"Most used: {stats['most_used_search']['name']}")
```

---

## 5. Search Suggestions

### Overview

Auto-suggest queries based on user search history. Provides autocomplete and popular query suggestions.

### Basic Usage

```python
from hippocampai.search import SearchSuggestionEngine

# Initialize engine
suggestion_engine = SearchSuggestionEngine(
    min_frequency=2,  # Minimum times a query must appear
    history_days=90   # Days of history to consider
)

# Record queries as user searches
suggestion_engine.record_query("user_001", "What are my goals?")
suggestion_engine.record_query("user_001", "What are my work projects?")
suggestion_engine.record_query("user_001", "What are my goals?")  # Repeated
```

### Getting Suggestions

```python
# Get top suggestions
suggestions = suggestion_engine.get_suggestions("user_001", limit=5)

for sugg in suggestions:
    print(f"Query: '{sugg.query}'")
    print(f"  Confidence: {sugg.confidence:.2f}")
    print(f"  Frequency: {sugg.frequency}")
    print(f"  Last used: {sugg.last_used}")
```

### Autocomplete

```python
# Get autocomplete suggestions based on prefix
autocomplete = suggestion_engine.get_suggestions(
    user_id="user_001",
    prefix="what are",
    limit=5
)

for sugg in autocomplete:
    print(f"  → {sugg.query}")
```

### Popular Queries

```python
# Get most popular queries
popular = suggestion_engine.get_popular_queries("user_001", limit=10)

for i, sugg in enumerate(popular, 1):
    print(f"{i}. '{sugg.query}' ({sugg.frequency} uses)")
```

### Recent Queries

```python
# Get recent unique queries
recent = suggestion_engine.get_recent_queries("user_001", limit=10)

print("Recent queries:")
for query in recent:
    print(f"  • {query}")
```

### Managing History

```python
# Clear old history
suggestion_engine.clear_history("user_001", days_to_keep=30)

# Clear all history
suggestion_engine.clear_history("user_001")

# Get statistics
stats = suggestion_engine.get_statistics("user_001")
print(f"Total queries: {stats['total_queries']}")
print(f"Unique queries: {stats['unique_queries']}")
print(f"Avg frequency: {stats['avg_frequency']:.2f}")
```

---

## Complete Example

```python
from hippocampai.client import MemoryClient
from hippocampai.models.search import SearchMode
from hippocampai.search import SavedSearchManager, SearchSuggestionEngine

# Initialize client and managers
client = MemoryClient()
search_mgr = SavedSearchManager()
suggestion_engine = SearchSuggestionEngine()

# 1. Save a frequently used search
weekly_review = search_mgr.save_search(
    name="Weekly Review",
    query="What did I accomplish this week?",
    user_id="user_001",
    search_mode=SearchMode.HYBRID,
    enable_reranking=True,
    k=10,
    tags=["weekly", "productivity"]
)

# 2. Record query for suggestions
suggestion_engine.record_query("user_001", weekly_review.query)

# 3. Execute search with custom mode
results = client.recall(
    query=weekly_review.query,
    user_id="user_001",
    search_mode=SearchMode.VECTOR_ONLY,  # Override mode
    enable_reranking=False,  # Faster search
    enable_score_breakdown=True,
    k=10
)

# 4. Display results with breakdowns
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.memory.text[:60]}...")
    print(f"   Score: {result.score:.3f}")

    if result.breakdown:
        print(f"   Breakdown:")
        print(f"     • Similarity: {result.breakdown['sim']:.3f}")
        print(f"     • Recency: {result.breakdown['recency']:.3f}")
        print(f"     • Mode: {result.breakdown['search_mode']}")

# 5. Get search suggestions for autocomplete
suggestions = suggestion_engine.get_suggestions(
    user_id="user_001",
    prefix="what",
    limit=5
)

print("\n\nAutocomplete suggestions:")
for sugg in suggestions:
    print(f"  → {sugg.query} (confidence: {sugg.confidence:.2f})")
```

---

## API Reference

### SearchMode Enum

```python
class SearchMode(str, Enum):
    HYBRID = "hybrid"          # Vector + BM25 with RRF fusion
    VECTOR_ONLY = "vector_only"  # Semantic search only
    KEYWORD_ONLY = "keyword_only"  # BM25 keyword search only
```

### MemoryClient.recall() Parameters

```python
def recall(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[dict] = None,
    search_mode: SearchMode = SearchMode.HYBRID,
    enable_reranking: bool = True,
    enable_score_breakdown: bool = True,
) -> list[RetrievalResult]:
```

---

## Best Practices

### 1. Choose the Right Search Mode

- **HYBRID**: Default for most queries
- **VECTOR_ONLY**: Conceptual, semantic queries
- **KEYWORD_ONLY**: Exact term matching

### 2. Use Reranking Wisely

- Enable for user-facing queries (better accuracy)
- Disable for bulk operations (better performance)

### 3. Leverage Saved Searches

- Save commonly used queries with optimal settings
- Use tags to organize searches by category
- Monitor usage statistics to identify patterns

### 4. Implement Autocomplete

- Record all user queries for better suggestions
- Use prefix matching for real-time autocomplete
- Show popular queries as defaults

---

## Performance Considerations

| Feature | Latency Impact | Memory Impact |
|---------|---------------|---------------|
| **Hybrid Mode** | Baseline | Baseline |
| **Vector Only** | -20% (faster) | Same |
| **Keyword Only** | -30% (fastest) | Same |
| **Reranking ON** | +50% | +10% |
| **Reranking OFF** | Baseline | Baseline |
| **Score Breakdown** | +5% | +5% |

---

## Support

For questions or issues, please check:

- Main README: `README.md`
- API Reference: `docs/API_COMPLETE_REFERENCE.md`
- Issue Tracker: GitHub Issues

---

**Generated**: 2025-10-28
**Version**: 1.0.0
