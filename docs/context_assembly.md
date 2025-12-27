# Automated Context Assembly

HippocampAI provides automated context assembly so agents don't need to manually pick memory items. The `assemble_context()` method handles retrieval, ranking, deduplication, and token budgeting automatically.

## Overview

Context assembly takes a user query and returns a ready-to-use `ContextPack` containing:
- Formatted context text for LLM prompts
- Citations (memory IDs) for attribution
- Structured list of selected memories
- Dropped items with reasons (for debugging)

## Quick Start

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store some memories
client.remember("I love coffee", user_id="alice", type="preference")
client.remember("I prefer dark roast", user_id="alice", type="preference")
client.remember("I work at Google", user_id="alice", type="fact")

# Assemble context for a query
pack = client.assemble_context(
    query="What are my coffee preferences?",
    user_id="alice",
)

# Use in LLM prompt
prompt = f"""
{pack.final_context_text}

User: What are my coffee preferences?
Assistant:"""

print(f"Selected {len(pack.selected_items)} memories")
print(f"Total tokens: {pack.total_tokens}")
print(f"Citations: {pack.citations}")
```

## API Reference

### `assemble_context()`

```python
pack = client.assemble_context(
    query="user query",
    user_id="user123",
    session_id=None,              # Optional: scope to session
    token_budget=4000,            # Max tokens for context
    max_items=20,                 # Max memory items
    recency_bias=0.3,             # Weight for recent memories (0-1)
    entity_focus=None,            # Entities to prioritize
    type_filter=None,             # Memory types to include
    min_relevance=0.1,            # Minimum relevance score (0-1)
    allow_summaries=True,         # Summarize when budget exceeded
    include_citations=True,       # Include memory IDs
    deduplicate=True,             # Remove duplicates
    time_range_days=None,         # Limit to last N days
)
```

### ContextPack Response

```python
class ContextPack:
    final_context_text: str       # Ready-to-use context string
    citations: list[str]          # Memory IDs for attribution
    selected_items: list[SelectedItem]  # Structured selected memories
    dropped_items: list[DroppedItem]    # Items dropped (for debugging)
    total_tokens: int             # Estimated token count
    query: str                    # Original query
    user_id: str
    session_id: Optional[str]
    constraints: ContextConstraints
    assembled_at: datetime
    metadata: dict
```

### SelectedItem

```python
class SelectedItem:
    memory_id: str
    text: str
    memory_type: str
    relevance_score: float
    importance: float
    created_at: datetime
    token_count: int
    tags: list[str]
    metadata: dict
```

### DroppedItem

```python
class DroppedItem:
    memory_id: str
    text_preview: str      # First 100 chars
    reason: DropReason     # Why it was dropped
    relevance_score: float
    details: str           # Additional info
```

Drop reasons:
- `TOKEN_BUDGET`: Would exceed token limit
- `LOW_RELEVANCE`: Below min_relevance threshold
- `DUPLICATE`: Similar to already selected item
- `EXPIRED`: Memory has expired
- `FILTERED`: Didn't match type/time filters
- `SUMMARIZED`: Was summarized to fit budget

## REST API

### Assemble Context (Full Response)

```bash
curl -X POST http://localhost:8000/v1/context:assemble \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coffee preferences",
    "user_id": "alice",
    "token_budget": 2000,
    "type_filter": ["preference"]
  }'
```

### Assemble Context (Text Only)

```bash
curl -X POST http://localhost:8000/v1/context:assemble/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coffee preferences",
    "user_id": "alice"
  }'
```

Response:
```json
{
  "context": "1. I love coffee\n2. I prefer dark roast",
  "citations": ["mem-123", "mem-456"],
  "token_count": 25,
  "item_count": 2
}
```

## Use Cases

### 1. Chatbot Context Injection

```python
def get_response(user_message: str, user_id: str) -> str:
    # Assemble relevant context
    pack = client.assemble_context(
        query=user_message,
        user_id=user_id,
        token_budget=2000,
    )
    
    # Build prompt with context
    prompt = f"""You are a helpful assistant. Use the following context about the user:

{pack.get_context_for_prompt()}

User: {user_message}
Assistant:"""
    
    # Call LLM
    response = llm.generate(prompt)
    
    # Optionally include citations
    if pack.citations:
        response += f"\n\n[Based on: {', '.join(pack.citations[:3])}]"
    
    return response
```

### 2. RAG with Token Budget

```python
# Strict token budget for context window
pack = client.assemble_context(
    query="quarterly sales report",
    user_id="analyst",
    token_budget=8000,  # Leave room for response
    allow_summaries=True,  # Summarize long memories
)

# Check if we got enough context
if pack.total_tokens < 100:
    print("Warning: Limited context available")
```

### 3. Recent Events Only

```python
# Only include memories from last 7 days
pack = client.assemble_context(
    query="recent updates",
    user_id="alice",
    time_range_days=7,
    type_filter=["event", "fact"],
)
```

### 4. Entity-Focused Context

```python
# Prioritize memories about specific entities
pack = client.assemble_context(
    query="project status",
    user_id="alice",
    entity_focus=["Project Alpha", "Team Lead"],
)
```

### 5. Debugging Context Selection

```python
pack = client.assemble_context(
    query="preferences",
    user_id="alice",
)

# See what was dropped and why
for dropped in pack.dropped_items:
    print(f"Dropped: {dropped.text_preview}")
    print(f"  Reason: {dropped.reason}")
    print(f"  Details: {dropped.details}")
```

## Configuration

### Token Budget Guidelines

| Use Case | Recommended Budget |
|----------|-------------------|
| Quick responses | 1000-2000 |
| Detailed analysis | 4000-8000 |
| Long-form generation | 8000-16000 |

### Recency Bias

- `0.0`: Pure relevance ranking
- `0.3`: Balanced (default)
- `0.7`: Strongly prefer recent
- `1.0`: Only recency matters

### Type Filters

Available memory types:
- `fact`: Personal information, statements
- `preference`: Likes, dislikes, opinions
- `goal`: Intentions, aspirations
- `habit`: Routines, regular activities
- `event`: Specific occurrences
- `context`: General conversation

## Best Practices

1. **Set appropriate token budgets** - Leave room for the LLM response
2. **Use type filters** when you know what kind of context you need
3. **Enable deduplication** to avoid repetitive context
4. **Check dropped_items** when debugging unexpected results
5. **Use citations** for transparency and attribution
6. **Adjust recency_bias** based on whether recent or historical context matters more

## Troubleshooting

### Empty context returned

- Check that memories exist for the user
- Lower the `min_relevance` threshold
- Remove restrictive `type_filter`
- Increase `time_range_days` or remove it

### Context too short

- Increase `token_budget`
- Increase `max_items`
- Lower `min_relevance`
- Disable `deduplicate` if memories are being incorrectly merged

### Irrelevant memories included

- Increase `min_relevance`
- Add `type_filter` to focus on specific types
- Use `entity_focus` to prioritize specific entities

### Token budget exceeded

This shouldn't happen - the assembler respects the budget. If you see more tokens than expected:
- Token estimation is approximate (4 chars ≈ 1 token)
- For precise counting, use tiktoken in production
