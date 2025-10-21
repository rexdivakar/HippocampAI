# HippocampAI API Reference

Quick reference for all HippocampAI APIs.

## MemoryClient

### Core Operations

#### remember()
```python
client.remember(
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    ttl_days: Optional[int] = None
) -> Memory
```

Store a memory with automatic size tracking.

**Returns:** Memory object with `id`, `text_length`, `token_count`, etc.

#### recall()
```python
client.recall(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]
```

Retrieve memories using hybrid search.

#### update_memory()
```python
client.update_memory(
    memory_id: str,
    text: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expires_at: Optional[datetime] = None
) -> Optional[Memory]
```

Update an existing memory. Size metrics are recalculated if text changes.

#### delete_memory()
```python
client.delete_memory(
    memory_id: str,
    user_id: Optional[str] = None
) -> bool
```

Delete a memory by ID.

#### get_memories()
```python
client.get_memories(
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100
) -> List[Memory]
```

Get memories with filtering (no semantic search).

**Filters:**
- `type`: Memory type (str or list)
- `tags`: Tags to filter by (str or list)
- `session_id`: Session ID
- `min_importance`: Minimum importance score
- `max_importance`: Maximum importance score
- `include_expired`: Include expired memories (default: False)

### Batch Operations

#### add_memories()
```python
client.add_memories(
    memories: List[Dict[str, Any]],
    user_id: str,
    session_id: Optional[str] = None
) -> List[Memory]
```

Batch add multiple memories.

#### delete_memories()
```python
client.delete_memories(
    memory_ids: List[str],
    user_id: Optional[str] = None
) -> int
```

Batch delete memories. Returns count of deleted memories.

### Memory Statistics

#### get_memory_statistics()
```python
client.get_memory_statistics(
    user_id: str
) -> Dict[str, Any]
```

Get memory size and usage statistics.

**Returns:**
```python
{
    "total_memories": int,
    "total_characters": int,
    "total_tokens": int,
    "avg_memory_size_chars": float,
    "avg_memory_size_tokens": float,
    "largest_memory_chars": int,
    "smallest_memory_chars": int,
    "by_type": Dict[str, Dict[str, Any]]  # Statistics per memory type
}
```

### Graph Operations

#### add_relationship()
```python
client.add_relationship(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    weight: float = 1.0
) -> bool
```

Add a relationship between two memories.

#### get_related_memories()
```python
client.get_related_memories(
    memory_id: str,
    relation_types: Optional[List[RelationType]] = None,
    max_depth: int = 1
) -> List[tuple]
```

Get related memories. Returns list of (memory_id, relation_type, weight) tuples.

#### get_memory_clusters()
```python
client.get_memory_clusters(
    user_id: str
) -> List[set]
```

Find clusters of related memories.

#### export_graph_to_json()
```python
client.export_graph_to_json(
    file_path: str,
    user_id: Optional[str] = None,
    indent: int = 2
) -> str
```

Export memory graph to a JSON file for backup or transfer.

**Args:**
- `file_path`: Path where the JSON file will be saved
- `user_id`: Optional user ID to export only a specific user's graph
- `indent`: JSON indentation level (default: 2)

**Returns:** File path where the graph was saved

**Example:**
```python
# Export full graph
client.export_graph_to_json("memory_graph.json")

# Export user-specific graph
client.export_graph_to_json("alice_graph.json", user_id="alice")
```

#### import_graph_from_json()
```python
client.import_graph_from_json(
    file_path: str,
    merge: bool = True
) -> Dict[str, Any]
```

Import memory graph from a JSON file.

**Args:**
- `file_path`: Path to the JSON file to import
- `merge`: If True, merge with existing graph; if False, replace existing graph

**Returns:** Dictionary with import statistics:
```python
{
    "file_path": str,          # Path that was imported
    "nodes_before": int,       # Node count before import
    "nodes_after": int,        # Node count after import
    "edges_before": int,       # Edge count before import
    "edges_after": int,        # Edge count after import
    "nodes_imported": int,     # Nodes in the file
    "edges_imported": int,     # Edges in the file
    "merged": bool             # Whether it was merged or replaced
}
```

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `ValueError`: If the file contains invalid graph format

**Example:**
```python
# Import and merge with existing graph
stats = client.import_graph_from_json("memory_graph.json")
print(f"Imported {stats['nodes_imported']} nodes")

# Replace existing graph
stats = client.import_graph_from_json("backup.json", merge=False)
```

### Version Control

#### get_memory_history()
```python
client.get_memory_history(
    memory_id: str
) -> List[MemoryVersion]
```

Get version history for a memory.

#### rollback_memory()
```python
client.rollback_memory(
    memory_id: str,
    version_number: int
) -> Optional[Memory]
```

Rollback memory to a previous version.

### Context Injection

#### inject_context()
```python
client.inject_context(
    prompt: str,
    query: str,
    user_id: str,
    k: int = 5,
    template: str = "default"
) -> str
```

Inject relevant memories into a prompt for LLMs.

**Templates:** `"default"`, `"minimal"`, `"detailed"`

### Telemetry

#### get_telemetry_metrics()
```python
client.get_telemetry_metrics() -> Dict[str, Any]
```

Get performance metrics including size tracking.

#### get_recent_operations()
```python
client.get_recent_operations(
    limit: int = 10,
    operation: Optional[str] = None
) -> List[MemoryTrace]
```

Get recent memory operations with traces.

#### export_telemetry()
```python
client.export_telemetry(
    trace_ids: Optional[List[str]] = None
) -> List[Dict]
```

Export telemetry data for external analysis.

---

## AsyncMemoryClient

All methods have `_async` suffix and return coroutines.

### Core Operations

```python
await client.remember_async(...)
await client.recall_async(...)
await client.update_memory_async(...)
await client.delete_memory_async(...)
await client.get_memories_async(...)
```

### Batch Operations

```python
await client.add_memories_async(...)
await client.delete_memories_async(...)
```

### Utilities

```python
await client.get_memory_statistics_async(...)
await client.inject_context_async(...)
await client.extract_from_conversation_async(...)
```

### Concurrent Operations

```python
import asyncio

# Store multiple memories concurrently
tasks = [
    client.remember_async(f"Memory {i}", user_id="alice")
    for i in range(10)
]
memories = await asyncio.gather(*tasks)
```

---

## Memory Model

```python
class Memory(BaseModel):
    id: str                          # UUID
    text: str                        # Memory text
    user_id: str                     # User ID
    session_id: Optional[str]        # Session ID
    type: MemoryType                 # preference, fact, goal, habit, event, context
    importance: float                # 0.0-10.0
    confidence: float                # 0.0-1.0
    tags: List[str]                  # Tags
    created_at: datetime             # Creation time
    updated_at: datetime             # Last update time
    expires_at: Optional[datetime]   # Expiration (TTL)
    access_count: int                # Access tracking
    text_length: int                 # Character count (NEW)
    token_count: int                 # Token estimate (NEW)
    metadata: Dict[str, Any]         # Custom metadata
```

---

## Configuration

### Presets

```python
# Local (fully self-hosted)
client = MemoryClient.from_preset("local")

# Cloud LLM with local vector DB
client = MemoryClient.from_preset("cloud")

# Production optimized
client = MemoryClient.from_preset("production")

# Development (fast)
client = MemoryClient.from_preset("development")
```

### Custom Configuration

```python
from hippocampai import MemoryClient, Config

config = Config(
    qdrant_url="http://localhost:6333",
    embed_model="BAAI/bge-small-en-v1.5",
    llm_provider="ollama",
    llm_model="qwen2.5:7b-instruct",
    top_k_final=20,
    weights={
        "sim": 0.55,
        "rerank": 0.20,
        "recency": 0.15,
        "importance": 0.10
    }
)

client = MemoryClient(config=config)
```

---

## Enums

### MemoryType
```python
class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"
```

### RelationType
```python
class RelationType(str, Enum):
    RELATED_TO = "related_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
```

### OperationType
```python
class OperationType(str, Enum):
    REMEMBER = "remember"
    RECALL = "recall"
    EXTRACT = "extract"
    UPDATE = "update"
    DELETE = "delete"
    GET = "get"
    EXPIRE = "expire"
```

---

For complete examples, see [FEATURES.md](FEATURES.md) or the `examples/` directory.
