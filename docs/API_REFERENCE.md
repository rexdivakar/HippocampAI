# HippocampAI API Reference

Complete API reference for HippocampAI v0.2.5. This document covers all public methods in `MemoryClient`, `AsyncMemoryClient`, and `UnifiedMemoryClient`.

## Table of Contents

- [Client Initialization](#client-initialization)
- [Core Memory Operations](#core-memory-operations)
- [Memory Retrieval](#memory-retrieval)
- [Memory Management](#memory-management)
- [Session Management](#session-management)
- [Multi-Agent Support](#multi-agent-support)
- [Intelligence Features](#intelligence-features)
- [Temporal & Scheduling](#temporal--scheduling)
- [Graph & Relationships](#graph--relationships)
- [Version Control & Audit](#version-control--audit)
- [Analytics & Insights](#analytics--insights)
- [Telemetry & Monitoring](#telemetry--monitoring)
- [Background Jobs](#background-jobs)
- [Async Operations](#async-operations)
- [Unified Client](#unified-client)

---

## Client Initialization

### MemoryClient

```python
from hippocampai import MemoryClient

client = MemoryClient(
    qdrant_url: str = "http://localhost:6333",
    redis_url: str = "redis://localhost:6379",
    llm_provider: Optional[LLMProvider] = None,
    config: Optional[Config] = None,
    **kwargs
)
```

**Parameters:**
- `qdrant_url` (str): Qdrant vector database URL. Default: `"http://localhost:6333"`
- `redis_url` (str): Redis cache URL. Default: `"redis://localhost:6379"`
- `llm_provider` (LLMProvider): LLM provider instance (OllamaProvider, OpenAIProvider, etc.)
- `config` (Config): Configuration object. If not provided, loads from environment
- `**kwargs`: Additional config overrides (weights, top_k, etc.)

**Returns:** MemoryClient instance

**Example:**
```python
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaProvider

client = MemoryClient(
    llm_provider=OllamaProvider(base_url="http://localhost:11434"),
    qdrant_url="http://localhost:6333"
)
```

---

### MemoryClient.from_preset()

```python
client = MemoryClient.from_preset(
    preset: PresetType,
    **overrides
)
```

**Parameters:**
- `preset` (PresetType): Preset configuration name. Options: `"local"`, `"cloud"`, `"production"`
- `**overrides`: Override any preset configuration values

**Returns:** MemoryClient instance

**Example:**
```python
# Local development preset
client = MemoryClient.from_preset("local")

# Cloud preset with overrides
client = MemoryClient.from_preset(
    "cloud",
    qdrant_url="https://my-qdrant.cloud"
)
```

---

## Core Memory Operations

### remember()

Store a new memory with metadata. Automatic semantic enrichment and categorization is applied.

```python
memory = client.remember(
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    ttl_days: Optional[int] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    visibility: Optional[str] = None
) -> Memory
```

**Parameters:**
- `text` (str): Memory content to store
- `user_id` (str): User identifier
- `session_id` (str): Session identifier for conversation tracking
- `type` (str): Memory type. Options: `"fact"`, `"preference"`, `"goal"`, `"habit"`, `"event"`, `"context"`
- `importance` (float): Importance score (0-10). If None, auto-calculated
- `tags` (list[str]): Tags for categorization
- `ttl_days` (int): Time-to-live in days. Memory expires after this period
- `agent_id` (str): Agent identifier for multi-agent systems
- `run_id` (str): Run identifier for tracking execution context
- `visibility` (str): Visibility level. Options: `"private"`, `"shared"`, `"public"`

**Returns:** Memory object with generated ID and metadata

**Raises:**
- `ValueError`: If text is empty or user_id is missing
- `QdrantException`: If storage fails

**Example:**
```python
memory = client.remember(
    text="I prefer oat milk in my coffee",
    user_id="alice",
    type="preference",
    importance=7.5,
    tags=["beverages", "preferences"],
    ttl_days=365  # Expires in 1 year
)
print(f"Stored memory: {memory.id}")
print(f"Type: {memory.type}")  # Auto-enriched type
print(f"Importance: {memory.importance}")
```

---

### recall()

Retrieve memories using hybrid search (vector + BM25 + reranking).

```python
results = client.recall(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[dict[str, Any]] = None
) -> list[RetrievalResult]
```

**Parameters:**
- `query` (str): Search query
- `user_id` (str): User identifier
- `session_id` (str): Filter by session (optional)
- `k` (int): Number of results to return. Default: 5
- `filters` (dict): Optional filters dictionary. Can include:
  - `type` (str): Filter by memory type ("fact", "preference", etc.)
  - `tags` (list[str]): Filter by tags
  - `agent_id` (str): Filter by agent
  - `min_importance` (float): Minimum importance threshold
  - `created_after` (datetime): Filter by creation time
  - `created_before` (datetime): Filter by creation time
  - Any other metadata fields

**Returns:** List of RetrievalResult objects with scores and breakdowns

**Example:**
```python
# Simple recall
results = client.recall(
    query="coffee preferences",
    user_id="alice",
    k=3
)

# With filters
results = client.recall(
    query="work schedule",
    user_id="alice",
    k=5,
    filters={
        "type": "preference",
        "tags": ["work"],
        "min_importance": 5.0
    }
)

for result in results:
    print(f"Memory: {result.memory.text}")
    print(f"Score: {result.score:.3f}")
    print(f"Breakdown: {result.breakdown}")
```

---

### update_memory()

Update an existing memory's content or metadata.

```python
updated = client.update_memory(
    memory_id: str,
    text: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    user_id: Optional[str] = None,
    type: Optional[MemoryType] = None
) -> Optional[Memory]
```

**Parameters:**
- `memory_id` (str): Memory ID to update
- `text` (str): New memory text (creates new version)
- `importance` (float): New importance score
- `tags` (list[str]): New tags (replaces existing)
- `metadata` (dict): New metadata (merges with existing)
- `user_id` (str): User ID for validation
- `type` (MemoryType): New memory type

**Returns:** Updated Memory object or None if not found

**Example:**
```python
updated = client.update_memory(
    memory_id="mem_123",
    importance=9.0,
    tags=["beverages", "preferences", "important"]
)
```

---

### delete_memory()

Delete a memory by ID.

```python
success = client.delete_memory(
    memory_id: str,
    user_id: Optional[str] = None
) -> bool
```

**Parameters:**
- `memory_id` (str): Memory ID to delete
- `user_id` (str): User ID for validation (optional)

**Returns:** True if deleted, False if not found

**Example:**
```python
if client.delete_memory("mem_123", user_id="alice"):
    print("Memory deleted successfully")
```

---

### extract_from_conversation()

Extract memories from a conversation using LLM.

```python
memories = client.extract_from_conversation(
    conversation: str,
    user_id: str,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None
) -> list[Memory]
```

**Parameters:**
- `conversation` (str): Conversation text (multi-turn dialogue)
- `user_id` (str): User identifier
- `session_id` (str): Session identifier
- `metadata` (dict): Additional metadata for all extracted memories

**Returns:** List of extracted Memory objects

**Example:**
```python
conversation = """
User: I love dark chocolate.
Assistant: That's great!
User: But I'm allergic to nuts.
"""

memories = client.extract_from_conversation(
    conversation=conversation,
    user_id="alice",
    session_id="session_001"
)
print(f"Extracted {len(memories)} memories")
```

---

## Memory Management

### get_memories()

Retrieve memories with advanced filtering and sorting.

```python
memories = client.get_memories(
    user_id: str,
    type: Optional[MemoryType] = None,
    tags: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",
    order: str = "desc"
) -> list[Memory]
```

**Parameters:**
- `user_id` (str): User identifier
- `type` (MemoryType): Filter by type
- `tags` (list[str]): Filter by tags (AND logic)
- `session_id` (str): Filter by session
- `agent_id` (str): Filter by agent
- `limit` (int): Maximum results. Default: 100
- `offset` (int): Pagination offset. Default: 0
- `sort_by` (str): Sort field. Options: `"created_at"`, `"importance"`, `"accessed_at"`
- `order` (str): Sort order. Options: `"asc"`, `"desc"`

**Returns:** List of Memory objects

**Example:**
```python
# Get top 10 most important preferences
memories = client.get_memories(
    user_id="alice",
    type=MemoryType.PREFERENCE,
    limit=10,
    sort_by="importance",
    order="desc"
)
```

---

### add_memories()

Batch add multiple memories efficiently.

```python
memories = client.add_memories(
    memories_data: list[dict],
    user_id: str
) -> list[Memory]
```

**Parameters:**
- `memories_data` (list[dict]): List of memory dictionaries with keys: `text`, `type`, `importance`, `tags`, `metadata`
- `user_id` (str): User identifier

**Returns:** List of created Memory objects

**Example:**
```python
memories_data = [
    {"text": "I like Python", "tags": ["programming"]},
    {"text": "I prefer dark mode", "type": "preference"},
    {"text": "Learn machine learning", "type": "goal"}
]

memories = client.add_memories(memories_data, user_id="alice")
print(f"Created {len(memories)} memories")
```

---

### delete_memories()

Batch delete multiple memories.

```python
count = client.delete_memories(
    memory_ids: list[str],
    user_id: Optional[str] = None
) -> int
```

**Parameters:**
- `memory_ids` (list[str]): List of memory IDs to delete
- `user_id` (str): User ID for validation

**Returns:** Number of memories deleted

**Example:**
```python
deleted = client.delete_memories(
    ["mem_1", "mem_2", "mem_3"],
    user_id="alice"
)
print(f"Deleted {deleted} memories")
```

---

### expire_memories()

Delete expired memories based on TTL.

```python
count = client.expire_memories(
    user_id: Optional[str] = None
) -> int
```

**Parameters:**
- `user_id` (str): User ID to clean up (None = all users)

**Returns:** Number of expired memories deleted

**Example:**
```python
expired = client.expire_memories(user_id="alice")
print(f"Cleaned up {expired} expired memories")
```

---

### get_memory_statistics()

Get statistics about user's memories.

```python
stats = client.get_memory_statistics(
    user_id: str
) -> dict[str, Any]
```

**Parameters:**
- `user_id` (str): User identifier

**Returns:** Dictionary with statistics:
- `total_memories`: Total count
- `by_type`: Count by memory type
- `by_tag`: Count by tag
- `total_characters`: Total text length
- `total_tokens`: Estimated token count
- `avg_importance`: Average importance score
- `most_recent`: Most recent memory timestamp

**Example:**
```python
stats = client.get_memory_statistics("alice")
print(f"Total memories: {stats['total_memories']}")
print(f"By type: {stats['by_type']}")
print(f"Total tokens: {stats['total_tokens']}")
```

---

## Session Management

### create_session()

Create a new conversation session.

```python
session = client.create_session(
    user_id: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    parent_session_id: Optional[str] = None
) -> Session
```

**Parameters:**
- `user_id` (str): User identifier
- `title` (str): Session title
- `metadata` (dict): Additional metadata
- `parent_session_id` (str): Parent session for hierarchical conversations

**Returns:** Session object

**Example:**
```python
session = client.create_session(
    user_id="alice",
    title="Coffee Preferences Discussion",
    metadata={"channel": "web"}
)
```

---

### get_session()

Retrieve a session by ID.

```python
session = client.get_session(
    session_id: str
) -> Optional[Session]
```

**Parameters:**
- `session_id` (str): Session identifier

**Returns:** Session object or None

---

### update_session()

Update session metadata or title.

```python
session = client.update_session(
    session_id: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None
) -> Optional[Session]
```

**Parameters:**
- `session_id` (str): Session identifier
- `title` (str): New title
- `metadata` (dict): Metadata updates (merges)

**Returns:** Updated Session object

---

### track_session_message()

Track a message in a session.

```python
client.track_session_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None
)
```

**Parameters:**
- `session_id` (str): Session identifier
- `role` (str): Message role (`"user"`, `"assistant"`, `"system"`)
- `content` (str): Message content
- `metadata` (dict): Message metadata

**Example:**
```python
client.track_session_message(
    session_id="sess_123",
    role="user",
    content="What's the weather like?"
)
```

---

### complete_session()

Mark a session as completed and optionally generate summary.

```python
session = client.complete_session(
    session_id: str,
    generate_summary: bool = True
) -> Optional[Session]
```

**Parameters:**
- `session_id` (str): Session identifier
- `generate_summary` (bool): Auto-generate summary. Default: True

**Returns:** Updated Session with summary

---

### summarize_session()

Generate or retrieve session summary.

```python
summary = client.summarize_session(
    session_id: str,
    force: bool = False
) -> Optional[str]
```

**Parameters:**
- `session_id` (str): Session identifier
- `force` (bool): Force regeneration even if summary exists

**Returns:** Session summary text

---

### get_session_memories()

Get all memories associated with a session.

```python
memories = client.get_session_memories(
    session_id: str,
    limit: int = 100
) -> list[Memory]
```

**Parameters:**
- `session_id` (str): Session identifier
- `limit` (int): Maximum results

**Returns:** List of Memory objects

---

### get_user_sessions()

Get all sessions for a user.

```python
sessions = client.get_user_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> list[Session]
```

**Parameters:**
- `user_id` (str): User identifier
- `limit` (int): Maximum results
- `offset` (int): Pagination offset

**Returns:** List of Session objects

---

### get_child_sessions()

Get child sessions in hierarchical conversation.

```python
children = client.get_child_sessions(
    parent_session_id: str
) -> list[Session]
```

**Parameters:**
- `parent_session_id` (str): Parent session ID

**Returns:** List of child Session objects

---

### delete_session()

Delete a session and optionally its memories.

```python
success = client.delete_session(
    session_id: str
) -> bool
```

**Parameters:**
- `session_id` (str): Session identifier

**Returns:** True if deleted

---

## Multi-Agent Support

### check_agent_permission()

Check if an agent has permission for an operation.

```python
has_permission = client.check_agent_permission(
    agent_id: str,
    user_id: str,
    operation: str
) -> bool
```

**Parameters:**
- `agent_id` (str): Agent identifier
- `user_id` (str): User identifier
- `operation` (str): Operation to check (`"read"`, `"write"`, `"delete"`)

**Returns:** True if agent has permission

---

### transfer_memory()

Transfer memory ownership between agents.

```python
client.transfer_memory(
    memory_id: str,
    from_agent_id: str,
    to_agent_id: str,
    user_id: str
)
```

**Parameters:**
- `memory_id` (str): Memory to transfer
- `from_agent_id` (str): Source agent
- `to_agent_id` (str): Destination agent
- `user_id` (str): User identifier

---

### get_agent_memories()

Get memories owned by an agent.

```python
memories = client.get_agent_memories(
    agent_id: str,
    user_id: str,
    limit: int = 100
) -> list[Memory]
```

**Parameters:**
- `agent_id` (str): Agent identifier
- `user_id` (str): User identifier
- `limit` (int): Maximum results

**Returns:** List of Memory objects

---

## Intelligence Features

### extract_facts()

Extract structured facts from text.

```python
from hippocampai import client.extract_facts(
    text: str,
    source: Optional[str] = None,
    confidence_threshold: float = 0.7
) -> list[Fact]
```

**Parameters:**
- `text` (str): Text to analyze
- `source` (str): Source identifier
- `confidence_threshold` (float): Minimum confidence score

**Returns:** List of Fact objects with categories and confidence

**Example:**
```python
facts = client.extract_facts(
    "John works at Google in San Francisco",
    source="profile"
)

for fact in facts:
    print(f"[{fact.category}] {fact.fact} (confidence: {fact.confidence:.2f})")
```

---

### extract_entities()

Extract named entities from text.

```python
entities = client.extract_entities(
    text: str
) -> list[Entity]
```

**Parameters:**
- `text` (str): Text to analyze

**Returns:** List of Entity objects with types (person, organization, location, etc.)

---

### extract_relationships()

Extract relationships between entities.

```python
relationships = client.extract_relationships(
    text: str,
    entities: list[Entity]
) -> list[Relationship]
```

**Parameters:**
- `text` (str): Text to analyze
- `entities` (list[Entity]): Extracted entities

**Returns:** List of Relationship objects

---

### cluster_user_memories()

Automatically cluster memories by semantic similarity.

```python
clusters = client.cluster_user_memories(
    user_id: str,
    max_clusters: int = 10
) -> list[MemoryCluster]
```

**Parameters:**
- `user_id` (str): User identifier
- `max_clusters` (int): Maximum number of clusters

**Returns:** List of MemoryCluster objects with topics and memories

---

### suggest_memory_tags()

Get AI-suggested tags for a memory.

```python
tags = client.suggest_memory_tags(
    memory: Memory,
    max_tags: int = 5
) -> list[str]
```

**Parameters:**
- `memory` (Memory): Memory object
- `max_tags` (int): Maximum tags to suggest

**Returns:** List of suggested tags

---

### detect_topic_shift()

Detect topic changes in conversation.

```python
shift = client.detect_topic_shift(
    user_id: str,
    window_size: int = 10
) -> Optional[TopicShift]
```

**Parameters:**
- `user_id` (str): User identifier
- `window_size` (int): Number of recent memories to analyze

**Returns:** TopicShift object or None

---

## Temporal & Scheduling

### get_memories_by_time_range()

Get memories from a specific time period.

```python
from hippocampai import TimeRange

memories = client.get_memories_by_time_range(
    user_id: str,
    time_range: TimeRange
) -> list[Memory]
```

**Parameters:**
- `user_id` (str): User identifier
- `time_range` (TimeRange): Time period (LAST_HOUR, LAST_DAY, LAST_WEEK, LAST_MONTH, LAST_YEAR)

**Returns:** List of Memory objects

---

### schedule_memory()

Schedule a memory for future activation.

```python
from datetime import datetime, timedelta, timezone

scheduled = client.schedule_memory(
    text: str,
    user_id: str,
    scheduled_for: datetime,
    recurrence: Optional[str] = None
) -> Memory
```

**Parameters:**
- `text` (str): Memory content
- `user_id` (str): User identifier
- `scheduled_for` (datetime): Activation time
- `recurrence` (str): Recurrence pattern (`"daily"`, `"weekly"`, `"monthly"`)

**Example:**
```python
tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
scheduled = client.schedule_memory(
    text="Follow up on proposal",
    user_id="alice",
    scheduled_for=tomorrow,
    recurrence="daily"
)
```

---

### get_due_scheduled_memories()

Get scheduled memories that are due.

```python
memories = client.get_due_scheduled_memories(
    user_id: str
) -> list[Memory]
```

---

### build_memory_narrative()

Build chronological narrative from memories.

```python
narrative = client.build_memory_narrative(
    user_id: str,
    time_range: TimeRange,
    title: Optional[str] = None
) -> str
```

**Returns:** Formatted narrative text

---

### create_memory_timeline()

Create timeline of events.

```python
timeline = client.create_memory_timeline(
    user_id: str,
    time_range: TimeRange,
    title: Optional[str] = None
) -> Timeline
```

**Returns:** Timeline object with events

---

## Graph & Relationships

### add_relationship()

Add relationship between two memories.

```python
from hippocampai import RelationType

client.add_relationship(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    metadata: Optional[dict] = None
)
```

**Parameters:**
- `source_id` (str): Source memory ID
- `target_id` (str): Target memory ID
- `relation_type` (RelationType): Relationship type
- `metadata` (dict): Additional metadata

---

### get_related_memories()

Get memories related to a specific memory.

```python
related = client.get_related_memories(
    memory_id: str,
    relation_type: Optional[RelationType] = None,
    max_depth: int = 1
) -> list[Memory]
```

---

### export_graph_to_json()

Export knowledge graph to JSON file.

```python
stats = client.export_graph_to_json(
    file_path: str,
    user_id: Optional[str] = None
) -> dict
```

**Returns:** Statistics about exported graph

---

### import_graph_from_json()

Import knowledge graph from JSON file.

```python
stats = client.import_graph_from_json(
    file_path: str,
    merge: bool = True
) -> dict
```

**Parameters:**
- `file_path` (str): Path to JSON file
- `merge` (bool): Merge with existing graph vs replace

---

## Version Control & Audit

### get_memory_history()

Get version history of a memory.

```python
history = client.get_memory_history(
    memory_id: str
) -> list[MemoryVersion]
```

**Returns:** List of MemoryVersion objects

---

### rollback_memory()

Rollback memory to a previous version.

```python
memory = client.rollback_memory(
    memory_id: str,
    version_number: int
) -> Optional[Memory]
```

---

### get_audit_trail()

Get audit trail for user or memory.

```python
trail = client.get_audit_trail(
    user_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    limit: int = 100
) -> list[AuditEntry]
```

---

### create_snapshot()

Create backup snapshot of collection.

```python
snapshot_name = client.create_snapshot(
    collection: str = "facts"
) -> str
```

**Returns:** Snapshot identifier

---

## Analytics & Insights

### detect_patterns()

Detect behavioral patterns across sessions.

```python
patterns = client.detect_patterns(
    user_id: str,
    min_occurrences: int = 3
) -> list[Pattern]
```

**Returns:** List of detected Pattern objects

---

### track_behavior_changes()

Track changes in user behavior over time.

```python
changes = client.track_behavior_changes(
    user_id: str,
    comparison_days: int = 30
) -> list[BehaviorChange]
```

---

### analyze_preference_drift()

Analyze how preferences change over time.

```python
drifts = client.analyze_preference_drift(
    user_id: str
) -> list[PreferenceDrift]
```

---

### detect_habits()

Detect habit formation and consistency.

```python
habits = client.detect_habits(
    user_id: str,
    min_occurrences: int = 5
) -> list[Habit]
```

---

### analyze_trends()

Analyze long-term trends in memories.

```python
trends = client.analyze_trends(
    user_id: str,
    window_days: int = 30
) -> list[Trend]
```

---

## Telemetry & Monitoring

### get_telemetry_metrics()

Get performance metrics.

```python
metrics = client.get_telemetry_metrics() -> dict[str, Any]
```

**Returns:** Dictionary with metrics:
- `recall_duration`: Recall latency stats
- `remember_duration`: Store latency stats
- `memory_size_chars`: Memory size stats
- `cache_hit_rate`: Cache performance

---

### get_recent_operations()

Get recent operations log.

```python
operations = client.get_recent_operations(
    limit: int = 10,
    operation: Optional[str] = None
) -> list[Operation]
```

---

## Background Jobs

### start_scheduler()

Start background job scheduler.

```python
client.start_scheduler()
```

---

### stop_scheduler()

Stop background job scheduler.

```python
client.stop_scheduler()
```

---

### get_scheduler_status()

Get scheduler status and job info.

```python
status = client.get_scheduler_status() -> dict
```

---

### consolidate_all_memories()

Manually trigger memory consolidation.

```python
count = client.consolidate_all_memories(
    similarity_threshold: float = 0.85
) -> int
```

**Returns:** Number of memories consolidated

---

### apply_importance_decay()

Manually apply importance decay.

```python
count = client.apply_importance_decay() -> int
```

**Returns:** Number of memories decayed

---

## Async Operations

`AsyncMemoryClient` provides async variants for **core memory operations only**:

```python
from hippocampai import AsyncMemoryClient

async_client = AsyncMemoryClient()

# Available async methods (core operations only)
await async_client.remember_async(text, user_id)
await async_client.recall_async(query, user_id)
await async_client.update_memory_async(memory_id, ...)
await async_client.delete_memory_async(memory_id)
await async_client.get_memories_async(user_id)
await async_client.extract_from_conversation_async(conversation, user_id)
await async_client.add_memories_async(memories_data, user_id)
await async_client.delete_memories_async(memory_ids)
await async_client.get_memory_statistics_async(user_id)
await async_client.inject_context_async(prompt, query, user_id)
```

**⚠️ Note:** AsyncMemoryClient only supports the core methods listed above. Advanced features (sessions, intelligence, temporal, analytics, graphs) are **NOT available** in async mode. Use the synchronous `MemoryClient` for full functionality

---

## Unified Client

Use `UnifiedMemoryClient` for basic operations with mode switching between local and remote:

```python
from hippocampai import UnifiedMemoryClient

# Local mode
client = UnifiedMemoryClient(mode="local")

# Remote mode
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

# Available methods (basic operations only)
memory = client.remember(text, user_id)
results = client.recall(query, user_id)
memory = client.get_memory(memory_id)
memories = client.get_memories(user_id, filters)
client.update_memory(memory_id, updates)
client.delete_memory(memory_id)
memories = client.batch_remember(texts, user_id)
memories = client.batch_get_memories(user_ids)
client.batch_delete_memories(memory_ids)
count = client.consolidate_memories(user_id)
count = client.cleanup_expired_memories(user_id)
analytics = client.get_memory_analytics(user_id)
health = client.health_check()
```

**⚠️ Note:** UnifiedMemoryClient provides a **limited subset of functionality** for basic operations. Advanced features (sessions, intelligence, temporal reasoning, analytics, graphs, version control) are **NOT available**. For full functionality, use `MemoryClient` directly

---

## Error Handling

All methods may raise:

- `ValueError`: Invalid parameters
- `QdrantException`: Vector database errors
- `RedisError`: Cache errors
- `LLMError`: LLM provider errors
- `AuthenticationError`: Authentication failures

**Example:**
```python
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    memory = client.remember("text", user_id="alice")
except ValueError as e:
    print(f"Invalid input: {e}")
except UnexpectedResponse as e:
    print(f"Database error: {e}")
```

---

## Type Definitions

### Memory

```python
@dataclass
class Memory:
    id: str
    text: str
    user_id: str
    type: MemoryType
    importance: float
    tags: list[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    access_count: int
    session_id: Optional[str]
    agent_id: Optional[str]
    text_length: int
    token_count: int
    extracted_facts: Optional[list]
    extracted_entities: Optional[dict]
```

### MemoryResult

```python
@dataclass
class MemoryResult:
    memory: Memory
    score: float
    breakdown: dict  # {"sim": float, "rerank": float, "recency": float, "importance": float}
```

### Session

```python
@dataclass
class Session:
    session_id: str
    user_id: str
    title: Optional[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    parent_session_id: Optional[str]
    summary: Optional[str]
    message_count: int
```

### MemoryType (Enum)

```python
class MemoryType(Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"
```

### TimeRange (Enum)

```python
class TimeRange(Enum):
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
```

### RelationType (Enum)

```python
class RelationType(Enum):
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
```

---

## Configuration Reference

See [CONFIGURATION.md](CONFIGURATION.md) for complete configuration options.

---

## Further Reading

- **[Getting Started](GETTING_STARTED.md)** - Setup and basic usage
- **[Features Guide](FEATURES.md)** - Detailed feature documentation
- **[User Guide](USER_GUIDE.md)** - Production deployment
- **[Examples](../examples/)** - 25+ working code examples

---

**Version:** v0.2.5
**Last Updated:** November 2025
