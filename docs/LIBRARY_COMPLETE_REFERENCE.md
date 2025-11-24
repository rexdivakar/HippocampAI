# HippocampAI Library Complete Function Reference

**Version**: v0.3.0
**Last Updated**: 2025-11-24
**Total Functions**: 102+

Complete reference for all HippocampAI library functions with usage examples.

---

## 📋 Table of Contents

1. [Initialization & Configuration](#initialization--configuration)
2. [Core Memory Operations](#core-memory-operations)
3. [Memory Retrieval & Search](#memory-retrieval--search)
4. [Batch Operations](#batch-operations)
5. [Memory Management](#memory-management)
6. [Graph & Relationships](#graph--relationships)
7. [Version Control & Audit](#version-control--audit)
8. [Session Management](#session-management)
9. [Intelligence Features](#intelligence-features)
10. [Temporal Analytics](#temporal-analytics)
11. [Cross-Session Insights](#cross-session-insights)
12. [Semantic Clustering](#semantic-clustering)
13. [Multi-Agent Features](#multi-agent-features)
14. [Scheduling & Automation](#scheduling--automation)
15. [Telemetry & Monitoring](#telemetry--monitoring)
16. [Utility Functions](#utility-functions)

---

## Quick Reference Table

| Category | Method Count | Key Functions |
|----------|--------------|---------------|
| Core Operations | 8 | remember, recall, extract_from_conversation |
| Memory Management | 12 | update_memory, delete_memory, get_memories |
| Session Management | 15 | create_session, track_session_message, summarize_session |
| Intelligence Features | 18 | extract_facts, extract_entities, enrich_memory_with_intelligence |
| Temporal Analytics | 10 | get_memories_by_time_range, build_memory_narrative, schedule_memory |
| Cross-Session Insights | 8 | detect_patterns, track_behavior_changes, detect_habits |
| Semantic Clustering | 6 | cluster_user_memories, suggest_memory_tags, detect_topic_shift |
| Graph & Relationships | 8 | add_relationship, get_related_memories, export_graph_to_json |
| Multi-Agent | 7 | create_agent_space, transfer_memory_between_agents |
| Version Control | 6 | get_memory_history, rollback_memory, create_snapshot |
| Telemetry | 4 | get_telemetry_metrics, get_recent_operations |

---

## 🚀 Initialization & Configuration

### MemoryClient(...)

Initialize the memory client.

**Signature:**

```python
def __init__(
    self,
    qdrant_url: str = "http://localhost:6333",
    redis_url: str = "redis://localhost:6379",
    llm_provider: Optional[BaseLLM] = None,
    config: Optional[Config] = None,
    **kwargs
)
```

**Parameters:**

- `qdrant_url` (str): Qdrant vector database URL
- `redis_url` (str): Redis cache URL
- `llm_provider` (BaseLLM): LLM provider instance (OllamaLLM, GroqLLM, etc.)
- `config` (Config): Configuration object
- `**kwargs`: Additional config overrides (weights, top_k, etc.)

**Returns:** MemoryClient instance

**Example:**

```python
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaLLM

# Basic initialization
client = MemoryClient()

# With LLM provider
client = MemoryClient(
    llm_provider=OllamaLLM(base_url="http://localhost:11434")
)

# With full configuration
client = MemoryClient(
    qdrant_url="http://localhost:6333",
    redis_url="redis://localhost:6379",
    llm_provider=OllamaLLM(),
    weights={"sim": 0.6, "rerank": 0.2, "recency": 0.1, "importance": 0.1}
)
```

---

### MemoryClient.from_preset(...)

Create client from configuration preset.

**Signature:**

```python
@classmethod
def from_preset(
    cls,
    preset: PresetType,
    **overrides
) -> "MemoryClient"
```

**Parameters:**

- `preset` (PresetType): "local", "cloud", "performance", or "quality"
- `**overrides`: Override specific settings

**Returns:** Configured MemoryClient

**Example:**

```python
# Local development preset
client = MemoryClient.from_preset("local")

# Cloud preset with overrides
client = MemoryClient.from_preset(
    "cloud",
    llm_provider=GroqLLM(api_key="your-key")
)

# Performance-optimized preset
client = MemoryClient.from_preset("performance")
```

---

## 💾 Core Memory Operations

### remember(...)

Store a new memory.

**Signature:**

```python
def remember(
    self,
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    ttl_days: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None
) -> Memory
```

**Parameters:**

- `text` (str): Memory content
- `user_id` (str): User identifier
- `session_id` (str, optional): Session identifier
- `type` (str): Memory type - "fact", "preference", "goal", "habit", "event"
- `importance` (float, optional): Importance score 0-10
- `tags` (list[str], optional): Memory tags
- `ttl_days` (int, optional): Time-to-live in days
- `metadata` (dict, optional): Additional metadata

**Returns:** Memory object with ID, timestamps, extracted facts

**Example:**

```python
# Basic memory
memory = client.remember(
    text="I prefer oat milk in my coffee",
    user_id="alice"
)

# Advanced memory with all options
memory = client.remember(
    text="Complete the project proposal by Friday",
    user_id="alice",
    session_id="work_session_123",
    type="goal",
    importance=9.0,
    tags=["work", "deadline", "project"],
    ttl_days=7,
    metadata={"project": "Q4_planning", "priority": "high"}
)

print(f"Stored memory: {memory.id}")
print(f"Extracted facts: {memory.extracted_facts}")
print(f"Memory size: {memory.text_length} chars, {memory.token_count} tokens")
```

---

### recall(...)

Retrieve relevant memories using hybrid search.

**Signature:**

```python
def recall(
    self,
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[dict[str, Any]] = None,
    min_score: float = 0.0,
    include_scores: bool = True
) -> list[RetrievalResult]
```

**Parameters:**

- `query` (str): Search query
- `user_id` (str): User identifier
- `session_id` (str, optional): Filter by session
- `k` (int): Number of results (default: 5)
- `filters` (dict, optional): Additional filters
- `min_score` (float): Minimum relevance score
- `include_scores` (bool): Include score breakdown

**Returns:** List of RetrievalResult with memory and scores

**Example:**

```python
# Basic recall
results = client.recall(
    query="coffee preferences",
    user_id="alice",
    k=5
)

for result in results:
    print(f"Memory: {result.memory.text}")
    print(f"Score: {result.score:.3f}")
    print(f"Breakdown: {result.breakdown}")

# Advanced recall with filters
results = client.recall(
    query="work deadlines",
    user_id="alice",
    k=10,
    filters={
        "type": "goal",
        "tags": ["work"],
        "min_importance": 7.0
    },
    min_score=0.6
)

# Access score components
for result in results:
    breakdown = result.breakdown
    print(f"Similarity: {breakdown['sim']:.2f}")
    print(f"Rerank: {breakdown['rerank']:.2f}")
    print(f"Recency: {breakdown['recency']:.2f}")
    print(f"Importance: {breakdown['importance']:.2f}")
```

---

### extract_from_conversation(...)

Extract memories from conversation text.

**Signature:**

```python
def extract_from_conversation(
    self,
    conversation: str,
    user_id: str,
    session_id: Optional[str] = None,
    extract_method: str = "llm"  # "llm" or "heuristic"
) -> list[Memory]
```

**Parameters:**

- `conversation` (str): Conversation text
- `user_id` (str): User identifier
- `session_id` (str, optional): Session identifier
- `extract_method` (str): Extraction method

**Returns:** List of extracted Memory objects

**Example:**

```python
conversation = """
User: I really enjoy drinking green tea in the morning.
Assistant: That's great! Green tea is healthy.
User: Yes, and I usually have it without sugar.
User: I also like to exercise at 6 AM before work.
"""

memories = client.extract_from_conversation(
    conversation=conversation,
    user_id="bob",
    session_id="chat_001",
    extract_method="llm"
)

print(f"Extracted {len(memories)} memories:")
for mem in memories:
    print(f"- {mem.text} (type: {mem.type.value})")

# Output:
# - User enjoys green tea in the morning (type: preference)
# - User prefers tea without sugar (type: preference)
# - User exercises at 6 AM before work (type: habit)
```

---

## 🔍 Memory Retrieval & Search

### get_memories(...)

Get memories with advanced filtering.

**Signature:**

```python
def get_memories(
    self,
    user_id: str,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",  # "created_at", "importance", "updated_at"
    sort_order: str = "desc"       # "asc" or "desc"
) -> list[Memory]
```

**Parameters:**

- `user_id` (str): User identifier
- `filters` (dict, optional): Filter conditions
- `limit` (int): Maximum results
- `offset` (int): Pagination offset
- `sort_by` (str): Sort field
- `sort_order` (str): Sort direction

**Returns:** List of Memory objects

**Example:**

```python
# Get all preferences
memories = client.get_memories(
    user_id="alice",
    filters={"type": "preference"}
)

# Advanced filtering
memories = client.get_memories(
    user_id="alice",
    filters={
        "type": "goal",
        "tags": ["work"],
        "min_importance": 8.0,
        "created_after": "2025-11-01T00:00:00Z",
        "session_id": "work_session_123"
    },
    limit=50,
    sort_by="importance",
    sort_order="desc"
)

# Pagination
page_1 = client.get_memories(user_id="alice", limit=10, offset=0)
page_2 = client.get_memories(user_id="alice", limit=10, offset=10)
```

---

### get_memories_advanced(...)

Advanced memory search with complex queries.

**Signature:**

```python
def get_memories_advanced(
    self,
    user_id: str,
    query: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
    semantic_threshold: float = 0.7,
    combine_mode: str = "and"  # "and" or "or"
) -> list[Memory]
```

**Example:**

```python
memories = client.get_memories_advanced(
    user_id="alice",
    query="programming languages",
    filters={"type": "fact", "tags": ["tech"]},
    semantic_threshold=0.75,
    combine_mode="and"
)
```

---

### get_memory_statistics(...)

Get comprehensive memory statistics.

**Signature:**

```python
def get_memory_statistics(
    self,
    user_id: str
) -> dict[str, Any]
```

**Returns:** Statistics including counts, sizes, types

**Example:**

```python
stats = client.get_memory_statistics(user_id="alice")

print(f"Total memories: {stats['total_memories']}")
print(f"Total size: {stats['total_characters']} chars")
print(f"Total tokens: {stats['total_tokens']}")
print(f"By type: {stats['by_type']}")
print(f"By tags: {stats['by_tags']}")
print(f"Average importance: {stats['avg_importance']:.2f}")

# Example output:
# {
#     "total_memories": 150,
#     "total_characters": 45000,
#     "total_tokens": 12000,
#     "by_type": {
#         "fact": {"count": 80, "chars": 24000, "tokens": 6400},
#         "preference": {"count": 50, "chars": 15000, "tokens": 4000},
#         "goal": {"count": 20, "chars": 6000, "tokens": 1600}
#     },
#     "by_tags": {
#         "work": 45,
#         "personal": 35,
#         "tech": 30
#     },
#     "avg_importance": 7.2
# }
```

---

## ✏️ Memory Management

### update_memory(...)

Update an existing memory.

**Signature:**

```python
def update_memory(
    self,
    memory_id: str,
    text: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    expires_at: Optional[datetime] = None
) -> Optional[Memory]
```

**Parameters:**

- `memory_id` (str): Memory identifier
- `text` (str, optional): New text content
- `importance` (float, optional): New importance score
- `tags` (list[str], optional): New tags
- `metadata` (dict, optional): Updated metadata
- `expires_at` (datetime, optional): New expiration time

**Returns:** Updated Memory or None if not found

**Example:**

```python
# Update text
updated = client.update_memory(
    memory_id="mem_abc123",
    text="I strongly prefer oat milk in my coffee"
)

# Update importance and tags
updated = client.update_memory(
    memory_id="mem_abc123",
    importance=9.5,
    tags=["food", "beverages", "health", "dairy-free"]
)

# Set expiration
from datetime import datetime, timedelta, timezone
expires = datetime.now(timezone.utc) + timedelta(days=30)
updated = client.update_memory(
    memory_id="mem_abc123",
    expires_at=expires
)
```

---

### delete_memory(...)

Delete a memory.

**Signature:**

```python
def delete_memory(
    self,
    memory_id: str,
    user_id: Optional[str] = None
) -> bool
```

**Parameters:**

- `memory_id` (str): Memory identifier
- `user_id` (str, optional): User ID for authorization

**Returns:** True if deleted, False if not found

**Example:**

```python
deleted = client.delete_memory(memory_id="mem_abc123", user_id="alice")
if deleted:
    print("Memory deleted successfully")
```

---

### expire_memories(...)

Clean up expired memories.

**Signature:**

```python
def expire_memories(
    self,
    user_id: Optional[str] = None
) -> int
```

**Parameters:**

- `user_id` (str, optional): User ID, if None expires all users

**Returns:** Count of expired memories

**Example:**

```python
# Expire for specific user
count = client.expire_memories(user_id="alice")
print(f"Expired {count} memories for alice")

# Expire for all users
count = client.expire_memories()
print(f"Expired {count} total memories")
```

---

## 📦 Batch Operations

### add_memories(...)

Batch create multiple memories.

**Signature:**

```python
def add_memories(
    self,
    memories_data: list[dict[str, Any]],
    user_id: str,
    check_duplicates: bool = True
) -> list[Memory]
```

**Parameters:**

- `memories_data` (list[dict]): List of memory data
- `user_id` (str): User identifier
- `check_duplicates` (bool): Check for duplicates

**Returns:** List of created Memory objects

**Example:**

```python
memories_data = [
    {"text": "I love Python", "type": "fact", "tags": ["programming"]},
    {"text": "I prefer dark mode", "type": "preference", "tags": ["UI"]},
    {"text": "Learn Rust by end of year", "type": "goal", "importance": 8.0}
]

created = client.add_memories(
    memories_data=memories_data,
    user_id="alice",
    check_duplicates=True
)

print(f"Created {len(created)} memories")
for mem in created:
    print(f"- {mem.id}: {mem.text}")
```

---

### delete_memories(...)

Batch delete multiple memories.

**Signature:**

```python
def delete_memories(
    self,
    memory_ids: list[str],
    user_id: Optional[str] = None
) -> int
```

**Parameters:**

- `memory_ids` (list[str]): List of memory IDs
- `user_id` (str, optional): User ID for authorization

**Returns:** Count of deleted memories

**Example:**

```python
memory_ids = ["mem_abc123", "mem_def456", "mem_ghi789"]
deleted_count = client.delete_memories(
    memory_ids=memory_ids,
    user_id="alice"
)
print(f"Deleted {deleted_count} memories")
```

---

## 🕸️ Graph & Relationships

### add_relationship(...)

Add relationship between memories.

**Signature:**

```python
def add_relationship(
    self,
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    strength: float = 1.0,
    metadata: Optional[dict[str, Any]] = None
) -> bool
```

**Parameters:**

- `source_id` (str): Source memory ID
- `target_id` (str): Target memory ID
- `relation_type` (RelationType): Relationship type
- `strength` (float): Relationship strength 0-1
- `metadata` (dict, optional): Additional metadata

**Returns:** True if added successfully

**Example:**

```python
from hippocampai.models.memory import RelationType

# Add relationship
client.add_relationship(
    source_id="mem_abc123",
    target_id="mem_def456",
    relation_type=RelationType.RELATED_TO,
    strength=0.85,
    metadata={"reason": "same_topic"}
)

# Supported relationship types:
# - RelationType.RELATED_TO
# - RelationType.CAUSES
# - RelationType.CONTRADICTS
# - RelationType.SUPPORTS
# - RelationType.TEMPORAL_BEFORE
# - RelationType.TEMPORAL_AFTER
```

---

### get_related_memories(...)

Get memories related to a memory.

**Signature:**

```python
def get_related_memories(
    self,
    memory_id: str,
    relation_type: Optional[RelationType] = None,
    min_strength: float = 0.0,
    max_depth: int = 1
) -> list[Memory]
```

**Parameters:**

- `memory_id` (str): Source memory ID
- `relation_type` (RelationType, optional): Filter by type
- `min_strength` (float): Minimum relationship strength
- `max_depth` (int): Graph traversal depth

**Returns:** List of related Memory objects

**Example:**

```python
# Get directly related memories
related = client.get_related_memories(
    memory_id="mem_abc123",
    min_strength=0.7
)

# Get memories with specific relationship
contradicting = client.get_related_memories(
    memory_id="mem_abc123",
    relation_type=RelationType.CONTRADICTS
)

# Traverse relationship graph (depth 2)
extended_network = client.get_related_memories(
    memory_id="mem_abc123",
    max_depth=2,
    min_strength=0.6
)
```

---

### export_graph_to_json(...)

Export memory graph to JSON file.

**Signature:**

```python
def export_graph_to_json(
    self,
    file_path: str,
    user_id: Optional[str] = None,
    include_vectors: bool = False
) -> dict[str, int]
```

**Parameters:**

- `file_path` (str): Output file path
- `user_id` (str, optional): Export for specific user
- `include_vectors` (bool): Include vector embeddings

**Returns:** Export statistics

**Example:**

```python
stats = client.export_graph_to_json(
    file_path="memory_graph_alice.json",
    user_id="alice",
    include_vectors=False
)

print(f"Exported {stats['memories']} memories")
print(f"Exported {stats['relationships']} relationships")
```

---

### import_graph_from_json(...)

Import memory graph from JSON file.

**Signature:**

```python
def import_graph_from_json(
    self,
    file_path: str,
    merge: bool = True
) -> dict[str, int]
```

**Parameters:**

- `file_path` (str): Input file path
- `merge` (bool): Merge with existing data

**Returns:** Import statistics

**Example:**

```python
stats = client.import_graph_from_json(
    file_path="memory_graph_alice.json",
    merge=True
)

print(f"Imported {stats['memories']} memories")
print(f"Imported {stats['relationships']} relationships")
```

---

## 📜 Version Control & Audit

### get_memory_history(...)

Get version history of a memory.

**Signature:**

```python
def get_memory_history(
    self,
    memory_id: str
) -> list[dict[str, Any]]
```

**Parameters:**

- `memory_id` (str): Memory identifier

**Returns:** List of versions with diffs

**Example:**

```python
history = client.get_memory_history(memory_id="mem_abc123")

for version in history:
    print(f"Version {version['version']}")
    print(f"Timestamp: {version['timestamp']}")
    print(f"Changes: {version['changes']}")
    print(f"Changed by: {version['changed_by']}")
```

---

### rollback_memory(...)

Rollback memory to previous version.

**Signature:**

```python
def rollback_memory(
    self,
    memory_id: str,
    version_number: int
) -> Optional[Memory]
```

**Parameters:**

- `memory_id` (str): Memory identifier
- `version_number` (int): Target version number

**Returns:** Rolled back Memory or None

**Example:**

```python
# Rollback to version 2
memory = client.rollback_memory(
    memory_id="mem_abc123",
    version_number=2
)

if memory:
    print(f"Rolled back to: {memory.text}")
```

---

### get_audit_trail(...)

Get audit trail for user or memory.

**Signature:**

```python
def get_audit_trail(
    self,
    user_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    limit: int = 100
) -> list[dict[str, Any]]
```

**Parameters:**

- `user_id` (str, optional): Filter by user
- `memory_id` (str, optional): Filter by memory
- `limit` (int): Maximum results

**Returns:** List of audit log entries

**Example:**

```python
# Get user audit trail
audit = client.get_audit_trail(user_id="alice", limit=50)

for entry in audit:
    print(f"{entry['timestamp']}: {entry['operation']}")
    print(f"Memory: {entry['memory_id']}")
    print(f"Details: {entry['details']}")

# Get memory-specific audit trail
audit = client.get_audit_trail(memory_id="mem_abc123")
```

---

### create_snapshot(...)

Create snapshot of collection.

**Signature:**

```python
def create_snapshot(
    self,
    collection: str = "facts"
) -> str
```

**Parameters:**

- `collection` (str): Collection name

**Returns:** Snapshot ID

**Example:**

```python
snapshot_id = client.create_snapshot(collection="facts")
print(f"Created snapshot: {snapshot_id}")
```

---

## 🗂️ Session Management

### create_session(...)

Create a new conversation session.

**Signature:**

```python
def create_session(
    self,
    user_id: str,
    session_id: Optional[str] = None,
    title: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None
) -> Session
```

**Parameters:**

- `user_id` (str): User identifier
- `session_id` (str, optional): Custom session ID
- `title` (str, optional): Session title
- `parent_session_id` (str, optional): Parent session for hierarchical sessions
- `metadata` (dict, optional): Session metadata

**Returns:** Session object

**Example:**

```python
# Basic session
session = client.create_session(
    user_id="alice",
    title="Project Planning Discussion"
)

# Hierarchical session
child_session = client.create_session(
    user_id="alice",
    title="Budget Details",
    parent_session_id=session.id,
    metadata={"project": "Q4_planning"}
)

print(f"Created session: {session.id}")
```

---

### track_session_message(...)

Track a message in a session.

**Signature:**

```python
def track_session_message(
    self,
    session_id: str,
    role: str,  # "user" or "assistant"
    content: str,
    metadata: Optional[dict[str, Any]] = None
) -> None
```

**Example:**

```python
client.track_session_message(
    session_id=session.id,
    role="user",
    content="What's the status of the project?"
)

client.track_session_message(
    session_id=session.id,
    role="assistant",
    content="The project is 75% complete and on track."
)
```

---

### complete_session(...)

Mark session as complete and generate summary.

**Signature:**

```python
def complete_session(
    self,
    session_id: str,
    generate_summary: bool = True
) -> Optional[Session]
```

**Parameters:**

- `session_id` (str): Session identifier
- `generate_summary` (bool): Auto-generate summary

**Returns:** Updated Session with summary

**Example:**

```python
session = client.complete_session(
    session_id="session_123",
    generate_summary=True
)

print(f"Session Summary: {session.summary}")
print(f"Key Topics: {session.topics}")
print(f"Message Count: {session.message_count}")
```

---

### summarize_session(...)

Generate summary for a session.

**Signature:**

```python
def summarize_session(
    self,
    session_id: str,
    force: bool = False
) -> Optional[str]
```

**Parameters:**

- `session_id` (str): Session identifier
- `force` (bool): Regenerate even if exists

**Returns:** Session summary

**Example:**

```python
summary = client.summarize_session(session_id="session_123")
print(f"Summary: {summary}")
```

---

### get_session_memories(...)

Get all memories associated with a session.

**Signature:**

```python
def get_session_memories(
    self,
    session_id: str,
    limit: int = 100
) -> list[Memory]
```

**Example:**

```python
memories = client.get_session_memories(session_id="session_123")
print(f"Found {len(memories)} memories in session")
```

---

### search_sessions(...)

Search sessions by query.

**Signature:**

```python
def search_sessions(
    self,
    query: str,
    user_id: str,
    limit: int = 10
) -> list[Session]
```

**Example:**

```python
sessions = client.search_sessions(
    query="project planning",
    user_id="alice",
    limit=5
)
```

---

## 🧠 Intelligence Features

### extract_facts(...)

Extract structured facts from text.

**Signature:**

```python
def extract_facts(
    self,
    text: str,
    source: str = "manual",
    user_id: Optional[str] = None
) -> list[Fact]
```

**Parameters:**

- `text` (str): Text to analyze
- `source` (str): Source identifier
- `user_id` (str, optional): User context

**Returns:** List of extracted facts with confidence scores

**Example:**

```python
facts = client.extract_facts(
    text="John works at Google in San Francisco. He studied CS at MIT and knows Python.",
    source="profile",
    user_id="john"
)

for fact in facts:
    print(f"[{fact.category.value}] {fact.fact}")
    print(f"Confidence: {fact.confidence:.2f}")
    print(f"Quality: {fact.quality_score:.2f}")

# Output:
# [employment] John works at Google
# Confidence: 0.95
# [location] Work location: San Francisco
# Confidence: 0.90
# [education] MIT Computer Science
# Confidence: 0.88
# [skills] Python
# Confidence: 0.93
```

---

### extract_entities(...)

Extract named entities from text.

**Signature:**

```python
def extract_entities(
    self,
    text: str,
    context: Optional[dict[str, Any]] = None
) -> list[Entity]
```

**Parameters:**

- `text` (str): Text to analyze
- `context` (dict, optional): Additional context

**Returns:** List of Entity objects

**Example:**

```python
entities = client.extract_entities(
    text="Elon Musk founded SpaceX in California. Tesla produces electric vehicles.",
    context={"domain": "technology"}
)

for entity in entities:
    print(f"{entity.canonical_name} ({entity.entity_type.value})")
    print(f"Confidence: {entity.confidence:.2f}")

# Output:
# Elon Musk (person)
# SpaceX (organization)
# California (location)
# Tesla (organization)
```

---

### extract_relationships(...)

Extract relationships between entities.

**Signature:**

```python
def extract_relationships(
    self,
    text: str,
    entities: list[Entity]
) -> list[EntityRelationship]
```

**Example:**

```python
entities = client.extract_entities(text)
relationships = client.extract_relationships(text, entities)

for rel in relationships:
    print(f"{rel.source_entity} --[{rel.relation_type.value}]--> {rel.target_entity}")
    print(f"Strength: {rel.strength:.2f}")

# Output:
# Elon Musk --[founded]--> SpaceX
# Strength: 0.95
```

---

### enrich_memory_with_intelligence(...)

Enrich memory with all intelligence features.

**Signature:**

```python
def enrich_memory_with_intelligence(
    self,
    memory: Memory,
    add_to_graph: bool = True
) -> dict[str, Any]
```

**Parameters:**

- `memory` (Memory): Memory to enrich
- `add_to_graph` (bool): Add entities to knowledge graph

**Returns:** Enrichment data with facts, entities, relationships

**Example:**

```python
memory = client.remember(
    "Marie Curie was a physicist who won two Nobel Prizes",
    user_id="alice"
)

enrichment = client.enrich_memory_with_intelligence(
    memory=memory,
    add_to_graph=True
)

print(f"Facts: {enrichment['facts']}")
print(f"Entities: {enrichment['entities']}")
print(f"Relationships: {enrichment['relationships']}")
```

---

### infer_knowledge(...)

Infer new knowledge from existing patterns.

**Signature:**

```python
def infer_knowledge(
    self,
    user_id: str,
    min_confidence: float = 0.7
) -> list[dict[str, Any]]
```

**Parameters:**

- `user_id` (str): User identifier
- `min_confidence` (float): Minimum confidence threshold

**Returns:** List of inferred facts

**Example:**

```python
inferred = client.infer_knowledge(user_id="alice", min_confidence=0.75)

for fact in inferred:
    print(f"{fact['fact']} (confidence: {fact['confidence']:.2f})")
    print(f"Evidence: {fact['evidence']}")
```

---

## ⏰ Temporal Analytics

### get_memories_by_time_range(...)

Get memories within time range.

**Signature:**

```python
def get_memories_by_time_range(
    self,
    user_id: str,
    time_range: TimeRange,
    custom_start: Optional[datetime] = None,
    custom_end: Optional[datetime] = None
) -> list[Memory]
```

**Parameters:**

- `user_id` (str): User identifier
- `time_range` (TimeRange): Predefined range (LAST_HOUR, LAST_DAY, LAST_WEEK, LAST_MONTH, CUSTOM)
- `custom_start` (datetime, optional): Custom start time
- `custom_end` (datetime, optional): Custom end time

**Returns:** List of Memory objects in time range

**Example:**

```python
from hippocampai.models.temporal import TimeRange
from datetime import datetime, timedelta, timezone

# Last week's memories
memories = client.get_memories_by_time_range(
    user_id="alice",
    time_range=TimeRange.LAST_WEEK
)

# Custom time range
start = datetime.now(timezone.utc) - timedelta(days=7)
end = datetime.now(timezone.utc)
memories = client.get_memories_by_time_range(
    user_id="alice",
    time_range=TimeRange.CUSTOM,
    custom_start=start,
    custom_end=end
)
```

---

### build_memory_narrative(...)

Build chronological narrative from memories.

**Signature:**

```python
def build_memory_narrative(
    self,
    user_id: str,
    time_range: TimeRange,
    title: str = "Memory Timeline",
    style: str = "paragraph"  # "paragraph" or "bullet_points"
) -> str
```

**Parameters:**

- `user_id` (str): User identifier
- `time_range` (TimeRange): Time range
- `title` (str): Narrative title
- `style` (str): Output style

**Returns:** Formatted narrative string

**Example:**

```python
narrative = client.build_memory_narrative(
    user_id="alice",
    time_range=TimeRange.LAST_MONTH,
    title="My Month in Review",
    style="paragraph"
)

print(narrative)

# Output:
# My Month in Review
# =================
#
# During this month, you focused on completing the project proposal...
# On November 1st, you decided to learn Rust...
# Throughout the month, you maintained your preference for...
```

---

### create_memory_timeline(...)

Create structured timeline of memories.

**Signature:**

```python
def create_memory_timeline(
    self,
    user_id: str,
    title: str,
    time_range: TimeRange
) -> MemoryTimeline
```

**Returns:** MemoryTimeline object with events

**Example:**

```python
timeline = client.create_memory_timeline(
    user_id="alice",
    title="Last Week's Journey",
    time_range=TimeRange.LAST_WEEK
)

print(f"Timeline: {timeline.title}")
print(f"Period: {timeline.start_time} to {timeline.end_time}")
print(f"Events: {len(timeline.events)}")

for event in timeline.events:
    print(f"- {event.timestamp}: {event.description}")
```

---

### analyze_event_sequences(...)

Analyze sequential event patterns.

**Signature:**

```python
def analyze_event_sequences(
    self,
    user_id: str,
    max_gap_hours: int = 24
) -> list[EventSequence]
```

**Parameters:**

- `user_id` (str): User identifier
- `max_gap_hours` (int): Maximum gap between events

**Returns:** List of EventSequence objects

**Example:**

```python
sequences = client.analyze_event_sequences(
    user_id="alice",
    max_gap_hours=24
)

for seq in sequences:
    print(f"Sequence: {seq.pattern}")
    print(f"Events: {len(seq.events)}")
    print(f"Duration: {seq.duration_hours} hours")
```

---

### schedule_memory(...)

Schedule future memory with recurrence.

**Signature:**

```python
def schedule_memory(
    self,
    text: str,
    user_id: str,
    scheduled_for: datetime,
    recurrence: Optional[str] = None  # "daily", "weekly", "monthly"
) -> Memory
```

**Parameters:**

- `text` (str): Memory content
- `user_id` (str): User identifier
- `scheduled_for` (datetime): Scheduled time
- `recurrence` (str, optional): Recurrence pattern

**Returns:** Scheduled Memory

**Example:**

```python
from datetime import datetime, timedelta, timezone

tomorrow = datetime.now(timezone.utc) + timedelta(days=1)

# One-time scheduled memory
scheduled = client.schedule_memory(
    text="Follow up on project proposal",
    user_id="alice",
    scheduled_for=tomorrow
)

# Recurring memory
recurring = client.schedule_memory(
    text="Review weekly goals",
    user_id="alice",
    scheduled_for=tomorrow,
    recurrence="weekly"
)
```

---

### get_temporal_summary(...)

Get temporal activity summary.

**Signature:**

```python
def get_temporal_summary(
    self,
    user_id: str
) -> dict[str, Any]
```

**Returns:** Temporal statistics and patterns

**Example:**

```python
stats = client.get_temporal_summary(user_id="alice")

print(f"Peak activity hour: {stats['peak_activity_hour']}")
print(f"Most active day: {stats['most_active_day']}")
print(f"Average memories per day: {stats['avg_memories_per_day']:.1f}")
print(f"Busiest time period: {stats['busiest_period']}")
```

---

## 🔍 Cross-Session Insights

### detect_patterns(...)

Detect behavioral patterns across sessions.

**Signature:**

```python
def detect_patterns(
    self,
    user_id: str,
    min_occurrences: int = 3
) -> list[Pattern]
```

**Parameters:**

- `user_id` (str): User identifier
- `min_occurrences` (int): Minimum pattern occurrences

**Returns:** List of detected patterns

**Example:**

```python
patterns = client.detect_patterns(user_id="alice", min_occurrences=5)

for pattern in patterns[:5]:
    print(f"Pattern: {pattern.pattern_type.value}")
    print(f"Description: {pattern.description}")
    print(f"Confidence: {pattern.confidence:.2f}")
    print(f"Occurrences: {pattern.occurrences}")
    print(f"First seen: {pattern.first_occurrence}")
    print(f"Last seen: {pattern.last_occurrence}")

# Example output:
# Pattern: routine
# Description: User typically creates memories about work at 9 AM
# Confidence: 0.85
# Occurrences: 23
```

---

### track_behavior_changes(...)

Track changes in user behavior.

**Signature:**

```python
def track_behavior_changes(
    self,
    user_id: str,
    comparison_days: int = 30
) -> list[BehaviorChange]
```

**Parameters:**

- `user_id` (str): User identifier
- `comparison_days` (int): Days to compare (recent vs older)

**Returns:** List of behavior changes

**Example:**

```python
changes = client.track_behavior_changes(
    user_id="alice",
    comparison_days=30
)

for change in changes:
    print(f"Change: {change.change_type.value}")
    print(f"Description: {change.description}")
    print(f"Confidence: {change.confidence:.2f}")
    print(f"Detected: {change.detected_at}")

# Example output:
# Change: preference_shift
# Description: User's coffee preference changed from regular milk to oat milk
# Confidence: 0.88
```

---

### analyze_preference_drift(...)

Analyze how preferences change over time.

**Signature:**

```python
def analyze_preference_drift(
    self,
    user_id: str
) -> list[PreferenceDrift]
```

**Returns:** List of preference drift analyses

**Example:**

```python
drifts = client.analyze_preference_drift(user_id="alice")

for drift in drifts:
    print(f"Category: {drift.category}")
    print(f"Original: {drift.original_preference}")
    print(f"Current: {drift.current_preference}")
    print(f"Drift score: {drift.drift_score:.2f}")
    print(f"Time span: {drift.time_span_days} days")
```

---

### detect_habits(...)

Detect habit formation.

**Signature:**

```python
def detect_habits(
    self,
    user_id: str,
    min_occurrences: int = 5
) -> list[Habit]
```

**Parameters:**

- `user_id` (str): User identifier
- `min_occurrences` (int): Minimum occurrences to qualify as habit

**Returns:** List of detected habits

**Example:**

```python
habits = client.detect_habits(user_id="alice", min_occurrences=7)

for habit in habits[:3]:
    print(f"Behavior: {habit.behavior}")
    print(f"Habit score: {habit.habit_score:.2f}")
    print(f"Status: {habit.status}")  # forming, established, weakening
    print(f"Frequency: {habit.frequency}")  # daily, weekly, etc.
    print(f"Consistency: {habit.consistency:.2f}")
    print(f"Occurrences: {habit.occurrences}")

# Example output:
# Behavior: Exercises at 6 AM
# Habit score: 0.92
# Status: established
# Frequency: daily
# Consistency: 0.95
# Occurrences: 28
```

---

### analyze_trends(...)

Analyze long-term trends.

**Signature:**

```python
def analyze_trends(
    self,
    user_id: str,
    window_days: int = 30
) -> list[Trend]
```

**Parameters:**

- `user_id` (str): User identifier
- `window_days` (int): Analysis window

**Returns:** List of trends

**Example:**

```python
trends = client.analyze_trends(user_id="alice", window_days=30)

for trend in trends:
    print(f"Category: {trend.category}")
    print(f"Trend: {trend.trend_type}")  # increasing, decreasing, stable
    print(f"Direction: {trend.direction}")
    print(f"Strength: {trend.strength:.2f}")
    print(f"Confidence: {trend.confidence:.2f}")
```

---

## 🏷️ Semantic Clustering

### cluster_user_memories(...)

Cluster memories by semantic similarity.

**Signature:**

```python
def cluster_user_memories(
    self,
    user_id: str,
    max_clusters: int = 10
) -> list[MemoryCluster]
```

**Parameters:**

- `user_id` (str): User identifier
- `max_clusters` (int): Maximum number of clusters

**Returns:** List of MemoryCluster objects

**Example:**

```python
clusters = client.cluster_user_memories(
    user_id="alice",
    max_clusters=10
)

for cluster in clusters:
    print(f"Topic: {cluster.topic}")
    print(f"Size: {len(cluster.memories)} memories")
    print(f"Tags: {cluster.tags}")
    print("Sample memories:")
    for mem in cluster.memories[:3]:
        print(f"  - {mem.text}")

# Example output:
# Topic: programming_languages
# Size: 15 memories
# Tags: ['programming', 'languages', 'tech']
# Sample memories:
#   - I love Python programming
#   - JavaScript is great for web dev
#   - Learning Rust for systems programming
```

---

### suggest_memory_tags(...)

Suggest tags for a memory.

**Signature:**

```python
def suggest_memory_tags(
    self,
    memory: Memory,
    max_tags: int = 5
) -> list[str]
```

**Parameters:**

- `memory` (Memory): Memory object
- `max_tags` (int): Maximum tags to suggest

**Returns:** List of suggested tags

**Example:**

```python
memory = client.remember(
    "I completed the React tutorial and built my first app",
    user_id="alice"
)

suggested_tags = client.suggest_memory_tags(memory, max_tags=5)
print(f"Suggested tags: {suggested_tags}")

# Output: ['programming', 'web_development', 'react', 'learning', 'achievement']
```

---

### detect_topic_shift(...)

Detect topic shifts in conversation.

**Signature:**

```python
def detect_topic_shift(
    self,
    user_id: str,
    window_size: int = 10
) -> list[TopicShift]
```

**Parameters:**

- `user_id` (str): User identifier
- `window_size` (int): Analysis window size

**Returns:** List of detected topic shifts

**Example:**

```python
shifts = client.detect_topic_shift(user_id="alice", window_size=10)

for shift in shifts:
    print(f"Shift at: {shift.timestamp}")
    print(f"From topic: {shift.from_topic}")
    print(f"To topic: {shift.to_topic}")
    print(f"Shift strength: {shift.strength:.2f}")
```

---

## 👥 Multi-Agent Features

### create_agent_space(...)

Create memory space for an agent.

**Signature:**

```python
def create_agent_space(
    self,
    agent_id: str,
    agent_name: str,
    capabilities: list[str],
    permissions: dict[str, Any]
) -> AgentSpace
```

**Parameters:**

- `agent_id` (str): Agent identifier
- `agent_name` (str): Agent name
- `capabilities` (list[str]): Agent capabilities
- `permissions` (dict): Access permissions

**Returns:** AgentSpace object

**Example:**

```python
agent_space = client.create_agent_space(
    agent_id="agent_researcher",
    agent_name="Research Assistant",
    capabilities=["web_search", "data_analysis", "summarization"],
    permissions={
        "can_read": True,
        "can_write": True,
        "can_delete": False,
        "scope": "research"
    }
)

print(f"Created agent space: {agent_space.agent_id}")
```

---

### transfer_memory_between_agents(...)

Transfer memory from one agent to another.

**Signature:**

```python
def transfer_memory_between_agents(
    self,
    memory_id: str,
    from_agent_id: str,
    to_agent_id: str,
    preserve_original: bool = True
) -> bool
```

**Parameters:**

- `memory_id` (str): Memory to transfer
- `from_agent_id` (str): Source agent
- `to_agent_id` (str): Target agent
- `preserve_original` (bool): Keep original copy

**Returns:** True if successful

**Example:**

```python
transferred = client.transfer_memory_between_agents(
    memory_id="mem_abc123",
    from_agent_id="agent_researcher",
    to_agent_id="agent_writer",
    preserve_original=True
)
```

---

### get_shared_memory_context(...)

Get shared memories across agents.

**Signature:**

```python
def get_shared_memory_context(
    self,
    agent_ids: list[str],
    query: Optional[str] = None,
    k: int = 10
) -> list[Memory]
```

**Parameters:**

- `agent_ids` (list[str]): List of agent IDs
- `query` (str, optional): Filter by query
- `k` (int): Maximum results

**Returns:** List of shared memories

**Example:**

```python
shared_memories = client.get_shared_memory_context(
    agent_ids=["agent_researcher", "agent_writer", "agent_editor"],
    query="project documentation",
    k=10
)
```

---

## ⏱️ Scheduling & Automation

### start_scheduler(...)

Start background task scheduler.

**Signature:**

```python
def start_scheduler(self) -> None
```

**Example:**

```python
client.start_scheduler()
print("Scheduler started")
```

---

### stop_scheduler(...)

Stop background task scheduler.

**Signature:**

```python
def stop_scheduler(self) -> None
```

**Example:**

```python
client.stop_scheduler()
print("Scheduler stopped")
```

---

### get_scheduler_status(...)

Get scheduler status and scheduled jobs.

**Signature:**

```python
def get_scheduler_status(self) -> dict[str, Any]
```

**Returns:** Scheduler status information

**Example:**

```python
status = client.get_scheduler_status()

print(f"Running: {status['running']}")
print(f"Jobs: {len(status['jobs'])}")
for job in status['jobs']:
    print(f"  - {job['name']}: {job['next_run']}")
```

---

### consolidate_all_memories(...)

Consolidate similar memories.

**Signature:**

```python
def consolidate_all_memories(
    self,
    similarity_threshold: float = 0.85
) -> int
```

**Parameters:**

- `similarity_threshold` (float): Similarity threshold for consolidation

**Returns:** Number of memories consolidated

**Example:**

```python
count = client.consolidate_all_memories(similarity_threshold=0.85)
print(f"Consolidated {count} memories")
```

---

### apply_importance_decay(...)

Apply importance decay to all memories.

**Signature:**

```python
def apply_importance_decay(self) -> int
```

**Returns:** Number of memories updated

**Example:**

```python
updated = client.apply_importance_decay()
print(f"Applied decay to {updated} memories")
```

---

## 📊 Telemetry & Monitoring

### get_telemetry_metrics(...)

Get comprehensive telemetry metrics.

**Signature:**

```python
def get_telemetry_metrics(self) -> dict[str, Any]
```

**Returns:** Telemetry data including operation counts, durations, sizes

**Example:**

```python
metrics = client.get_telemetry_metrics()

print("Operation counts:")
for op, count in metrics['operation_counts'].items():
    print(f"  {op}: {count}")

print(f"\\nAverage recall time: {metrics['recall_duration']['avg']:.2f}ms")
print(f"Average memory size: {metrics['memory_size_chars']['avg']:.1f} chars")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

---

### get_recent_operations(...)

Get recent operations with details.

**Signature:**

```python
def get_recent_operations(
    self,
    limit: int = 10,
    operation: Optional[str] = None
) -> list[OperationLog]
```

**Parameters:**

- `limit` (int): Maximum results
- `operation` (str, optional): Filter by operation type

**Returns:** List of operation logs

**Example:**

```python
# Get all recent operations
operations = client.get_recent_operations(limit=20)

for op in operations:
    print(f"{op.operation.value}: {op.duration_ms:.2f}ms ({op.status})")
    if op.metadata:
        print(f"  Metadata: {op.metadata}")

# Get recent recalls only
recalls = client.get_recent_operations(limit=10, operation="recall")
```

---

### export_telemetry(...)

Export telemetry data.

**Signature:**

```python
def export_telemetry(
    self,
    trace_ids: Optional[list[str]] = None
) -> list[dict[str, Any]]
```

**Parameters:**

- `trace_ids` (list[str], optional): Specific trace IDs to export

**Returns:** List of telemetry data

**Example:**

```python
telemetry_data = client.export_telemetry()

# Save to file
import json
with open('telemetry_export.json', 'w') as f:
    json.dump(telemetry_data, f, indent=2)
```

---

## 🛠️ Utility Functions

### inject_context(...)

Inject memory context into LLM prompt.

**Signature:**

```python
def inject_context(
    self,
    prompt: str,
    query: str,
    user_id: str,
    k: int = 5,
    template: str = "default"
) -> str
```

**Parameters:**

- `prompt` (str): Original prompt
- `query` (str): Query for relevant memories
- `user_id` (str): User identifier
- `k` (int): Number of memories to inject
- `template` (str): Context template

**Returns:** Enhanced prompt with context

**Example:**

```python
enhanced_prompt = client.inject_context(
    prompt="What are my preferences?",
    query="user preferences",
    user_id="alice",
    k=5
)

print(enhanced_prompt)

# Output:
# Based on what I know about you:
# - You prefer oat milk in your coffee
# - You work remotely on Tuesdays
# - You enjoy green tea in the morning
# - You exercise at 6 AM
# - You prefer dark mode
#
# What are my preferences?
```

---

### reconcile_user_memories(...)

Reconcile and deduplicate user memories.

**Signature:**

```python
def reconcile_user_memories(
    self,
    user_id: str
) -> list[Memory]
```

**Parameters:**

- `user_id` (str): User identifier

**Returns:** List of reconciled memories

**Example:**

```python
reconciled = client.reconcile_user_memories(user_id="alice")
print(f"Reconciled {len(reconciled)} memories")
```

---

## 📝 Complete Usage Example

Here's a comprehensive example using multiple features:

```python
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaLLM
from hippocampai.models.temporal import TimeRange

# Initialize
client = MemoryClient(
    llm_provider=OllamaLLM(base_url="http://localhost:11434")
)

# Create session
session = client.create_session(
    user_id="alice",
    title="Project Planning"
)

# Store memories
client.track_session_message(
    session_id=session.id,
    role="user",
    content="I need to finish the React project by Friday"
)

memory1 = client.remember(
    text="Complete React project by Friday",
    user_id="alice",
    session_id=session.id,
    type="goal",
    importance=9.0,
    tags=["work", "deadline"]
)

# Extract from conversation
conversation = """
User: I prefer working in the mornings
Assistant: That's good for productivity
User: Yes, I usually start at 6 AM
"""

memories = client.extract_from_conversation(
    conversation=conversation,
    user_id="alice",
    session_id=session.id
)

# Recall relevant information
results = client.recall(
    query="work schedule and deadlines",
    user_id="alice",
    k=5
)

# Analyze patterns
patterns = client.detect_patterns(user_id="alice")
habits = client.detect_habits(user_id="alice", min_occurrences=5)

# Generate insights
summary = client.summarize_session(session_id=session.id)
facts = client.extract_facts(summary, source="session_summary")

# Get statistics
stats = client.get_memory_statistics(user_id="alice")
telemetry = client.get_telemetry_metrics()

# Build narrative
narrative = client.build_memory_narrative(
    user_id="alice",
    time_range=TimeRange.LAST_WEEK,
    title="This Week's Progress"
)

print(narrative)

# Complete session
client.complete_session(session_id=session.id, generate_summary=True)

# Export data
client.export_graph_to_json(
    file_path="alice_memory_graph.json",
    user_id="alice"
)
```

---

## 📚 Additional Resources

- **[API Reference](API_REFERENCE.md)** - REST API documentation
- **[SaaS Guide](SAAS_GUIDE.md)** - Complete SaaS platform guide
- **[Configuration](CONFIGURATION.md)** - Configuration options
- **[Examples](../examples/)** - Working code examples
- **[User Guide](USER_GUIDE.md)** - Production deployment

---

**Last Updated**: 2025-11-24
**Library Version**: v0.3.0
**Total Functions**: 102+
**Status**: Production Ready
