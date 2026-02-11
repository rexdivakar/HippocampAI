# HippocampAI - Complete Feature Documentation

This document provides comprehensive documentation for all memory management features in HippocampAI, including core capabilities and advanced features.

## Table of Contents

1. [Core CRUD Operations](#core-crud-operations)
2. [Advanced Filtering & Search](#advanced-filtering--search)
3. [Memory Lifecycle Management](#memory-lifecycle-management)
4. [Graph & Relationships](#graph--relationships)
5. [Version Control & Audit](#version-control--audit)
6. [Batch Operations](#batch-operations)
7. [Context Injection](#context-injection)
8. [Session Management](#session-management)
9. [Temporal Reasoning](#temporal-reasoning)
10. [Cross-Session Insights](#cross-session-insights)
11. [Real-Time Incremental Knowledge Graph](#real-time-incremental-knowledge-graph) **NEW v0.5.0**
12. [Graph-Aware Retrieval](#graph-aware-retrieval) **NEW v0.5.0**
13. [Memory Relevance Feedback Loop](#memory-relevance-feedback-loop) **NEW v0.5.0**
14. [Memory Triggers / Event-Driven Actions](#memory-triggers--event-driven-actions) **NEW v0.5.0**
15. [Procedural Memory / Prompt Self-Optimization](#procedural-memory--prompt-self-optimization) **NEW v0.5.0**
16. [Embedding Model Migration](#embedding-model-migration) **NEW v0.5.0**
17. [Storage & Caching](#storage--caching)
18. [Monitoring & Telemetry](#monitoring--telemetry)
19. [API Reference](#api-reference)

---

## Core CRUD Operations

### 1. Create Memory - `remember()`

Store a new memory with optional metadata, tags, and TTL.

**Signature:**
```python
def remember(
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    ttl_days: Optional[int] = None,
) -> Memory
```

**Example:**
```python
from hippocampai import MemoryClient

client = MemoryClient()

# Basic memory
memory = client.remember(
    text="Python is great for ML",
    user_id="alice"
)

# Memory with tags and TTL
memory = client.remember(
    text="Meeting notes for Q1 planning",
    user_id="alice",
    tags=["work", "meetings", "q1"],
    importance=8.5,
    ttl_days=90  # Expires in 90 days
)
```

**Location:** `src/hippocampai/client.py:207`

---

### 2. Read Memory - `recall()` & `get_memories()`

#### Semantic Search with `recall()`

Hybrid retrieval combining BM25, vector search, and reranking.

**Signature:**
```python
def recall(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RetrievalResult]
```

**Example:**
```python
# Basic semantic search
results = client.recall(
    query="What are my coffee preferences?",
    user_id="alice",
    k=5
)

# With tag filtering
results = client.recall(
    query="morning routines",
    user_id="alice",
    k=10,
    filters={"tags": ["morning", "routine"]}
)

for result in results:
    print(f"{result.memory.text}")
    print(f"  Relevance: {result.score:.3f}")
```

**Location:** `src/hippocampai/client.py:283`

#### Advanced Filtering with `get_memories()`

Retrieve memories with complex filters without semantic search.

**Signature:**
```python
def get_memories(
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> List[Memory]
```

**Supported Filters:**
- `type`: Memory type ("fact", "preference", "goal", "habit", "event", "context")
- `tags`: Tag filtering (str or list) - matches ANY
- `session_id`: Session ID
- `min_importance`: Minimum importance score (0-10)
- `max_importance`: Maximum importance score (0-10)
- `include_expired`: Include expired memories (default: False)

**Example:**
```python
# High-importance work memories
memories = client.get_memories(
    user_id="alice",
    filters={
        "tags": ["work"],
        "min_importance": 7.0,
        "type": "fact"
    },
    limit=50
)

# All preferences (no semantic search)
prefs = client.get_memories(
    user_id="alice",
    filters={"type": "preference"}
)
```

**Location:** `src/hippocampai/client.py:507`

---

### 3. Update Memory - `update_memory()`

Modify any field of an existing memory with automatic re-embedding.

**Signature:**
```python
def update_memory(
    memory_id: str,
    text: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expires_at: Optional[datetime] = None,
) -> Optional[Memory]
```

**Features:**
- Automatic re-embedding if text changes
- Preserves user ownership
- Updates `updated_at` timestamp
- Full telemetry tracking
- Returns updated Memory or None if not found

**Example:**
```python
# Update text and importance
updated = client.update_memory(
    memory_id="abc-123",
    text="Updated memory content",
    importance=9.0
)

# Update tags only
updated = client.update_memory(
    memory_id="abc-123",
    tags=["new", "tags"]
)

# Update metadata
updated = client.update_memory(
    memory_id="abc-123",
    metadata={"project": "HippocampAI", "status": "active"}
)
```

**Location:** `src/hippocampai/client.py:382`

---

### 4. Delete Memory - `delete_memory()`

Remove a memory by ID with optional user verification.

**Signature:**
```python
def delete_memory(
    memory_id: str,
    user_id: Optional[str] = None
) -> bool
```

**Features:**
- Optional user ownership verification
- Searches both collections (facts & prefs)
- Full telemetry tracking
- Returns True if deleted, False if not found

**Example:**
```python
# Delete with user verification
deleted = client.delete_memory(
    memory_id="abc-123",
    user_id="alice"
)

if deleted:
    print("Memory deleted successfully")
else:
    print("Memory not found or unauthorized")

# Delete without verification (admin use)
deleted = client.delete_memory(memory_id="abc-123")
```

**Location:** `src/hippocampai/client.py:464`

---

## Advanced Filtering & Search

### Advanced Filters - `get_memories_advanced()`

Enhanced filtering with metadata support and custom sorting.

**Signature:**
```python
def get_memories_advanced(
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "created_at",  # created_at, importance, access_count
    sort_order: str = "desc",
    limit: int = 100,
) -> List[Memory]
```

**Supported Filters:**
- All `get_memories()` filters
- `metadata`: Dictionary of key-value pairs for exact matching

**Example:**
```python
# Filter by custom metadata
memories = client.get_memories_advanced(
    user_id="alice",
    filters={
        "tags": "work",
        "min_importance": 7.0,
        "metadata": {"project": "HippocampAI", "status": "active"}
    },
    sort_by="importance",
    sort_order="desc"
)

# Most accessed memories
popular = client.get_memories_advanced(
    user_id="alice",
    sort_by="access_count",
    sort_order="desc",
    limit=10
)
```

**Location:** `src/hippocampai/client.py:937`

---

## Memory Lifecycle Management

### Memory TTL (Time-To-Live)

Automatic expiration support for temporary memories.

**Features:**
- Set expiration during creation with `ttl_days`
- Auto-exclude expired in `get_memories()` by default
- Manual cleanup with `expire_memories()`
- `is_expired()` method for checking status

**Example:**
```python
# Create memory that expires in 7 days
memory = client.remember(
    text="Temporary reminder for next week",
    user_id="alice",
    ttl_days=7
)

# Check if expired
if memory.is_expired():
    print("Memory has expired")

# Manual cleanup
expired_count = client.expire_memories(user_id="alice")
print(f"Removed {expired_count} expired memories")

# Get all including expired
all_memories = client.get_memories(
    user_id="alice",
    filters={"include_expired": True}
)
```

**Cleanup Method:**
```python
def expire_memories(user_id: Optional[str] = None) -> int
```

**Locations:**
- Memory model: `src/hippocampai/models/memory.py:41`
- Cleanup: `src/hippocampai/client.py:595`

---

### Memory Access Tracking

Automatic tracking of memory usage with access counts.

**Features:**
- Increments `access_count` on retrieval
- Tracks `last_accessed_at` timestamp
- Automatic audit trail entry
- No performance overhead

**Methods:**
```python
def track_memory_access(memory_id: str, user_id: str)
```

**Example:**
```python
# Manual tracking
client.track_memory_access(memory_id="abc-123", user_id="alice")

# Automatic tracking via inject_context
prompt = client.inject_context(
    prompt="What are my preferences?",
    query="preferences",
    user_id="alice"
)
# ↑ Automatically tracks access for all retrieved memories

# Query most accessed memories
popular = client.get_memories_advanced(
    user_id="alice",
    sort_by="access_count",
    sort_order="desc"
)
```

**Location:** `src/hippocampai/client.py:855`

---

## Graph & Relationships

### Architecture: In-Memory Graph with JSON Persistence

HippocampAI uses **NetworkX** (an in-memory directed graph library) rather than a dedicated graph database. This is a deliberate architectural choice for v0.5.0:

**How it works:**

```
┌─────────────────────────────────────────────────────────────┐
│                   Triple-Store Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Qdrant (Vector DB)     NetworkX (Graph)     BM25 (Text)   │
│   ├─ Embeddings          ├─ Entities          ├─ Keywords   │
│   ├─ Similarity search   ├─ Relationships     ├─ Full-text  │
│   └─ Payload filtering   ├─ Facts & Topics    └─ Ranking    │
│                          └─ Graph traversal                  │
│                                                              │
│   Storage: Qdrant        Storage: JSON file   Storage: RAM   │
│   collections            (auto-saved)                        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│   Score Fusion (6 weights):                                  │
│   sim + rerank + recency + importance + graph + feedback     │
└─────────────────────────────────────────────────────────────┘
```

**Class hierarchy:**
- `nx.DiGraph` → `MemoryGraph` (base, memory-to-memory relationships) → `KnowledgeGraph` (entities, facts, topics, inference)

**Persistence:**
- Graph state is serialized to JSON at `GRAPH_PERSISTENCE_PATH` (default: `data/knowledge_graph.json`)
- Auto-save runs every `GRAPH_AUTO_SAVE_INTERVAL` seconds (default: 300)
- On startup, the graph is loaded from JSON back into memory

**Why not a graph database (yet)?**
- Zero external dependencies — no Neo4j/ArangoDB to deploy
- Sub-millisecond traversal for typical graph sizes (< 100K nodes)
- JSON persistence is sufficient for single-instance deployments
- Neo4j integration is on the roadmap for multi-instance and large-scale deployments (see [Roadmap](#roadmap))

**Location:** `src/hippocampai/graph/memory_graph.py`, `src/hippocampai/graph/knowledge_graph.py`, `src/hippocampai/graph/graph_persistence.py`

### Graph Index for Memory Relationships

NetworkX-based directed graph for mapping relationships between memories.

**Relationship Types:**
- `RELATED_TO` - General association
- `CAUSED_BY` - Causal relationship
- `LEADS_TO` - Sequential relationship
- `CONTRADICTS` - Conflicting information
- `SUPPORTS` - Supporting evidence
- `PART_OF` - Hierarchical relationship
- `SIMILAR_TO` - Similarity relationship
- `SUPERSEDES` - Replacement relationship

**Methods:**
```python
def add_relationship(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    weight: float = 1.0
) -> bool

def get_related_memories(
    memory_id: str,
    relation_types: Optional[List[RelationType]] = None,
    max_depth: int = 1
) -> List[Tuple[str, str, float]]

def get_memory_clusters(user_id: str) -> List[Set[str]]
```

**Example:**
```python
from hippocampai import RelationType

# Create memories
m1 = client.remember("Python is a programming language", user_id="alice")
m2 = client.remember("I use Python for ML", user_id="alice")
m3 = client.remember("TensorFlow is a Python framework", user_id="alice")

# Add to graph
client.graph.add_memory(m1.id, "alice", {})
client.graph.add_memory(m2.id, "alice", {})
client.graph.add_memory(m3.id, "alice", {})

# Create relationships
client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO, weight=0.9)
client.add_relationship(m2.id, m3.id, RelationType.SUPPORTS)

# Get related memories (depth 1)
related = client.get_related_memories(m1.id, max_depth=1)
for memory_id, relation_type, weight in related:
    print(f"Related: {memory_id} ({relation_type}, weight: {weight})")

# Get related memories (depth 2) - transitive relationships
deep_related = client.get_related_memories(m1.id, max_depth=2)

# Find memory clusters
clusters = client.get_memory_clusters("alice")
for cluster in clusters:
    print(f"Cluster: {len(cluster)} memories")
```

**Location:** `src/hippocampai/graph/memory_graph.py`

**Dependencies:** NetworkX (`pip install networkx`)

---

## Version Control & Audit

### Memory Version Control

Git-like versioning system for tracking memory changes over time.

**Features:**
- Track up to N versions per memory (configurable, default: 10)
- Compare versions with diff
- Rollback to previous versions
- Version history with metadata
- Change summaries
- Automatic audit trail integration

**Methods:**
```python
def get_memory_history(memory_id: str) -> List[MemoryVersion]

def rollback_memory(memory_id: str, version_number: int) -> Optional[Memory]
```

**Example:**
```python
# Create memory
memory = client.remember("Version 1 text", user_id="alice")

# Create version manually
version = client.version_control.create_version(
    memory.id,
    memory.model_dump(mode="json"),
    created_by="alice",
    change_summary="Initial version"
)

# Update memory
updated = client.update_memory(memory.id, text="Version 2 text", importance=9.0)

# Create another version
version2 = client.version_control.create_version(
    memory.id,
    updated.model_dump(mode="json"),
    created_by="alice",
    change_summary="Updated text and importance"
)

# View history
history = client.get_memory_history(memory.id)
for v in history:
    print(f"v{v.version_number}: {v.change_summary}")

# Compare versions
diff = client.version_control.compare_versions(memory.id, 1, 2)
print(f"Changes: {diff}")

# Rollback to version 1
rolled_back = client.rollback_memory(memory.id, version_number=1)
```

**Location:** `src/hippocampai/versioning/memory_versioning.py`

---

### Audit Trail

Complete change tracking with timestamps and metadata.

**Change Types:**
- `CREATED` - Memory created
- `UPDATED` - Memory modified
- `DELETED` - Memory removed
- `ACCESSED` - Memory retrieved
- `RELATIONSHIP_ADDED` - Graph edge added
- `RELATIONSHIP_REMOVED` - Graph edge removed

**Methods:**
```python
def get_audit_trail(
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    change_type: Optional[ChangeType] = None,
    limit: int = 100
) -> List[AuditEntry]
```

**Example:**
```python
from hippocampai import ChangeType

# Get audit trail for specific memory
entries = client.get_audit_trail(memory_id="abc-123", limit=50)
for entry in entries:
    print(f"{entry.timestamp}: {entry.change_type.value}")
    print(f"  User: {entry.user_id}")
    print(f"  Changes: {entry.changes}")

# Get all user activity
user_activity = client.get_audit_trail(user_id="alice", limit=100)

# Filter by change type
updates = client.get_audit_trail(
    change_type=ChangeType.UPDATED,
    limit=50
)

# Relationship changes
relationships = client.get_audit_trail(
    change_type=ChangeType.RELATIONSHIP_ADDED
)
```

**Location:** `src/hippocampai/versioning/memory_versioning.py:203`

---

## Batch Operations

### Batch Add - `add_memories()`

Add multiple memories at once for efficiency.

**Signature:**
```python
def add_memories(
    memories: List[Dict[str, Any]],
    user_id: str,
    session_id: Optional[str] = None,
) -> List[Memory]
```

**Example:**
```python
memories_data = [
    {
        "text": "Python is great for ML",
        "tags": ["programming", "ml"],
        "importance": 8.5
    },
    {
        "text": "I prefer dark mode",
        "type": "preference",
        "importance": 7.0
    },
    {
        "text": "Finish project by Friday",
        "type": "goal",
        "importance": 9.5,
        "ttl_days": 7
    }
]

created = client.add_memories(memories_data, user_id="alice")
print(f"Created {len(created)} memories")
```

**Location:** `src/hippocampai/client.py:658`

---

### Batch Delete - `delete_memories()`

Delete multiple memories at once.

**Signature:**
```python
def delete_memories(
    memory_ids: List[str],
    user_id: Optional[str] = None
) -> int
```

**Example:**
```python
# Delete multiple memories
ids_to_delete = ["id1", "id2", "id3"]
deleted_count = client.delete_memories(ids_to_delete, user_id="alice")
print(f"Deleted {deleted_count}/{len(ids_to_delete)} memories")
```

**Location:** `src/hippocampai/client.py:705`

---

## Context Injection

### LLM Prompt Context Injection

Helper for injecting relevant memories into LLM prompts with formatting.

**Templates:**
- `default` - Clean, numbered list with metadata
- `minimal` - Compact, pipe-separated
- `detailed` - Full metadata including importance and dates

**Methods:**
```python
def inject_context(
    prompt: str,
    query: str,
    user_id: str,
    k: int = 5,
    template: str = "default"
) -> str
```

**Example:**
```python
# Default template
prompt = client.inject_context(
    prompt="What should I have for breakfast?",
    query="breakfast preferences food",
    user_id="alice",
    k=3,
    template="default"
)

# Minimal template (compact)
prompt = client.inject_context(
    prompt="Remind me about my goals",
    query="goals objectives",
    user_id="alice",
    k=5,
    template="minimal"
)

# Detailed template (full metadata)
prompt = client.inject_context(
    prompt="What are my important work tasks?",
    query="work tasks important",
    user_id="alice",
    k=10,
    template="detailed"
)

# Send to LLM
# response = llm.generate(prompt)
```

**Direct Usage:**
```python
from hippocampai import ContextInjector

injector = ContextInjector(max_tokens=2000, template="default")

# With conversation history
prompt = injector.create_prompt_with_history(
    current_query="What's next?",
    conversation_history=[
        {"role": "user", "content": "Tell me about my goals"},
        {"role": "assistant", "content": "Your goals include..."}
    ],
    memories=memories,
    max_history_turns=5
)

# Token-aware truncation
selected_memories = injector.truncate_to_token_limit(
    memories=all_memories,
    current_prompt=base_prompt
)
```

**Location:** `src/hippocampai/utils/context_injection.py`

---

## Session Management

HippocampAI's session management system allows you to organize conversations into structured sessions with automatic summarization, entity tracking, and semantic search capabilities.

### Overview

**Key Features:**
- Session lifecycle management (create, update, complete, delete)
- Automatic LLM-powered summarization
- Entity extraction and tracking (people, technologies, organizations, etc.)
- Fact extraction with confidence scores
- Semantic session search
- Hierarchical sessions (parent-child relationships)
- Automatic topic boundary detection
- Rich metadata and tagging

### Core Models

#### Session
```python
class Session(BaseModel):
    id: str                              # Unique session ID
    user_id: str                         # Owner user ID
    title: Optional[str]                 # Session title
    summary: Optional[str]               # Auto-generated summary
    status: SessionStatus                # active, inactive, completed, archived
    parent_session_id: Optional[str]     # For hierarchical sessions
    child_session_ids: List[str]         # Child sessions
    message_count: int                   # Total messages tracked
    memory_count: int                    # Associated memories
    entities: Dict[str, Entity]          # Extracted entities
    facts: List[SessionFact]             # Extracted facts
    metadata: Dict[str, Any]             # Custom metadata
    tags: List[str]                      # Tags for filtering
    started_at: datetime                 # Session start time
    last_activity_at: datetime           # Last message time
    ended_at: Optional[datetime]         # Completion time
```

#### Entity
```python
class Entity(BaseModel):
    name: str                    # Entity name
    type: str                    # person, organization, technology, etc.
    mentions: int                # Number of times mentioned
    first_mentioned_at: datetime # First occurrence
    last_mentioned_at: datetime  # Most recent occurrence
    metadata: Dict[str, Any]     # Additional context
```

#### SessionFact
```python
class SessionFact(BaseModel):
    fact: str                    # Fact statement
    confidence: float            # 0.0-1.0 confidence score
    extracted_at: datetime       # Extraction timestamp
    sources: List[str]           # Memory IDs supporting this fact
```

### 1. Create Session - `create_session()`

Create a new conversation session with optional metadata and hierarchy.

**Signature:**
```python
def create_session(
    user_id: str,
    title: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Session
```

**Example:**
```python
from hippocampai import MemoryClient

client = MemoryClient()

# Basic session
session = client.create_session(
    user_id="alice",
    title="ML Project Discussion"
)

# Session with metadata and tags
session = client.create_session(
    user_id="alice",
    title="Q1 Planning - Sentiment Analysis",
    tags=["work", "ml", "quarterly"],
    metadata={
        "project": "sentiment-analysis",
        "team": "data-science",
        "priority": "high"
    }
)

# Hierarchical session (child)
child_session = client.create_session(
    user_id="alice",
    title="Deep Dive: Model Architecture",
    parent_session_id=session.id,
    tags=["technical", "deep-dive"]
)
```

**Location:** `src/hippocampai/client.py:1043`

---

### 2. Track Messages - `track_session_message()`

Track messages within a session with automatic entity extraction.

**Signature:**
```python
def track_session_message(
    session_id: str,
    text: str,
    user_id: str,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    auto_boundary_detect: bool = False,
) -> Session
```

**Features:**
- Creates memory and associates with session
- Updates session statistics
- Auto-extracts entities
- Optional boundary detection (creates new session if topic changes)
- Triggers auto-summarization at threshold

**Example:**
```python
# Basic message tracking
session = client.track_session_message(
    session_id=session.id,
    text="We're building a sentiment analysis model using BERT",
    user_id="alice"
)

# With importance and tags
session = client.track_session_message(
    session_id=session.id,
    text="Deadline is Friday - this is critical",
    user_id="alice",
    type="fact",
    importance=9.5,
    tags=["deadline", "critical"]
)

# With boundary detection
result = client.track_session_message(
    session_id=session.id,
    text="Let's switch topics to cloud infrastructure",
    user_id="alice",
    auto_boundary_detect=True  # May create new session
)

if result.id != session.id:
    print(f"Topic changed - new session created: {result.id}")
```

**Location:** `src/hippocampai/client.py:1095`

---

### 3. Session Summarization - `summarize_session()`

Generate LLM-powered summaries of sessions.

**Signature:**
```python
def summarize_session(
    session_id: str,
    force: bool = False
) -> Optional[str]
```

**Features:**
- Automatic summarization at message threshold (default: 10)
- Manual summary generation
- Force re-summarization
- Graceful fallback if LLM unavailable

**Example:**
```python
# Generate summary
summary = client.summarize_session(session.id)
if summary:
    print(f"Summary: {summary}")
else:
    print("LLM not available or insufficient messages")

# Force re-summarization
summary = client.summarize_session(session.id, force=True)
```

**Location:** `src/hippocampai/client.py:1210`

---

### 4. Entity Extraction - `extract_session_entities()`

Extract and track named entities from session messages.

**Signature:**
```python
def extract_session_entities(
    session_id: str,
    force: bool = False
) -> Dict[str, Any]
```

**Entity Types:**
- `person` - People and names
- `organization` - Companies, institutions
- `technology` - Technologies, frameworks, tools
- `location` - Places and locations
- `product` - Products and services
- `concept` - Abstract concepts
- `event` - Events and occurrences

**Example:**
```python
# Extract entities
entities = client.extract_session_entities(session.id)

if entities:
    for name, entity in entities.items():
        print(f"{name} ({entity.type})")
        print(f"  Mentions: {entity.mentions}")
        print(f"  First seen: {entity.first_mentioned_at}")
        print(f"  Last seen: {entity.last_mentioned_at}")

# Force re-extraction
entities = client.extract_session_entities(session.id, force=True)
```

**Location:** `src/hippocampai/client.py:1241`

**Fallback:** Uses regex-based extraction if LLM unavailable

---

### 5. Fact Extraction - `extract_session_facts()`

Extract key facts with confidence scores from sessions.

**Signature:**
```python
def extract_session_facts(
    session_id: str,
    force: bool = False
) -> List[SessionFact]
```

**Example:**
```python
# Extract facts
facts = client.extract_session_facts(session.id)

if facts:
    for fact in facts:
        print(f"Fact: {fact.fact}")
        print(f"  Confidence: {fact.confidence:.2f}")
        print(f"  Sources: {len(fact.sources)} memories")
```

**Location:** `src/hippocampai/client.py:1225`

---

### 6. Session Search - `search_sessions()`

Semantic search across all sessions.

**Signature:**
```python
def search_sessions(
    query: str,
    user_id: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[SessionSearchResult]
```

**Supported Filters:**
- `tags`: Tag filtering (str or list)
- `metadata`: Metadata key-value matching
- `status`: Session status filtering

**Example:**
```python
# Basic search
results = client.search_sessions(
    query="machine learning tensorflow discussions",
    user_id="alice",
    k=5
)

for result in results:
    print(f"{result.session.title}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Messages: {result.session.message_count}")

# Search with filters
results = client.search_sessions(
    query="project planning",
    user_id="alice",
    k=10,
    filters={
        "tags": ["work"],
        "metadata": {"priority": "high"}
    }
)
```

**Location:** `src/hippocampai/client.py:1139`

---

### 7. Hierarchical Sessions - `get_child_sessions()`

Manage parent-child session relationships.

**Signature:**
```python
def get_child_sessions(
    parent_session_id: str
) -> List[Session]
```

**Example:**
```python
# Create parent session
parent = client.create_session(
    user_id="alice",
    title="ML Project - Overall Planning"
)

# Create child sessions
arch_session = client.create_session(
    user_id="alice",
    title="Deep Dive: Architecture",
    parent_session_id=parent.id
)

data_session = client.create_session(
    user_id="alice",
    title="Deep Dive: Data Pipeline",
    parent_session_id=parent.id
)

# Get all children
children = client.get_child_sessions(parent.id)
for child in children:
    print(f"- {child.title} ({child.message_count} messages)")
```

**Location:** `src/hippocampai/client.py:1197`

---

### 8. Session Statistics - `get_session_statistics()`

Get comprehensive analytics for a session.

**Signature:**
```python
def get_session_statistics(
    session_id: str
) -> Dict[str, Any]
```

**Returns:**
```python
{
    "message_count": int,
    "memory_count": int,
    "duration_seconds": float,
    "entity_count": int,
    "fact_count": int,
    "avg_importance": float,
    "top_entities": List[Dict],  # Top 10 entities by mentions
    "memory_types": Dict[str, int]  # Breakdown by type
}
```

**Example:**
```python
stats = client.get_session_statistics(session.id)

print(f"Messages: {stats['message_count']}")
print(f"Duration: {stats['duration_seconds']:.1f}s")
print(f"Entities: {stats['entity_count']}")
print(f"Facts: {stats['fact_count']}")
print(f"Avg Importance: {stats['avg_importance']:.2f}")

# Top entities
for entity in stats['top_entities']:
    print(f"  - {entity['name']} ({entity['type']}): {entity['mentions']} mentions")
```

**Location:** `src/hippocampai/client.py:1255`

---

### 9. User Sessions - `get_user_sessions()`

Retrieve all sessions for a user with optional filtering.

**Signature:**
```python
def get_user_sessions(
    user_id: str,
    status: Optional[SessionStatus] = None,
    limit: int = 50
) -> List[Session]
```

**Example:**
```python
from hippocampai import SessionStatus

# All sessions
all_sessions = client.get_user_sessions(user_id="alice")

# Filter by status
active_sessions = client.get_user_sessions(
    user_id="alice",
    status=SessionStatus.ACTIVE
)

completed_sessions = client.get_user_sessions(
    user_id="alice",
    status=SessionStatus.COMPLETED,
    limit=100
)
```

**Location:** `src/hippocampai/client.py:1153`

---

### 10. Complete Session - `complete_session()`

Mark a session as completed with optional final summary.

**Signature:**
```python
def complete_session(
    session_id: str,
    generate_summary: bool = False
) -> Optional[Session]
```

**Example:**
```python
# Complete without summary
completed = client.complete_session(session_id)

# Complete with final summary
completed = client.complete_session(
    session_id=session.id,
    generate_summary=True
)

if completed:
    print(f"Duration: {completed.duration_seconds():.1f}s")
    print(f"Summary: {completed.summary}")
```

**Location:** `src/hippocampai/client.py:1122`

---

### 11. Session Memories - `get_session_memories()`

Retrieve all memories associated with a session.

**Signature:**
```python
def get_session_memories(
    session_id: str,
    limit: int = 100
) -> List[Memory]
```

**Example:**
```python
# Get all session memories
memories = client.get_session_memories(session.id)

for memory in memories:
    print(f"[{memory.type}] {memory.text}")
    print(f"  Importance: {memory.importance:.1f}")
```

**Location:** `src/hippocampai/client.py:1181`

---

### Complete Session Management Example

```python
from hippocampai import MemoryClient, SessionStatus

client = MemoryClient()
user_id = "alice"

# 1. Create session
session = client.create_session(
    user_id=user_id,
    title="ML Project - Requirements",
    tags=["work", "ml"],
    metadata={"project": "sentiment-analysis"}
)

# 2. Track conversation
messages = [
    "We need sentiment analysis for customer reviews",
    "Dataset should have 100k+ reviews",
    "Using TensorFlow and BERT",
    "Deadline is end of Q1"
]

for msg in messages:
    session = client.track_session_message(
        session_id=session.id,
        text=msg,
        user_id=user_id
    )

# 3. Generate summary
summary = client.summarize_session(session.id)
print(f"Summary: {summary}")

# 4. Extract insights
facts = client.extract_session_facts(session.id)
entities = client.extract_session_entities(session.id)

print(f"Facts: {len(facts)}")
for fact in facts:
    print(f"  - {fact.fact} (confidence: {fact.confidence:.2f})")

print(f"Entities: {len(entities)}")
for name, entity in entities.items():
    print(f"  - {name} ({entity.type}): {entity.mentions} mentions")

# 5. Get statistics
stats = client.get_session_statistics(session.id)
print(f"Duration: {stats['duration_seconds']:.1f}s")
print(f"Avg Importance: {stats['avg_importance']:.2f}")

# 6. Search similar sessions
similar = client.search_sessions(
    query="machine learning projects",
    user_id=user_id,
    k=5
)

# 7. Complete session
completed = client.complete_session(
    session_id=session.id,
    generate_summary=True
)
```

**Full Example:** `examples/10_session_management_demo.py`

---

### Automatic Boundary Detection

Sessions can automatically detect topic changes and create new sessions.

**How it works:**
1. **LLM-based** (preferred): Compares new message with session summary
2. **Fallback**: Entity overlap analysis + inactivity timeout

**Example:**
```python
# Track messages with boundary detection
result = client.track_session_message(
    session_id=current_session.id,
    text="Let's switch to discussing cloud infrastructure",
    user_id="alice",
    auto_boundary_detect=True
)

# Check if new session was created
if result.id != current_session.id:
    print("Topic changed - new session created")
    print(f"Old: {current_session.id}")
    print(f"New: {result.id}")
```

**Configuration:**
```python
from hippocampai import SessionManager

session_manager = SessionManager(
    qdrant_store=client.qdrant,
    embedder=client.embedder,
    llm=client.llm,
    collection_name="sessions",
    auto_summarize_threshold=10,      # Summarize after 10 messages
    inactivity_threshold_minutes=30   # Boundary after 30 min inactivity
)
```

---

### Use Cases

**1. Customer Support:**
```python
support_session = client.create_session(
    user_id="customer_123",
    title="Support Ticket #12345",
    tags=["support", "billing"],
    metadata={
        "ticket_id": "12345",
        "category": "billing",
        "assigned_to": "agent_42"
    }
)
```

**2. Educational Tutoring:**
```python
course_session = client.create_session(
    user_id="student_alice",
    title="Python Programming - Week 1",
    tags=["education", "programming"],
    metadata={
        "course": "python-101",
        "instructor": "bob"
    }
)

# Create lesson sub-sessions
for lesson in ["Variables", "Functions", "Loops"]:
    client.create_session(
        user_id="student_alice",
        title=f"Lesson: {lesson}",
        parent_session_id=course_session.id
    )
```

**3. Project Management:**
```python
project = client.create_session(
    user_id="alice",
    title="Q1 2024 - ML Infrastructure",
    tags=["work", "infrastructure"],
    metadata={"team": "ml-platform", "quarter": "Q1-2024"}
)

# Weekly planning sessions
weekly = client.create_session(
    user_id="alice",
    title="Week 1 - Requirements",
    parent_session_id=project.id
)
```

---

### Session Management Benefits

- **Organization**: Structure conversations logically
- **Context Retention**: Full conversation history with summaries
- **Entity Tracking**: Automatic identification of key entities
- **Fact Extraction**: Distill key information with confidence
- **Semantic Search**: Find relevant past sessions easily
- **Hierarchy**: Organize complex discussions with parent-child
- **Analytics**: Rich statistics on session activity
- **Topic Detection**: Automatic boundary detection

---

### Dependencies

- **Required**: Core HippocampAI dependencies (Qdrant, embedder)
- **Optional**: LLM provider (Ollama/OpenAI) for:
  - Summarization
  - Fact extraction
  - LLM-based entity extraction
  - LLM-based boundary detection

**Fallback:** Works without LLM using regex-based entity extraction

---

**For complete documentation, see:** [SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md)

---

## Temporal Reasoning

HippocampAI's temporal reasoning system enables time-based memory operations, chronological analysis, event sequencing, and future memory scheduling.

### Overview

**Key Features:**
- Time-range based memory queries (last week, last month, etc.)
- Chronological narrative construction
- Memory timeline creation with event extraction
- Event sequence analysis for related activities
- Memory scheduling with recurrence support
- Temporal statistics and summaries

### Time Range Queries - `get_memories_by_time_range()`

Query memories using predefined or custom time ranges.

**Signature:**
```python
def get_memories_by_time_range(
    user_id: str,
    time_range: Optional[TimeRange] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> List[Memory]
```

**Time Range Options:**
- `LAST_HOUR` - Past 60 minutes
- `LAST_DAY` - Past 24 hours
- `LAST_WEEK` - Past 7 days
- `LAST_MONTH` - Past 30 days
- `LAST_YEAR` - Past 365 days
- `TODAY` - Current day (midnight to now)
- `YESTERDAY` - Previous day
- `THIS_WEEK` - Monday to now
- `THIS_MONTH` - Month start to now
- `THIS_YEAR` - Year start to now

**Example:**
```python
from hippocampai import MemoryClient, TimeRange

client = MemoryClient()

# Get memories from last week
last_week = client.get_memories_by_time_range(
    user_id="alice",
    time_range=TimeRange.LAST_WEEK
)

# Custom time range
from datetime import datetime, timedelta, timezone
start = datetime.now(timezone.utc) - timedelta(days=3)
end = datetime.now(timezone.utc)

custom_range = client.get_memories_by_time_range(
    user_id="alice",
    start_time=start,
    end_time=end
)

# With additional filters
work_memories = client.get_memories_by_time_range(
    user_id="alice",
    time_range=TimeRange.THIS_MONTH,
    filters={"tags": ["work"]},
    limit=50
)
```

**Location:** `src/hippocampai/client.py:2052`

---

### Chronological Narratives - `build_memory_narrative()`

Generate human-readable chronological narratives from memories.

**Signature:**
```python
def build_memory_narrative(
    user_id: str,
    time_range: Optional[TimeRange] = None,
    title: Optional[str] = None
) -> str
```

**Example:**
```python
# Generate narrative for last week
narrative = client.build_memory_narrative(
    user_id="alice",
    time_range=TimeRange.LAST_WEEK,
    title="My Week in Review"
)
print(narrative)
# Output:
# My Week in Review
#
# 2024-01-15 09:00 - Morning standup meeting
# 2024-01-15 14:30 - Code review session
# 2024-01-16 10:00 - Project planning meeting
# ...
```

**Location:** `src/hippocampai/client.py:2095`

---

### Memory Timelines - `create_memory_timeline()`

Create structured timelines with extracted temporal events.

**Signature:**
```python
def create_memory_timeline(
    user_id: str,
    title: str = "Memory Timeline",
    time_range: Optional[TimeRange] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Timeline
```

**Timeline Model:**
```python
class Timeline(BaseModel):
    id: str
    user_id: str
    title: str
    events: List[TemporalEvent]
    start_time: datetime
    end_time: datetime

    def get_duration(self) -> timedelta:
        """Get timeline duration"""
```

**Temporal Event Model:**
```python
class TemporalEvent(BaseModel):
    id: str
    memory_id: str
    text: str
    timestamp: datetime
    event_type: str  # "meeting", "task", "milestone", "note"
    participants: List[str]
    location: Optional[str]
    duration: Optional[int]  # minutes
    metadata: Dict[str, Any]
```

**Example:**
```python
# Create timeline
timeline = client.create_memory_timeline(
    user_id="alice",
    title="Last Month's Activities",
    time_range=TimeRange.LAST_MONTH
)

print(f"Timeline: {timeline.title}")
print(f"Events: {len(timeline.events)}")
print(f"Duration: {timeline.get_duration()}")

# Access events
for event in timeline.events[:5]:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  {event.text}")
    if event.participants:
        print(f"  With: {', '.join(event.participants)}")
```

**Location:** `src/hippocampai/client.py:2119`

---

### Event Sequence Analysis - `analyze_event_sequences()`

Identify sequences of related events within time windows.

**Signature:**
```python
def analyze_event_sequences(
    user_id: str,
    max_gap_hours: int = 24
) -> List[List[Memory]]
```

**Example:**
```python
# Find event sequences (max 24 hour gap)
sequences = client.analyze_event_sequences(
    user_id="alice",
    max_gap_hours=24
)

print(f"Found {len(sequences)} sequences")

for i, sequence in enumerate(sequences, 1):
    print(f"\nSequence {i}:")
    print(f"  Events: {len(sequence)}")
    print(f"  Timespan: {sequence[0].created_at} to {sequence[-1].created_at}")

    for memory in sequence:
        print(f"    • {memory.text}")
```

**Use Cases:**
- Identify multi-step workflows
- Track project progression
- Detect behavioral patterns
- Group related activities

**Location:** `src/hippocampai/client.py:2154`

---

### Memory Scheduling - `schedule_memory()`

Schedule memories for future creation with optional recurrence.

**Signature:**
```python
def schedule_memory(
    text: str,
    user_id: str,
    scheduled_for: datetime,
    type: str = "fact",
    tags: Optional[List[str]] = None,
    recurrence: Optional[str] = None,  # "daily", "weekly", "monthly"
    reminder_offset: Optional[int] = None,  # minutes before
    metadata: Optional[Dict[str, Any]] = None,
) -> ScheduledMemory
```

**Scheduled Memory Model:**
```python
class ScheduledMemory(BaseModel):
    id: str
    user_id: str
    text: str
    type: MemoryType
    scheduled_for: datetime
    triggered: bool
    recurrence: Optional[str]
    reminder_offset: Optional[int]
    metadata: Dict[str, Any]
```

**Example:**
```python
from datetime import datetime, timedelta, timezone

# One-time scheduled memory
tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
scheduled = client.schedule_memory(
    text="Follow up on project proposal",
    user_id="alice",
    scheduled_for=tomorrow,
    tags=["reminder", "work"]
)

# Daily recurring memory
morning_standup = client.schedule_memory(
    text="Morning standup meeting",
    user_id="alice",
    scheduled_for=datetime.now(timezone.utc).replace(hour=9, minute=0),
    type="event",
    recurrence="daily",
    reminder_offset=15  # 15 minutes before
)

# Weekly recurring memory
next_week = datetime.now(timezone.utc) + timedelta(days=7)
weekly_review = client.schedule_memory(
    text="Weekly team retrospective",
    user_id="alice",
    scheduled_for=next_week,
    recurrence="weekly",
    tags=["meeting", "team"]
)
```

**Location:** `src/hippocampai/client.py:2176`

---

### Getting Due Memories - `get_due_scheduled_memories()`

Retrieve all scheduled memories that are due for creation.

**Signature:**
```python
def get_due_scheduled_memories() -> List[ScheduledMemory]
```

**Example:**
```python
# Check for due memories
due_memories = client.get_due_scheduled_memories()

for scheduled in due_memories:
    print(f"Due: {scheduled.text}")
    print(f"  Scheduled for: {scheduled.scheduled_for}")
    print(f"  Recurrence: {scheduled.recurrence or 'None'}")

    # Create the memory
    memory = client.remember(
        scheduled.text,
        scheduled.user_id,
        type=scheduled.type.value,
        tags=["scheduled"]
    )

    # Mark as triggered (handles recurrence automatically)
    client.trigger_scheduled_memory(scheduled.id)
```

**Location:** `src/hippocampai/client.py:2223`

---

### Triggering Scheduled Memories - `trigger_scheduled_memory()`

Mark a scheduled memory as triggered and handle recurrence.

**Signature:**
```python
def trigger_scheduled_memory(scheduled_id: str) -> bool
```

**Behavior:**
- Marks memory as triggered
- If recurrence is set, creates next occurrence:
  - `daily`: Schedules for next day
  - `weekly`: Schedules for next week
  - `monthly`: Schedules for next month

**Location:** `src/hippocampai/client.py:2235`

---

### Temporal Statistics - `get_temporal_summary()`

Get comprehensive temporal statistics for user's memories.

**Signature:**
```python
def get_temporal_summary(user_id: str) -> Dict[str, Any]
```

**Returns:**
```python
{
    "total_memories": int,
    "time_span_days": int,
    "first_memory": datetime,
    "most_recent": datetime,
    "peak_activity_hour": int,  # 0-23
    "daily_distribution": Dict[str, int],  # day -> count
    "memory_type_distribution": Dict[str, int],
    "avg_memories_per_day": float
}
```

**Example:**
```python
stats = client.get_temporal_summary(user_id="alice")

print(f"Total memories: {stats['total_memories']}")
print(f"Time span: {stats['time_span_days']} days")
print(f"Peak activity: {stats['peak_activity_hour']}:00")

# Daily distribution
for day, count in stats['daily_distribution'].items():
    bar = '█' * (count // 10)
    print(f"{day}: {count:3d} {bar}")
```

**Location:** `src/hippocampai/client.py:2246`

---

### Temporal Reasoning Use Cases

**1. Personal Memory Journal:**
```python
# Build monthly narratives
narrative = client.build_memory_narrative(
    user_id="alice",
    time_range=TimeRange.THIS_MONTH,
    title="January 2024 - Month in Review"
)
```

**2. Task Management:**
```python
# Schedule recurring reminders
daily_checkin = client.schedule_memory(
    text="Daily progress update",
    user_id="alice",
    scheduled_for=datetime.now(timezone.utc).replace(hour=17, minute=0),
    recurrence="daily"
)
```

**3. Activity Analysis:**
```python
# Analyze work patterns
sequences = client.analyze_event_sequences(
    user_id="alice",
    max_gap_hours=4
)
print(f"Found {len(sequences)} work sessions")
```

**4. Time-based Insights:**
```python
# Peak productivity analysis
stats = client.get_temporal_summary(user_id="alice")
print(f"Most active hour: {stats['peak_activity_hour']}:00")
```

---

## Cross-Session Insights

HippocampAI's insight system analyzes memories across sessions to detect behavioral patterns, track changes, identify habits, and analyze long-term trends.

### Overview

**Key Features:**
- Pattern detection (recurring, sequential, correlational)
- Behavioral change tracking between time periods
- Preference drift analysis with timeline evolution
- Habit formation detection with multi-factor scoring
- Long-term trend analysis with direction and strength

### Pattern Detection - `detect_patterns()`

Detect behavioral patterns across memories and sessions.

**Signature:**
```python
def detect_patterns(
    user_id: str,
    session_ids: Optional[List[str]] = None
) -> List[Pattern]
```

**Pattern Types:**
- `recurring` - Repeated behaviors or activities
- `sequential` - Ordered sequences of events
- `correlational` - Co-occurring activities

**Pattern Model:**
```python
class Pattern(BaseModel):
    id: str
    user_id: str
    pattern_type: str
    description: str
    confidence: float  # 0.0-1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    memory_ids: List[str]
    session_ids: List[str]
    frequency: Optional[str]  # "daily", "weekly", etc.
    metadata: Dict[str, Any]
```

**Example:**
```python
# Detect all patterns
patterns = client.detect_patterns(user_id="alice")

for pattern in patterns[:5]:
    print(f"{pattern.pattern_type.upper()}: {pattern.description}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Occurrences: {pattern.occurrences}")
    print(f"  Frequency: {pattern.frequency or 'irregular'}")
    print(f"  First seen: {pattern.first_seen}")

# Detect patterns in specific sessions
patterns = client.detect_patterns(
    user_id="alice",
    session_ids=["session_1", "session_2"]
)
```

**Location:** `src/hippocampai/client.py:2265`

---

### Behavioral Change Tracking - `track_behavior_changes()`

Track changes in user behavior between time periods.

**Signature:**
```python
def track_behavior_changes(
    user_id: str,
    comparison_days: int = 30,
) -> List[BehaviorChange]
```

**Change Types:**
- `PREFERENCE_SHIFT` - Changed preferences
- `HABIT_FORMED` - New habit established
- `HABIT_BROKEN` - Habit discontinued
- `GOAL_ACHIEVED` - Goal completed
- `GOAL_ABANDONED` - Goal dropped
- `INTEREST_GAINED` - New interest area
- `INTEREST_LOST` - Reduced interest
- `BEHAVIOR_PATTERN` - General behavior change

**BehaviorChange Model:**
```python
class BehaviorChange(BaseModel):
    id: str
    user_id: str
    change_type: ChangeType
    description: str
    confidence: float
    before_value: Optional[str]
    after_value: Optional[str]
    change_magnitude: Optional[float]
    detected_at: datetime
    evidence_memory_ids: List[str]
    evidence_session_ids: List[str]
    metadata: Dict[str, Any]
```

**Example:**
```python
# Compare last 30 days vs previous period
changes = client.track_behavior_changes(
    user_id="alice",
    comparison_days=30
)

for change in changes:
    print(f"{change.change_type.value.upper()}")
    print(f"  {change.description}")
    print(f"  Confidence: {change.confidence:.2f}")

    if change.before_value and change.after_value:
        print(f"  Before: {change.before_value}")
        print(f"  After: {change.after_value}")

    if change.change_magnitude:
        print(f"  Magnitude: {change.change_magnitude:.2f}")
```

**Location:** `src/hippocampai/client.py:2295`

---

### Preference Drift Analysis - `analyze_preference_drift()`

Analyze how user preferences evolve over time.

**Signature:**
```python
def analyze_preference_drift(
    user_id: str,
    category: Optional[str] = None
) -> List[PreferenceDrift]
```

**PreferenceDrift Model:**
```python
class PreferenceDrift(BaseModel):
    id: str
    user_id: str
    category: str
    original_preference: str
    current_preference: str
    drift_score: float  # 0.0 (stable) to 1.0 (complete change)
    timeline: List[Tuple[datetime, str]]  # Evolution history
    memory_ids: List[str]
    first_recorded: datetime
    last_updated: datetime
    metadata: Dict[str, Any]
```

**Example:**
```python
# Analyze all preference drifts
drifts = client.analyze_preference_drift(user_id="alice")

for drift in drifts:
    print(f"Category: {drift.category}")
    print(f"  Original: {drift.original_preference}")
    print(f"  Current: {drift.current_preference}")
    print(f"  Drift score: {drift.drift_score:.2f}")
    print(f"  Timeline points: {len(drift.timeline)}")

    # Show evolution
    for timestamp, value in drift.timeline:
        print(f"    {timestamp.date()}: {value}")

# Analyze specific category
food_drifts = client.analyze_preference_drift(
    user_id="alice",
    category="food"
)
```

**Location:** `src/hippocampai/client.py:2325`

---

### Habit Detection - `detect_habits()`

Detect and score potential habits from behavioral patterns.

**Signature:**
```python
def detect_habits(
    user_id: str,
    min_occurrences: int = 5
) -> List[HabitScore]
```

**HabitScore Model:**
```python
class HabitScore(BaseModel):
    id: str
    user_id: str
    behavior: str
    habit_score: float  # 0.0-1.0
    consistency: float  # Regularity of occurrence
    frequency: int  # Number of occurrences
    recency: float  # How recent
    duration: int  # Days active
    status: str  # "forming", "established", "breaking", "broken"
    occurrences: List[datetime]
    memory_ids: List[str]
    detected_at: datetime
    metadata: Dict[str, Any]
```

**Scoring Factors:**
- **Consistency** (40%): Regularity of occurrence
- **Frequency** (30%): Number of times performed
- **Recency** (20%): How recently performed
- **Duration** (10%): How long the behavior has been tracked

**Example:**
```python
# Detect habits (minimum 5 occurrences)
habits = client.detect_habits(
    user_id="alice",
    min_occurrences=5
)

# Habits are sorted by score (highest first)
for habit in habits[:5]:
    print(f"Behavior: {habit.behavior}")
    print(f"  Habit score: {habit.habit_score:.2f}")
    print(f"  Status: {habit.status.upper()}")
    print(f"  Consistency: {habit.consistency:.2f}")
    print(f"  Frequency: {habit.frequency} times")
    print(f"  Duration: {habit.duration} days")

    # Show recent occurrences
    recent = habit.occurrences[-3:]
    for occ in recent:
        print(f"    • {occ.date()}")
```

**Status Classification:**
- `forming`: score < 0.4, duration < 21 days
- `established`: score >= 0.7
- `breaking`: score declining, recent occurrences reduced
- `broken`: no recent occurrences

**Location:** `src/hippocampai/client.py:2348`

---

### Trend Analysis - `analyze_trends()`

Analyze long-term trends in user behavior.

**Signature:**
```python
def analyze_trends(
    user_id: str,
    window_days: int = 30
) -> List[Trend]
```

**Trend Model:**
```python
class Trend(BaseModel):
    id: str
    user_id: str
    category: str
    trend_type: str  # "increasing", "decreasing", "stable", "cyclical"
    description: str
    confidence: float
    data_points: List[Tuple[datetime, float]]
    direction: str  # "up", "down", "flat"
    strength: float  # 0.0-1.0
    detected_at: datetime
    metadata: Dict[str, Any]
```

**Example:**
```python
# Analyze trends over last 30 days
trends = client.analyze_trends(
    user_id="alice",
    window_days=30
)

for trend in trends:
    print(f"Category: {trend.category}")
    print(f"  Trend: {trend.trend_type}")
    print(f"  Direction: {trend.direction.upper()}")
    print(f"  Strength: {trend.strength:.2f}")
    print(f"  Confidence: {trend.confidence:.2f}")
    print(f"  Description: {trend.description}")

    # Visual indicator
    if trend.direction == "up":
        print("  📈 ↗️")
    elif trend.direction == "down":
        print("  📉 ↘️")
    else:
        print("  📊 ➡️")
```

**Location:** `src/hippocampai/client.py:2370`

---

### Cross-Session Insights Use Cases

**1. Personalized Recommendations:**
```python
# Detect patterns to recommend activities
patterns = client.detect_patterns(user_id="alice")
recurring = [p for p in patterns if p.pattern_type == "recurring"]
for pattern in recurring:
    recommend_based_on_pattern(pattern)
```

**2. Behavior Change Interventions:**
```python
# Detect broken habits and intervene
habits = client.detect_habits(user_id="alice")
breaking = [h for h in habits if h.status == "breaking"]
for habit in breaking:
    send_motivation_message(habit.behavior)
```

**3. Preference Adaptation:**
```python
# Adapt UI based on preference drift
drifts = client.analyze_preference_drift(user_id="alice")
for drift in drifts:
    if drift.drift_score > 0.7:  # Significant change
        update_recommendations(drift.category, drift.current_preference)
```

**4. Progress Tracking:**
```python
# Track behavioral changes for goal achievement
changes = client.track_behavior_changes(user_id="alice")
achievements = [c for c in changes if c.change_type == ChangeType.GOAL_ACHIEVED]
for achievement in achievements:
    celebrate_milestone(achievement.description)
```

**5. Long-term Analytics:**
```python
# Analyze trends for insights
trends = client.analyze_trends(user_id="alice", window_days=90)
positive_trends = [t for t in trends if t.direction == "up" and t.strength > 0.7]
for trend in positive_trends:
    report_positive_change(trend.category, trend.description)
```

---

### Insights Integration Example

```python
from hippocampai import MemoryClient

client = MemoryClient()
user_id = "alice"

# 1. Detect patterns
patterns = client.detect_patterns(user_id)
print(f"Found {len(patterns)} patterns")

# 2. Track changes
changes = client.track_behavior_changes(user_id, comparison_days=30)
print(f"Detected {len(changes)} behavioral changes")

# 3. Analyze preferences
drifts = client.analyze_preference_drift(user_id)
print(f"Found {len(drifts)} preference drifts")

# 4. Identify habits
habits = client.detect_habits(user_id, min_occurrences=5)
established = [h for h in habits if h.status == "established"]
print(f"{len(established)} established habits")

# 5. Analyze trends
trends = client.analyze_trends(user_id, window_days=30)
increasing = [t for t in trends if t.direction == "up"]
print(f"{len(increasing)} increasing trends")

# Generate comprehensive report
report = {
    "patterns": len(patterns),
    "changes": len(changes),
    "preference_drifts": len(drifts),
    "established_habits": len(established),
    "positive_trends": len(increasing)
}
```

**Full Examples:**
- `examples/13_temporal_reasoning_demo.py`
- `examples/14_cross_session_insights_demo.py`

---

## Real-Time Incremental Knowledge Graph

**NEW in v0.5.0** — Automatic entity and relationship extraction on every `remember()` call, building a persistent knowledge graph alongside the vector store.

### Overview

When `ENABLE_REALTIME_GRAPH=true`, every call to `client.remember()` automatically extracts entities, facts, relationships, and topics from the memory text and adds them to a `KnowledgeGraph`. The graph supports three node types: **entity**, **fact**, and **topic**.

### Data Flow

```
remember("Alice works at Google on TensorFlow")
    │
    ├──► Qdrant: store embedding vector + payload
    │
    ├──► BM25: index text for keyword search
    │
    └──► KnowledgeGraph (NetworkX):
         ├─ Extract entities: Alice (person), Google (org), TensorFlow (tech)
         ├─ Extract relationships: Alice→works_at→Google, Alice→works_on→TensorFlow
         ├─ Link memory ID to each entity node
         └─ Auto-save graph to JSON (every 300s)

recall("What frameworks does Alice use?")
    │
    ├──► Vector search (Qdrant) → ranked list #1
    ├──► BM25 keyword search → ranked list #2
    ├──► Graph traversal (NetworkX):
    │    ├─ Extract "Alice" from query
    │    ├─ Find Alice node → traverse edges (max_depth=2)
    │    ├─ Score: entity_confidence * edge_weight / (1 + hop_distance)
    │    └─ Return ranked list #3
    │
    └──► Reciprocal Rank Fusion (3-way) → final results
         Weights: sim + rerank + recency + importance + graph + feedback
```

### Key Methods

```python
from hippocampai.graph.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()

# Add entities and link to memories
kg.add_entity(name="Python", entity_type="technology", metadata={})
kg.link_memory_to_entity(memory_id="mem_123", entity_name="Python")

# Query entity timeline
timeline = kg.get_entity_timeline("Python")

# Infer new facts from graph structure
inferred = kg.infer_new_facts(user_id="alice")
```

### Persistence

The graph is persisted to JSON at the path specified by `GRAPH_PERSISTENCE_PATH` (default: `data/knowledge_graph.json`). Auto-save runs every `GRAPH_AUTO_SAVE_INTERVAL` seconds (default: 300).

### Extraction Modes

- **`pattern`** (default): Fast regex-based extraction of entities and relationships. No LLM required.
- **`llm`**: Uses the configured LLM provider for deeper semantic extraction. Higher quality but slower.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REALTIME_GRAPH` | `true` | Enable auto-extraction |
| `GRAPH_EXTRACTION_MODE` | `pattern` | `pattern` or `llm` |
| `GRAPH_PERSISTENCE_PATH` | `data/knowledge_graph.json` | Persistence file path |
| `GRAPH_AUTO_SAVE_INTERVAL` | `300` | Auto-save interval (seconds) |

### Example

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Storing a memory automatically extracts entities and relationships
memory = client.remember(
    "Alice works at Google on the TensorFlow team",
    user_id="alice"
)
# Knowledge graph now contains:
#   entities: Alice (person), Google (organization), TensorFlow (technology)
#   relationships: Alice -> works_at -> Google, Alice -> works_on -> TensorFlow
```

---

## Graph-Aware Retrieval

**NEW in v0.5.0** — Augment vector + BM25 retrieval with graph-based scoring for deeper contextual recall.

### Overview

When `ENABLE_GRAPH_RETRIEVAL=true`, the `GraphRetriever` adds a third scoring signal alongside vector similarity and BM25. The process:

1. Extract entities from the query
2. Traverse the knowledge graph from matched entities up to `GRAPH_RETRIEVAL_MAX_DEPTH` hops
3. Score connected memories: `entity_confidence * edge_weight / (1 + hop_distance)`
4. Fuse all three signals via Reciprocal Rank Fusion (RRF): **vector + BM25 + graph**

### Search Mode

A new `GRAPH_HYBRID` search mode is available in the `SearchMode` enum, enabling 3-way RRF fusion.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_GRAPH_RETRIEVAL` | `false` | Enable graph retrieval |
| `GRAPH_RETRIEVAL_MAX_DEPTH` | `2` | Max traversal depth |
| `WEIGHT_GRAPH` | `0.0` | Graph weight in score fusion |

### Example

```python
from hippocampai import MemoryClient

client = MemoryClient()

# With graph retrieval enabled, recall uses 3-way fusion
results = client.recall(
    query="What frameworks does Alice use?",
    user_id="alice",
    k=5
)

# Each result includes graph score in breakdown
for r in results:
    print(f"{r.memory.text} (score: {r.score:.3f})")
    print(f"  Breakdown: {r.breakdown}")
    # breakdown includes: sim, rerank, recency, importance, graph
```

---

## Memory Relevance Feedback Loop

**NEW in v0.5.0** — Collect user feedback on retrieved memories to improve future retrieval quality.

### Overview

Users can rate retrieved memories as `relevant`, `not_relevant`, `partially_relevant`, or `outdated`. Feedback is aggregated using an exponentially-weighted rolling average with a 30-day half-life and integrated into retrieval scoring via `WEIGHT_FEEDBACK`.

### Feedback Types

| Type | Effect |
|------|--------|
| `relevant` | Boosts memory score (1.0) |
| `not_relevant` | Penalizes memory score (0.0) |
| `partially_relevant` | Mild boost (0.5) |
| `outdated` | Penalizes and flags for review (0.0) |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/memories/{memory_id}/feedback` | Submit feedback |
| `GET` | `/v1/memories/{memory_id}/feedback` | Get aggregated score |
| `GET` | `/v1/feedback/stats` | Get user feedback statistics |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHT_FEEDBACK` | `0.1` | Feedback weight in score fusion |
| `FEEDBACK_WINDOW_DAYS` | `90` | Lookback window for feedback events |

### Example

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Recall and get results
results = client.recall("coffee preferences", user_id="alice")

# Rate the first result as relevant
client.rate_recall(
    memory_id=results[0].memory.id,
    user_id="alice",
    query="coffee preferences",
    feedback_type="relevant"
)

# Future recalls will boost this memory's score
```

---

## Memory Triggers / Event-Driven Actions

**NEW in v0.5.0** — Register triggers that fire actions (webhooks, websocket messages, logs) when memory events occur.

### Overview

Triggers allow you to react to memory lifecycle events with configurable conditions and actions.

### Trigger Events

| Event | Fires When |
|-------|------------|
| `on_remember` | New memory created |
| `on_recall` | Memory retrieved |
| `on_update` | Memory updated |
| `on_delete` | Memory deleted |
| `on_conflict` | Conflict detected |
| `on_expire` | Memory expired |

### Trigger Actions

| Action | Description |
|--------|-------------|
| `webhook` | POST to configured URL |
| `websocket` | Send via WebSocket |
| `log` | Write to application log |

### Condition Operators

Triggers support conditions with operators: `eq`, `gt`, `lt`, `contains`, `matches`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/triggers` | Create a trigger |
| `GET` | `/v1/triggers?user_id=...` | List triggers |
| `DELETE` | `/v1/triggers/{trigger_id}?user_id=...` | Delete a trigger |
| `GET` | `/v1/triggers/{trigger_id}/history` | Get fire history |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRIGGERS` | `true` | Enable trigger system |
| `TRIGGER_WEBHOOK_TIMEOUT` | `10` | Webhook timeout (seconds) |

### Example

```python
import httpx

# Create a webhook trigger for high-importance memories
response = httpx.post("http://localhost:8000/v1/triggers", json={
    "name": "High importance alert",
    "user_id": "alice",
    "event": "on_remember",
    "conditions": [
        {"field": "importance", "operator": "gt", "value": 8.0}
    ],
    "action": "webhook",
    "action_config": {
        "url": "https://hooks.example.com/memory-alert"
    }
})

trigger = response.json()
print(f"Trigger created: {trigger['id']}")
```

---

## Procedural Memory / Prompt Self-Optimization

**NEW in v0.5.0** — Extract, store, and inject behavioral rules that optimize LLM prompts based on interaction history.

### Overview

Procedural memory automatically learns "how to" rules from user interactions and injects them into prompts to improve response quality over time.

### Key Capabilities

1. **Rule Extraction**: Analyze interactions to extract behavioral rules (LLM-based with heuristic fallback)
2. **Rule Injection**: Inject relevant rules into prompts before LLM calls
3. **Effectiveness Tracking**: Track rule success via EMA: `0.9 * old_rate + 0.1 * new_outcome`
4. **Rule Consolidation**: Merge redundant rules via LLM to stay under `PROCEDURAL_RULE_MAX_COUNT`

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/procedural/rules?user_id=...` | List active rules |
| `POST` | `/v1/procedural/extract` | Extract rules from interactions |
| `POST` | `/v1/procedural/inject` | Inject rules into a prompt |
| `PUT` | `/v1/procedural/rules/{rule_id}/feedback` | Update rule effectiveness |
| `POST` | `/v1/procedural/consolidate?user_id=...` | Consolidate redundant rules |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PROCEDURAL_MEMORY` | `false` | Enable procedural memory |
| `PROCEDURAL_RULE_MAX_COUNT` | `50` | Max rules per user |
| `HALF_LIFE_PROCEDURAL` | `180` | Rule decay half-life (days) |

### Example

```python
import httpx

# Extract rules from recent interactions
response = httpx.post("http://localhost:8000/v1/procedural/extract", json={
    "user_id": "alice",
    "interactions": [
        "User prefers concise answers under 100 words",
        "User always wants code examples in Python",
        "User dislikes technical jargon"
    ]
})

rules = response.json()
print(f"Extracted {len(rules)} rules")

# Inject rules into a prompt
response = httpx.post("http://localhost:8000/v1/procedural/inject", json={
    "user_id": "alice",
    "base_prompt": "Explain how neural networks work",
    "max_rules": 3
})

result = response.json()
print(f"Enhanced prompt with {result['rules_injected']} rules")
print(result["prompt"])
```

---

## Embedding Model Migration

**NEW in v0.5.0** — Safely migrate all stored embeddings when changing the embedding model.

### Overview

When you change `EMBED_MODEL`, existing vectors become incompatible with new queries. The migration system provides:

1. **Model change detection**: `detect_model_change()` compares the current model against `EMBED_MODEL_VERSION`
2. **Migration workflow**: Start a migration that re-encodes all memories in batches
3. **Celery background task**: `migrate_embeddings_task` with 1-hour soft time limit
4. **Progress tracking**: Track `migrated_count`, `failed_count`, and status

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/admin/embeddings/migrate` | Start migration |
| `GET` | `/v1/admin/embeddings/migration/{id}` | Check status |
| `POST` | `/v1/admin/embeddings/migration/{id}/cancel` | Cancel migration |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL_VERSION` | `1` | Current model version tag |

### Example

```python
import httpx

# Start migration to a new model
response = httpx.post("http://localhost:8000/v1/admin/embeddings/migrate", json={
    "new_model": "BAAI/bge-base-en-v1.5",
    "new_dimension": 768
})

migration = response.json()
print(f"Migration started: {migration['id']}")
print(f"Status: {migration['status']}")
print(f"Total memories: {migration['total_memories']}")

# Check progress
status = httpx.get(f"http://localhost:8000/v1/admin/embeddings/migration/{migration['id']}")
print(f"Migrated: {status.json()['migrated_count']}")
print(f"Failed: {status.json()['failed_count']}")
```

### Migration States

| Status | Description |
|--------|-------------|
| `pending` | Migration created but not started |
| `running` | Re-encoding in progress |
| `completed` | All memories re-encoded |
| `cancelled` | Migration cancelled by user |
| `failed` | Migration failed (check logs) |

---

## Storage & Caching

### Key-Value Store

In-memory KV store with TTL support for fast lookups.

**Features:**
- O(1) access by memory ID
- User and tag indexing
- Automatic cache expiration
- Statistics tracking
- Memory-specific optimizations

**Methods:**
```python
# Automatic usage (internal)
client.kv_store.set_memory(memory_id, memory_data)
retrieved = client.kv_store.get_memory(memory_id)

# Index queries
user_memories = client.kv_store.get_user_memories("alice")
tagged_memories = client.kv_store.get_memories_by_tag("work")

# Statistics
stats = client.kv_store.get_statistics()
print(f"Cache size: {stats['total_entries']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

**Location:** `src/hippocampai/storage/kv_store.py`

---

### Memory Snapshots

Point-in-time backups using Qdrant's native snapshot functionality.

**Method:**
```python
def create_snapshot(collection: str = "facts") -> str
```

**Example:**
```python
# Create snapshot of facts collection
snapshot_id = client.create_snapshot("facts")
print(f"Snapshot created: {snapshot_id}")

# Create snapshot of preferences
snapshot_id = client.create_snapshot("prefs")

# Restore using Qdrant CLI
# qdrant-console snapshot restore <snapshot_id>
```

**Location:** `src/hippocampai/client.py:883`

---

## Monitoring & Telemetry

### Operation Tracking

Full telemetry for all memory operations with traces, events, and metrics.

**Operation Types:**
- `REMEMBER` - Memory creation
- `RECALL` - Memory retrieval
- `UPDATE` - Memory modification
- `DELETE` - Memory deletion
- `GET` - get_memories() calls
- `EXPIRE` - TTL cleanup
- `EXTRACT` - Conversation extraction

**Methods:**
```python
# Get metrics summary
metrics = client.get_telemetry_metrics()
print(f"Total operations: {metrics['total_operations']}")
print(f"Average duration: {metrics['avg_duration_ms']:.2f}ms")

# Get recent operations
operations = client.get_recent_operations(limit=10)
for op in operations:
    print(f"{op['operation']}: {op['duration_ms']:.2f}ms")

# Filter by operation type
recalls = client.get_recent_operations(
    limit=20,
    operation="recall"
)

# Export for analysis
telemetry_data = client.export_telemetry()
```

**Location:** `src/hippocampai/telemetry.py`

---

## API Reference

### Complete Method List

#### Memory CRUD
| Method | Purpose | Returns |
|--------|---------|---------|
| `remember()` | Create memory | `Memory` |
| `recall()` | Semantic search | `List[RetrievalResult]` |
| `get_memories()` | Filter-based retrieval | `List[Memory]` |
| `update_memory()` | Modify memory | `Optional[Memory]` |
| `delete_memory()` | Remove memory | `bool` |

#### Batch Operations
| Method | Purpose | Returns |
|--------|---------|---------|
| `add_memories()` | Batch create | `List[Memory]` |
| `delete_memories()` | Batch delete | `int` |

#### Advanced Filtering
| Method | Purpose | Returns |
|--------|---------|---------|
| `get_memories_advanced()` | Filter + sort | `List[Memory]` |
| `expire_memories()` | Remove expired | `int` |
| `track_memory_access()` | Update access count | `None` |

#### Graph Operations
| Method | Purpose | Returns |
|--------|---------|---------|
| `add_relationship()` | Create edge | `bool` |
| `get_related_memories()` | Traverse graph | `List[Tuple]` |
| `get_memory_clusters()` | Find communities | `List[Set]` |

#### Version Control
| Method | Purpose | Returns |
|--------|---------|---------|
| `get_memory_history()` | Version list | `List[MemoryVersion]` |
| `rollback_memory()` | Revert changes | `Optional[Memory]` |
| `get_audit_trail()` | Change log | `List[AuditEntry]` |

#### Context & Utilities
| Method | Purpose | Returns |
|--------|---------|---------|
| `inject_context()` | Format for LLM | `str` |
| `create_snapshot()` | Backup collection | `str` |
| `extract_from_conversation()` | Parse conversation | `List[Memory]` |

#### Telemetry
| Method | Purpose | Returns |
|--------|---------|---------|
| `get_telemetry_metrics()` | Usage stats | `Dict` |
| `get_recent_operations()` | Operation history | `List[Dict]` |
| `export_telemetry()` | Export data | `List[Dict]` |

---

## Performance Notes

### Operation Complexity
- **Create**: O(1) + embedding time
- **Update**: O(1) lookup + O(n) re-embedding (if text changes)
- **Delete**: O(1) per collection
- **Get with filters**: O(n) scan with indexed filters
- **Semantic search**: O(log n) + reranking
- **Graph traversal**: O(V + E) where V = vertices, E = edges
- **Version control**: O(1) per version
- **Batch operations**: O(n) where n = batch size

### Optimization Tips
1. Use `get_memories()` for filtering without semantic search
2. Leverage KV store for frequent ID-based lookups
3. Batch operations when creating/deleting multiple memories
4. Use tag indices for fast tag-based filtering
5. Limit graph depth for performance
6. Configure version history size based on needs

---

## Dependencies

### Core Dependencies
- `qdrant-client` - Vector database
- `sentence-transformers` - Embedding generation
- `rank-bm25` - Keyword search
- `pydantic` - Data validation

### Advanced Features
- `networkx` - Graph operations (install with `pip install networkx`)

### Optional
- `ollama` or `openai` - LLM providers for extraction/scoring

---

## Testing

### Test Coverage
**Total:** 47 comprehensive tests (100% passing)

**Test Suites:**
1. `tests/test_new_features.py` - 21 tests for core CRUD operations
2. `tests/test_advanced_features.py` - 26 tests for advanced features

**Run Tests:**
```bash
# All tests
pytest tests/ -v

# Core features only
pytest tests/test_new_features.py -v

# Advanced features only
pytest tests/test_advanced_features.py -v

# With coverage
pytest tests/ --cov=hippocampai --cov-report=html
```

---

## Examples

### Quick Start Examples

#### Example 1: Basic CRUD
```python
from hippocampai import MemoryClient

client = MemoryClient()

# Create
memory = client.remember(
    "I love Python programming",
    user_id="alice",
    tags=["programming", "preferences"]
)

# Read
results = client.recall("programming", user_id="alice")

# Update
updated = client.update_memory(
    memory.id,
    text="I love Python and machine learning",
    importance=9.0
)

# Delete
deleted = client.delete_memory(memory.id, user_id="alice")
```

#### Example 2: Advanced Filtering
```python
# Complex filters
memories = client.get_memories_advanced(
    user_id="alice",
    filters={
        "tags": ["work"],
        "min_importance": 7.0,
        "metadata": {"project": "HippocampAI"}
    },
    sort_by="access_count",
    sort_order="desc",
    limit=20
)
```

#### Example 3: Graph Relationships
```python
from hippocampai import RelationType

# Create related memories
m1 = client.remember("Deep learning uses neural networks", user_id="alice")
m2 = client.remember("TensorFlow is a DL framework", user_id="alice")

# Add to graph
client.graph.add_memory(m1.id, "alice", {})
client.graph.add_memory(m2.id, "alice", {})

# Create relationship
client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO)

# Find related
related = client.get_related_memories(m1.id, max_depth=2)
```

### Full Examples
- `examples/06_advanced_memory_management.py` - Core CRUD operations
- `examples/07_advanced_features_demo.py` - All advanced features

**Run Examples:**
```bash
python examples/06_advanced_memory_management.py
python examples/07_advanced_features_demo.py
```

---

## Migration Guide

### From v0.1.5 to V0.2.5

**All changes are backward compatible.** Existing code will work without modifications.

#### New Capabilities
```python
# Before: Limited retrieval
results = client.recall("coffee", user_id="alice")

# After: Advanced filtering
results = client.recall(
    "coffee",
    user_id="alice",
    filters={"tags": "beverages", "min_importance": 7.0}
)

# After: Direct memory management
client.update_memory(memory_id, importance=9.0)
client.delete_memory(memory_id, user_id="alice")

# After: Batch operations
client.add_memories([...], user_id="alice")
client.delete_memories([id1, id2, id3], user_id="alice")
```

---

## Change Log

### Version 0.2.5 (Current)

**Added - Core Features:**
- ✅ `update_memory()` - Modify existing memories
- ✅ `delete_memory()` - Remove memories by ID
- ✅ `get_memories()` - Advanced filtering without semantic search
- ✅ Tag-based filtering in `recall()`
- ✅ Memory TTL with `ttl_days` and `expire_memories()`

**Added - Advanced Features:**
- ✅ Graph index with 8 relationship types
- ✅ Key-value store for O(1) lookups
- ✅ Version control with diff and rollback
- ✅ Context injection helpers for LLM prompts
- ✅ Batch operations (`add_memories`, `delete_memories`)
- ✅ Memory access tracking
- ✅ Advanced metadata filtering and sorting
- ✅ Snapshot management
- ✅ Complete audit trail

**Enhanced:**
- Memory model with `expires_at` field
- QdrantStore with `get()` and `update()` methods
- Tag payload indexing
- Telemetry with 4 new operation types
- Comprehensive test coverage (47 tests)

**Files Modified/Created:**
- Core: `client.py`, `models/memory.py`, `vector/qdrant_store.py`, `telemetry.py`
- New modules: `graph/`, `storage/`, `versioning/`, `utils/context_injection.py`
- Tests: `test_new_features.py`, `test_advanced_features.py`
- Examples: `06_advanced_memory_management.py`, `07_advanced_features_demo.py`
- Docs: `FEATURES.md` (this file)

---

## Roadmap

### Completed ✅
- [x] Core CRUD operations
- [x] Advanced filtering and search
- [x] Memory TTL and lifecycle
- [x] Graph relationships
- [x] Version control and audit
- [x] Batch operations
- [x] Context injection
- [x] KV store caching
- [x] Comprehensive testing
- [x] Real-Time Incremental Knowledge Graph (v0.5.0)
- [x] Graph-Aware Retrieval with 3-way RRF fusion (v0.5.0)
- [x] Memory Relevance Feedback Loop (v0.5.0)
- [x] Memory Triggers / Event-Driven Actions (v0.5.0)
- [x] Procedural Memory / Prompt Self-Optimization (v0.5.0)
- [x] Embedding Model Migration (v0.5.0)

### Future Enhancements 🚀
- [ ] Persistent graph storage (Neo4j integration) — **WIP** (currently uses NetworkX in-memory + JSON persistence)
- [ ] Redis backend for KV store
- [ ] Multi-user permission system
- [ ] Advanced analytics dashboard

---

## Summary

**All 15 features successfully implemented and tested:**

✅ **High Priority (5)**
1. update_memory() - Modify memories
2. delete_memory() - Remove memories
3. get_memories() - Advanced filtering
4. Tag-based filtering - In recall()
5. Memory TTL - Automatic expiration

✅ **Medium Priority (4)**
6. Graph Index - Relationship mapping
7. Key-Value Store - Fast lookups
8. Version Control - Change tracking
9. Context Injection - LLM prompt helpers

✅ **Low Priority (6)**
10. Batch Operations - Efficiency
11. Access Tracking - Usage stats
12. Advanced Filters - Metadata support
13. Snapshots - Point-in-time backups
14. Audit Trail - Complete changelog
15. get_memories_advanced() - Enhanced filtering

**Status:** Production-ready with 47/47 tests passing ✓

---

*For more details, see individual examples and test files.*
