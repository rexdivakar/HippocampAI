# Session Management Guide

Comprehensive guide for HippocampAI's session management features, including conversation tracking, summarization, entity extraction, and hierarchical organization.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [Creating & Managing Sessions](#creating--managing-sessions)
5. [Message Tracking](#message-tracking)
6. [Auto-Summarization](#auto-summarization)
7. [Entity & Fact Extraction](#entity--fact-extraction)
8. [Session Search](#session-search)
9. [Hierarchical Sessions](#hierarchical-sessions)
10. [Boundary Detection](#boundary-detection)
11. [Session Analytics](#session-analytics)
12. [Advanced Usage](#advanced-usage)
13. [Best Practices](#best-practices)
14. [API Reference](#api-reference)

---

## Overview

HippocampAI's session management system allows you to organize conversations into structured sessions with automatic summarization, entity tracking, and semantic search capabilities.

### Key Features

- **Session Lifecycle Management** - Create, update, complete, and delete sessions
- **Automatic Summarization** - LLM-powered summaries when sessions reach threshold
- **Entity Extraction** - Track people, technologies, locations, organizations automatically
- **Fact Extraction** - Extract key facts with confidence scores
- **Session Search** - Semantic similarity search across all sessions
- **Hierarchical Organization** - Parent-child relationships for complex discussions
- **Boundary Detection** - Automatic topic change detection
- **Rich Metadata** - Tags, custom metadata, and statistics

### When to Use Sessions

Sessions are ideal for:
- Multi-turn conversations with clear topics
- Project-based discussions
- Meeting notes and transcripts
- Customer support interactions
- Educational tutoring sessions
- Complex workflows with sub-tasks

---

## Core Concepts

### Session

A container for related messages and memories with metadata.

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

### Session Status

```python
class SessionStatus(str, Enum):
    ACTIVE = "active"          # Currently active
    INACTIVE = "inactive"      # No recent activity
    COMPLETED = "completed"    # Explicitly completed
    ARCHIVED = "archived"      # Archived for long-term storage
```

### Entity

Represents a named entity mentioned in the session.

```python
class Entity(BaseModel):
    name: str                    # Entity name
    type: str                    # person, organization, technology, etc.
    mentions: int                # Number of times mentioned
    first_mentioned_at: datetime # First occurrence
    last_mentioned_at: datetime  # Most recent occurrence
    metadata: Dict[str, Any]     # Additional context
```

### SessionFact

Key fact extracted from the session.

```python
class SessionFact(BaseModel):
    fact: str                    # Fact statement
    confidence: float            # 0.0-1.0 confidence score
    extracted_at: datetime       # Extraction timestamp
    sources: List[str]           # Memory IDs supporting this fact
```

---

## Quick Start

### Basic Session Workflow

```python
from hippocampai import MemoryClient, SessionStatus

# Initialize client
client = MemoryClient()
user_id = "alice"

# 1. Create a new session
session = client.create_session(
    user_id=user_id,
    title="Project Planning Meeting",
    tags=["work", "planning"]
)

# 2. Track messages
for message in conversation_messages:
    session = client.track_session_message(
        session_id=session.id,
        text=message,
        user_id=user_id
    )

# 3. Generate summary
summary = client.summarize_session(session.id)
print(f"Summary: {summary}")

# 4. Extract facts and entities
facts = client.extract_session_facts(session.id)
entities = client.extract_session_entities(session.id)

# 5. Complete the session
completed = client.complete_session(
    session_id=session.id,
    generate_summary=True
)
```

---

## Creating & Managing Sessions

### Create a Session

```python
# Minimal creation
session = client.create_session(
    user_id="alice",
    title="Quick Discussion"
)

# With metadata and tags
session = client.create_session(
    user_id="alice",
    title="Q1 Planning - ML Project",
    tags=["work", "machine-learning", "quarterly"],
    metadata={
        "project": "sentiment-analysis",
        "team": "data-science",
        "priority": "high"
    }
)

# With parent session (hierarchical)
child_session = client.create_session(
    user_id="alice",
    title="Deep Dive: Model Architecture",
    parent_session_id=parent_session.id,
    tags=["technical", "deep-dive"]
)
```

**Location:** `src/hippocampai/client.py:1043`

### Get a Session

```python
# Retrieve by ID
session = client.get_session(session_id)

if session:
    print(f"Title: {session.title}")
    print(f"Status: {session.status.value}")
    print(f"Messages: {session.message_count}")
    print(f"Entities: {len(session.entities)}")
```

**Location:** `src/hippocampai/client.py:1060`

### Update Session Metadata

```python
# Update title and tags
updated = client.update_session(
    session_id=session.id,
    title="Q1 Planning - ML Project (Updated)",
    tags=["work", "ml", "q1", "completed"]
)

# Update metadata only
updated = client.update_session(
    session_id=session.id,
    metadata={
        "status": "in-review",
        "reviewed_by": "bob",
        "rating": 4.5
    }
)

# Update status
updated = client.update_session(
    session_id=session.id,
    status=SessionStatus.ARCHIVED
)
```

**Location:** `src/hippocampai/client.py:1075`

### Complete a Session

```python
# Complete without summary
completed = client.complete_session(session_id=session.id)

# Complete with final summary generation
completed = client.complete_session(
    session_id=session.id,
    generate_summary=True
)

if completed:
    print(f"Duration: {completed.duration_seconds():.1f} seconds")
    print(f"Final summary: {completed.summary}")
```

**Location:** `src/hippocampai/client.py:1122`

### Delete a Session

```python
# Delete session and all associated data
deleted = client.delete_session(session_id)

if deleted:
    print("Session deleted successfully")
```

**Location:** `src/hippocampai/client.py:1271`

---

## Message Tracking

### Track Messages

```python
# Basic message tracking
session = client.track_session_message(
    session_id=session.id,
    text="I'm working on a sentiment analysis project",
    user_id="alice"
)

# With type and importance
session = client.track_session_message(
    session_id=session.id,
    text="The deadline is Friday, this is critical",
    user_id="alice",
    type="fact",
    importance=9.5,
    tags=["deadline", "critical"]
)

# With boundary detection
session = client.track_session_message(
    session_id=session.id,
    text="Let's switch topics and discuss cloud infrastructure",
    user_id="alice",
    auto_boundary_detect=True  # May create new session if topic changed
)
```

**Location:** `src/hippocampai/client.py:1095`

### How Message Tracking Works

1. Creates a memory using `remember()`
2. Associates memory with session via `session_id`
3. Updates session statistics (message count, last activity)
4. Optionally extracts entities automatically
5. Checks for session boundary if `auto_boundary_detect=True`
6. Triggers auto-summarization if threshold reached

### Get Session Memories

```python
# Retrieve all memories for a session
memories = client.get_session_memories(session_id, limit=100)

for memory in memories:
    print(f"[{memory.type}] {memory.text}")
    print(f"  Importance: {memory.importance:.1f}")
```

**Location:** `src/hippocampai/client.py:1181`

---

## Auto-Summarization

### Automatic Summarization

Sessions are automatically summarized when:
- Message count reaches `auto_summarize_threshold` (default: 10 messages)
- Session is completed with `generate_summary=True`
- `summarize_session()` is called manually

```python
# Automatic summary on threshold
for i in range(15):  # Will trigger auto-summary at 10 messages
    client.track_session_message(
        session_id=session.id,
        text=f"Message {i}",
        user_id="alice"
    )

# Manual summary generation
summary = client.summarize_session(session_id)
if summary:
    print(f"Summary: {summary}")
else:
    print("LLM not available or insufficient messages")

# Force re-summarization
summary = client.summarize_session(session_id, force=True)
```

**Location:** `src/hippocampai/client.py:1210`

### Summary Configuration

```python
# Custom threshold during SessionManager initialization
from hippocampai import SessionManager

session_manager = SessionManager(
    qdrant_store=client.qdrant,
    embedder=client.embedder,
    llm=client.llm,
    collection_name="sessions",
    auto_summarize_threshold=5  # Summarize after 5 messages
)
```

### LLM Requirements

- Requires configured LLM (Ollama or OpenAI)
- Falls back gracefully if LLM unavailable
- Minimum 3 messages required for meaningful summary

---

## Entity & Fact Extraction

### Extract Entities

Entities are automatically extracted during message tracking, or manually:

```python
# Extract entities from session
entities = client.extract_session_entities(session_id)

if entities:
    for name, entity in entities.items():
        print(f"{name} ({entity.type})")
        print(f"  Mentions: {entity.mentions}")
        print(f"  First seen: {entity.first_mentioned_at}")
        print(f"  Last seen: {entity.last_mentioned_at}")
```

**Supported Entity Types:**
- `person` - People and names
- `organization` - Companies and institutions
- `technology` - Technologies, frameworks, tools
- `location` - Places and locations
- `product` - Products and services
- `concept` - Abstract concepts
- `event` - Events and occurrences

**Location:** `src/hippocampai/client.py:1241`

### Extract Facts

```python
# Extract key facts with confidence scores
facts = client.extract_session_facts(session_id)

if facts:
    for fact in facts:
        print(f"Fact: {fact.fact}")
        print(f"  Confidence: {fact.confidence:.2f}")
        print(f"  Sources: {len(fact.sources)} memories")
        print(f"  Extracted: {fact.extracted_at}")
```

**Location:** `src/hippocampai/client.py:1225`

### Fallback Extraction

If LLM is unavailable:
- **Entities**: Uses regex patterns for common technologies and capitalized words
- **Facts**: Not extracted (requires LLM)

---

## Session Search

### Semantic Session Search

```python
# Search sessions by semantic similarity
results = client.search_sessions(
    query="machine learning tensorflow discussions",
    user_id="alice",
    k=5
)

for result in results:
    session = result.session
    print(f"\nTitle: {session.title}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Messages: {session.message_count}")
    print(f"  Tags: {', '.join(session.tags)}")
    if session.summary:
        print(f"  Summary: {session.summary[:100]}...")

# Search with tag filtering
results = client.search_sessions(
    query="project planning",
    user_id="alice",
    k=10,
    filters={"tags": ["work", "planning"]}
)

# Search with metadata filtering
results = client.search_sessions(
    query="high priority tasks",
    user_id="alice",
    filters={
        "metadata": {"priority": "high"}
    }
)
```

**Location:** `src/hippocampai/client.py:1139`

### How Search Works

1. Creates embedding from query
2. Performs vector search in Qdrant
3. Returns sessions ranked by similarity
4. Applies filters if specified
5. Includes session metadata and statistics

---

## Hierarchical Sessions

### Parent-Child Relationships

```python
# Create parent session
parent = client.create_session(
    user_id="alice",
    title="ML Project - Overall Planning"
)

# Create child sessions
architecture = client.create_session(
    user_id="alice",
    title="Deep Dive: Model Architecture",
    parent_session_id=parent.id,
    tags=["technical"]
)

data_pipeline = client.create_session(
    user_id="alice",
    title="Deep Dive: Data Pipeline",
    parent_session_id=parent.id,
    tags=["engineering"]
)

# Get all child sessions
children = client.get_child_sessions(parent.id)
for child in children:
    print(f"- {child.title} ({child.message_count} messages)")
```

**Location:** `src/hippocampai/client.py:1197`

### Use Cases for Hierarchical Sessions

- **Project organization**: Main project → Sub-tasks
- **Meeting notes**: Quarterly planning → Weekly updates
- **Education**: Course → Lessons
- **Support tickets**: Main issue → Follow-ups
- **Research**: Research topic → Experiments

---

## Boundary Detection

Automatic session boundary detection helps identify when a conversation topic changes significantly.

### Enable Boundary Detection

```python
# Create initial session
session = client.create_session(
    user_id="alice",
    title="Current Discussion"
)

# Track messages normally
client.track_session_message(
    session_id=session.id,
    text="Python is great for data science",
    user_id="alice",
    auto_boundary_detect=False
)

client.track_session_message(
    session_id=session.id,
    text="I use pandas and numpy daily",
    user_id="alice",
    auto_boundary_detect=False
)

# Message with topic change
result = client.track_session_message(
    session_id=session.id,
    text="Let's switch to discussing cloud infrastructure",
    user_id="alice",
    auto_boundary_detect=True  # Enable detection
)

# Check if new session was created
if result.id != session.id:
    print("Boundary detected! New session created.")
    print(f"Old session: {session.id}")
    print(f"New session: {result.id}")
```

### Detection Methods

1. **LLM-based** (preferred):
   - Compares new message semantically with session summary
   - Returns similarity score and boundary decision
   - Threshold: 0.7 (configurable)

2. **Fallback (no LLM)**:
   - Entity overlap analysis
   - Inactivity timeout (default: 30 minutes)
   - Keyword-based heuristics

### Configuration

```python
from hippocampai import SessionManager

session_manager = SessionManager(
    qdrant_store=client.qdrant,
    embedder=client.embedder,
    llm=client.llm,
    collection_name="sessions",
    inactivity_threshold_minutes=60  # 1 hour timeout
)
```

---

## Session Analytics

### Session Statistics

```python
# Get comprehensive statistics
stats = client.get_session_statistics(session_id)

print(f"Messages: {stats['message_count']}")
print(f"Memories: {stats['memory_count']}")
print(f"Duration: {stats['duration_seconds']:.1f} seconds")
print(f"Entities: {stats['entity_count']}")
print(f"Facts: {stats['fact_count']}")
print(f"Average importance: {stats['avg_importance']:.2f}")

# Top entities
if stats['top_entities']:
    print("\nTop entities:")
    for entity in stats['top_entities']:
        print(f"  - {entity['name']} ({entity['type']}): {entity['mentions']} mentions")

# Memory type breakdown
if stats['memory_types']:
    print("\nMemory types:")
    for mem_type, count in stats['memory_types'].items():
        print(f"  - {mem_type}: {count}")
```

**Location:** `src/hippocampai/client.py:1255`

### Get User Sessions

```python
# Get all sessions for a user
all_sessions = client.get_user_sessions(
    user_id="alice",
    limit=50
)

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

# Display sessions
for session in all_sessions:
    print(f"{session.title} - {session.status.value}")
    print(f"  {session.message_count} messages")
    print(f"  Started: {session.started_at}")
    if session.ended_at:
        print(f"  Ended: {session.ended_at}")
```

**Location:** `src/hippocampai/client.py:1153`

---

## Advanced Usage

### Custom Session Workflows

```python
# Workflow 1: Customer Support Session
support_session = client.create_session(
    user_id="customer_123",
    title="Support Ticket #12345",
    tags=["support", "billing"],
    metadata={
        "ticket_id": "12345",
        "category": "billing",
        "priority": "high",
        "assigned_to": "agent_42"
    }
)

# Track support conversation
for message in support_transcript:
    client.track_session_message(
        session_id=support_session.id,
        text=message,
        user_id="customer_123"
    )

# Extract resolution facts
facts = client.extract_session_facts(support_session.id)
resolution = [f.fact for f in facts if "resolved" in f.fact.lower()]

# Complete with summary
client.complete_session(
    session_id=support_session.id,
    generate_summary=True
)
```

```python
# Workflow 2: Educational Tutoring
course_session = client.create_session(
    user_id="student_alice",
    title="Python Programming - Week 1",
    tags=["education", "programming"],
    metadata={
        "course": "python-101",
        "week": 1,
        "instructor": "bob"
    }
)

# Create lesson sub-sessions
for lesson_title in ["Variables", "Functions", "Loops"]:
    lesson = client.create_session(
        user_id="student_alice",
        title=f"Lesson: {lesson_title}",
        parent_session_id=course_session.id,
        tags=["lesson"]
    )
```

### Integration with Memory Retrieval

```python
# Retrieve memories from specific session
session_id = "abc-123"

# Method 1: Using filters in recall()
results = client.recall(
    query="What did we discuss about architecture?",
    user_id="alice",
    filters={"session_id": session_id}
)

# Method 2: Using get_session_memories()
memories = client.get_session_memories(session_id)

# Method 3: Combined search across sessions
session_results = client.search_sessions(
    query="architecture discussions",
    user_id="alice",
    k=3
)

for result in session_results:
    session_memories = client.get_session_memories(result.session.id)
    print(f"\n{result.session.title}:")
    for mem in session_memories[:5]:
        print(f"  - {mem.text[:60]}...")
```

### Exporting Session Data

```python
# Export session summary and key information
session = client.get_session(session_id)

export_data = {
    "session_id": session.id,
    "title": session.title,
    "status": session.status.value,
    "duration_seconds": session.duration_seconds(),
    "message_count": session.message_count,
    "summary": session.summary,
    "entities": {
        name: {
            "type": entity.type,
            "mentions": entity.mentions
        }
        for name, entity in session.entities.items()
    },
    "facts": [
        {
            "fact": fact.fact,
            "confidence": fact.confidence
        }
        for fact in session.facts
    ],
    "metadata": session.metadata,
    "tags": session.tags
}

# Save to JSON
import json
with open(f"session_{session_id}.json", "w") as f:
    json.dump(export_data, f, indent=2, default=str)
```

---

## Best Practices

### 1. Session Naming

```python
# ✅ Good: Descriptive, specific titles
client.create_session(
    user_id="alice",
    title="Q1 2024 - ML Project Architecture Review"
)

# ❌ Bad: Generic titles
client.create_session(
    user_id="alice",
    title="Discussion"
)
```

### 2. Tagging Strategy

```python
# ✅ Good: Consistent, hierarchical tags
tags=["work", "ml", "q1-2024", "planning"]

# ❌ Bad: Inconsistent, redundant
tags=["Work", "work", "ml project", "planning meeting"]
```

### 3. Metadata Usage

```python
# ✅ Good: Structured, queryable metadata
metadata={
    "project_id": "proj-123",
    "priority": "high",
    "team": "data-science",
    "budget": 50000,
    "deadline": "2024-03-31"
}

# ❌ Bad: Unstructured or redundant
metadata={
    "notes": "some random notes here...",
    "title": "duplicates title field"
}
```

### 4. Session Lifecycle

```python
# ✅ Good: Explicitly complete sessions
session = client.create_session(...)
# ... track messages ...
client.complete_session(session.id, generate_summary=True)

# ❌ Bad: Leave sessions in active state indefinitely
session = client.create_session(...)
# ... track messages ...
# Never completed - stays active forever
```

### 5. Hierarchical Organization

```python
# ✅ Good: Logical parent-child hierarchy
parent = client.create_session(user_id="alice", title="Q1 Planning")
child1 = client.create_session(
    user_id="alice",
    title="Week 1 - Requirements",
    parent_session_id=parent.id
)

# ❌ Bad: Too many levels or unclear relationships
# Avoid more than 2-3 levels of nesting
```

### 6. Boundary Detection

```python
# ✅ Good: Enable selectively for long conversations
client.track_session_message(
    session_id=session.id,
    text=message,
    user_id="alice",
    auto_boundary_detect=(len(messages) > 10)  # Only after 10 messages
)

# ❌ Bad: Enable for every message (performance overhead)
client.track_session_message(
    session_id=session.id,
    text=message,
    user_id="alice",
    auto_boundary_detect=True  # Every single message
)
```

---

## API Reference

### Core Session Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `create_session()` | Create new session | `Session` |
| `get_session()` | Retrieve session by ID | `Optional[Session]` |
| `update_session()` | Update session metadata | `Optional[Session]` |
| `delete_session()` | Delete session | `bool` |
| `complete_session()` | Mark session complete | `Optional[Session]` |

### Message Tracking

| Method | Purpose | Returns |
|--------|---------|---------|
| `track_session_message()` | Track message in session | `Session` |
| `get_session_memories()` | Get session memories | `List[Memory]` |

### Analysis & Extraction

| Method | Purpose | Returns |
|--------|---------|---------|
| `summarize_session()` | Generate summary | `Optional[str]` |
| `extract_session_facts()` | Extract facts | `List[SessionFact]` |
| `extract_session_entities()` | Extract entities | `Dict[str, Entity]` |
| `get_session_statistics()` | Get analytics | `Dict[str, Any]` |

### Search & Discovery

| Method | Purpose | Returns |
|--------|---------|---------|
| `search_sessions()` | Semantic search | `List[SessionSearchResult]` |
| `get_user_sessions()` | Get user's sessions | `List[Session]` |
| `get_child_sessions()` | Get child sessions | `List[Session]` |

---

## Complete Example

```python
from hippocampai import MemoryClient, SessionStatus

# Initialize
client = MemoryClient()
user_id = "alice"

# 1. Create session
session = client.create_session(
    user_id=user_id,
    title="ML Project - Requirements Gathering",
    tags=["work", "ml", "planning"],
    metadata={"project": "sentiment-analysis", "phase": "planning"}
)

# 2. Track conversation
messages = [
    "We need to analyze customer reviews for sentiment",
    "Dataset should have at least 100k reviews",
    "Priority is accuracy over speed",
    "Team will use TensorFlow and BERT",
    "Deadline is end of Q1"
]

for msg in messages:
    session = client.track_session_message(
        session_id=session.id,
        text=msg,
        user_id=user_id,
        type="fact"
    )

# 3. Generate summary
summary = client.summarize_session(session.id)
print(f"Summary: {summary}")

# 4. Extract insights
facts = client.extract_session_facts(session.id)
print(f"\nKey Facts ({len(facts)}):")
for fact in facts:
    print(f"  - {fact.fact} (confidence: {fact.confidence:.2f})")

entities = client.extract_session_entities(session.id)
print(f"\nEntities ({len(entities)}):")
for name, entity in entities.items():
    print(f"  - {name} ({entity.type}): {entity.mentions} mentions")

# 5. Create child session for deep dive
deep_dive = client.create_session(
    user_id=user_id,
    title="Deep Dive: Model Selection",
    parent_session_id=session.id,
    tags=["technical", "deep-dive"]
)

# 6. Get statistics
stats = client.get_session_statistics(session.id)
print(f"\nSession Statistics:")
print(f"  Duration: {stats['duration_seconds']:.1f}s")
print(f"  Messages: {stats['message_count']}")
print(f"  Avg Importance: {stats['avg_importance']:.2f}")

# 7. Search similar sessions
similar = client.search_sessions(
    query="machine learning project planning",
    user_id=user_id,
    k=5
)
print(f"\nSimilar Sessions ({len(similar)}):")
for result in similar:
    print(f"  - {result.session.title} (score: {result.score:.3f})")

# 8. Complete session
completed = client.complete_session(
    session_id=session.id,
    generate_summary=True
)
print(f"\nSession completed: {completed.status.value}")
```

---

## Troubleshooting

### LLM Not Available

If LLM is not configured or unavailable:
- Summaries will not be generated (returns `None`)
- Facts extraction will fail (returns empty list)
- Entity extraction falls back to regex-based extraction
- Boundary detection uses heuristic-based approach

**Solution**: Configure LLM in your HippocampAI config:
```python
from hippocampai import MemoryClient, Config

config = Config(
    llm_provider="ollama",  # or "openai"
    llm_model="qwen2.5:7b-instruct"
)
client = MemoryClient(config=config)
```

### Session Not Found

If `get_session()` returns `None`:
- Verify session ID is correct
- Check if session was deleted
- Ensure Qdrant is running and accessible

### Boundary Detection Not Working

If topic changes aren't detected:
- Enable with `auto_boundary_detect=True`
- Ensure LLM is configured for better detection
- Check inactivity threshold settings
- Verify messages have sufficient content difference

---

## Performance Considerations

### Embeddings

- Each session creates one embedding (title + summary + tags)
- Embeddings updated when session is summarized
- Use meaningful titles and tags for better search

### LLM Calls

- Summarization: 1 LLM call per session
- Fact extraction: 1 LLM call per session
- Entity extraction: 1 LLM call per session
- Boundary detection: 1 LLM call per message (if enabled)

**Optimization**: Batch session operations and use boundary detection sparingly.

### Storage

- Sessions stored in separate Qdrant collection
- Minimal overhead per session (~1-2 KB)
- Entities and facts stored inline with session
- No impact on existing memory storage

---

For more examples, see:
- `examples/10_session_management_demo.py` - Comprehensive demo
- [FEATURES.md](FEATURES.md) - Complete feature documentation
- [API_REFERENCE.md](API_REFERENCE.md) - Quick API reference
