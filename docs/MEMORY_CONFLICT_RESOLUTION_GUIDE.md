# Memory Conflict Resolution Guide

Complete guide to HippocampAI's automatic conflict detection and resolution system.

## Overview

HippocampAI includes a sophisticated conflict resolution system that automatically detects contradictory memories and resolves them using configurable strategies. This ensures your memory store remains consistent and accurate over time.

## Table of Contents

- [Quick Start](#quick-start)
- [Conflict Detection](#conflict-detection)
- [Resolution Strategies](#resolution-strategies)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Quick Start

### Basic Conflict Detection

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store conflicting memories
client.remember("I love coffee", user_id="alice", type="preference")
client.remember("I hate coffee", user_id="alice", type="preference")

# Detect conflicts
conflicts = client.detect_memory_conflicts("alice", check_llm=True)

print(f"Found {len(conflicts)} conflicts")
for conflict in conflicts:
    print(f"{conflict.conflict_type}:")
    print(f"  Memory 1: {conflict.memory_1.text}")
    print(f"  Memory 2: {conflict.memory_2.text}")
    print(f"  Confidence: {conflict.confidence_score}")
```

### Auto-Resolve Conflicts

```python
# Automatically resolve all conflicts using temporal strategy (newest wins)
result = client.auto_resolve_conflicts("alice", strategy="temporal")

print(f"Resolved: {result['resolved_count']} conflicts")
print(f"Deleted: {result['deleted_count']} memories")
print(f"Merged: {result['merged_count']} memories")
```

---

## Conflict Detection

### How It Works

The conflict detection system uses a multi-stage approach:

1. **Semantic Similarity**: Find memories that are semantically similar (configurable threshold: 0.75)
2. **Pattern Matching**: Check for contradiction patterns (e.g., "love" vs "hate")
3. **LLM Analysis** (optional): Deep analysis using LLM for accurate contradiction detection
4. **Confidence Scoring**: Assign confidence score to detected conflicts (0.0-1.0)

### Conflict Types

```python
from hippocampai.pipeline.conflict_resolution import ConflictType

# Available conflict types:
ConflictType.DIRECT_CONTRADICTION      # "I love X" vs "I hate X"
ConflictType.VALUE_CHANGE              # Age 25 vs Age 26
ConflictType.FACTUAL_INCONSISTENCY     # Conflicting facts
ConflictType.TEMPORAL_INCONSISTENCY    # Timeline doesn't make sense
```

### Detection Methods

#### 1. Pattern-Based (Fast)

```python
# Quick pattern-based detection (no LLM required)
conflicts = client.detect_memory_conflicts(
    "alice",
    check_llm=False  # Fast, pattern-based only
)
```

**Detected patterns:**
- love ↔ hate
- like ↔ dislike
- enjoy ↔ don't enjoy
- prefer ↔ don't prefer
- allergic to ↔ not allergic
- vegetarian ↔ not vegetarian
- yes ↔ no
- always ↔ never

#### 2. LLM-Based (Accurate)

```python
# Deep LLM-based analysis (more accurate, slower)
conflicts = client.detect_memory_conflicts(
    "alice",
    check_llm=True  # Use LLM for deep analysis
)
```

**Advantages:**
- Understands context and nuance
- Detects implicit contradictions
- Handles complex relationships
- More accurate confidence scores

#### 3. Type-Filtered Detection

```python
# Only check specific memory types
conflicts = client.detect_memory_conflicts(
    "alice",
    check_llm=True,
    memory_type="preference"  # Only check preferences
)
```

---

## Resolution Strategies

### Strategy Overview

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **TEMPORAL** | Latest memory wins | Default, handles preference changes |
| **CONFIDENCE** | Higher confidence wins | When confidence scores are reliable |
| **IMPORTANCE** | Higher importance wins | For important vs trivial information |
| **USER_REVIEW** | Flag for manual review | Critical data, user wants control |
| **AUTO_MERGE** | LLM merges both memories | Capture evolution of information |
| **KEEP_BOTH** | Keep both, mark as conflicting | When both perspectives are valid |

### 1. TEMPORAL Strategy (Default)

**Rule**: Keep the most recent memory

```python
result = client.auto_resolve_conflicts("alice", strategy="temporal")
```

**Example:**
```
Memory 1: "I love coffee" (created: 2024-01-01, confidence: 0.9)
Memory 2: "I hate coffee" (created: 2024-02-01, confidence: 0.95)
→ Result: Keep Memory 2 (newer), delete Memory 1
```

**Best for:**
- Preference changes over time
- Updated information
- Default behavior

### 2. CONFIDENCE Strategy

**Rule**: Keep the memory with higher confidence score

```python
result = client.auto_resolve_conflicts("alice", strategy="confidence")
```

**Example:**
```
Memory 1: "I work at Google" (confidence: 0.95)
Memory 2: "I work at Facebook" (confidence: 0.7)
→ Result: Keep Memory 1 (higher confidence)
```

**Best for:**
- When confidence scores are well-calibrated
- Extracted vs manually entered data
- Verified vs unverified information

### 3. IMPORTANCE Strategy

**Rule**: Keep the memory with higher importance score

```python
result = client.auto_resolve_conflicts("alice", strategy="importance")
```

**Example:**
```
Memory 1: "Allergic to peanuts" (importance: 9.5)
Memory 2: "Not allergic to peanuts" (importance: 5.0)
→ Result: Keep Memory 1 (more important)
```

**Best for:**
- Safety-critical information (allergies, medical)
- Core preferences vs casual mentions
- Mission-critical data

### 4. USER_REVIEW Strategy

**Rule**: Flag both memories for manual review, don't auto-resolve

```python
result = client.auto_resolve_conflicts("alice", strategy="user_review")
```

**Behavior:**
- Both memories kept
- `has_conflict` flag set in metadata
- `conflict_id` and `conflict_with` stored
- Can be reviewed later

**Best for:**
- Critical business data
- Legal/compliance requirements
- When user wants final say

### 5. AUTO_MERGE Strategy

**Rule**: Use LLM to merge conflicting information into single memory

```python
result = client.auto_resolve_conflicts("alice", strategy="auto_merge")
```

**Example:**
```
Memory 1: "I love coffee" (2 months ago)
Memory 2: "I hate coffee" (yesterday)
→ Result: "I used to love coffee but now hate it"
```

**Merged memory properties:**
- Importance: max(mem1, mem2)
- Confidence: min(mem1, mem2) * 0.9
- Tags: union of both
- Metadata includes `merged_from` list

**Best for:**
- Capturing evolution of preferences
- Historical context is important
- Rich narrative memory

### 6. KEEP_BOTH Strategy

**Rule**: Keep both memories but mark them as conflicting

```python
result = client.auto_resolve_conflicts("alice", strategy="keep_both")
```

**Behavior:**
- Both memories preserved
- Conflict metadata attached
- Can query conflicting memories later

**Best for:**
- Uncertainty about which is correct
- Multiple valid perspectives
- Research/analysis scenarios

---

## API Reference

### detect_memory_conflicts()

Detect conflicts in user's memories.

```python
conflicts = client.detect_memory_conflicts(
    user_id="alice",
    check_llm=True,           # Use LLM for deep analysis
    memory_type="preference"   # Optional type filter
)
```

**Returns**: List of `MemoryConflict` objects

**MemoryConflict Fields:**
```python
conflict.id                    # Unique conflict ID
conflict.memory_1              # First conflicting memory
conflict.memory_2              # Second conflicting memory
conflict.conflict_type         # Type of conflict
conflict.confidence_score      # Confidence in detection (0-1)
conflict.similarity_score      # Semantic similarity (0-1)
conflict.detected_at           # Detection timestamp
conflict.resolved              # Whether resolved
conflict.resolution_strategy   # Strategy used
conflict.winner_memory_id      # ID of winning memory
conflict.resolution_notes      # Explanation
```

### auto_resolve_conflicts()

Automatically detect and resolve all conflicts.

```python
result = client.auto_resolve_conflicts(
    user_id="alice",
    strategy="temporal",      # Resolution strategy
    memory_type=None          # Optional type filter
)
```

**Returns**: Dictionary with resolution summary

```python
{
    "user_id": "alice",
    "conflicts_found": 5,
    "resolved_count": 5,
    "deleted_count": 4,
    "merged_count": 1,
    "deleted_memory_ids": ["id1", "id2", ...]
}
```

---

## Examples

### Example 1: Food Preferences

```python
from hippocampai import MemoryClient
import time

client = MemoryClient()

# User changes their mind over time
client.remember("I love pizza", user_id="alice", type="preference", importance=7)
time.sleep(1)  # Simulate time passage
client.remember("I hate pizza now", user_id="alice", type="preference", importance=7)

# Detect and resolve
conflicts = client.detect_memory_conflicts("alice")
print(f"Conflict detected: {conflicts[0].conflict_type}")

result = client.auto_resolve_conflicts("alice", strategy="temporal")
print(f"Resolution: Kept newer preference, deleted older one")
```

### Example 2: Medical Information (High Stakes)

```python
# Critical medical information - use USER_REVIEW strategy
client.remember("Allergic to peanuts", user_id="bob", type="fact", importance=10)
client.remember("Not allergic to peanuts", user_id="bob", type="fact", importance=10)

# Flag for manual review instead of auto-resolving
result = client.auto_resolve_conflicts("bob", strategy="user_review")

# Both memories are kept with conflict flags
memories = client.get_memories("bob")
for mem in memories:
    if mem.metadata.get("has_conflict"):
        print(f"⚠️ REVIEW REQUIRED: {mem.text}")
        print(f"   Conflicts with: {mem.metadata['conflict_with']}")
```

### Example 3: Evolution of Information

```python
# Track how information evolves over time
client.remember(
    "I work at Google",
    user_id="carol",
    type="fact",
    importance=8
)

time.sleep(1)

client.remember(
    "I work at Facebook",
    user_id="carol",
    type="fact",
    importance=8
)

# Use auto-merge to capture the transition
result = client.auto_resolve_conflicts("carol", strategy="auto_merge")

# Merged memory might be: "I used to work at Google but now work at Facebook"
memories = client.get_memories("carol")
print(f"Merged memory: {memories[0].text}")
print(f"Merged from: {memories[0].metadata['merged_from']}")
```

---

## Best Practices

### 1. Choose the Right Strategy

```python
# For preferences that change over time
client.auto_resolve_conflicts(user_id, strategy="temporal")

# For critical/medical information
client.auto_resolve_conflicts(user_id, strategy="user_review")

# For rich historical context
client.auto_resolve_conflicts(user_id, strategy="auto_merge")

# When confidence scores are reliable
client.auto_resolve_conflicts(user_id, strategy="confidence")
```

### 2. Regular Conflict Checks

```python
# Run periodic conflict detection and resolution
import schedule

def check_conflicts():
    for user_id in active_users:
        conflicts = client.detect_memory_conflicts(user_id)
        if conflicts:
            client.auto_resolve_conflicts(user_id, strategy="temporal")

# Run daily
schedule.every().day.at("02:00").do(check_conflicts)
```

### 3. Monitor Conflict Metrics

```python
# Track conflict detection over time
conflicts = client.detect_memory_conflicts("alice")

metrics = {
    "total_conflicts": len(conflicts),
    "by_type": {},
    "avg_confidence": sum(c.confidence_score for c in conflicts) / len(conflicts) if conflicts else 0
}

for conflict in conflicts:
    conflict_type = conflict.conflict_type
    metrics["by_type"][conflict_type] = metrics["by_type"].get(conflict_type, 0) + 1

print(f"Conflict metrics: {metrics}")
```

---

## See Also

- [Memory Provenance & Lineage Guide](MEMORY_PROVENANCE_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Architecture Guide](ARCHITECTURE.md)
