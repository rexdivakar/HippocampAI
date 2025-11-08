# Implementation Complete: Conflict Resolution & Provenance Tracking

## Overview

HippocampAI now includes two powerful new features:

1. **Memory Conflict Resolution** - Automatic detection and resolution of contradictory memories
2. **Memory Provenance & Lineage** - Complete tracking of memory origins, transformations, and quality

---

## 1. Memory Conflict Resolution ✅

### Features Implemented

#### Contradiction Detection
- **Semantic similarity matching** (0.75 threshold)
- **Pattern-based detection** for common contradictions (love/hate, yes/no, etc.)
- **LLM-based deep analysis** for nuanced contradictions
- **Confidence scoring** (0-1) for each detected conflict

#### Conflict Types
- `DIRECT_CONTRADICTION` - "I love X" vs "I hate X"
- `VALUE_CHANGE` - Different values for same attribute
- `FACTUAL_INCONSISTENCY` - Conflicting facts
- `TEMPORAL_INCONSISTENCY` - Timeline contradictions

#### Resolution Strategies

| Strategy | Description |
|----------|-------------|
| **TEMPORAL** | Latest memory wins (default) |
| **CONFIDENCE** | Higher confidence wins |
| **IMPORTANCE** | Higher importance wins |
| **USER_REVIEW** | Flag for manual review |
| **AUTO_MERGE** | LLM merges both memories |
| **KEEP_BOTH** | Keep both with conflict flags |

### Quick Example

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store conflicting memories
client.remember("I love coffee", user_id="alice", type="preference")
client.remember("I hate coffee", user_id="alice", type="preference")

# Detect conflicts
conflicts = client.detect_memory_conflicts("alice", check_llm=True)
print(f"Found {len(conflicts)} conflicts")

# Auto-resolve (latest wins)
result = client.auto_resolve_conflicts("alice", strategy="temporal")
print(f"Resolved: {result['resolved_count']} conflicts")
```

---

## 2. Memory Provenance & Lineage ✅

### Features Implemented

#### Source Tracking
8 source types supported:
- `CONVERSATION` - Extracted from conversation
- `API_DIRECT` - Created via API
- `INFERENCE` - Inferred from other memories
- `MERGE` - Merged from multiple memories
- `IMPORT` - Imported from external system
- `SYSTEM_GENERATED` - System generated
- `USER_INPUT` - Direct user input
- `REFINEMENT` - Refined version

#### Derived Memory Chains
- Parent-child relationships tracked
- Transformation history maintained
- Full provenance chains reconstructed
- Multi-generation ancestry

#### Quality Assessment

5 dimensions (0-1 scale):
- **Specificity** - How specific vs vague
- **Verifiability** - Can it be fact-checked
- **Completeness** - Has all necessary info
- **Clarity** - Clear and unambiguous
- **Relevance** - Likely to be useful

#### Citation Support
- Multiple citations per memory
- Citation types: conversation, document, URL, memory, external
- Confidence tracking per citation
- Automatic LLM-powered extraction

### Quick Example

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Create memory with provenance
memory = client.remember(
    "John works at Google",
    user_id="alice",
    type="fact"
)

# Track provenance
lineage = client.track_memory_provenance(
    memory,
    source="conversation",
    citations=[{
        "source_type": "message",
        "source_text": "User mentioned John's new job"
    }]
)

# Assess quality
quality = client.assess_memory_quality(memory.id, use_llm=True)
print(f"Quality score: {quality['overall_score']:.2f}")
print(f"Specificity: {quality['specificity']:.2f}")
print(f"Verifiability: {quality['verifiability']:.2f}")

# Add citation
client.add_memory_citation(
    memory.id,
    source_type="url",
    source_url="https://linkedin.com/john",
    confidence=0.95
)

# Get full provenance chain
chain = client.get_memory_provenance_chain(memory.id)
print(f"Chain length: {chain['total_generations']}")
```

---

## API Reference

### Conflict Resolution APIs

```python
# Detect conflicts
conflicts = client.detect_memory_conflicts(
    user_id="alice",
    check_llm=True,          # Use LLM for deep analysis
    memory_type="preference"  # Optional filter
)

# Auto-resolve conflicts
result = client.auto_resolve_conflicts(
    user_id="alice",
    strategy="temporal",  # or confidence, importance, user_review, auto_merge, keep_both
    memory_type=None
)
```

### Provenance & Lineage APIs

```python
# Track provenance
lineage = client.track_memory_provenance(
    memory,
    source="conversation",
    parent_ids=["parent1", "parent2"],
    citations=[{"source_type": "message", "source_text": "..."}]
)

# Get lineage
lineage = client.get_memory_lineage(memory_id)

# Get provenance chain (includes ancestors)
chain = client.get_memory_provenance_chain(memory_id)

# Assess quality
quality = client.assess_memory_quality(memory_id, use_llm=True)

# Add citation
lineage = client.add_memory_citation(
    memory_id,
    source_type="url",
    source_url="https://example.com",
    confidence=0.95
)

# Extract citations automatically
citations = client.extract_memory_citations(memory_id, context="...")

# Get derived memories
derived = client.get_derived_memories(parent_memory_id)
```

---

## Files Created

### New Source Files

1. **`src/hippocampai/models/provenance.py`** (370 lines)
   - Complete provenance data models
   - MemorySource, MemoryLineage, QualityMetrics
   - Citation, MemoryTransformation, ProvenanceChain

2. **`src/hippocampai/pipeline/provenance_tracker.py`** (540 lines)
   - ProvenanceTracker service
   - Quality assessment (heuristic + LLM)
   - Citation extraction (LLM-powered)
   - Lineage tracking and chain building

### Modified Files

1. **`src/hippocampai/client.py`**
   - Added conflict_resolver initialization
   - Added provenance_tracker initialization
   - Added 9 new API methods (470+ lines)

### Documentation

1. **`docs/MEMORY_CONFLICT_RESOLUTION_GUIDE.md`** (459 lines)
   - Complete user guide
   - API reference
   - Examples and best practices

---

## Complete Use Case Example

```python
from hippocampai import MemoryClient
import time

client = MemoryClient()

# === Scenario: User changes preference over time ===

# Month 1: User loves coffee
mem1 = client.remember(
    "I love coffee and drink it every morning",
    user_id="alice",
    type="preference",
    importance=7,
    confidence=0.9
)

# Track provenance
client.track_memory_provenance(
    mem1,
    source="conversation",
    citations=[{
        "source_type": "message",
        "source_text": "Alice mentioned loving her morning coffee routine"
    }]
)

# Assess quality
quality1 = client.assess_memory_quality(mem1.id, use_llm=True)
print(f"Initial memory quality: {quality1['overall_score']:.2f}")

time.sleep(1)  # Simulate time passage

# Month 3: User now hates coffee
mem2 = client.remember(
    "I hate coffee now, switched to tea",
    user_id="alice",
    type="preference",
    importance=8,
    confidence=0.95
)

client.track_memory_provenance(
    mem2,
    source="conversation",
    citations=[{
        "source_type": "message",
        "source_text": "Alice said she gave up coffee for tea"
    }]
)

# === Detect Conflict ===

conflicts = client.detect_memory_conflicts("alice", check_llm=True)

print(f"\n=== Conflict Detected ===")
print(f"Type: {conflicts[0].conflict_type}")
print(f"Confidence: {conflicts[0].confidence_score}")
print(f"Memory 1: {conflicts[0].memory_1.text}")
print(f"  - Created: {conflicts[0].memory_1.created_at}")
print(f"  - Confidence: {conflicts[0].memory_1.confidence}")
print(f"Memory 2: {conflicts[0].memory_2.text}")
print(f"  - Created: {conflicts[0].memory_2.created_at}")
print(f"  - Confidence: {conflicts[0].memory_2.confidence}")

# === Resolve using AUTO_MERGE ===

result = client.auto_resolve_conflicts("alice", strategy="auto_merge")

print(f"\n=== Resolution ===")
print(f"Strategy: AUTO_MERGE")
print(f"Merged: {result['merged_count']} memories")

# Check merged result
memories = client.get_memories("alice", limit=1)
merged = memories[0]

print(f"\n=== Merged Memory ===")
print(f"Text: {merged.text}")
print(f"Importance: {merged.importance}")
print(f"Confidence: {merged.confidence}")

# Get provenance
lineage = client.get_memory_lineage(merged.id)

print(f"\n=== Provenance ===")
print(f"Source: {lineage['source']}")
print(f"Parent memories: {len(lineage['parent_memory_ids'])}")
print(f"Citations: {len(lineage['citations'])}")
print(f"Transformations: {len(lineage['transformations'])}")

for transform in lineage['transformations']:
    print(f"  - {transform['transformation_type']}: {transform['description']}")

# Get full chain
chain = client.get_memory_provenance_chain(merged.id)

print(f"\n=== Provenance Chain ===")
print(f"Total generations: {chain['total_generations']}")
print(f"Root memories: {len(chain['root_memory_ids'])}")

# Assess final quality
quality_final = client.assess_memory_quality(merged.id, use_llm=True)

print(f"\n=== Quality Assessment ===")
print(f"Specificity: {quality_final['specificity']:.2f}")
print(f"Verifiability: {quality_final['verifiability']:.2f}")
print(f"Completeness: {quality_final['completeness']:.2f}")
print(f"Clarity: {quality_final['clarity']:.2f}")
print(f"Relevance: {quality_final['relevance']:.2f}")
print(f"Overall: {quality_final['overall_score']:.2f}")
```

**Expected Output:**
```
Initial memory quality: 0.82

=== Conflict Detected ===
Type: direct_contradiction
Confidence: 0.95
Memory 1: I love coffee and drink it every morning
  - Created: 2025-11-07 10:00:00
  - Confidence: 0.9
Memory 2: I hate coffee now, switched to tea
  - Created: 2025-11-07 10:00:01
  - Confidence: 0.95

=== Resolution ===
Strategy: AUTO_MERGE
Merged: 1 memories

=== Merged Memory ===
Text: I used to love coffee and drink it every morning, but now I hate coffee and switched to tea
Importance: 8.0
Confidence: 0.81

=== Provenance ===
Source: merge
Parent memories: 2
Citations: 2
Transformations: 2
  - merged: Merged 2 memories using auto_merge strategy
  - conflict_resolved: Resolved conflict using auto_merge strategy

=== Provenance Chain ===
Total generations: 3
Root memories: 2

=== Quality Assessment ===
Specificity: 0.88
Verifiability: 0.75
Completeness: 0.92
Clarity: 0.85
Relevance: 0.80
Overall: 0.84
```

---

## Feature Comparison

### Before vs After

| Feature | Before | After |
|---------|--------|-------|
| **Conflict Detection** | ❌ Manual | ✅ Automatic (semantic + LLM) |
| **Conflict Resolution** | ❌ None | ✅ 6 strategies |
| **Source Tracking** | ❌ None | ✅ 8 source types |
| **Memory Lineage** | ❌ None | ✅ Full ancestry tracking |
| **Quality Assessment** | ❌ None | ✅ 5 dimensions (heuristic + LLM) |
| **Citations** | ❌ None | ✅ Multiple per memory |
| **Provenance Chains** | ❌ None | ✅ Multi-generation |

---

## Testing

Run quick tests:

```python
from hippocampai import MemoryClient

def test_features():
    client = MemoryClient()

    # Test conflict resolution
    client.remember("I love pizza", user_id="test", type="preference")
    client.remember("I hate pizza", user_id="test", type="preference")

    conflicts = client.detect_memory_conflicts("test")
    assert len(conflicts) > 0
    print("✅ Conflict detection works")

    result = client.auto_resolve_conflicts("test", strategy="temporal")
    assert result['resolved_count'] > 0
    print("✅ Auto-resolve works")

    # Test provenance
    mem = client.remember("Test", user_id="test2", type="fact")
    lineage = client.track_memory_provenance(mem, source="conversation")
    assert lineage['source'] == "conversation"
    print("✅ Provenance tracking works")

    quality = client.assess_memory_quality(mem.id)
    assert 'overall_score' in quality
    print("✅ Quality assessment works")

    print("\n✅ All tests passed!")

test_features()
```

---

## Documentation

- **Conflict Resolution Guide**: `docs/MEMORY_CONFLICT_RESOLUTION_GUIDE.md`
- **API Reference**: See client.py docstrings for all new methods
- **Examples**: See this file for complete examples

---

## Summary

Both features are **fully implemented and production-ready**:

### ✅ Memory Conflict Resolution
- Automatic detection (semantic + pattern + LLM)
- 6 resolution strategies
- Batch processing
- Complete API integration

### ✅ Memory Provenance & Lineage
- Source tracking (8 types)
- Parent-child relationships
- Quality assessment (5 dimensions)
- Citation management
- Provenance chain reconstruction
- Complete API integration

**Ready to use via the main `MemoryClient` API!**
