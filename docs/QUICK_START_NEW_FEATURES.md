# Quick Start: Conflict Resolution & Provenance

## Installation

```bash
pip install hippocampai
```

## 1. Conflict Resolution

### Detect Conflicts

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store conflicting memories
client.remember("I love coffee", user_id="alice", type="preference")
client.remember("I hate coffee", user_id="alice", type="preference")

# Detect
conflicts = client.detect_memory_conflicts("alice", check_llm=True)
print(f"Found {len(conflicts)} conflicts")
```

### Auto-Resolve

```python
# Latest wins (default)
result = client.auto_resolve_conflicts("alice", strategy="temporal")

# Higher confidence wins
result = client.auto_resolve_conflicts("alice", strategy="confidence")

# Merge both
result = client.auto_resolve_conflicts("alice", strategy="auto_merge")

# Flag for review
result = client.auto_resolve_conflicts("alice", strategy="user_review")
```

## 2. Provenance Tracking

### Track Source

```python
memory = client.remember("John works at Google", user_id="alice", type="fact")

# Add provenance
lineage = client.track_memory_provenance(
    memory,
    source="conversation",
    citations=[{
        "source_type": "message",
        "source_text": "User mentioned John's job"
    }]
)
```

### Quality Assessment

```python
quality = client.assess_memory_quality(memory_id, use_llm=True)

print(f"Specificity: {quality['specificity']:.2f}")
print(f"Overall: {quality['overall_score']:.2f}")
```

### Add Citations

```python
client.add_memory_citation(
    memory_id,
    source_type="url",
    source_url="https://example.com",
    confidence=0.95
)
```

### Get Provenance

```python
# Get lineage
lineage = client.get_memory_lineage(memory_id)

# Get full chain (with ancestors)
chain = client.get_memory_provenance_chain(memory_id)
```

## 3. Resolution Strategies

| Strategy | When to Use |
|----------|-------------|
| `temporal` | Preferences that change over time |
| `confidence` | When confidence scores are reliable |
| `importance` | For critical vs trivial information |
| `user_review` | Manual review needed |
| `auto_merge` | Capture evolution of info |
| `keep_both` | Both perspectives valid |

## 4. Quality Dimensions

- **Specificity** (0-1): How specific vs vague
- **Verifiability** (0-1): Can it be fact-checked
- **Completeness** (0-1): Has all necessary info
- **Clarity** (0-1): Clear and unambiguous
- **Relevance** (0-1): Likely to be useful

## 5. Source Types

- `conversation` - From user conversation
- `api_direct` - Direct API creation
- `inference` - Inferred from other memories
- `merge` - Merged memories
- `import` - External import
- `system_generated` - System created
- `user_input` - Direct user input
- `refinement` - Refined version

## Run Example

```bash
python examples/conflict_and_provenance_demo.py
```

## Documentation

- **Conflict Resolution**: `docs/MEMORY_CONFLICT_RESOLUTION_GUIDE.md`
- **Full Implementation**: `NEW_FEATURES_IMPLEMENTATION.md`
- **API Reference**: See `client.py` docstrings

## Complete Example

```python
from hippocampai import MemoryClient
import time

client = MemoryClient()

# 1. Create memory with provenance
mem1 = client.remember(
    "I love coffee",
    user_id="alice",
    type="preference",
    importance=7
)

client.track_memory_provenance(
    mem1,
    source="conversation",
    citations=[{"source_type": "message", "source_text": "User said..."}]
)

time.sleep(1)

# 2. Create conflicting memory
mem2 = client.remember(
    "I hate coffee now",
    user_id="alice",
    type="preference",
    importance=8
)

# 3. Detect and resolve
conflicts = client.detect_memory_conflicts("alice", check_llm=True)
result = client.auto_resolve_conflicts("alice", strategy="auto_merge")

# 4. Check result
memories = client.get_memories("alice", limit=1)
merged = memories[0]

# 5. Assess quality
quality = client.assess_memory_quality(merged.id)
print(f"Quality: {quality['overall_score']:.2f}")

# 6. Get provenance
lineage = client.get_memory_lineage(merged.id)
print(f"Citations: {len(lineage['citations'])}")
print(f"Transformations: {len(lineage['transformations'])}")
```

## Key Benefits

✅ **Automatic conflict detection** - No manual checking needed
✅ **Flexible resolution** - 6 strategies for different use cases
✅ **Complete provenance** - Track memory origins and transformations
✅ **Quality scores** - Assess memory reliability
✅ **Citation support** - Link memories to sources
✅ **Lineage tracking** - See how memories evolved

All features work seamlessly with existing HippocampAI functionality!
