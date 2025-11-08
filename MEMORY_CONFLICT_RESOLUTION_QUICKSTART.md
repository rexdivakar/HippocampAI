# Memory Conflict Resolution - Quick Start Guide

## What Is It?

Memory Conflict Resolution automatically detects and resolves contradictory memories in HippocampAI.

**Example Conflict:**
- Memory 1 (2 months ago): "I love coffee" ☕
- Memory 2 (today): "I hate coffee" 🚫

**What happens:** System detects the contradiction and automatically resolves it based on your chosen strategy.

## Installation

No additional installation needed! Conflict resolution is built into HippocampAI.

## Quick Setup

### 1. Basic Configuration (`.env` file)

```bash
# Enable conflict resolution (default: true)
ENABLE_CONFLICT_RESOLUTION=true

# Choose resolution strategy (default: temporal)
# Options: temporal, confidence, importance, user_review, auto_merge, keep_both
CONFLICT_RESOLUTION_STRATEGY=temporal

# Auto-resolve conflicts (default: true)
AUTO_RESOLVE_CONFLICTS=true
```

### 2. Basic Usage

```python
from hippocampai.unified_client import UnifiedHippocampAI

# Initialize client (uses config from .env)
client = UnifiedHippocampAI()

# Add memories - conflicts are automatically detected and resolved!
await client.add("I love coffee", user_id="user123", memory_type="preference")
await client.add("I hate coffee", user_id="user123", memory_type="preference")

# ✅ Automatically resolved! Latest memory ("I hate coffee") is kept
```

## Resolution Strategies

### Temporal (Default) - Latest Wins
```python
# Newest memory always wins
CONFLICT_RESOLUTION_STRATEGY=temporal
```
**Use when:** General purpose, preferences change over time

### Confidence - Higher Confidence Wins
```python
# Memory with higher confidence score wins
CONFLICT_RESOLUTION_STRATEGY=confidence
```
**Use when:** You track confidence scores

### Importance - Higher Importance Wins
```python
# Memory with higher importance score wins
CONFLICT_RESOLUTION_STRATEGY=importance
```
**Use when:** You track importance scores

### User Review - Flag for Manual Review
```python
# Don't auto-resolve, flag both memories
CONFLICT_RESOLUTION_STRATEGY=user_review
```
**Use when:** Sensitive data, need human oversight

### Auto Merge - AI Merges Conflicts
```python
# LLM creates merged memory
CONFLICT_RESOLUTION_STRATEGY=auto_merge
```
**Use when:** You have an LLM and want smart merging

### Keep Both - Store Both with Flags
```python
# Keep both memories with conflict markers
CONFLICT_RESOLUTION_STRATEGY=keep_both
```
**Use when:** Both may be valid in different contexts

## Common Examples

### Example 1: Changing Preferences (Temporal)

```python
# Monday: User loves coffee
await client.add("I love coffee", user_id="alice", memory_type="preference")

# Friday: User's taste changes
await client.add("I hate coffee", user_id="alice", memory_type="preference")

# Result: "I hate coffee" is kept (newer), "I love coffee" is deleted
```

### Example 2: Fact Verification (Confidence)

```python
client = UnifiedHippocampAI(
    conflict_resolution_strategy=ConflictResolutionStrategy.CONFIDENCE
)

# Low confidence fact
await client.add(
    "I might be allergic to peanuts",
    user_id="bob",
    memory_type="fact",
    confidence=0.5
)

# High confidence fact (after allergy test)
await client.add(
    "I am NOT allergic to peanuts",
    user_id="bob",
    memory_type="fact",
    confidence=0.98
)

# Result: High confidence memory wins
```

### Example 3: Detecting Conflicts (No Auto-Resolve)

```python
# Just detect conflicts, don't resolve them
conflicts = await client.memory_service.detect_memory_conflicts(
    user_id="charlie",
    memory_type="preference"
)

for conflict in conflicts:
    print(f"Found conflict: {conflict['memory_1']['text']} vs {conflict['memory_2']['text']}")
```

### Example 4: Manual Resolution

```python
# Dry run to see what would happen
result = await client.memory_service.resolve_memory_conflicts(
    user_id="david",
    strategy=ConflictResolutionStrategy.TEMPORAL,
    dry_run=True  # Preview only
)

print(f"Would resolve {result['conflicts_found']} conflicts")

# Actually resolve
result = await client.memory_service.resolve_memory_conflicts(
    user_id="david",
    strategy=ConflictResolutionStrategy.TEMPORAL,
    dry_run=False  # Actually do it
)

print(f"Resolved {result['conflicts_resolved']} conflicts")
```

## When to Use What Strategy

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| User preferences | `temporal` | Preferences evolve over time |
| Facts with confidence | `confidence` | Trust more confident data |
| Critical data | `importance` | Prioritize important info |
| Medical records | `user_review` | Need human verification |
| Complex evolution | `auto_merge` | AI captures nuance |
| Context-dependent | `keep_both` | Both may be valid |

## Troubleshooting

### Problem: Conflicts not being detected

**Solution 1:** Lower similarity threshold
```bash
CONFLICT_SIMILARITY_THRESHOLD=0.70  # Default is 0.75
```

**Solution 2:** Enable LLM checking
```bash
CONFLICT_CHECK_LLM=true
```

### Problem: Too many false positives

**Solution:** Increase similarity threshold
```bash
CONFLICT_SIMILARITY_THRESHOLD=0.85
```

### Problem: Wrong memory is kept

**Solution:** Change resolution strategy
```bash
# Try confidence instead of temporal
CONFLICT_RESOLUTION_STRATEGY=confidence
```

### Problem: Performance is slow

**Solution:** Disable LLM checking
```bash
CONFLICT_CHECK_LLM=false
```

## Advanced: Custom Configuration per Operation

```python
# Disable conflict checking for specific memory
await client.add(
    "I love coffee",
    user_id="user123",
    memory_type="preference",
    check_conflicts=False,  # Skip conflict detection
    auto_resolve_conflicts=False  # Skip resolution
)

# Use different strategy for specific operation
from hippocampai.pipeline.conflict_resolution import ConflictResolutionStrategy

result = await client.memory_service.resolve_memory_conflicts(
    user_id="user123",
    strategy=ConflictResolutionStrategy.CONFIDENCE,  # Override default
    memory_type="fact",  # Only facts
    check_llm=True  # Use LLM for deep analysis
)
```

## Complete Working Example

```python
from hippocampai.unified_client import UnifiedHippocampAI
from hippocampai.pipeline.conflict_resolution import ConflictResolutionStrategy

async def main():
    # Initialize with conflict resolution
    client = UnifiedHippocampAI(
        enable_conflict_resolution=True,
        conflict_resolution_strategy=ConflictResolutionStrategy.TEMPORAL
    )

    user_id = "alice"

    # Add initial preference
    print("Adding: 'I love coffee'")
    await client.add(
        text="I love coffee",
        user_id=user_id,
        memory_type="preference",
        confidence=0.9
    )

    # Add conflicting preference
    print("Adding: 'I hate coffee'")
    await client.add(
        text="I hate coffee",
        user_id=user_id,
        memory_type="preference",
        confidence=0.95
    )

    # Recall to see which one won
    memories = await client.recall(
        query="coffee",
        user_id=user_id,
        k=10
    )

    print(f"\nActive memory: {memories[0].text}")
    # Output: "I hate coffee" (newer memory won)

    # Check for any conflicts
    conflicts = await client.memory_service.detect_memory_conflicts(
        user_id=user_id
    )

    print(f"\nConflicts detected: {len(conflicts)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Next Steps

- **Read full guide:** See `docs/MEMORY_CONFLICT_RESOLUTION_GUIDE.md` for detailed documentation
- **Try different strategies:** Experiment with confidence, importance, and auto_merge
- **Configure for your use case:** Adjust thresholds and settings in `.env`
- **Monitor conflicts:** Periodically run `detect_memory_conflicts()` to check for issues

## Key Points to Remember

1. ✅ **Enabled by default** - No setup needed
2. ✅ **Automatic** - Conflicts are detected and resolved on memory creation
3. ✅ **Configurable** - Choose strategy and thresholds to fit your needs
4. ✅ **Safe** - Use `dry_run=True` to preview changes
5. ✅ **Fast** - Pattern-based detection is quick, LLM is optional

---

**Ready to go!** Just add memories and let HippocampAI handle the conflicts automatically. 🎉
