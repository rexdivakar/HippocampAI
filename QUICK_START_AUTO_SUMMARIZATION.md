# 🚀 Auto-Summarization Quick Start

## Installation

The auto-summarization features are already included in HippocampAI. No additional installation required!

## Enable Features

Add to your `.env` file:

```bash
# Enable auto-summarization features
AUTO_SUMMARIZATION_ENABLED=true
HIERARCHICAL_SUMMARIZATION_ENABLED=true
SLIDING_WINDOW_ENABLED=true
IMPORTANCE_DECAY_ENABLED=true

# Memory tiering (keep recent hot, compress old)
HOT_THRESHOLD_DAYS=7
WARM_THRESHOLD_DAYS=30
COLD_THRESHOLD_DAYS=90

# Decay settings
DECAY_FUNCTION=exponential
MIN_IMPORTANCE_THRESHOLD=1.0
```

## Basic Usage

```python
from hippocampai import MemoryClient
from hippocampai.pipeline import (
    AutoSummarizer,
    ImportanceDecayEngine,
    AutoConsolidator,
)

# Initialize client
client = MemoryClient()

# Get memories
memories = client.get_memories(user_id="alice", limit=100)

# 1️⃣ CREATE HIERARCHICAL SUMMARY
summarizer = AutoSummarizer(llm=client.llm)
summaries = summarizer.create_hierarchical_summary(
    memories, 
    user_id="alice",
    max_levels=3
)

print(f"Created {len(summaries)} hierarchical summaries")
for summary in summaries[:3]:
    print(f"Level {summary.level}: {summary.content[:100]}...")

# 2️⃣ COMPRESS OLD MEMORIES (Sliding Window)
result = summarizer.sliding_window_compression(
    memories,
    window_size=10,      # Process 10 at a time
    keep_recent=5,       # Keep last 5 uncompressed
    user_id="alice"
)

print(f"💾 Saved {result['stats']['tokens_saved']} tokens")
print(f"📊 Compression ratio: {result['stats']['compression_ratio']:.1%}")

# 3️⃣ APPLY IMPORTANCE DECAY
decay_engine = ImportanceDecayEngine()
decay_results = decay_engine.apply_decay_batch(memories)

print(f"⏱️ Decayed {decay_results['stats']['updated_count']} memories")
print(f"📉 Average decay: {decay_results['stats']['avg_decay']:.2f}")

# 4️⃣ IDENTIFY PRUNING CANDIDATES
pruning = decay_engine.identify_pruning_candidates(memories)

print(f"🗑️ Prune candidates: {pruning['stats']['prune_candidates']}")
print(f"💡 Recommendation: {pruning['summary']['action']}")

# 5️⃣ CONSOLIDATE SIMILAR MEMORIES
consolidator = AutoConsolidator(consolidator=client.consolidator)
result = consolidator.auto_consolidate(memories)

if result:
    print(f"🔄 Consolidated {result.memories_consolidated} memories")
    print(f"✅ Efficiency: {result.calculate_efficiency():.1%}")
```

## Common Use Cases

### Use Case 1: Token Budget Management

```python
# Stay within token budget
MAX_TOKENS = 50000

current_tokens = sum(Memory.estimate_tokens(m.text) for m in memories)
print(f"Current: {current_tokens} tokens")

if current_tokens > MAX_TOKENS:
    # Compress old memories
    result = summarizer.sliding_window_compression(
        memories, window_size=15, keep_recent=10
    )
    print(f"Saved: {result['stats']['tokens_saved']} tokens")
```

### Use Case 2: Health-Based Cleanup

```python
# Generate health report
report = decay_engine.generate_maintenance_report(memories)

print(f"Average health: {report['summary']['average_health']:.1f}/10")
print(f"Keep: {report['summary']['keep_count']}")
print(f"Decay: {report['summary']['decay_count']}")
print(f"Prune: {report['summary']['prune_count']}")

# Prune unhealthy memories
for candidate in pruning['candidates']:
    if candidate.health_score < 2.0:
        print(f"Pruning: {candidate.memory_id}")
        client.delete_memory(candidate.memory_id)
```

### Use Case 3: Analyze Before Acting

```python
# 1. Analyze compression opportunities
analysis = summarizer.analyze_compression_opportunities(memories)
print(f"Potential savings: {analysis['potential_tokens_saved']} tokens")

# 2. Estimate consolidation impact
estimate = consolidator.estimate_consolidation_impact(memories)
print(f"Would consolidate: {estimate['memories_in_groups']} memories")

# 3. Only proceed if beneficial
if analysis['potential_tokens_saved'] > 1000:
    result = summarizer.sliding_window_compression(memories)
```

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `AUTO_SUMMARIZATION_ENABLED` | `true` | Enable auto-summarization |
| `HIERARCHICAL_SUMMARIZATION_ENABLED` | `true` | Enable hierarchical summaries |
| `SLIDING_WINDOW_ENABLED` | `true` | Enable sliding window compression |
| `SLIDING_WINDOW_SIZE` | `10` | Memories per compression window |
| `SLIDING_WINDOW_KEEP_RECENT` | `5` | Recent memories to keep uncompressed |
| `HOT_THRESHOLD_DAYS` | `7` | Days for HOT tier (no compression) |
| `WARM_THRESHOLD_DAYS` | `30` | Days for WARM tier (light compression) |
| `COLD_THRESHOLD_DAYS` | `90` | Days for COLD tier (heavy compression) |
| `IMPORTANCE_DECAY_ENABLED` | `true` | Enable importance decay |
| `DECAY_FUNCTION` | `exponential` | Decay function type |
| `MIN_IMPORTANCE_THRESHOLD` | `1.0` | Minimum importance floor |
| `AUTO_PRUNING_ENABLED` | `false` | Enable automatic pruning (manual by default) |
| `PRUNING_STRATEGY` | `comprehensive` | Pruning strategy |
| `MIN_HEALTH_THRESHOLD` | `3.0` | Minimum health to keep (0-10 scale) |

## Memory Tiers

Memories are automatically classified into tiers:

| Tier | Age | Access | Compression | Token Retention |
|------|-----|--------|-------------|-----------------|
| **HOT** | < 7 days OR 10+ accesses | Frequent | None | 100% |
| **WARM** | 7-30 days | Moderate | Light | 80% |
| **COLD** | 30-90 days | Rare | Medium | 50% |
| **ARCHIVED** | > 90 days | Very rare | Heavy | 25% |

## Decay Functions

Choose the right decay function for your use case:

- **`exponential`** (default) - Natural decay, similar to forgetting curve
- **`linear`** - Predictable, uniform decay over time
- **`logarithmic`** - Slower, gentler decay for long-term retention
- **`step`** - Discrete intervals, simpler calculation
- **`hybrid`** - Adaptive decay based on access patterns (recommended)

## Pruning Strategies

- **`comprehensive`** (default) - Multi-factor analysis (importance + recency + access + confidence)
- **`importance_only`** - Based on importance score only
- **`age_based`** - Based on age thresholds
- **`access_based`** - Based on access frequency
- **`conservative`** - Only prune very low-value memories (safest)

## Health Score Interpretation

| Score | Recommendation | Action |
|-------|----------------|--------|
| 8-10 | **Keep** | Preserve as-is |
| 6-8 | **Keep** | Monitor for changes |
| 4-6 | **Decay** | Allow natural decay |
| 2-4 | **Archive** | Consider compression |
| 0-2 | **Prune** | Safe to delete |

## Best Practices

1. **Start Conservative**
   - Use `PRUNING_STRATEGY=conservative`
   - Set high `MIN_HEALTH_THRESHOLD` (4.0+)
   - Disable `AUTO_PRUNING_ENABLED` initially

2. **Monitor First**
   - Generate maintenance reports weekly
   - Analyze compression opportunities
   - Estimate consolidation impact
   - Only enable automation after validation

3. **Protect Important Memories**
   - High importance (>8) rarely pruned
   - Frequently accessed memories protected
   - Recent memories always kept
   - High confidence weighted heavily

4. **Regular Maintenance**
   - Apply decay weekly
   - Consolidate monthly
   - Compress as needed
   - Prune quarterly

## Troubleshooting

### Too Many Memories Pruned?

```python
# Increase health threshold
decay_engine.identify_pruning_candidates(
    memories,
    min_health_threshold=4.0  # Higher = more conservative
)
```

### Not Saving Enough Tokens?

```python
# Use more aggressive compression
result = summarizer.sliding_window_compression(
    memories,
    window_size=20,      # Larger windows
    keep_recent=3,       # Keep fewer recent
)
```

### Decay Too Fast?

```python
# Use slower decay function
config = DecayConfig(decay_function=DecayFunction.LOGARITHMIC)
decay_engine = ImportanceDecayEngine(config)
```

## Documentation

- **Full Guide**: `docs/AUTO_SUMMARIZATION_GUIDE.md`
- **Implementation Summary**: `AUTO_SUMMARIZATION_IMPLEMENTATION_SUMMARY.md`
- **Tests**: `tests/test_auto_*.py`

## Support

For issues or questions:
1. Check the full documentation in `docs/AUTO_SUMMARIZATION_GUIDE.md`
2. Review test examples in `tests/test_auto_*.py`
3. File an issue on GitHub

---

**🎉 You're ready to use auto-summarization!**
