# Auto-Summarization Enhancement Guide

## Overview

HippocampAI now includes comprehensive auto-summarization features that automatically manage memory lifecycle, compress older memories, and intelligently prune low-value information. This guide covers the four major components:

1. **Recursive/Hierarchical Summarization** - Multi-level memory summarization
2. **Sliding Window Compression** - Progressive compression of older memories
3. **Automatic Consolidation** - Scheduled memory consolidation
4. **Importance Decay & Pruning** - Intelligent memory lifecycle management

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Recursive Hierarchical Summarization](#recursive-hierarchical-summarization)
- [Sliding Window Compression](#sliding-window-compression)
- [Automatic Consolidation](#automatic-consolidation)
- [Importance Decay and Pruning](#importance-decay-and-pruning)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Features

### ✅ Recursive/Hierarchical Summarization
- Multi-level memory hierarchy (configurable up to N levels)
- Batch-based summarization with configurable batch sizes
- LLM-powered or heuristic-based summarization
- Parent-child relationship tracking
- Token-aware compression

### ✅ Sliding Window Compression
- Keep recent memories verbatim
- Progressively compress older memories
- Configurable window sizes and retention
- Token budget management
- Compression statistics and analysis

### ✅ Memory Tiering
- **HOT** - Frequently accessed, kept verbatim
- **WARM** - Occasionally accessed, lightly compressed (80%)
- **COLD** - Rarely accessed, heavily compressed (50%)
- **ARCHIVED** - Very old, maximum compression (25%)

### ✅ Importance Decay
- Multiple decay functions (exponential, linear, logarithmic, step, hybrid)
- Access-based boost (reduces decay for frequently used memories)
- Confidence-weighted decay
- Type-specific half-lives
- Configurable minimum importance threshold

### ✅ Intelligent Pruning
- Multiple strategies (comprehensive, importance-only, age-based, access-based, conservative)
- Health scoring system (0-10 scale)
- Recommendations (keep, decay, archive, prune)
- Target-based pruning
- Maintenance reports

### ✅ Automatic Consolidation
- Scheduled consolidation (time-based)
- Threshold-based triggers (memory count)
- Token budget triggers
- Similarity-based grouping
- Consolidation history and statistics

---

## Quick Start

```python
from hippocampai import MemoryClient
from hippocampai.pipeline import (
    AutoSummarizer,
    ImportanceDecayEngine,
    AutoConsolidator,
    DecayConfig,
    DecayFunction,
)

# Initialize client
client = MemoryClient()

# 1. Auto-Summarization
summarizer = AutoSummarizer(
    llm=client.llm,
    hot_threshold_days=7,
    warm_threshold_days=30,
    cold_threshold_days=90,
)

# Create hierarchical summary
memories = client.get_memories(user_id="alice")
summaries = summarizer.create_hierarchical_summary(
    memories, user_id="alice", max_levels=3
)

# 2. Sliding Window Compression
result = summarizer.sliding_window_compression(
    memories,
    window_size=10,
    keep_recent=5,
    user_id="alice"
)
print(f"Tokens saved: {result['stats']['tokens_saved']}")

# 3. Importance Decay
decay_engine = ImportanceDecayEngine(
    config=DecayConfig(decay_function=DecayFunction.EXPONENTIAL)
)

# Apply decay
decay_results = decay_engine.apply_decay_batch(memories)
print(f"Memories decayed: {decay_results['stats']['updated_count']}")

# 4. Pruning Analysis
pruning_report = decay_engine.identify_pruning_candidates(memories)
print(f"Prune candidates: {pruning_report['stats']['prune_candidates']}")

# 5. Auto-Consolidation
from hippocampai.pipeline import ConsolidationSchedule

consolidator = AutoConsolidator(
    schedule=ConsolidationSchedule(
        enabled=True,
        interval_hours=168,  # Weekly
        min_memories_threshold=50,
    ),
    consolidator=client.consolidator,
)

# Check if consolidation needed
should_run, trigger = consolidator.should_consolidate(memories)
if should_run:
    result = consolidator.consolidate_batch(memories, trigger)
    print(f"Consolidated: {result.memories_consolidated} memories")
```

---

## Recursive Hierarchical Summarization

### Concept

Hierarchical summarization creates multiple levels of summaries, starting from individual memories (level 0) and progressively grouping and summarizing them into higher-level summaries.

```
Level 0: [Memory1, Memory2, Memory3, Memory4, Memory5, Memory6]
           ↓        ↓        ↓        ↓        ↓        ↓
Level 1:    [Summary1-2]  [Summary3-4]  [Summary5-6]
                  ↓              ↓              ↓
Level 2:            [Summary1-4]    [Summary5-6]
                          ↓              ↓
Level 3:                [Summary1-6]
```

### Usage

```python
from hippocampai.pipeline import AutoSummarizer

summarizer = AutoSummarizer(
    llm=client.llm,
    max_tokens_per_summary=150,
    hierarchical_batch_size=5,
)

# Create hierarchical summary
memories = client.get_memories(user_id="alice", limit=20)
summaries = summarizer.create_hierarchical_summary(
    memories=memories,
    user_id="alice",
    max_levels=3
)

# Explore hierarchy
for summary in summaries:
    print(f"Level {summary.level}: {summary.content[:100]}...")
    print(f"  Covers {len(summary.memory_ids)} memories")
    print(f"  Token count: {summary.token_count}")
    if summary.parent_id:
        print(f"  Parent: {summary.parent_id}")
```

### Parameters

- `max_levels` - Maximum hierarchy depth (default: 3)
- `hierarchical_batch_size` - Memories per batch (default: 5)
- `max_tokens_per_summary` - Token limit per summary (default: 150)

---

## Sliding Window Compression

### Concept

Sliding window compression keeps recent memories uncompressed while progressively compressing older memories based on their age and access patterns.

```
[Old memories........][Older memories...][Recent memories (verbatim)]
     ↓ HEAVY              ↓ MEDIUM              ↓ NONE
  Compressed 75%      Compressed 50%         Kept 100%
```

### Usage

```python
# Apply sliding window compression
result = summarizer.sliding_window_compression(
    memories=memories,
    window_size=10,       # Process 10 memories at a time
    keep_recent=5,        # Keep last 5 uncompressed
    user_id="alice"
)

# Review results
print(f"Total memories: {result['stats']['total_memories']}")
print(f"Compression ratio: {result['stats']['compression_ratio']:.2%}")
print(f"Tokens saved: {result['stats']['tokens_saved']}")

# Access compressed memories
for compressed in result['compressed_memories']:
    print(f"Summary: {compressed.summary_text}")
    print(f"  Original tokens: {compressed.original_token_count}")
    print(f"  Compressed tokens: {compressed.compressed_token_count}")
    print(f"  Compression ratio: {compressed.compression_ratio:.2%}")
```

### Analyze Compression Opportunities

```python
# Analyze before compressing
analysis = summarizer.analyze_compression_opportunities(memories)

print(f"Total tokens: {analysis['total_tokens']}")
print(f"Potential savings: {analysis['potential_tokens_saved']}")
print(f"Recommendation: {analysis['recommended_action']}")

# Review by tier
for opportunity in analysis['opportunities']:
    print(f"{opportunity['tier']}:")
    print(f"  Memories: {opportunity['memory_count']}")
    print(f"  Current tokens: {opportunity['current_tokens']}")
    print(f"  After compression: {opportunity['compressed_tokens']}")
    print(f"  Savings: {opportunity['tokens_saved']}")
```

---

## Memory Tiering

### Automatic Tier Classification

```python
# Determine tier for a memory
tier = summarizer.determine_memory_tier(memory)
print(f"Memory tier: {tier.value}")  # HOT, WARM, COLD, or ARCHIVED

# Get compression level for tier
compression_level = summarizer.get_compression_level_for_tier(tier)
print(f"Compression: {compression_level.value}")

# Compress memory
if compression_level != CompressionLevel.NONE:
    compressed = summarizer.compress_memory(memory, compression_level)
    print(f"Compressed from {compressed.original_token_count} to {compressed.compressed_token_count} tokens")
```

### Tier Thresholds

Tiers are determined by:
- **Age**: Days since creation
- **Access patterns**: Access count and recency
- **Importance**: Memory importance score

Default thresholds (configurable):
- HOT: < 7 days OR access_count >= 10
- WARM: 7-30 days AND access_count >= 3
- COLD: 30-90 days
- ARCHIVED: > 90 days

---

## Automatic Consolidation

### Setup

```python
from hippocampai.pipeline import (
    AutoConsolidator,
    ConsolidationSchedule,
    ConsolidationTrigger,
)

# Create schedule
schedule = ConsolidationSchedule(
    enabled=True,
    interval_hours=168,  # Weekly
    min_memories_threshold=50,
    similarity_threshold=0.85,
    max_batch_size=100,
    token_budget_threshold=10000,
)

# Initialize consolidator
consolidator = AutoConsolidator(
    schedule=schedule,
    consolidator=client.consolidator,
    similarity_calculator=None,  # Uses heuristics
)
```

### Manual Consolidation

```python
# Check if consolidation is needed
should_run, trigger = consolidator.should_consolidate(memories)

if should_run:
    print(f"Consolidation triggered by: {trigger.value}")

    # Run consolidation
    result = consolidator.consolidate_batch(
        memories,
        trigger=ConsolidationTrigger.MANUAL
    )

    print(f"Status: {result.status.value}")
    print(f"Memories analyzed: {result.memories_analyzed}")
    print(f"Memories consolidated: {result.memories_consolidated}")
    print(f"Groups created: {result.consolidation_groups}")
    print(f"Tokens before: {result.tokens_before}")
    print(f"Tokens after: {result.tokens_after}")
    print(f"Efficiency: {result.calculate_efficiency():.2%}")
```

### Automatic Consolidation

```python
# Auto-consolidate if conditions met
result = consolidator.auto_consolidate(memories)

if result:
    print(f"Auto-consolidation completed: {result.status.value}")
else:
    print("Consolidation not needed yet")
```

### Consolidation Statistics

```python
# Get historical statistics
stats = consolidator.get_consolidation_stats()

print(f"Total runs: {stats['total_runs']}")
print(f"Completed: {stats['completed_runs']}")
print(f"Failed: {stats['failed_runs']}")
print(f"Total memories consolidated: {stats['total_memories_consolidated']}")
print(f"Total tokens saved: {stats['total_tokens_saved']}")
print(f"Average efficiency: {stats['average_efficiency']:.2%}")

# Get recent results
recent = consolidator.get_recent_results(limit=5)
for result in recent:
    print(f"{result.started_at}: {result.status.value}")
```

### Estimate Impact

```python
# Estimate before running
estimate = consolidator.estimate_consolidation_impact(memories)

print(f"Total memories: {estimate['total_memories']}")
print(f"Estimated groups: {estimate['estimated_groups']}")
print(f"Current tokens: {estimate['current_tokens']}")
print(f"Estimated after: {estimate['estimated_tokens_after']}")
print(f"Estimated savings: {estimate['estimated_tokens_saved']}")
print(f"Recommendation: {estimate['recommendation']}")
```

---

## Importance Decay and Pruning

### Importance Decay

```python
from hippocampai.pipeline import (
    ImportanceDecayEngine,
    DecayConfig,
    DecayFunction,
)

# Configure decay
config = DecayConfig(
    decay_function=DecayFunction.EXPONENTIAL,
    half_life_days={
        "preference": 90,
        "goal": 60,
        "fact": 30,
        "event": 14,
        "context": 30,
        "habit": 90,
    },
    min_importance=1.0,
    access_boost_factor=0.5,
    confidence_weight=0.3,
)

decay_engine = ImportanceDecayEngine(config)

# Calculate decayed importance for a memory
decayed = decay_engine.calculate_decayed_importance(memory)
print(f"Original importance: {memory.importance}")
print(f"Decayed importance: {decayed}")

# Apply decay to all memories
result = decay_engine.apply_decay_batch(memories)

print(f"Memories updated: {result['stats']['updated_count']}")
print(f"Total decay: {result['stats']['total_decay']:.2f}")
print(f"Significant decay (>10%): {result['stats']['significant_decay_count']}")

# Review updates
for memory_id, update in result['updates'].items():
    if update['decay_percentage'] > 10:
        print(f"Memory {memory_id}: {update['decay_percentage']:.1f}% decay")
```

### Memory Health Scoring

```python
# Calculate health score
health = decay_engine.calculate_memory_health(memory)

print(f"Health score: {health.health_score:.1f}/10")
print(f"Recommendation: {health.recommendation}")
print(f"Components:")
print(f"  Importance: {health.importance_score:.1f}/10")
print(f"  Recency: {health.recency_score:.1f}/10")
print(f"  Access: {health.access_score:.1f}/10")
print(f"  Confidence: {health.confidence_score:.1f}/10")
```

### Pruning

```python
from hippocampai.pipeline import PruningStrategy

# Identify pruning candidates
result = decay_engine.identify_pruning_candidates(
    memories,
    strategy=PruningStrategy.COMPREHENSIVE,
    min_health_threshold=3.0,
)

print(f"Total memories: {result['stats']['total_memories']}")
print(f"Prune candidates: {result['stats']['prune_candidates']}")
print(f"Prune percentage: {result['stats']['prune_percentage']:.1f}%")

# Review candidates
for candidate in result['candidates']:
    print(f"Memory {candidate.memory_id}:")
    print(f"  Health: {candidate.health_score:.1f}/10")
    print(f"  Recommendation: {candidate.recommendation}")
    print(f"  Age: {candidate.factors['age_days']} days")
    print(f"  Access count: {candidate.factors['access_count']}")
```

### Maintenance Report

```python
# Generate comprehensive maintenance report
report = decay_engine.generate_maintenance_report(memories)

print("=" * 50)
print("MEMORY MAINTENANCE REPORT")
print("=" * 50)

# Summary
summary = report['summary']
print(f"\nTotal memories: {summary['total_memories']}")
print(f"Average health: {summary['average_health']:.1f}/10")
print(f"Keep: {summary['keep_count']}")
print(f"Decay: {summary['decay_count']}")
print(f"Archive: {summary['archive_count']}")
print(f"Prune: {summary['prune_count']}")

# Health distribution
print("\nHealth Distribution:")
for category, count in report['health_distribution'].items():
    print(f"  {category}: {count}")

# Recommended actions
print("\nImmediate Actions:")
for action in report['actions']['immediate']:
    print(f"  • {action}")

print("\nRecommended Actions:")
for action in report['actions']['recommended']:
    print(f"  • {action}")
```

---

## Configuration

### Environment Variables

Add to `.env` file:

```bash
# Auto-Summarization
AUTO_SUMMARIZATION_ENABLED=true
HIERARCHICAL_SUMMARIZATION_ENABLED=true
SLIDING_WINDOW_ENABLED=true
SLIDING_WINDOW_SIZE=10
SLIDING_WINDOW_KEEP_RECENT=5
MAX_TOKENS_PER_SUMMARY=150
HIERARCHICAL_BATCH_SIZE=5
HIERARCHICAL_MAX_LEVELS=3

# Memory Tiering
HOT_THRESHOLD_DAYS=7
WARM_THRESHOLD_DAYS=30
COLD_THRESHOLD_DAYS=90
HOT_ACCESS_COUNT_THRESHOLD=10

# Importance Decay
IMPORTANCE_DECAY_ENABLED=true
DECAY_FUNCTION=exponential  # linear, exponential, logarithmic, step, hybrid
DECAY_INTERVAL_HOURS=24
MIN_IMPORTANCE_THRESHOLD=1.0
ACCESS_BOOST_FACTOR=0.5

# Pruning
AUTO_PRUNING_ENABLED=false  # Manual by default for safety
PRUNING_INTERVAL_HOURS=168  # Weekly
PRUNING_STRATEGY=comprehensive
MIN_HEALTH_THRESHOLD=3.0
PRUNING_TARGET_PERCENTAGE=0.1

# Consolidation
AUTO_CONSOLIDATION_ENABLED=false  # Manual by default
CONSOLIDATION_INTERVAL_HOURS=168
CONSOLIDATION_THRESHOLD=0.85
```

### Programmatic Configuration

```python
from hippocampai import Config

config = Config()

# Update settings
config.auto_summarization_enabled = True
config.hierarchical_summarization_enabled = True
config.sliding_window_size = 15
config.hot_threshold_days = 10
config.decay_function = "hybrid"
config.min_health_threshold = 4.0

# Initialize with custom config
client = MemoryClient(config=config)
```

---

## Best Practices

### 1. Start Conservative

```python
# Begin with conservative settings
config = DecayConfig(
    decay_function=DecayFunction.LOGARITHMIC,  # Slowest decay
    min_importance=2.0,  # Higher minimum
    access_boost_factor=1.0,  # More boost for accessed memories
)

# Use conservative pruning
result = decay_engine.identify_pruning_candidates(
    memories,
    strategy=PruningStrategy.CONSERVATIVE,
)
```

### 2. Monitor Before Automating

```python
# 1. Analyze first
analysis = summarizer.analyze_compression_opportunities(memories)
print(f"Would save: {analysis['potential_tokens_saved']} tokens")

# 2. Generate maintenance report
report = decay_engine.generate_maintenance_report(memories)
print(f"Would prune: {report['summary']['prune_count']} memories")

# 3. Estimate consolidation impact
estimate = consolidator.estimate_consolidation_impact(memories)
print(f"Would consolidate: {estimate['memories_in_groups']} memories")

# 4. Only then enable automation
if analysis['potential_tokens_saved'] > 1000:
    config.auto_summarization_enabled = True
```

### 3. Regular Maintenance Schedule

```python
# Weekly maintenance routine
def weekly_maintenance():
    memories = client.get_memories(user_id="alice")

    # 1. Apply decay
    decay_results = decay_engine.apply_decay_batch(memories)

    # 2. Generate report
    report = decay_engine.generate_maintenance_report(memories)

    # 3. Prune if needed
    if report['summary']['prune_count'] > 100:
        candidates = decay_engine.identify_pruning_candidates(
            memories,
            strategy=PruningStrategy.COMPREHENSIVE,
        )
        # Prune candidates with health < 2.0
        for candidate in candidates:
            if candidate.health_score < 2.0:
                client.delete_memory(candidate.memory_id)

    # 4. Consolidate
    consolidator.auto_consolidate(memories)

    # 5. Compress
    result = summarizer.sliding_window_compression(memories)

    return report
```

### 4. Preserve Important Memories

```python
# Protect high-value memories from aggressive pruning
def identify_safe_to_prune(memories):
    candidates = []

    for memory in memories:
        health = decay_engine.calculate_memory_health(memory)

        # Don't prune if:
        # - High importance (>8)
        # - Frequently accessed (>10 times)
        # - Recent (< 30 days)
        # - High confidence (>0.9)
        if (memory.importance <= 8 and
            memory.access_count <= 10 and
            (datetime.now(timezone.utc) - memory.created_at).days > 30 and
            memory.confidence <= 0.9):

            if health.health_score < 3.0:
                candidates.append(memory)

    return candidates
```

### 5. Test on Subset First

```python
# Test on small subset before applying to all memories
test_memories = memories[:100]  # First 100 memories

# Test compression
test_result = summarizer.sliding_window_compression(test_memories)
if test_result['stats']['compression_ratio'] < 0.7:  # Good compression
    # Apply to all
    full_result = summarizer.sliding_window_compression(memories)
```

---

## Examples

### Example 1: Full Lifecycle Management

```python
from hippocampai import MemoryClient
from hippocampai.pipeline import (
    AutoSummarizer,
    ImportanceDecayEngine,
    AutoConsolidator,
    DecayFunction,
    PruningStrategy,
)

# Initialize
client = MemoryClient()
memories = client.get_memories(user_id="alice")

# 1. Analyze current state
summarizer = AutoSummarizer(llm=client.llm)
analysis = summarizer.analyze_compression_opportunities(memories)
print(f"Current state: {len(memories)} memories, {analysis['total_tokens']} tokens")

# 2. Apply importance decay
decay_engine = ImportanceDecayEngine()
decay_results = decay_engine.apply_decay_batch(memories)
print(f"Decayed {decay_results['stats']['updated_count']} memories")

# 3. Identify pruning candidates
pruning = decay_engine.identify_pruning_candidates(
    memories,
    strategy=PruningStrategy.COMPREHENSIVE,
    min_health_threshold=3.0,
)
print(f"Found {pruning['stats']['prune_candidates']} candidates for pruning")

# 4. Prune low-value memories
for candidate in pruning['candidates']:
    if candidate.health_score < 2.0:
        print(f"Pruning memory {candidate.memory_id} (health: {candidate.health_score:.1f})")
        client.delete_memory(candidate.memory_id)

# 5. Consolidate similar memories
consolidator = AutoConsolidator(consolidator=client.consolidator)
result = consolidator.auto_consolidate(memories)
if result:
    print(f"Consolidated {result.memories_consolidated} memories")

# 6. Compress older memories
compression_result = summarizer.sliding_window_compression(
    memories,
    window_size=10,
    keep_recent=5,
)
print(f"Compression saved {compression_result['stats']['tokens_saved']} tokens")

# 7. Create hierarchical summary
summaries = summarizer.create_hierarchical_summary(memories, user_id="alice")
print(f"Created {len(summaries)} hierarchical summaries")
```

### Example 2: Token Budget Management

```python
# Manage memory within token budget
MAX_TOKENS = 50000

def optimize_for_token_budget(memories, max_tokens):
    summarizer = AutoSummarizer(llm=client.llm)

    # Current token count
    current_tokens = sum(Memory.estimate_tokens(m.text) for m in memories)
    print(f"Current tokens: {current_tokens}")

    if current_tokens <= max_tokens:
        print("Within budget!")
        return memories

    # Calculate how much to save
    tokens_to_save = current_tokens - max_tokens
    print(f"Need to save: {tokens_to_save} tokens")

    # Apply progressive compression
    result = summarizer.sliding_window_compression(
        memories,
        window_size=15,
        keep_recent=10,
    )

    if result['stats']['tokens_saved'] >= tokens_to_save:
        print(f"Success! Saved {result['stats']['tokens_saved']} tokens")
        return result

    # If still over budget, prune low-value memories
    decay_engine = ImportanceDecayEngine()
    pruning = decay_engine.identify_pruning_candidates(
        memories,
        strategy=PruningStrategy.COMPREHENSIVE,
    )

    # Calculate tokens that would be freed
    prune_tokens = 0
    for candidate in sorted(pruning['candidates'], key=lambda c: c.health_score):
        memory = next(m for m in memories if m.id == candidate.memory_id)
        prune_tokens += Memory.estimate_tokens(memory.text)

        if prune_tokens >= tokens_to_save:
            break

    print(f"Would prune {len(pruning['candidates'])} memories to save {prune_tokens} tokens")
    return result

# Execute
optimize_for_token_budget(memories, MAX_TOKENS)
```

### Example 3: Health-Based Memory Triage

```python
def triage_memories(memories):
    decay_engine = ImportanceDecayEngine()

    # Categorize by health
    excellent = []
    good = []
    fair = []
    poor = []
    critical = []

    for memory in memories:
        health = decay_engine.calculate_memory_health(memory)

        if health.health_score >= 8:
            excellent.append((memory, health))
        elif health.health_score >= 6:
            good.append((memory, health))
        elif health.health_score >= 4:
            fair.append((memory, health))
        elif health.health_score >= 2:
            poor.append((memory, health))
        else:
            critical.append((memory, health))

    print("Memory Health Triage:")
    print(f"  Excellent (8-10): {len(excellent)} - Keep as-is")
    print(f"  Good (6-8): {len(good)} - Monitor")
    print(f"  Fair (4-6): {len(fair)} - Light compression")
    print(f"  Poor (2-4): {len(poor)} - Heavy compression or archive")
    print(f"  Critical (0-2): {len(critical)} - Prune immediately")

    # Take action
    summarizer = AutoSummarizer()

    # Compress fair memories
    for memory, health in fair:
        compressed = summarizer.compress_memory(memory, CompressionLevel.LIGHT)

    # Heavy compress poor memories
    for memory, health in poor:
        compressed = summarizer.compress_memory(memory, CompressionLevel.HEAVY)

    # Prune critical
    for memory, health in critical:
        print(f"Pruning: {memory.text[:50]}... (health: {health.health_score:.1f})")
        client.delete_memory(memory.id)

# Execute triage
triage_memories(memories)
```

---

## API Reference

For complete API documentation, see:
- `src/hippocampai/pipeline/auto_summarization.py`
- `src/hippocampai/pipeline/importance_decay.py`
- `src/hippocampai/pipeline/auto_consolidation.py`

---

## Troubleshooting

### Issue: Decay increasing importance

**Solution**: This is expected! The hybrid decay function includes access boosts and confidence weights that can increase importance for frequently-used, high-confidence memories. This is by design to protect valuable memories.

### Issue: Too many memories being pruned

**Solution**: Increase `min_health_threshold` or use `PruningStrategy.CONSERVATIVE`:

```python
result = decay_engine.identify_pruning_candidates(
    memories,
    strategy=PruningStrategy.CONSERVATIVE,
    min_health_threshold=2.0,  # Only prune very unhealthy
)
```

### Issue: Compression not saving enough tokens

**Solution**: Adjust thresholds or use more aggressive compression:

```python
# Make tiering more aggressive
summarizer = AutoSummarizer(
    hot_threshold_days=3,   # Shorter HOT period
    warm_threshold_days=14,
    cold_threshold_days=45,
)

# Or compress manually with HEAVY level
for memory in old_memories:
    compressed = summarizer.compress_memory(memory, CompressionLevel.HEAVY)
```

---

## Next Steps

1. **Monitor**: Use maintenance reports to understand your memory patterns
2. **Tune**: Adjust thresholds based on your usage patterns
3. **Automate**: Once comfortable, enable automatic features
4. **Scale**: These features are designed for production use with millions of memories

For questions or issues, please file an issue on GitHub.
