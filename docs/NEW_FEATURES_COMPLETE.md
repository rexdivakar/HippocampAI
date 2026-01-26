# HippocampAI - New Features Documentation

This document covers the advanced features added to HippocampAI for enterprise-grade memory management.

## Table of Contents

1. [Bi-Temporal Fact Tracking](#bi-temporal-fact-tracking)
2. [Automated Context Assembly](#automated-context-assembly)
3. [Custom Schema Support](#custom-schema-support)
4. [Agentic Memory Classification](#agentic-memory-classification)
5. [Benchmark Suite](#benchmark-suite)

---

## Bi-Temporal Fact Tracking

### What It Does

Bi-temporal fact tracking enables you to track not just *what* you know, but *when* you knew it and *when* it was true. This is essential for:

- **Audit trails**: "What did we believe about Alice's employer on January 1st?"
- **Historical queries**: "What was valid during Q1 2024?"
- **Change tracking**: "Show me the history of changes to this fact"
- **Compliance**: Maintain complete history without data loss

### Time Dimensions

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Event Time** | When the fact occurred or was stated | "Alice said she works at Google on Jan 15" |
| **Valid Time** | The interval during which the fact is/was true | "Alice worked at Google from 2020-2023" |
| **System Time** | When HippocampAI recorded the fact (automatic) | "We learned this on Jan 15 at 3:00 PM" |

### Python API

```python
from hippocampai import MemoryClient
from datetime import datetime, timezone

client = MemoryClient()

# Store a bi-temporal fact
fact = client.store_bitemporal_fact(
    text="Alice works at Google",
    user_id="alice",
    entity_id="alice",
    property_name="employer",
    valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
)

# Revise a fact (creates new version, preserves history)
new_fact = client.revise_bitemporal_fact(
    original_fact_id=fact.id,
    new_text="Alice works at Microsoft",
    user_id="alice",
)

# Query facts
result = client.query_bitemporal_facts(
    user_id="alice",
    entity_id="alice",
    property_name="employer",
)

# Get full history
history = client.get_bitemporal_fact_history(fact.fact_id)

# Retract a fact (mark as invalid without deleting)
client.retract_bitemporal_fact(fact_id=fact.id, user_id="alice")
```

### REST API

```bash
# Store fact
curl -X POST http://localhost:8000/v1/bitemporal/facts:store \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice works at Google", "user_id": "alice", "entity_id": "alice", "property_name": "employer"}'

# Query facts
curl -X POST http://localhost:8000/v1/bitemporal/facts:query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "include_superseded": true}'

# Revise fact
curl -X POST http://localhost:8000/v1/bitemporal/facts:revise \
  -H "Content-Type: application/json" \
  -d '{"original_fact_id": "fact-id", "new_text": "Alice works at Microsoft", "user_id": "alice"}'

# Get history
curl -X POST http://localhost:8000/v1/bitemporal/facts:history \
  -H "Content-Type: application/json" \
  -d '{"fact_id": "logical-fact-id"}'
```

### Use Cases

1. **Employment History**: Track job changes over time with full audit trail
2. **Preference Evolution**: See how user preferences change
3. **Compliance Auditing**: Query what was known at any point in time
4. **Data Corrections**: Fix errors while preserving original records

---

## Automated Context Assembly

### What It Does

Context assembly automatically retrieves, ranks, deduplicates, and formats relevant memories for LLM prompts. No manual memory selection needed.

### Benefits

- **Token Budget Management**: Automatically fits context within token limits
- **Relevance Scoring**: Ranks memories by semantic relevance
- **Deduplication**: Removes redundant information
- **Citation Tracking**: Provides memory IDs for attribution
- **Recency Bias**: Optionally prioritize recent memories

### Python API

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Basic context assembly
pack = client.assemble_context(
    query="What are my coffee preferences?",
    user_id="alice",
)

# Use in LLM prompt
prompt = f"""
{pack.final_context_text}

User: What are my coffee preferences?
Assistant:"""

# Advanced options
pack = client.assemble_context(
    query="Recent work updates",
    user_id="alice",
    token_budget=2000,        # Max tokens for context
    max_items=10,             # Max memory items
    recency_bias=0.5,         # Weight for recent memories (0-1)
    type_filter=["fact", "event"],  # Only these types
    min_relevance=0.3,        # Minimum relevance score
    include_citations=True,   # Include memory IDs
    deduplicate=True,         # Remove duplicates
)

# Access structured data
print(f"Selected: {len(pack.selected_items)} items")
print(f"Dropped: {len(pack.dropped_items)} items")
print(f"Total tokens: {pack.total_tokens}")
print(f"Citations: {pack.citations}")
```

### REST API

```bash
# Full context assembly
curl -X POST http://localhost:8000/v1/context:assemble \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coffee preferences",
    "user_id": "alice",
    "token_budget": 2000,
    "type_filter": ["preference"]
  }'

# Text-only response
curl -X POST http://localhost:8000/v1/context:assemble/text \
  -H "Content-Type: application/json" \
  -d '{"query": "coffee preferences", "user_id": "alice"}'
```

### Response Structure

```python
class ContextPack:
    final_context_text: str       # Ready-to-use context string
    citations: list[str]          # Memory IDs for attribution
    selected_items: list[SelectedItem]  # Structured selected memories
    dropped_items: list[DroppedItem]    # Items dropped with reasons
    total_tokens: int             # Estimated token count
    query: str                    # Original query
    user_id: str
    constraints: ContextConstraints
    assembled_at: datetime
```

### Drop Reasons

When memories are excluded, the reason is tracked:
- `TOKEN_BUDGET`: Would exceed token limit
- `LOW_RELEVANCE`: Below min_relevance threshold
- `DUPLICATE`: Similar to already selected item
- `MAX_ITEMS`: Maximum items reached
- `TYPE_FILTER`: Didn't match type filter

---

## Custom Schema Support

### What It Does

Define domain-specific entity types and relationships with validation, without modifying code.

### Benefits

- **Type Safety**: Validate memory payloads against schemas
- **Domain Modeling**: Define custom entity types for your domain
- **Relationship Constraints**: Specify valid relationship endpoints
- **Backwards Compatible**: Default schema when none provided

### Python API

```python
from hippocampai.schema import (
    SchemaRegistry,
    SchemaDefinition,
    EntityTypeDefinition,
    AttributeDefinition,
)

# Get global registry
registry = SchemaRegistry()

# Define custom schema
schema = SchemaDefinition(
    name="crm_schema",
    version="1.0.0",
    entity_types=[
        EntityTypeDefinition(
            name="customer",
            attributes=[
                AttributeDefinition(name="customer_id", type="string", required=True),
                AttributeDefinition(name="name", type="string", required=True),
                AttributeDefinition(name="tier", type="string", enum_values=["bronze", "silver", "gold"]),
            ],
        ),
    ],
    relationship_types=[],
)

# Register and activate
registry.register_schema(schema)
registry.set_active_schema("crm_schema")

# Validate entities
result = registry.validate_entity(
    "customer",
    {"customer_id": "C001", "name": "Acme Corp", "tier": "gold"}
)
print(f"Valid: {result.valid}")
print(f"Errors: {result.errors}")
```

### Load from File

```python
# From JSON
schema = registry.load_schema_from_file("schemas/crm.json", set_active=True)

# From YAML
schema = registry.load_schema_from_file("schemas/crm.yaml", set_active=True)
```

### Attribute Types

| Type | Python Type | Description |
|------|-------------|-------------|
| `string` | `str` | Text values |
| `integer` | `int` | Whole numbers |
| `float` | `float` | Decimal numbers |
| `boolean` | `bool` | True/False |
| `datetime` | `datetime` | Date and time |
| `list` | `list` | Array of values |
| `dict` | `dict` | Key-value mapping |

### Constraints

```python
# Numeric constraints
AttributeDefinition(name="age", type="integer", min_value=0, max_value=150)

# String constraints
AttributeDefinition(name="code", type="string", min_length=3, max_length=10)

# Enum constraints
AttributeDefinition(name="status", type="string", enum_values=["active", "inactive"])
```

---

## Agentic Memory Classification

### What It Does

Uses LLM-based multi-step reasoning for accurate memory type classification, with fallback to pattern-based classification.

### Benefits

- **Higher Accuracy**: LLM reasoning outperforms simple pattern matching
- **Confidence Scores**: Know how certain the classification is
- **Alternative Types**: Get second-best classification when uncertain
- **Reasoning Explanation**: Understand why a type was chosen
- **Caching**: Consistent results with TTL-based cache
- **Validation**: Optional validation step for higher accuracy

### Memory Types

| Type | Description | Examples |
|------|-------------|----------|
| `FACT` | Personal info, identity, biographical data | "My name is Alex", "I work at Google" |
| `PREFERENCE` | Likes, dislikes, opinions | "I love pizza", "I prefer dark mode" |
| `GOAL` | Intentions, aspirations, plans | "I want to learn Python" |
| `HABIT` | Routines, regular activities | "I usually wake up at 7am" |
| `EVENT` | Specific occurrences, meetings | "I met John yesterday" |
| `CONTEXT` | General conversation, observations | "The weather is nice" |

### Python API

```python
from hippocampai.utils.agentic_classifier import (
    AgenticMemoryClassifier,
    classify_memory_agentic,
    classify_memory_agentic_with_confidence,
    classify_memory_agentic_with_details,
)

# Simple classification
memory_type = classify_memory_agentic("My name is Alex")
print(f"Type: {memory_type}")  # MemoryType.FACT

# With confidence
memory_type, confidence = classify_memory_agentic_with_confidence("I love pizza")
print(f"Type: {memory_type}, Confidence: {confidence}")

# Full details
result = classify_memory_agentic_with_details("I want to learn Python")
print(f"Type: {result.memory_type}")
print(f"Confidence: {result.confidence}")
print(f"Level: {result.confidence_level}")  # HIGH, MEDIUM, LOW, UNCERTAIN
print(f"Reasoning: {result.reasoning}")
print(f"Alternative: {result.alternative_type}")

# Custom classifier with validation
classifier = AgenticMemoryClassifier(
    use_cache=True,
    validate_classifications=True,  # Extra validation step
)
result = classifier.classify_with_details("Python is my favorite language")
```

### Confidence Levels

| Level | Score Range | Meaning |
|-------|-------------|---------|
| HIGH | 0.9+ | Very confident |
| MEDIUM | 0.7-0.9 | Reasonably confident |
| LOW | 0.5-0.7 | Uncertain |
| UNCERTAIN | <0.5 | Very uncertain |

### Caching

Classifications are cached for consistency (2-hour TTL, max 2000 entries):

```python
from hippocampai.utils.agentic_classifier import clear_agentic_cache

# Clear cache if needed
clear_agentic_cache()
```

---

## Benchmark Suite

### What It Does

Provides tools for benchmarking HippocampAI performance with synthetic data generation and metrics collection.

### Components

1. **Data Generator**: Create synthetic memories and queries
2. **Benchmark Runner**: Execute operations and collect metrics
3. **Results Export**: JSON and Markdown output

### Python API

```python
from bench.data_generator import generate_memories, generate_queries
from bench.runner import run_benchmark, BenchmarkSuite

# Generate test data
memories = list(generate_memories(count=1000, num_users=10))
queries = list(generate_queries(count=100))

# Run a benchmark
def my_operation():
    # Your operation here
    pass

result = run_benchmark(
    name="my_benchmark",
    operation=my_operation,
    iterations=100,
)

print(f"P50 latency: {result.latency_p50_ms:.2f}ms")
print(f"P99 latency: {result.latency_p99_ms:.2f}ms")
print(f"Throughput: {result.ops_per_second:.2f} ops/sec")

# Create benchmark suite
suite = BenchmarkSuite(name="full_benchmark")
suite.add_result(result)
suite.export_json("results/benchmark.json")
suite.export_markdown("results/benchmark.md")
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| `latency_p50_ms` | 50th percentile latency |
| `latency_p95_ms` | 95th percentile latency |
| `latency_p99_ms` | 99th percentile latency |
| `latency_min_ms` | Minimum latency |
| `latency_max_ms` | Maximum latency |
| `ops_per_second` | Operations per second |
| `errors` | Number of errors |

---

## Running Tests

### All New Features

```bash
QDRANT_URL=http://100.113.229.40:6333 python -m pytest \
  tests/test_bitemporal.py \
  tests/test_context_assembly.py \
  tests/test_custom_schema.py \
  tests/test_agentic_classifier.py \
  tests/test_benchmarks.py -v
```

### Integration Test

```bash
python scripts/test_all_features.py
```

This runs all features end-to-end and cleans up test collections on success.

---

## UI Support

All features are accessible via the web UI:

| Feature | URL Path | Description |
|---------|----------|-------------|
| Bi-Temporal Facts | `/bitemporal` | View, create, revise, and retract facts |
| Context Assembly | `/context` | Assemble context with visual feedback |
| Schema Management | `/schema` | Define and validate custom schemas |
| Agentic Classifier | `/classifier` | Test memory classification |

Access the UI at `http://localhost:3000` after starting the frontend.
