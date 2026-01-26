# HippocampAI Testing Guide

This guide covers how to run tests and validate all features implemented in HippocampAI.

## Prerequisites

### 1. Environment Setup

```bash
# Activate conda environment
conda activate hippo

# Ensure Qdrant is running (via Docker or remote)
# Default: http://100.113.229.40:6333

# Verify services are running
curl http://100.113.229.40:6333/readyz
```

### 2. Environment Variables

The test scripts use the default Qdrant URL `http://100.113.229.40:6333`. No environment variables needed.

---

## Quick Test Commands

### Run All New Feature Tests

```bash
python -m pytest tests/test_bitemporal.py tests/test_context_assembly.py tests/test_custom_schema.py tests/test_agentic_classifier.py tests/test_benchmarks.py -v
```

### Run Integration Test (All Features)

```bash
TOKENIZERS_PARALLELISM=false python scripts/test_all_features.py
```

This script:
- Tests all features end-to-end
- Uses `http://100.113.229.40:6333` for Qdrant
- Uses `http://100.113.229.40:8000` for API
- Cleans up test collections on success

### Run Individual Test Suites

```bash
# Bi-temporal facts (15 tests)
python -m pytest tests/test_bitemporal.py -v

# Context assembly (17 tests)
python -m pytest tests/test_context_assembly.py -v

# Custom schema (28 tests)
python -m pytest tests/test_custom_schema.py -v

# Agentic classifier (28 tests)
python -m pytest tests/test_agentic_classifier.py -v

# Benchmarks (14 tests)
python -m pytest tests/test_benchmarks.py -v
```

---

## Feature 1: Bi-Temporal Fact Tracking

### What It Does
Tracks facts with two time dimensions:
- **Valid Time**: When the fact was true in the real world
- **System Time**: When the fact was recorded in the system

### Test File
`tests/test_bitemporal.py`

### Manual API Test

```bash
# Store a bi-temporal fact
curl -X POST http://100.113.229.40:8000/v1/bitemporal/facts:store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Google",
    "user_id": "test_user_123",
    "entity_id": "alice",
    "property_name": "employer"
  }'

# Query facts
curl -X POST http://100.113.229.40:8000/v1/bitemporal/facts:query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_123"}'

# Revise a fact
curl -X POST http://100.113.229.40:8000/v1/bitemporal/facts:revise \
  -H "Content-Type: application/json" \
  -d '{
    "original_fact_id": "<fact_id>",
    "new_text": "Alice works at Microsoft",
    "user_id": "test_user_123"
  }'
```

---

## Feature 2: Automated Context Assembly

### What It Does
Automatically assembles relevant context from memories for LLM prompts with:
- Token budget management
- Relevance scoring
- Deduplication
- Citation tracking

### Test File
`tests/test_context_assembly.py`

### Manual API Test

```bash
# Assemble context
curl -X POST http://100.113.229.40:8000/v1/context:assemble \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "What does Alice like?",
    "max_tokens": 500
  }'

# Get context as plain text
curl -X POST http://100.113.229.40:8000/v1/context:assemble/text \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "Tell me about Alice"
  }'
```

---

## Feature 3: Custom Schema Support

### What It Does
Allows defining custom entity types and relationship types with validation:
- Entity type definitions with attributes
- Relationship type definitions with constraints
- Schema validation for entities and relationships

### Test File
`tests/test_custom_schema.py`

### Python Code Test

```python
from hippocampai.schema.models import (
    AttributeDefinition,
    EntityTypeDefinition,
    SchemaDefinition,
)
from hippocampai.schema.validator import SchemaValidator
from hippocampai.schema.registry import SchemaRegistry

# Create a custom schema
person_type = EntityTypeDefinition(
    name="Person",
    description="A person entity",
    attributes=[
        AttributeDefinition(name="name", type="string", required=True),
        AttributeDefinition(name="age", type="integer", required=False),
    ]
)

schema = SchemaDefinition(
    name="my_schema",
    version="1.0",
    entity_types=[person_type],
    relationship_types=[]
)

# Validate an entity
validator = SchemaValidator(schema)
result = validator.validate_entity(
    entity_type="Person",
    attributes={"name": "Alice", "age": 30}
)
print(f"Valid: {result.valid}")
```

---

## Feature 4: Agentic Memory Classification

### What It Does
Uses LLM-based multi-step reasoning for accurate memory type classification:
- Confidence scoring
- Alternative type suggestions
- Reasoning explanations
- Caching for consistency

### Test File
`tests/test_agentic_classifier.py`

### Python Code Test

```python
from hippocampai.utils.agentic_classifier import (
    classify_memory_agentic,
    classify_memory_agentic_with_confidence,
    classify_memory_agentic_with_details,
)

# Simple classification
memory_type = classify_memory_agentic("My name is Alex")
print(f"Type: {memory_type}")

# With confidence
memory_type, confidence = classify_memory_agentic_with_confidence("I love pizza")
print(f"Type: {memory_type}, Confidence: {confidence}")

# Full details
result = classify_memory_agentic_with_details("I want to learn Python")
print(f"Type: {result.memory_type}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

---

## Feature 5: Benchmark Suite

### What It Does
Provides tools for benchmarking HippocampAI performance:
- Synthetic data generation
- Benchmark runner with metrics
- Results export (JSON, Markdown)

### Test File
`tests/test_benchmarks.py`

### Python Code Test

```python
from bench.data_generator import generate_memories, generate_queries
from bench.runner import run_benchmark

# Generate test data
memories = list(generate_memories(count=100, num_users=2))
queries = list(generate_queries(count=10))

# Run a simple benchmark
import time

def sample_op():
    time.sleep(0.001)
    return True

result = run_benchmark("test", sample_op, iterations=10)
print(f"P50 latency: {result.latency_p50_ms:.2f}ms")
print(f"Throughput: {result.ops_per_second:.2f} ops/sec")
```

---

## UI Testing

Access the web UI at `http://localhost:3000` (or your configured frontend URL).

### New Feature Pages

| Feature | URL Path | Description |
|---------|----------|-------------|
| Bi-Temporal Facts | `/bitemporal` | View, create, revise, and retract facts with history |
| Context Assembly | `/context` | Assemble context with visual feedback and settings |
| Custom Schema | `/schema` | View schema definitions and validate entities |
| Memory Classifier | `/classifier` | Test memory classification with confidence scores |

### Navigation

All new features are accessible from the "Analyze" dropdown menu in the navigation bar.

---

## Code Quality Checks

### Ruff (Linting)

```bash
ruff check src/hippocampai/models/bitemporal.py \
           src/hippocampai/storage/bitemporal_store.py \
           src/hippocampai/schema/ \
           src/hippocampai/context/ \
           src/hippocampai/utils/agentic_classifier.py \
           bench/
```

### Mypy (Type Checking)

```bash
mypy src/hippocampai/models/bitemporal.py \
     src/hippocampai/storage/bitemporal_store.py \
     src/hippocampai/schema/ \
     src/hippocampai/context/ \
     src/hippocampai/utils/agentic_classifier.py \
     bench/ \
     --ignore-missing-imports
```

---

## Troubleshooting

### Connection Refused Error

```
httpx.ConnectError: [Errno 61] Connection refused
```

**Solution**: Ensure Qdrant is running at `http://100.113.229.40:6333`:
```bash
curl http://100.113.229.40:6333/readyz
```

### Tests Timeout

**Solution**: Run with increased timeout:
```bash
pytest tests/test_context_assembly.py -v --timeout=300
```

### Import Errors

**Solution**: Ensure you're in the correct conda environment:
```bash
conda activate hippo
pip install -e .
```

### Rate Limiting (Agentic Classifier)

The agentic classifier now handles rate limits automatically with:
- 5 retry attempts
- Exponential backoff up to 60 seconds
- Reduced token usage in prompts
