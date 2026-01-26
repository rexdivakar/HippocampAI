# Bi-temporal Fact Tracking

HippocampAI supports bi-temporal fact tracking, enabling you to track not just *what* you know, but *when* you knew it and *when* it was true.

## Overview

Bi-temporal modeling tracks two independent time dimensions:

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Event Time** | When the fact occurred or was stated | "Alice said she works at Google on Jan 15" |
| **Valid Time** | The interval during which the fact is/was true | "Alice worked at Google from 2020-2023" |
| **System Time** | When HippocampAI recorded the fact (automatic) | "We learned this on Jan 15 at 3:00 PM" |

This enables powerful queries like:
- "As of last month, what did we believe about Alice's employer?"
- "What was valid during Q1 2024?"
- "Show me the history of changes to this fact"

## Quick Start

### Store a Bi-temporal Fact

```python
from hippocampai import MemoryClient
from datetime import datetime, timezone

client = MemoryClient()

# Store a fact about employment
fact = client.store_bitemporal_fact(
    text="Alice works at Google",
    user_id="alice",
    entity_id="alice",
    property_name="employer",
    valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
)

print(f"Stored fact: {fact.id}")
print(f"Valid from: {fact.valid_from}")
print(f"System time: {fact.system_time}")
```

### Revise a Fact (Without Deleting History)

```python
# Alice changes jobs - revise the fact
new_fact = client.revise_bitemporal_fact(
    original_fact_id=fact.id,
    new_text="Alice works at Microsoft",
    user_id="alice",
    reason="job_change",
    new_valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
)

print(f"New fact: {new_fact.text}")
print(f"Supersedes: {new_fact.supersedes}")
```

### Query Facts

```python
# Get currently valid facts
result = client.query_bitemporal_facts(
    user_id="alice",
    entity_id="alice",
    property_name="employer",
)

for fact in result.facts:
    print(f"{fact.text} (valid: {fact.valid_from} - {fact.valid_to})")
```

## API Reference

### `store_bitemporal_fact()`

Store a fact with bi-temporal tracking.

```python
fact = client.store_bitemporal_fact(
    text="The fact content",
    user_id="user123",
    entity_id="entity_id",        # Optional: entity this fact is about
    property_name="property",      # Optional: property name (e.g., "employer")
    event_time=datetime.now(),     # Optional: when fact occurred
    valid_from=datetime.now(),     # Optional: validity start
    valid_to=None,                 # Optional: validity end (None = still valid)
    confidence=0.9,                # Optional: confidence score
    source="conversation",         # Optional: source of the fact
    metadata={"key": "value"},     # Optional: additional metadata
)
```

### `revise_bitemporal_fact()`

Create a revision that supersedes an existing fact.

```python
new_fact = client.revise_bitemporal_fact(
    original_fact_id="fact-id",
    new_text="Updated fact content",
    user_id="user123",
    new_valid_from=datetime.now(),  # Optional
    new_valid_to=None,              # Optional
    reason="correction",            # Reason for revision
    confidence=0.9,                 # Optional
    metadata={},                    # Optional
)
```

### `retract_bitemporal_fact()`

Mark a fact as invalid without deleting it.

```python
success = client.retract_bitemporal_fact(
    fact_id="fact-id",
    reason="incorrect_information",
)
```

### `query_bitemporal_facts()`

Query facts with temporal filters.

```python
result = client.query_bitemporal_facts(
    user_id="user123",
    query="semantic search query",   # Optional: semantic search
    entity_id="entity_id",           # Optional: filter by entity
    property_name="property",        # Optional: filter by property
    as_of_system_time=datetime(...), # Optional: "as-of" query
    valid_at=datetime(...),          # Optional: point-in-time validity
    valid_from=datetime(...),        # Optional: range start
    valid_to=datetime(...),          # Optional: range end
    include_superseded=False,        # Include superseded facts
    include_retracted=False,         # Include retracted facts
    limit=100,                       # Max results
)

for fact in result.facts:
    print(fact.text)
```

### `get_bitemporal_fact_history()`

Get all versions of a logical fact.

```python
history = client.get_bitemporal_fact_history(fact_id="logical-fact-id")

for version in history:
    print(f"{version.system_time}: {version.text}")
    print(f"  Status: {version.status}")
    print(f"  Valid: {version.valid_from} - {version.valid_to}")
```

### `get_latest_valid_fact()`

Get the most recent valid fact for an entity/property.

```python
fact = client.get_latest_valid_fact(
    user_id="user123",
    entity_id="alice",
    property_name="employer",
)

if fact:
    print(f"Current employer: {fact.text}")
```

## REST API

### Store Fact

```bash
curl -X POST http://localhost:8000/v1/bitemporal/facts:store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Google",
    "user_id": "alice",
    "entity_id": "alice",
    "property_name": "employer"
  }'
```

### Revise Fact

```bash
curl -X POST http://localhost:8000/v1/bitemporal/facts:revise \
  -H "Content-Type: application/json" \
  -d '{
    "original_fact_id": "fact-id-here",
    "new_text": "Alice works at Microsoft",
    "user_id": "alice",
    "reason": "job_change"
  }'
```

### Query Facts

```bash
curl -X POST http://localhost:8000/v1/bitemporal/facts:query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "entity_id": "alice",
    "property_name": "employer",
    "include_superseded": true
  }'
```

### Get Fact History

```bash
curl -X POST http://localhost:8000/v1/bitemporal/facts:history \
  -H "Content-Type: application/json" \
  -d '{
    "fact_id": "logical-fact-id"
  }'
```

## Use Cases

### 1. Employment History

Track job changes over time:

```python
# Initial employment
fact = client.store_bitemporal_fact(
    text="Alice works at Google as Software Engineer",
    user_id="alice",
    entity_id="alice",
    property_name="employment",
    valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
)

# Promotion
client.revise_bitemporal_fact(
    original_fact_id=fact.id,
    new_text="Alice works at Google as Senior Engineer",
    user_id="alice",
    reason="promotion",
    new_valid_from=datetime(2022, 6, 1, tzinfo=timezone.utc),
)

# Job change
client.revise_bitemporal_fact(
    original_fact_id=fact.id,
    new_text="Alice works at Microsoft as Principal Engineer",
    user_id="alice",
    reason="job_change",
    new_valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
)

# Query full history
history = client.get_bitemporal_fact_history(fact.fact_id)
```

### 2. Preference Evolution

Track how preferences change:

```python
# Initial preference
pref = client.store_bitemporal_fact(
    text="User prefers dark mode",
    user_id="user123",
    property_name="ui_preference",
)

# Preference changes
client.revise_bitemporal_fact(
    original_fact_id=pref.id,
    new_text="User prefers light mode",
    user_id="user123",
    reason="preference_change",
)
```

### 3. Audit Trail

Query what was known at a specific point in time:

```python
# What did we believe about Alice on Jan 1, 2024?
result = client.query_bitemporal_facts(
    user_id="alice",
    as_of_system_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
)
```

## Data Model

### BiTemporalFact

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique version ID |
| `fact_id` | str | Logical fact ID (same across revisions) |
| `text` | str | Fact content |
| `user_id` | str | Owner |
| `entity_id` | str? | Entity this fact is about |
| `property_name` | str? | Property name |
| `event_time` | datetime | When fact occurred |
| `valid_from` | datetime | Validity start |
| `valid_to` | datetime? | Validity end (None = still valid) |
| `system_time` | datetime | When recorded (immutable) |
| `status` | FactStatus | active/superseded/retracted |
| `superseded_by` | str? | ID of superseding fact |
| `supersedes` | str? | ID of superseded fact |
| `confidence` | float | Confidence score (0-1) |
| `source` | str | Source of the fact |
| `metadata` | dict | Additional metadata |

### FactStatus

- `active`: Currently valid
- `superseded`: Replaced by newer fact
- `retracted`: Explicitly invalidated
- `expired`: valid_to has passed

## Best Practices

1. **Use entity_id and property_name** for structured facts to enable efficient querying
2. **Always provide a reason** when revising facts for audit purposes
3. **Use valid_from/valid_to** to model real-world validity, not just when you learned it
4. **Query with include_superseded=True** when you need full history
5. **Use as_of_system_time** for audit queries ("what did we know then?")

## Troubleshooting

### Fact not found when revising

Ensure you're using the version ID (`fact.id`), not the logical fact ID (`fact.fact_id`).

### Query returns no results

Check that:
- The user_id matches
- The fact hasn't been retracted (or use `include_retracted=True`)
- The temporal filters match the fact's validity period

### Multiple active facts for same property

This is valid - bi-temporal allows overlapping validity periods. Use `get_latest_valid_fact()` to get the most recent one.
