# Production Resilience & Observability

HippocampAI provides production-grade resilience and observability features to ensure reliable operation in real-world deployments.

## Table of Contents

- [Automatic Retry Logic](#automatic-retry-logic)
- [Structured JSON Logging](#structured-json-logging)
- [Request ID Tracking](#request-id-tracking)
- [Production Best Practices](#production-best-practices)
- [Configuration](#configuration)
- [Monitoring](#monitoring)

## Automatic Retry Logic

All external service calls (Qdrant, LLM providers) automatically retry on transient failures using exponential backoff.

### Features

- **Automatic retries** on connection errors, timeouts, and network issues
- **Exponential backoff** with configurable wait times
- **Structured logging** of retry attempts
- **Zero configuration** required - works out of the box

### Retry Configuration

#### Qdrant Operations

All Qdrant vector store operations have automatic retry:

```python
from hippocampai.memory.client import MemoryClient

client = MemoryClient(qdrant_url="http://localhost:6333")

# These operations automatically retry on transient failures:
client.store(user_id="user123", content="...", memory_type="facts")  # Retries on failure
results = client.search(user_id="user123", query="...")              # Retries on failure
client.delete(user_id="user123", memory_ids=["id1", "id2"])          # Retries on failure
```

**Default Configuration:**

- Max attempts: 3
- Min wait: 1 second
- Max wait: 5 seconds
- Retry on: `ConnectionError`, `TimeoutError`, `OSError`

**Operations with retry:**

- `upsert()` - Insert/update vectors
- `search()` - Vector similarity search
- `scroll()` - Paginated retrieval
- `delete()` - Delete vectors
- `get()` - Retrieve by ID
- `update()` - Update metadata

#### LLM Operations

All LLM provider calls have automatic retry:

```python
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM

# Ollama
llm = OllamaLLM(model="qwen2.5:7b-instruct")
response = llm.generate(prompt="...", system="...")  # Retries on failure
response = llm.chat(messages=[...])                   # Retries on failure

# OpenAI
llm = OpenAILLM(api_key="...", model="gpt-4o-mini")
response = llm.generate(prompt="...", system="...")  # Retries on failure
response = llm.chat(messages=[...])                   # Retries on failure
```

**Default Configuration:**

- Max attempts: 3
- Min wait: 2 seconds
- Max wait: 10 seconds
- Retry on: `ConnectionError`, `TimeoutError`, `OSError`

### Retry Behavior

The retry logic uses **exponential backoff** with the following pattern:

1. **First attempt**: Immediate execution
2. **First retry**: Wait 1-2 seconds (depending on operation type)
3. **Second retry**: Wait 2-4 seconds (exponential increase)
4. **Third retry**: Wait 4-8 seconds (capped by max_wait)
5. **Final failure**: Raise exception after max attempts exhausted

**Example retry sequence:**

```
[Attempt 1] → Failure (ConnectionError)
  ↓ Wait 2 seconds
[Attempt 2] → Failure (TimeoutError)
  ↓ Wait 4 seconds (exponential backoff)
[Attempt 3] → Success ✓
```

### Custom Retry Configuration

You can create custom retry decorators for specific use cases:

```python
from hippocampai.utils.retry import retry_on_exception

@retry_on_exception(
    exception_types=(ConnectionError, TimeoutError),
    max_attempts=5,
    min_wait=1,
    max_wait=30,
)
def my_custom_operation():
    # Your code here
    pass
```

## Structured JSON Logging

HippocampAI uses structured JSON logging for production observability, making it easy to integrate with log aggregation tools.

### Setup

```python
from hippocampai.utils.structured_logging import setup_structured_logging
import logging

# Enable JSON-formatted structured logging
setup_structured_logging(
    level=logging.INFO,
    format_json=True,
    include_timestamp=True
)
```

### Features

- **JSON formatted logs** - Easy parsing and indexing
- **Rich metadata** - Timestamps, levels, modules, functions, line numbers
- **Process/thread info** - Process ID and thread ID for debugging
- **Exception tracking** - Full exception details in structured format
- **Request ID tracking** - Trace requests across operations

### Log Format

Each log entry includes:

```json
{
  "timestamp": "2026-02-11T10:30:45.123456+00:00",
  "level": "INFO",
  "logger": "hippocampai.memory.client",
  "module": "client",
  "function": "store",
  "line": 145,
  "message": "Memory stored successfully",
  "process_id": 12345,
  "thread_id": 67890,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "memory_type": "facts"
}
```

### Usage

```python
from hippocampai.utils.structured_logging import get_logger

logger = get_logger(__name__)

# Basic logging
logger.info("Operation completed")

# Logging with context
logger.info(
    "Memory stored",
    extra={
        "user_id": "user123",
        "memory_type": "facts",
        "tags": ["programming", "python"]
    }
)

# Error logging with exception
try:
    # ... operation ...
    pass
except Exception as e:
    logger.error("Operation failed", exc_info=True, extra={"user_id": "user123"})
```

## Request ID Tracking

Request IDs enable distributed tracing across operations, making it easy to track a single request through multiple service calls.

### Manual Request ID

```python
from hippocampai.utils.structured_logging import set_request_id, get_logger

logger = get_logger(__name__)

# Set request ID manually
request_id = set_request_id("custom-request-id")

# All logs will now include this request_id
logger.info("Processing request")  # Includes request_id

# Clear when done
clear_request_id()
```

### Request Context Manager

The recommended approach is to use `RequestContext`:

```python
from hippocampai.utils.structured_logging import RequestContext, get_logger

logger = get_logger(__name__)

# Automatic request tracking with context manager
with RequestContext(operation="user_query") as ctx:
    logger.info("Starting operation")  # Includes ctx.request_id

    # All operations within this context share the same request_id
    client.store(user_id="user123", content="...")
    results = client.search(user_id="user123", query="...")

    logger.info("Operation completed")  # Includes ctx.request_id

# Request ID automatically cleared after context exit
```

### Benefits

- **Distributed tracing** - Track requests across multiple services
- **Debugging** - Find all logs related to a specific request
- **Performance monitoring** - Measure end-to-end request latency
- **Error tracking** - Identify which requests failed and why

### Example Log Aggregation Query

With request IDs, you can easily query logs:

```bash
# ElasticSearch
GET /logs/_search
{
  "query": {
    "match": { "request_id": "550e8400-e29b-41d4-a716-446655440000" }
  }
}

# Splunk
index=hippocampai request_id="550e8400-e29b-41d4-a716-446655440000"

# CloudWatch Logs Insights
fields @timestamp, message, level
| filter request_id = "550e8400-e29b-41d4-a716-446655440000"
| sort @timestamp desc
```

## Production Best Practices

### 1. Always Use RequestContext

```python
from hippocampai.utils.structured_logging import RequestContext

# Good: Wrap operations in RequestContext
with RequestContext(operation="api_endpoint") as ctx:
    result = client.search(user_id=user_id, query=query)
    return result

# Bad: No request tracking
result = client.search(user_id=user_id, query=query)
return result
```

### 2. Log Important Events

```python
from hippocampai.utils.structured_logging import get_logger

logger = get_logger(__name__)

# Log key operations with context
logger.info(
    "Memory stored",
    extra={
        "user_id": user_id,
        "memory_type": memory_type,
        "memory_id": memory_id
    }
)
```

### 3. Handle Exceptions Properly

```python
try:
    result = client.search(user_id=user_id, query=query)
except Exception as e:
    logger.error(
        "Search operation failed",
        exc_info=True,  # Include full traceback
        extra={
            "user_id": user_id,
            "query": query,
            "error_type": type(e).__name__
        }
    )
    raise
```

### 4. Monitor Retry Attempts

The retry decorators automatically log warnings before each retry:

```json
{
  "level": "WARNING",
  "message": "Retrying hippocampai.vector.qdrant_store.QdrantStore.search in 2 seconds",
  "attempt_number": 2,
  "exception": "ConnectionError: Connection refused"
}
```

Monitor these warnings to detect:

- Frequent retries (may indicate infrastructure issues)
- Services nearing failure threshold
- Network instability

## Configuration

### Environment Variables

You can configure logging behavior via environment variables:

```bash
# Log level
export HIPPOCAMPAI_LOG_LEVEL=INFO

# JSON formatting
export HIPPOCAMPAI_LOG_JSON=true

# Log file output (in addition to console)
export HIPPOCAMPAI_LOG_FILE=/var/log/hippocampai.log
```

### Programmatic Configuration

```python
from hippocampai.utils.structured_logging import setup_structured_logging
import logging

# Development: Human-readable format
setup_structured_logging(
    level=logging.DEBUG,
    format_json=False,
    include_timestamp=True
)

# Production: JSON format for log aggregation
setup_structured_logging(
    level=logging.INFO,
    format_json=True,
    include_timestamp=True
)
```

## Monitoring

### Key Metrics to Track

1. **Retry Rate**
   - Count of retry attempts
   - Success rate after retries
   - Services with high retry rates

2. **Request Duration**
   - P50, P95, P99 latencies
   - Slow requests (> 1 second)
   - Operations timing out

3. **Error Rate**
   - Errors by type
   - Errors by operation
   - Errors by user

4. **Request Volume**
   - Requests per second
   - Peak load times
   - Traffic patterns

### Sample Queries

**Find requests with retries:**

```python
# Query your log aggregation tool
{
  "query": "Retrying",
  "level": "WARNING",
  "time_range": "last_1h"
}
```

**Find slow requests:**

```python
{
  "filter": {
    "duration_seconds": { "gte": 1.0 }
  },
  "time_range": "last_1h"
}
```

**Find failed requests:**

```python
{
  "filter": {
    "status": "error"
  },
  "time_range": "last_1h",
  "group_by": "operation"
}
```

### Alerting Recommendations

1. **High Retry Rate**
   - Alert if retry rate > 10% over 5 minutes
   - Indicates infrastructure issues

2. **Request Failures**
   - Alert if error rate > 1% over 5 minutes
   - Indicates service degradation

3. **Slow Requests**
   - Alert if P95 latency > 2 seconds
   - Indicates performance issues

4. **Service Unavailability**
   - Alert if any service is unreachable
   - Immediate action required

## See Also

- [examples/example_resilience.py](../examples/example_resilience.py) - Working code examples
- [src/hippocampai/utils/retry.py](../src/hippocampai/utils/retry.py) - Retry implementation
- [src/hippocampai/utils/structured_logging.py](../src/hippocampai/utils/structured_logging.py) - Logging implementation
