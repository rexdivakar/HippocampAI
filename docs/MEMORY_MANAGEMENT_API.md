# Memory Management API Documentation

Comprehensive API documentation for HippocampAI's Memory Management features.

## Overview

The Memory Management API provides a complete suite of operations for managing memories with:

- **CRUD Operations**: Create, Read, Update, Delete memories
- **Batch Operations**: Perform operations on multiple memories at once
- **Hybrid Search**: Vector + keyword search with customizable weights
- **Automatic Extraction**: Extract memories from conversation logs
- **Deduplication**: Automatically detect and handle duplicate memories
- **Consolidation**: Merge similar memories intelligently
- **TTL Support**: Time-to-live for automatic memory expiration
- **Redis Caching**: Fast retrieval with Redis backend

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. User isolation is enforced via `user_id` parameters.

---

## Endpoints

### Health & Status

#### `GET /healthz`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "service": "hippocampai",
  "version": "0.2.5"
}
```

#### `GET /stats`

Get Redis cache statistics.

**Response:**

```json
{
  "total_keys": 150,
  "memory_keys": 100,
  "user_indices": 25,
  "tag_indices": 25
}
```

---

## CRUD Operations

### Create Memory

#### `POST /v1/memories`

Create a new memory with optional deduplication check.

**Request Body:**

```json
{
  "text": "Paris is the capital of France",
  "user_id": "user123",
  "session_id": "session456",
  "type": "fact",
  "importance": 8.0,
  "tags": ["geography", "europe"],
  "ttl_days": 30,
  "metadata": {
    "source": "wikipedia"
  },
  "check_duplicate": true
}
```

**Response:** `201 Created`

```json
{
  "id": "mem_abc123",
  "text": "Paris is the capital of France",
  "user_id": "user123",
  "session_id": "session456",
  "type": "fact",
  "importance": 8.0,
  "confidence": 0.9,
  "tags": ["geography", "europe"],
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z",
  "expires_at": "2025-02-14T10:30:00Z",
  "access_count": 0,
  "text_length": 31,
  "token_count": 7,
  "metadata": {
    "source": "wikipedia"
  }
}
```

**Memory Types:**

- `fact`: Factual information
- `preference`: User preferences
- `goal`: User goals
- `habit`: User habits
- `event`: Events
- `context`: Contextual information

---

### Get Memory

#### `GET /v1/memories/{memory_id}`

Retrieve a specific memory by ID.

**Response:** `200 OK`

```json
{
  "id": "mem_abc123",
  "text": "Paris is the capital of France",
  ...
}
```

**Error:** `404 Not Found` if memory doesn't exist

---

### Update Memory

#### `PATCH /v1/memories/{memory_id}`

Update an existing memory.

**Request Body:**

```json
{
  "memory_id": "mem_abc123",
  "text": "Paris is the capital and largest city of France",
  "importance": 9.0,
  "tags": ["geography", "europe", "capitals"],
  "metadata": {
    "updated_by": "user"
  },
  "expires_at": "2025-12-31T23:59:59Z"
}
```

**Response:** `200 OK`

```json
{
  "id": "mem_abc123",
  "text": "Paris is the capital and largest city of France",
  "importance": 9.0,
  "updated_at": "2025-01-15T11:00:00Z",
  ...
}
```

---

### Delete Memory

#### `DELETE /v1/memories/{memory_id}`

Delete a memory. Optionally provide `user_id` for authorization.

**Query Parameters:**

- `user_id` (optional): User ID for authorization check

**Response:** `200 OK`

```json
{
  "success": true,
  "memory_id": "mem_abc123"
}
```

**Error:** `404 Not Found` if memory doesn't exist or unauthorized

---

### Query Memories

#### `POST /v1/memories/query`

Query memories with filters.

**Request Body:**

```json
{
  "user_id": "user123",
  "filters": {
    "type": "fact",
    "tags": ["geography"]
  },
  "limit": 100
}
```

**Response:** `200 OK`

```json
[
  {
    "id": "mem_abc123",
    "text": "Paris is the capital of France",
    ...
  },
  ...
]
```

---

## Batch Operations

### Batch Create

#### `POST /v1/memories/batch`

Create multiple memories at once.

**Request Body:**

```json
{
  "memories": [
    {
      "text": "Memory 1",
      "user_id": "user123",
      "type": "fact"
    },
    {
      "text": "Memory 2",
      "user_id": "user123",
      "type": "preference"
    }
  ],
  "check_duplicates": true
}
```

**Response:** `201 Created`

```json
[
  {
    "id": "mem_1",
    "text": "Memory 1",
    ...
  },
  {
    "id": "mem_2",
    "text": "Memory 2",
    ...
  }
]
```

---

### Batch Update

#### `PATCH /v1/memories/batch`

Update multiple memories at once.

**Request Body:**

```json
{
  "updates": [
    {
      "memory_id": "mem_1",
      "importance": 9.0
    },
    {
      "memory_id": "mem_2",
      "tags": ["updated"]
    }
  ]
}
```

**Response:** `200 OK`

```json
[
  {
    "id": "mem_1",
    "importance": 9.0,
    ...
  },
  ...
]
```

---

### Batch Delete

#### `DELETE /v1/memories/batch`

Delete multiple memories at once.

**Request Body:**

```json
{
  "memory_ids": ["mem_1", "mem_2", "mem_3"],
  "user_id": "user123"
}
```

**Response:** `200 OK`

```json
{
  "success": true,
  "deleted_count": 3,
  "total": 3,
  "results": {
    "mem_1": true,
    "mem_2": true,
    "mem_3": true
  }
}
```

---

## Retrieval & Search

### Recall Memories (Hybrid Search)

#### `POST /v1/memories/recall`

Perform hybrid search with customizable scoring weights.

**Request Body:**

```json
{
  "query": "programming languages",
  "user_id": "user123",
  "session_id": "session456",
  "k": 5,
  "filters": {
    "type": "fact"
  },
  "custom_weights": {
    "sim": 0.5,
    "rerank": 0.2,
    "recency": 0.2,
    "importance": 0.1
  }
}
```

**Scoring Weights:**

- `sim`: Semantic similarity (vector search)
- `rerank`: Cross-encoder reranking score
- `recency`: Time decay based on creation date
- `importance`: Memory importance score

**Response:** `200 OK`

```json
[
  {
    "memory": {
      "id": "mem_xyz",
      "text": "Python is a programming language",
      ...
    },
    "score": 0.892,
    "breakdown": {
      "sim": 0.95,
      "rerank": 0.88,
      "recency": 0.75,
      "importance": 0.80,
      "final": 0.892
    }
  },
  ...
]
```

---

## Extraction

### Extract from Conversation

#### `POST /v1/memories/extract`

Automatically extract memories from conversation logs using LLM.

**Request Body:**

```json
{
  "conversation": "User: I'm learning Python for data science.\nAssistant: That's great! Python is excellent for data science.",
  "user_id": "user123",
  "session_id": "session456"
}
```

**Response:** `200 OK`

```json
[
  {
    "id": "mem_ext1",
    "text": "User is learning Python for data science",
    "type": "goal",
    "importance": 8.0,
    "metadata": {
      "extracted_from": "conversation"
    },
    ...
  }
]
```

**Note:** Requires LLM to be configured.

---

## Deduplication

### Deduplicate User Memories

#### `POST /v1/memories/deduplicate`

Detect and optionally remove duplicate memories.

**Request Body:**

```json
{
  "user_id": "user123",
  "dry_run": true
}
```

**Response:** `200 OK`

```json
{
  "user_id": "user123",
  "total_memories": 150,
  "duplicates_found": 12,
  "memories_removed": 0,
  "dry_run": true,
  "details": [
    {
      "memory_id": "mem_123",
      "action": "skip",
      "duplicates": ["mem_456"]
    },
    ...
  ]
}
```

**Actions:**

- `skip`: Exact duplicate, should be skipped
- `update`: Similar memory, should be updated

---

## Consolidation

### Consolidate Similar Memories

#### `POST /v1/memories/consolidate`

Merge similar memories using LLM or heuristics.

**Request Body:**

```json
{
  "user_id": "user123",
  "similarity_threshold": 0.85,
  "dry_run": true
}
```

**Response:** `200 OK`

```json
{
  "user_id": "user123",
  "total_memories": 150,
  "groups_found": 5,
  "memories_consolidated": 0,
  "dry_run": true
}
```

---

## Maintenance

### Expire Memories

#### `POST /v1/memories/expire`

Remove expired memories based on TTL.

**Request Body:**

```json
{
  "user_id": "user123"
}
```

**Response:** `200 OK`

```json
{
  "success": true,
  "expired_count": 8
}
```

---

## Configuration

### Environment Variables

```bash
# Redis
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_CACHE_TTL=300

# Qdrant
QDRANT_URL=http://localhost:6333
COLLECTION_FACTS=hippocampai_facts
COLLECTION_PREFS=hippocampai_prefs

# Embeddings
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_QUANTIZED=false
EMBED_DIMENSION=384

# Scoring Weights
WEIGHT_SIM=0.55
WEIGHT_RERANK=0.20
WEIGHT_RECENCY=0.15
WEIGHT_IMPORTANCE=0.10
```

---

## Usage Examples

### Python Client

```python
import asyncio
import httpx

async def main():
    client = httpx.AsyncClient(base_url="http://localhost:8000")

    # Create memory
    response = await client.post("/v1/memories", json={
        "text": "I prefer dark mode",
        "user_id": "user123",
        "type": "preference",
        "importance": 7.0
    })
    memory = response.json()
    print(f"Created memory: {memory['id']}")

    # Recall memories
    response = await client.post("/v1/memories/recall", json={
        "query": "user preferences",
        "user_id": "user123",
        "k": 5
    })
    results = response.json()
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['memory']['text']}")

asyncio.run(main())
```

### cURL

```bash
# Create memory
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paris is the capital of France",
    "user_id": "user123",
    "type": "fact",
    "importance": 8.0
  }'

# Recall memories
curl -X POST http://localhost:8000/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "capital of France",
    "user_id": "user123",
    "k": 5
  }'

# Batch create
curl -X POST http://localhost:8000/v1/memories/batch \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      {"text": "Memory 1", "user_id": "user123", "type": "fact"},
      {"text": "Memory 2", "user_id": "user123", "type": "fact"}
    ]
  }'
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:

```json
{
  "detail": "Error message here"
}
```

---

## Performance Considerations

1. **Redis Caching**: Frequently accessed memories are cached in Redis with configurable TTL
2. **Batch Operations**: Use batch endpoints for multiple operations to reduce overhead
3. **BM25 Index**: Rebuild periodically for optimal keyword search performance
4. **Custom Weights**: Adjust scoring weights based on your use case
5. **Filters**: Use filters to narrow search scope and improve performance

---

## Best Practices

1. **Use Deduplication**: Enable `check_duplicate` when creating memories to avoid duplicates
2. **Set TTL**: Use `ttl_days` for temporary memories
3. **Tag Effectively**: Use meaningful tags for better filtering
4. **Batch Operations**: Prefer batch endpoints for multiple operations
5. **Custom Weights**: Adjust scoring weights based on your specific needs
6. **Regular Maintenance**: Run deduplication and consolidation periodically
7. **Monitor Cache**: Check `/stats` endpoint to monitor Redis cache performance

---

## Running the Server

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:latest

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Run server
python -m hippocampai.api.async_app
```

### Production

```bash
# With uvicorn
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose up -d
```

---

## API Changelog

### Version 0.2.5 (Current)

- Added async/await support
- Integrated Redis for caching
- Added batch operations
- Added deduplication service
- Added consolidation service
- Enhanced hybrid search with customizable weights
- Added TTL support
- Improved error handling

### Version 0.1.5

- Initial release
- Basic CRUD operations
- Hybrid retrieval
- Conversation extraction

---

## Support

For issues or questions:

- GitHub: <https://github.com/anthropics/hippocampai>
- Documentation: <https://docs.hippocampai.io>
