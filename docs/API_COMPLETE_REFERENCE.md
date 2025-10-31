# HippocampAI API Endpoints - Complete Reference

This document provides complete API endpoint details for all HippocampAI services, including the new Advanced Intelligence APIs.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Core Memory APIs](#core-memory-apis)
- [Advanced Intelligence APIs](#advanced-intelligence-apis)
- [Temporal Analytics APIs](#temporal-analytics-apis)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Base URL

```
http://localhost:8000  # Development
```

---

## Authentication

Currently, no authentication is required. Future versions will support:
- API Key authentication
- OAuth 2.0
- JWT tokens

---

## Core Memory APIs

### 1. Health Check

Check API server health status.

**Endpoint:** `GET /healthz`

**Response:**
```json
{
  "status": "ok"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/healthz
```

---

### 2. Store Memory (Remember)

Store a new memory.

**Endpoint:** `POST /v1/memories:remember`

**Request Body:**
```json
{
  "text": "I work at Google as a Senior Engineer",
  "user_id": "user_123",
  "session_id": "session_456",
  "type": "fact",
  "importance": 8.5,
  "tags": ["work", "career"],
  "ttl_days": 365
}
```

**Parameters:**
- `text` (string, required): Memory content
- `user_id` (string, required): User identifier
- `session_id` (string, optional): Session identifier
- `type` (string, optional): Memory type (default: "fact")
  - Options: `fact`, `preference`, `goal`, `habit`, `event`, `context`
- `importance` (float, optional): Importance score 0.0-10.0
- `tags` (array, optional): List of tags
- `ttl_days` (int, optional): Time-to-live in days

**Response:**
```json
{
  "id": "mem_abc123",
  "text": "I work at Google as a Senior Engineer",
  "user_id": "user_123",
  "session_id": "session_456",
  "type": "fact",
  "importance": 8.5,
  "confidence": 1.0,
  "tags": ["work", "career"],
  "created_at": "2025-01-28T10:30:00Z",
  "updated_at": "2025-01-28T10:30:00Z",
  "access_count": 0,
  "metadata": {}
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google as a Senior Engineer",
    "user_id": "user_123",
    "type": "fact",
    "importance": 8.5,
    "tags": ["work", "career"]
  }'
```

---

### 3. Retrieve Memories (Recall)

Retrieve relevant memories based on query.

**Endpoint:** `POST /v1/memories:recall`

**Request Body:**
```json
{
  "query": "Where do I work?",
  "user_id": "user_123",
  "session_id": "session_456",
  "k": 5,
  "filters": {
    "type": "fact",
    "tags": ["work"]
  }
}
```

**Parameters:**
- `query` (string, required): Search query
- `user_id` (string, required): User identifier
- `session_id` (string, optional): Session identifier
- `k` (int, optional): Number of results (default: 5)
- `filters` (object, optional): Filter criteria
  - `type`: Memory type filter
  - `tags`: Tag filter (array)
  - `importance_min`: Minimum importance
  - `importance_max`: Maximum importance

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_abc123",
      "text": "I work at Google as a Senior Engineer",
      "user_id": "user_123",
      "type": "fact",
      "importance": 8.5,
      "score": 0.95,
      "created_at": "2025-01-28T10:30:00Z"
    }
  ],
  "count": 1,
  "query": "Where do I work?"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/memories:recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where do I work?",
    "user_id": "user_123",
    "k": 5
  }'
```

---

### 4. Extract Memories from Conversation

Extract memories from conversation text.

**Endpoint:** `POST /v1/memories:extract`

**Request Body:**
```json
{
  "conversation": "User: I love playing tennis\nAssistant: That's great!",
  "user_id": "user_123",
  "session_id": "session_456"
}
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_xyz789",
      "text": "loves playing tennis",
      "type": "preference",
      "importance": 6.0
    }
  ],
  "count": 1
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/memories:extract \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": "User: I love playing tennis",
    "user_id": "user_123"
  }'
```

---

### 5. Update Memory

Update an existing memory.

**Endpoint:** `PATCH /v1/memories:update`

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "text": "I work at Google as a Staff Engineer",
  "importance": 9.0,
  "tags": ["work", "career", "promotion"],
  "metadata": {"updated_reason": "promotion"}
}
```

**cURL Example:**
```bash
curl -X PATCH http://localhost:8000/v1/memories:update \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "mem_abc123",
    "importance": 9.0,
    "tags": ["work", "career", "promotion"]
  }'
```

---

### 6. Delete Memory

Delete a specific memory.

**Endpoint:** `DELETE /v1/memories:delete`

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "deleted": true,
  "memory_id": "mem_abc123"
}
```

**cURL Example:**
```bash
curl -X DELETE http://localhost:8000/v1/memories:delete \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "mem_abc123",
    "user_id": "user_123"
  }'
```

---

## Advanced Intelligence APIs

### 1. Extract Facts

Extract structured facts from text with quality scoring.

**Endpoint:** `POST /v1/intelligence/facts:extract`

**Request Body:**
```json
{
  "text": "I work at Google as a Senior Engineer in Mountain View, California.",
  "source": "conversation",
  "user_id": "user_123",
  "with_quality": true
}
```

**Parameters:**
- `text` (string, required): Text to extract facts from
- `source` (string, optional): Source identifier (default: "api")
- `user_id` (string, optional): User identifier for context
- `with_quality` (bool, optional): Include quality metrics (default: true)

**Response:**
```json
{
  "facts": [
    {
      "fact": "works at Google",
      "category": "employment",
      "confidence": 0.92,
      "quality_score": 0.88,
      "entities": ["Google"],
      "temporal": null,
      "temporal_type": "present",
      "source": "conversation",
      "metadata": {
        "extraction_method": "pattern",
        "quality_metrics": {
          "specificity": 0.85,
          "verifiability": 0.90,
          "completeness": 0.88,
          "clarity": 0.92,
          "relevance": 0.87,
          "overall_quality": 0.88
        }
      },
      "extracted_at": "2025-01-28T10:30:00Z"
    },
    {
      "fact": "is a Senior Engineer",
      "category": "occupation",
      "confidence": 0.90,
      "quality_score": 0.85,
      "entities": ["Senior Engineer"],
      "temporal": null,
      "temporal_type": "present",
      "source": "conversation",
      "extracted_at": "2025-01-28T10:30:00Z"
    },
    {
      "fact": "located in Mountain View",
      "category": "location",
      "confidence": 0.88,
      "quality_score": 0.82,
      "entities": ["Mountain View", "California"],
      "temporal": null,
      "temporal_type": "present",
      "source": "conversation",
      "extracted_at": "2025-01-28T10:30:00Z"
    }
  ],
  "count": 3,
  "metadata": {
    "source": "conversation",
    "with_quality": true
  }
}
```

**Fact Categories:**
- `employment`, `occupation`, `location`, `education`
- `relationship`, `preference`, `skill`, `experience`
- `contact`, `event`, `goal`, `habit`, `opinion`
- `attribute`, `possession`, `other`

**Quality Metrics (0.0-1.0):**
- `specificity`: How specific the fact is
- `verifiability`: How verifiable the fact is
- `completeness`: How complete the fact is
- `clarity`: How clear the fact is
- `relevance`: How relevant the fact is
- `overall_quality`: Weighted average of all metrics

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/facts:extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google as a Senior Engineer in Mountain View.",
    "source": "api",
    "with_quality": true
  }'
```

---

### 2. Extract Entities

Extract and recognize named entities from text.

**Endpoint:** `POST /v1/intelligence/entities:extract`

**Request Body:**
```json
{
  "text": "John Smith works at Microsoft in Seattle. Email: john@microsoft.com",
  "context": {
    "source": "conversation",
    "user_id": "user_123"
  }
}
```

**Parameters:**
- `text` (string, required): Text to extract entities from
- `context` (object, optional): Context metadata

**Response:**
```json
{
  "entities": [
    {
      "text": "John Smith",
      "type": "person",
      "confidence": 0.90,
      "entity_id": "person_john_smith",
      "canonical_name": "John Smith",
      "metadata": {
        "extraction_method": "pattern"
      },
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z",
      "mention_count": 1
    },
    {
      "text": "Microsoft",
      "type": "organization",
      "confidence": 0.95,
      "entity_id": "organization_microsoft",
      "canonical_name": "Microsoft",
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z",
      "mention_count": 1
    },
    {
      "text": "Seattle",
      "type": "location",
      "confidence": 0.88,
      "entity_id": "location_seattle",
      "canonical_name": "Seattle",
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z",
      "mention_count": 1
    },
    {
      "text": "john@microsoft.com",
      "type": "email",
      "confidence": 0.98,
      "entity_id": "email_john_microsoft_com",
      "canonical_name": "john@microsoft.com",
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z",
      "mention_count": 1
    }
  ],
  "count": 4,
  "statistics": {
    "total_entities": 4,
    "by_type": {
      "person": 1,
      "organization": 1,
      "location": 1,
      "email": 1
    },
    "top_mentioned": [
      {
        "entity_id": "person_john_smith",
        "canonical_name": "John Smith",
        "type": "person",
        "mention_count": 1
      }
    ]
  }
}
```

**Entity Types:**
- **People**: `person`
- **Organizations**: `organization`
- **Places**: `location`
- **Time**: `date`, `time`
- **Contact**: `email`, `phone`, `url`
- **Technology**: `language`, `framework`, `tool`, `product`
- **Professional**: `skill`, `degree`, `certification`, `industry`
- **Other**: `event`, `money`, `topic`, `other`

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/entities:extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith works at Microsoft in Seattle. Email: john@microsoft.com"
  }'
```

---

### 3. Search Entities

Search for entities by query string.

**Endpoint:** `POST /v1/intelligence/entities:search`

**Request Body:**
```json
{
  "query": "john",
  "entity_type": "person",
  "min_mentions": 2
}
```

**Parameters:**
- `query` (string, required): Search query
- `entity_type` (string, optional): Filter by entity type
- `min_mentions` (int, optional): Minimum mention count (default: 1)

**Response:**
```json
{
  "entities": [
    {
      "entity_id": "person_john_smith",
      "canonical_name": "John Smith",
      "type": "person",
      "aliases": ["John", "J. Smith"],
      "mention_count": 5,
      "first_seen": "2025-01-20T10:00:00Z",
      "last_seen": "2025-01-28T10:30:00Z"
    }
  ],
  "count": 1
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/entities:search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "john",
    "entity_type": "person",
    "min_mentions": 2
  }'
```

---

### 4. Get Entity Profile

Get complete profile for a specific entity.

**Endpoint:** `GET /v1/intelligence/entities/{entity_id}`

**URL Parameters:**
- `entity_id` (string, required): Entity identifier

**Response:**
```json
{
  "entity": {
    "entity_id": "person_john_smith",
    "canonical_name": "John Smith",
    "type": "person",
    "aliases": ["John", "J. Smith", "John S."],
    "attributes": {},
    "relationships": [
      {
        "from_entity_id": "person_john_smith",
        "to_entity_id": "organization_microsoft",
        "relation_type": "works_at",
        "confidence": 0.95
      }
    ],
    "mentions": [
      {
        "text": "John Smith",
        "source": "User mentioned working with John...",
        "timestamp": "2025-01-28T10:30:00Z",
        "context": {}
      }
    ],
    "first_seen": "2025-01-20T10:00:00Z",
    "last_seen": "2025-01-28T10:30:00Z",
    "mention_count": 5
  },
  "timeline": [
    {
      "text": "John Smith",
      "source": "conversation...",
      "timestamp": "2025-01-20T10:00:00Z",
      "context": {}
    }
  ]
}
```

**cURL Example:**
```bash
curl http://localhost:8000/v1/intelligence/entities/person_john_smith
```

---

### 5. Analyze Relationships

Analyze relationships between entities.

**Endpoint:** `POST /v1/intelligence/relationships:analyze`

**Request Body:**
```json
{
  "text": "Alice Johnson works at Microsoft in Seattle. She graduated from Stanford University.",
  "entity_ids": ["person_alice_johnson", "organization_microsoft"]
}
```

**Parameters:**
- `text` (string, required): Text to extract relationships from
- `entity_ids` (array, optional): Specific entities to analyze

**Response:**
```json
{
  "relationships": [
    {
      "from_entity_id": "person_alice_johnson",
      "to_entity_id": "organization_microsoft",
      "relation_type": "works_at",
      "confidence": 0.95,
      "strength_score": 0.85,
      "strength_level": "very_strong",
      "co_occurrence_count": 3,
      "contexts": ["Alice Johnson works at Microsoft..."],
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z"
    },
    {
      "from_entity_id": "person_alice_johnson",
      "to_entity_id": "organization_stanford_university",
      "relation_type": "studied_at",
      "confidence": 0.92,
      "strength_score": 0.78,
      "strength_level": "strong",
      "co_occurrence_count": 1,
      "contexts": ["She graduated from Stanford University"],
      "first_seen": "2025-01-28T10:30:00Z",
      "last_seen": "2025-01-28T10:30:00Z"
    }
  ],
  "network": {
    "entities": ["person_alice_johnson", "organization_microsoft", "organization_stanford_university"],
    "relationships": [...],
    "clusters": [],
    "central_entities": [
      ["person_alice_johnson", 0.75],
      ["organization_microsoft", 0.60]
    ],
    "network_density": 0.45,
    "metadata": {
      "num_entities": 3,
      "num_relationships": 2,
      "avg_relationship_strength": 0.815
    }
  },
  "visualization_data": {
    "nodes": [
      {
        "id": "person_alice_johnson",
        "centrality": 0.75,
        "relationship_count": 2
      }
    ],
    "edges": [
      {
        "source": "person_alice_johnson",
        "target": "organization_microsoft",
        "relation_type": "works_at",
        "strength": 0.85,
        "strength_level": "very_strong",
        "co_occurrence_count": 3
      }
    ],
    "metadata": {
      "total_relationships": 2
    }
  }
}
```

**Relationship Types:**
- `works_at`, `located_in`, `founded_by`, `manages`
- `knows`, `studied_at`, `part_of`
- `similar_to`, `related_to`

**Strength Levels:**
- `very_weak` (0.0-0.2)
- `weak` (0.2-0.4)
- `moderate` (0.4-0.6)
- `strong` (0.6-0.8)
- `very_strong` (0.8-1.0)

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/relationships:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Microsoft in Seattle."
  }'
```

---

### 6. Get Entity Relationships

Get all relationships for a specific entity.

**Endpoint:** `GET /v1/intelligence/relationships/{entity_id}`

**URL Parameters:**
- `entity_id` (string, required): Entity identifier

**Query Parameters:**
- `relation_type` (string, optional): Filter by relation type
- `min_strength` (float, optional): Minimum strength score (0.0-1.0)

**Response:**
```json
{
  "entity_id": "person_alice_johnson",
  "relationships": [
    {
      "from_entity_id": "person_alice_johnson",
      "to_entity_id": "organization_microsoft",
      "relation_type": "works_at",
      "strength_score": 0.85,
      "strength_level": "very_strong",
      "co_occurrence_count": 3
    }
  ],
  "count": 1,
  "co_occurring_entities": [
    {
      "entity_id": "organization_microsoft",
      "count": 3
    },
    {
      "entity_id": "organization_stanford_university",
      "count": 1
    }
  ]
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/v1/intelligence/relationships/person_alice_johnson?min_strength=0.5"
```

---

### 7. Get Relationship Network

Get complete relationship network analysis.

**Endpoint:** `GET /v1/intelligence/relationships:network`

**Response:**
```json
{
  "network": {
    "entities": [...],
    "relationships": [...],
    "clusters": [
      {
        "cluster_id": "cluster_0",
        "entities": ["person_alice", "person_bob", "organization_acme"],
        "relationships": [...],
        "cohesion_score": 0.75,
        "metadata": {
          "size": 3,
          "edge_count": 3
        }
      }
    ],
    "central_entities": [
      ["person_alice", 0.85],
      ["organization_microsoft", 0.72]
    ],
    "network_density": 0.45
  },
  "visualization": {
    "nodes": [...],
    "edges": [...]
  },
  "statistics": {
    "total_relationships": 15,
    "total_entities": 10,
    "by_type": {
      "works_at": 5,
      "knows": 4,
      "studied_at": 3
    },
    "by_strength": {
      "very_strong": 3,
      "strong": 5,
      "moderate": 4,
      "weak": 2,
      "very_weak": 1
    },
    "avg_strength": 0.68,
    "avg_co_occurrence": 2.5
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/v1/intelligence/relationships:network
```

---

### 8. Cluster Memories

Cluster memories by semantic similarity.

**Endpoint:** `POST /v1/intelligence/clustering:analyze`

**Request Body:**
```json
{
  "memories": [
    {
      "id": "mem_1",
      "text": "Went for a morning run",
      "user_id": "user_123",
      "type": "event",
      "created_at": "2025-01-28T06:00:00Z"
    },
    {
      "id": "mem_2",
      "text": "Started a new healthy eating plan",
      "user_id": "user_123",
      "type": "habit",
      "created_at": "2025-01-28T08:00:00Z"
    }
  ],
  "max_clusters": 10,
  "hierarchical": false
}
```

**Parameters:**
- `memories` (array, required): List of memory objects
- `max_clusters` (int, optional): Maximum clusters (default: 10)
- `hierarchical` (bool, optional): Use hierarchical clustering (default: false)

**Response:**
```json
{
  "clusters": [
    {
      "topic": "health",
      "memories": [...],
      "tags": ["health", "fitness", "wellness"],
      "size": 2
    },
    {
      "topic": "work",
      "memories": [...],
      "tags": ["work", "project", "meeting"],
      "size": 3
    }
  ],
  "count": 2,
  "quality_metrics": {
    "per_cluster": {
      "cluster_0": {
        "cohesion": 0.85,
        "diversity": 0.65,
        "temporal_density": 0.72,
        "size": 2,
        "tag_count": 3
      },
      "cluster_1": {
        "cohesion": 0.78,
        "diversity": 0.70,
        "temporal_density": 0.65,
        "size": 3,
        "tag_count": 4
      }
    }
  }
}
```

**Quality Metrics (0.0-1.0):**
- `cohesion`: Intra-cluster similarity
- `diversity`: Type and tag diversity
- `temporal_density`: How close memories are in time
- `size`: Number of memories
- `tag_count`: Number of unique tags

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/clustering:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [...],
    "max_clusters": 5,
    "hierarchical": false
  }'
```

---

### 9. Optimize Cluster Count

Determine optimal number of clusters.

**Endpoint:** `POST /v1/intelligence/clustering:optimize`

**Request Body:**
```json
{
  "memories": [...],
  "min_k": 2,
  "max_k": 15
}
```

**Response:**
```json
{
  "optimal_cluster_count": 5,
  "min_k": 2,
  "max_k": 15
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/clustering:optimize \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [...],
    "min_k": 2,
    "max_k": 15
  }'
```

---

## Temporal Analytics APIs

### 1. General Temporal Analysis

Perform temporal analysis on memories.

**Endpoint:** `POST /v1/intelligence/temporal:analyze`

**Request Body:**
```json
{
  "memories": [...],
  "analysis_type": "peak_activity",
  "time_window_days": 30,
  "timezone_offset": -8
}
```

**Parameters:**
- `memories` (array, required): List of memory objects
- `analysis_type` (string, required): Type of analysis
  - Options: `peak_activity`, `patterns`, `trends`, `clusters`
- `time_window_days` (int, optional): Analysis window (default: 30)
- `timezone_offset` (int, optional): Timezone offset in hours (default: 0)

**Response for `peak_activity`:**
```json
{
  "analysis": {
    "peak_hour": 14,
    "peak_day": "wednesday",
    "peak_time_period": "afternoon",
    "hourly_distribution": {
      "14": 25,
      "15": 20,
      "10": 18
    },
    "daily_distribution": {
      "monday": 15,
      "wednesday": 30,
      "friday": 20
    },
    "time_period_distribution": {
      "afternoon": 45,
      "morning": 30,
      "evening": 25
    },
    "metadata": {
      "total_memories": 100,
      "peak_hour_count": 25,
      "peak_day_count": 30
    }
  },
  "metadata": {
    "analysis_type": "peak_activity"
  }
}
```

**Response for `patterns`:**
```json
{
  "analysis": {
    "patterns": [
      {
        "pattern_type": "daily",
        "description": "Activity around 14:00 daily",
        "frequency": 0.85,
        "confidence": 0.90,
        "occurrences": [...],
        "next_predicted": "2025-01-29T14:00:00Z",
        "regularity_score": 0.88
      },
      {
        "pattern_type": "weekly",
        "description": "Activity on wednesdays",
        "frequency": 0.95,
        "confidence": 0.85,
        "occurrences": [...],
        "next_predicted": "2025-01-29T00:00:00Z",
        "regularity_score": 0.82
      }
    ],
    "count": 2
  },
  "metadata": {
    "analysis_type": "patterns"
  }
}
```

**Response for `trends`:**
```json
{
  "analysis": {
    "activity_trend": {
      "metric": "activity",
      "time_window_days": 30,
      "direction": "increasing",
      "strength": 0.75,
      "change_rate": 0.15,
      "current_value": 3.5,
      "forecast": 3.8
    },
    "importance_trend": {
      "metric": "importance",
      "time_window_days": 30,
      "direction": "stable",
      "strength": 0.30,
      "change_rate": 0.02,
      "current_value": 6.8
    }
  },
  "metadata": {
    "analysis_type": "trends",
    "time_window_days": 30
  }
}
```

**Response for `clusters`:**
```json
{
  "analysis": {
    "clusters": [
      {
        "cluster_id": "temporal_cluster_0",
        "start_time": "2025-01-28T08:00:00Z",
        "end_time": "2025-01-28T12:00:00Z",
        "duration_hours": 4.0,
        "memories": [...],
        "density": 2.5,
        "dominant_type": "event",
        "tags": ["work", "meeting"]
      }
    ],
    "count": 5
  },
  "metadata": {
    "analysis_type": "clusters"
  }
}
```

**Time Periods:**
- `early_morning` (5am-9am)
- `late_morning` (9am-12pm)
- `afternoon` (12pm-5pm)
- `evening` (5pm-9pm)
- `night` (9pm-1am)
- `late_night` (1am-5am)

**Trend Directions:**
- `increasing`, `decreasing`, `stable`, `volatile`

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/temporal:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [...],
    "analysis_type": "peak_activity",
    "timezone_offset": -8
  }'
```

---

### 2. Peak Times Analysis

Get detailed peak activity analysis.

**Endpoint:** `POST /v1/intelligence/temporal:peak-times`

**Request Body:**
```json
{
  "memories": [...],
  "timezone_offset": -8
}
```

**Response:**
```json
{
  "peak_analysis": {
    "peak_hour": 14,
    "peak_day": "wednesday",
    "peak_time_period": "afternoon",
    "hourly_distribution": {...},
    "daily_distribution": {...},
    "time_period_distribution": {...}
  },
  "metadata": {
    "total_memories": 100,
    "timezone_offset": -8
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/intelligence/temporal:peak-times \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [...],
    "timezone_offset": -8
  }'
```

---

### 3. Intelligence Health Check

Check intelligence services health.

**Endpoint:** `GET /v1/intelligence/health`

**Response:**
```json
{
  "status": "healthy",
  "services": "Advanced Intelligence APIs",
  "version": "1.0.0"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/v1/intelligence/health
```

---

## Error Handling

All endpoints return consistent error responses:

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Example Error Response:**
```json
{
  "detail": "Entity person_unknown not found"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. Future versions will include:
- Per-user rate limits
- Per-endpoint rate limits
- Configurable rate limit tiers

---

## Best Practices

### 1. Request Optimization

- **Batch Operations**: Use batch endpoints when processing multiple items
- **Pagination**: Request only needed results (use `k` parameter)
- **Filtering**: Use filters to narrow results early

### 2. Quality Thresholds

- **Facts**: Use `quality_score >= 0.7` for reliable facts
- **Entities**: Use `confidence >= 0.8` for high-confidence entities
- **Relationships**: Use `min_strength >= 0.5` for significant relationships

### 3. Error Handling

- Always check HTTP status codes
- Implement retry logic with exponential backoff
- Log error details for debugging

### 4. Data Quality

- Provide context in extraction requests
- Use appropriate source identifiers
- Tag memories for better clustering

---

## Complete cURL Examples Collection

### Store and Retrieve Workflow

```bash
# 1. Store a memory
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google as a Senior Engineer in Mountain View",
    "user_id": "user_123",
    "type": "fact",
    "importance": 8.5,
    "tags": ["work", "career"]
  }'

# 2. Retrieve memories
curl -X POST http://localhost:8000/v1/memories:recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where do I work?",
    "user_id": "user_123",
    "k": 5
  }'

# 3. Extract facts
curl -X POST http://localhost:8000/v1/intelligence/facts:extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google as a Senior Engineer in Mountain View",
    "with_quality": true
  }'

# 4. Extract entities
curl -X POST http://localhost:8000/v1/intelligence/entities:extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google as a Senior Engineer in Mountain View"
  }'

# 5. Analyze relationships
curl -X POST http://localhost:8000/v1/intelligence/relationships:analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I work at Google in Mountain View"
  }'
```

---

## Summary

This API provides:
- **6 Core Memory Endpoints**: Store, retrieve, extract, update, delete memories
- **9 Advanced Intelligence Endpoints**: Facts, entities, relationships, clustering
- **3 Temporal Analytics Endpoints**: Patterns, trends, peak times

For implementation examples, see:
- `examples/advanced_intelligence_demo.py` - Complete Python examples
- `ADVANCED_INTELLIGENCE_FEATURES.md` - Detailed feature documentation

---

**Last Updated:** 2025-01-28
**Version:** 1.0.0
**API Status:** Production Ready
