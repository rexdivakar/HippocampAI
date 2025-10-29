# HippocampAI: Complete Deployment & Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Deployment Guide](#deployment-guide)
   - [Prerequisites](#prerequisites)
   - [Local Development Setup](#local-development-setup)
   - [Docker Deployment](#docker-deployment)
   - [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Library Usage Guide](#library-usage-guide)
6. [Feature Documentation](#feature-documentation)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

HippocampAI is a production-ready, intelligent memory management system designed as a SaaS platform. It provides semantic memory storage, retrieval, and intelligent processing capabilities for AI applications.

### Key Features

- **Semantic Memory Storage**: Vector-based storage with Qdrant
- **Fast Retrieval**: Hybrid search combining semantic and keyword-based approaches
- **Intelligent Processing**: Entity extraction, relationship mapping, fact extraction
- **Memory Consolidation**: Automatic deduplication and consolidation
- **Temporal Analytics**: Time-based insights and trend detection
- **Multi-tenant Support**: User and session isolation
- **RESTful API**: FastAPI-based HTTP interface
- **Async Task Processing**: Celery-based background jobs
- **Caching Layer**: Redis for high-performance lookups

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI REST API                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Memory     │  │ Intelligence │  │  Background  │      │
│  │   Routes     │  │   Routes     │  │   Tasks      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│   Qdrant     │ │  Redis   │ │  Celery  │
│   Vector DB  │ │  Cache   │ │  Worker  │
└──────────────┘ └──────────┘ └──────────┘
```

### Components

1. **FastAPI Application**: HTTP server handling requests
2. **Qdrant**: Vector database for semantic search
3. **Redis**: Caching and task queue
4. **Celery**: Asynchronous task processing
5. **LLM Adapters**: Support for Ollama, OpenAI, Groq

---

## Deployment Guide

### Prerequisites

#### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Disk**: 10GB+ free space
- **Docker**: Version 20.10+ (for containerized deployment)

#### Required Services

- Qdrant vector database
- Redis server
- LLM provider (Ollama/OpenAI/Groq)

---

### Local Development Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Step 4: Set Up Infrastructure Services

**Option A: Using Docker Compose (Recommended)**

```bash
# Start Qdrant and Redis
docker-compose up -d qdrant redis

# Verify services are running
docker-compose ps
```

**Option B: Manual Installation**

```bash
# Install Qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Install Redis
docker run -p 6379:6379 redis:7-alpine

# Or install locally via package manager
# Ubuntu/Debian:
sudo apt-get install redis-server
# macOS:
brew install redis
```

#### Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Minimal `.env` configuration:**

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
COLLECTION_FACTS=hippocampai_facts
COLLECTION_PREFS=hippocampai_prefs

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL=3600

# LLM Configuration
LLM_PROVIDER=ollama  # Options: ollama, openai, groq
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
ALLOW_CLOUD=false

# Embedding Configuration
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DIMENSION=384
EMBED_QUANTIZED=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

#### Step 6: Initialize Database Collections

```bash
# Run initialization script
python setup_initial.py
```

#### Step 7: Start the Application

```bash
# Start FastAPI server
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, start Celery worker (optional)
celery -A hippocampai.celery_app worker --loglevel=info
```

#### Step 8: Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# Run test suite
pytest tests/
```

---

### Docker Deployment

#### Complete Stack with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT_ALLOW_RECOVERY_MODE=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Ollama LLM (optional)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # HippocampAI API
  hippocampai-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - LLM_BASE_URL=http://ollama:11434
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    command: uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000

  # Celery Worker
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - LLM_BASE_URL=http://ollama:11434
    depends_on:
      - redis
      - qdrant
      - hippocampai-api
    command: celery -A hippocampai.celery_app worker --loglevel=info

volumes:
  qdrant_storage:
  redis_data:
  ollama_data:
```

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY setup.py .
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "hippocampai.api.async_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Deploy with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f hippocampai-api

# Scale Celery workers
docker-compose up -d --scale celery-worker=3

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

---

### Production Deployment

#### AWS Deployment

**Architecture:**

- **ECS Fargate**: For API and Celery workers
- **Elasticache**: For Redis
- **EC2**: For Qdrant (or managed vector DB)
- **ALB**: For load balancing
- **CloudWatch**: For logging and monitoring

**Deployment Steps:**

1. **Set up infrastructure:**

```bash
# Create VPC and subnets
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Create ECS cluster
aws ecs create-cluster --cluster-name hippocampai-cluster

# Create Elasticache for Redis
aws elasticache create-cache-cluster \
  --cache-cluster-id hippocampai-redis \
  --engine redis \
  --cache-node-type cache.t3.medium \
  --num-cache-nodes 1
```

2. **Push Docker image to ECR:**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t hippocampai .
docker tag hippocampai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hippocampai:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hippocampai:latest
```

3. **Create ECS task definition:**

```json
{
  "family": "hippocampai-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "hippocampai",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/hippocampai:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "QDRANT_URL",
          "value": "http://qdrant-instance:6333"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://redis.cache.amazonaws.com:6379"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hippocampai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

4. **Deploy with Terraform (optional):**

```hcl
# main.tf
resource "aws_ecs_service" "hippocampai" {
  name            = "hippocampai-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.hippocampai.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.hippocampai.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.hippocampai.arn
    container_name   = "hippocampai"
    container_port   = 8000
  }
}
```

#### Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hippocampai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hippocampai
  template:
    metadata:
      labels:
        app: hippocampai
    spec:
      containers:
      - name: hippocampai
        image: hippocampai:latest
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hippocampai-service
spec:
  selector:
    app: hippocampai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
```

---

## Configuration

### Environment Variables

Complete `.env` configuration:

```bash
# ==================== VECTOR DATABASE ====================
QDRANT_URL=http://localhost:6333
COLLECTION_FACTS=hippocampai_facts
COLLECTION_PREFS=hippocampai_prefs

# HNSW Parameters (for performance tuning)
HNSW_M=48              # Number of connections per node
EF_CONSTRUCTION=256    # Size of the dynamic candidate list
EF_SEARCH=128          # Size of the dynamic candidate list during search

# ==================== EMBEDDING MODEL ====================
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DIMENSION=384
EMBED_QUANTIZED=false
EMBED_BATCH_SIZE=32

# ==================== RERANKER ====================
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_CACHE_TTL=86400

# ==================== BM25 ====================
BM25_BACKEND=rank-bm25  # Options: rank-bm25

# ==================== LLM PROVIDER ====================
LLM_PROVIDER=ollama     # Options: ollama, openai, groq
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
ALLOW_CLOUD=false       # Set to true for OpenAI/Groq

# For OpenAI (if ALLOW_CLOUD=true)
# OPENAI_API_KEY=sk-...

# For Groq (if ALLOW_CLOUD=true)
# GROQ_API_KEY=gsk_...

# ==================== REDIS ====================
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL=3600

# ==================== RETRIEVAL ====================
TOP_K_QDRANT=200        # Initial retrieval size
TOP_K_FINAL=20          # Final result size after reranking
RRF_K=60                # Reciprocal Rank Fusion parameter

# ==================== SCORING WEIGHTS ====================
WEIGHT_SIM=0.35         # Semantic similarity weight
WEIGHT_RERANK=0.30      # Reranker score weight
WEIGHT_RECENCY=0.20     # Recency score weight
WEIGHT_IMPORTANCE=0.15  # Importance score weight

# ==================== DECAY HALF-LIVES ====================
HALF_LIFE_PREFS=90      # Preference decay (days)
HALF_LIFE_FACTS=30      # Fact decay (days)
HALF_LIFE_EVENTS=14     # Event decay (days)

# ==================== SCHEDULER ====================
ENABLE_SCHEDULER=true
DECAY_CRON=0 2 * * *           # Daily at 2 AM
CONSOLIDATE_CRON=0 3 * * 0     # Weekly on Sunday at 3 AM
SNAPSHOT_CRON=0 * * * *        # Hourly

# ==================== API ====================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
```

### Configuration via Python

```python
from hippocampai.config import Config

# Load from environment
config = Config()

# Or create custom configuration
custom_config = Config(
    qdrant_url="http://custom-qdrant:6333",
    redis_url="redis://custom-redis:6379",
    llm_provider="groq",
    llm_model="llama-3.3-70b-versatile",
    embed_model="BAAI/bge-large-en-v1.5",
    embed_dimension=1024
)
```

---

## Library Usage Guide

### Installation as Library

```bash
# Install from source
cd HippocampAI
pip install -e .

# Or install specific version
pip install git+https://github.com/rexdivakar/HippocampAI.git@v1.0.0
```

### Basic Usage

#### 1. Initialize Client

```python
from hippocampai import MemoryClient

# Initialize with default configuration
client = MemoryClient()

# Or with custom configuration
client = MemoryClient(
    qdrant_url="http://localhost:6333",
    redis_url="redis://localhost:6379",
    llm_provider="ollama"
)
```

#### 2. Store Memories

```python
# Store a simple memory
memory = client.remember(
    text="Paris is the capital of France",
    user_id="user123",
    type="fact"
)

print(f"Stored memory: {memory.id}")
# Output: Stored memory: mem_abc123xyz
```

#### 3. Recall Memories

```python
# Semantic search
results = client.recall(
    query="What is the capital of France?",
    user_id="user123",
    k=5
)

for result in results:
    print(f"Score: {result.score:.3f} - {result.memory.text}")
# Output: Score: 0.923 - Paris is the capital of France
```

#### 4. Update and Delete

```python
# Update a memory
updated = client.update_memory(
    memory_id=memory.id,
    text="Paris is the capital and largest city of France",
    importance=9.0
)

# Delete a memory
client.delete_memory(memory_id=memory.id, user_id="user123")
```

---

## Feature Documentation

### 1. Memory Storage and Retrieval

#### Creating Memories with Metadata

```python
from datetime import datetime, timezone

# Create a memory with full metadata
memory = client.remember(
    text="I prefer dark roast coffee with oat milk",
    user_id="user123",
    session_id="session_abc",
    type="preference",  # Types: fact, preference, event, context
    importance=8.5,     # Scale 0-10
    tags=["beverages", "coffee", "dietary"],
    metadata={
        "category": "food_preferences",
        "confidence": 0.95,
        "source": "user_profile"
    },
    ttl_days=90  # Auto-expire after 90 days
)

# Access memory properties
print(f"ID: {memory.id}")
print(f"Type: {memory.type}")
print(f"Created: {memory.created_at}")
print(f"Importance: {memory.importance}")
print(f"Tags: {memory.tags}")
```

#### Batch Operations

```python
# Batch create
memories_data = [
    {
        "text": "User loves pizza",
        "user_id": "user123",
        "type": "preference",
        "importance": 7.0
    },
    {
        "text": "User is allergic to peanuts",
        "user_id": "user123",
        "type": "fact",
        "importance": 10.0,
        "tags": ["health", "allergies"]
    },
    {
        "text": "User attended workshop on AI",
        "user_id": "user123",
        "type": "event",
        "importance": 6.0
    }
]

created_memories = client.batch_create_memories(memories_data)
print(f"Created {len(created_memories)} memories")

# Batch update
updates = [
    {
        "memory_id": created_memories[0].id,
        "importance": 8.0
    },
    {
        "memory_id": created_memories[1].id,
        "tags": ["health", "allergies", "critical"]
    }
]

updated_memories = client.batch_update_memories(updates)

# Batch delete
memory_ids = [mem.id for mem in created_memories[2:]]
results = client.batch_delete_memories(memory_ids)
```

### 2. Advanced Retrieval

#### Hybrid Search with Custom Weights

```python
# Standard recall with default weights
results = client.recall(
    query="What are my dietary preferences?",
    user_id="user123",
    k=10
)

# Custom weighted search
# Emphasize importance over recency
custom_results = client.recall(
    query="critical health information",
    user_id="user123",
    k=10,
    custom_weights={
        "sim": 0.30,        # Semantic similarity
        "rerank": 0.25,     # Reranker score
        "recency": 0.10,    # How recent
        "importance": 0.35  # User-defined importance
    }
)

for result in custom_results:
    print(f"Score: {result.score:.3f}")
    print(f"  Similarity: {result.breakdown['sim']:.3f}")
    print(f"  Importance: {result.breakdown['importance']:.3f}")
    print(f"  Text: {result.memory.text}")
```

#### Filtering and Pagination

```python
from datetime import timedelta

# Filter by tags
tagged_memories = client.get_memories(
    user_id="user123",
    filters={"tags": "health"}
)

# Filter by date range
recent_memories = client.get_memories(
    user_id="user123",
    created_after=datetime.now(timezone.utc) - timedelta(days=7)
)

# Filter by importance threshold
important_memories = client.get_memories(
    user_id="user123",
    importance_min=8.0
)

# Combined filters
filtered = client.get_memories(
    user_id="user123",
    filters={
        "type": "preference",
        "tags": "dietary",
        "min_importance": 7.0
    },
    created_after=datetime.now(timezone.utc) - timedelta(days=30)
)

# Text search across memories
search_results = client.get_memories(
    user_id="user123",
    search_text="coffee"
)

# Pagination
page1 = client.get_memories(
    user_id="user123",
    limit=20,
    offset=0
)

page2 = client.get_memories(
    user_id="user123",
    limit=20,
    offset=20
)
```

### 3. Intelligent Memory Processing

#### Fact Extraction from Text

```python
from hippocampai.pipeline.fact_extraction import FactExtractor
from hippocampai.adapters.provider_ollama import OllamaLLM

# Initialize fact extractor
llm = OllamaLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434")
fact_extractor = FactExtractor(llm=llm)

# Extract facts from conversation
conversation = """
User: Hi, I'm John. I live in San Francisco and work as a software engineer at TechCorp.
I love hiking on weekends and I'm currently learning Spanish.
"""

facts = fact_extractor.extract_facts(
    text=conversation,
    user_id="user123"
)

for fact in facts:
    print(f"Fact: {fact['text']}")
    print(f"  Category: {fact['category']}")
    print(f"  Confidence: {fact['confidence']}")
    print(f"  Importance: {fact['importance']}")

# Store extracted facts
for fact in facts:
    client.remember(
        text=fact['text'],
        user_id="user123",
        type="fact",
        importance=fact['importance'],
        metadata={"confidence": fact['confidence']}
    )
```

#### Entity Recognition

```python
from hippocampai.pipeline.entity_recognition import EntityRecognizer

# Initialize entity recognizer
entity_recognizer = EntityRecognizer(llm=llm)

# Extract entities from text
text = "Apple Inc. announced that Tim Cook will present at the conference in Cupertino next month."

entities = entity_recognizer.extract_entities(text)

for entity in entities:
    print(f"{entity['type']}: {entity['text']}")
    print(f"  Confidence: {entity['confidence']}")

# Output:
# ORGANIZATION: Apple Inc.
#   Confidence: 0.98
# PERSON: Tim Cook
#   Confidence: 0.95
# LOCATION: Cupertino
#   Confidence: 0.92
```

#### Relationship Mapping

```python
from hippocampai.pipeline.relationship_mapping import RelationshipMapper

# Initialize relationship mapper
relationship_mapper = RelationshipMapper(llm=llm)

# Extract relationships
text = "John works for TechCorp. He is managed by Sarah who is the CTO."

relationships = relationship_mapper.extract_relationships(text)

for rel in relationships:
    print(f"{rel['source']} --[{rel['type']}]--> {rel['target']}")
    print(f"  Confidence: {rel['confidence']}")

# Output:
# John --[WORKS_FOR]--> TechCorp
#   Confidence: 0.95
# John --[MANAGED_BY]--> Sarah
#   Confidence: 0.90
# Sarah --[HAS_ROLE]--> CTO
#   Confidence: 0.92
```

### 4. Memory Consolidation

#### Automatic Deduplication

```python
# Create similar memories
client.remember(
    text="Rome is the capital of Italy",
    user_id="user123",
    check_duplicate=False
)

client.remember(
    text="Rome is the capital city of Italy",
    user_id="user123",
    check_duplicate=False
)

# Run deduplication analysis
result = client.deduplicate_user_memories(
    user_id="user123",
    similarity_threshold=0.85,
    dry_run=True  # Preview without making changes
)

print(f"Total memories: {result['total_memories']}")
print(f"Duplicate groups: {result['groups_found']}")
print(f"Memories to remove: {result['would_remove']}")

# Actually perform deduplication
result = client.deduplicate_user_memories(
    user_id="user123",
    similarity_threshold=0.85,
    dry_run=False
)

print(f"Removed {result['removed']} duplicate memories")
```

#### Memory Consolidation

```python
# Create related memories
client.remember(
    text="I like coffee in the morning",
    user_id="user123",
    type="preference"
)

client.remember(
    text="I enjoy drinking coffee every morning",
    user_id="user123",
    type="preference"
)

client.remember(
    text="Morning coffee is my favorite",
    user_id="user123",
    type="preference"
)

# Consolidate similar memories
result = client.consolidate_memories(
    user_id="user123",
    similarity_threshold=0.75,
    dry_run=True
)

print(f"Groups to consolidate: {result['groups_found']}")

for group in result['groups']:
    print(f"\nGroup: {len(group['memories'])} memories")
    print(f"Suggested consolidation: {group['consolidated_text']}")

# Perform consolidation
result = client.consolidate_memories(
    user_id="user123",
    similarity_threshold=0.75,
    dry_run=False
)

print(f"Consolidated {result['consolidated']} memory groups")
```

### 5. Session Management

#### Creating and Using Sessions

```python
from hippocampai.session.session_manager import SessionManager

# Initialize session manager
session_manager = SessionManager(
    redis_store=client.redis,
    qdrant_store=client.qdrant,
    embedder=client.embedder
)

# Create a session
session = session_manager.create_session(
    user_id="user123",
    metadata={
        "platform": "web",
        "user_agent": "Mozilla/5.0...",
        "ip_address": "192.168.1.1"
    }
)

print(f"Session ID: {session.id}")

# Store memories in session context
memory = client.remember(
    text="User asked about pricing plans",
    user_id="user123",
    session_id=session.id,
    type="context"
)

# Retrieve session-specific memories
session_memories = client.get_memories(
    user_id="user123",
    session_id=session.id
)

# Update session
session_manager.update_session(
    session_id=session.id,
    metadata={"last_topic": "pricing"}
)

# End session
session_manager.end_session(session_id=session.id)
```

#### Cross-Session Insights

```python
from hippocampai.pipeline.insights import InsightGenerator

# Initialize insight generator
insight_gen = InsightGenerator(
    session_manager=session_manager,
    llm=llm
)

# Generate insights across sessions
insights = insight_gen.generate_user_insights(
    user_id="user123",
    insight_types=["patterns", "trends", "preferences"]
)

for insight in insights:
    print(f"Type: {insight['type']}")
    print(f"Insight: {insight['text']}")
    print(f"Confidence: {insight['confidence']}")
    print(f"Supporting sessions: {len(insight['sessions'])}")
```

### 6. Temporal Analytics

#### Time-Based Analysis

```python
from hippocampai.pipeline.temporal_analytics import TemporalAnalyzer

# Initialize temporal analyzer
temporal_analyzer = TemporalAnalyzer(llm=llm)

# Analyze activity patterns
patterns = temporal_analyzer.analyze_activity_patterns(
    user_id="user123",
    start_date=datetime.now(timezone.utc) - timedelta(days=30),
    end_date=datetime.now(timezone.utc)
)

print(f"Peak activity hour: {patterns['peak_hour']}")
print(f"Most active day: {patterns['peak_day']}")
print(f"Total interactions: {patterns['total_count']}")

# Detect trends
trends = temporal_analyzer.detect_trends(
    user_id="user123",
    topic="coffee preferences",
    time_window_days=90
)

for trend in trends:
    print(f"Trend: {trend['description']}")
    print(f"Confidence: {trend['confidence']}")
    print(f"Direction: {trend['direction']}")  # increasing, decreasing, stable

# Forecast future patterns
forecast = temporal_analyzer.forecast_activity(
    user_id="user123",
    days_ahead=7
)

print(f"Predicted activity next week: {forecast['predicted_count']}")
print(f"Confidence interval: {forecast['confidence_low']} - {forecast['confidence_high']}")
```

#### Memory Decay Management

```python
# Memories automatically decay based on configuration
# Check current decay scores
memories = client.get_memories(user_id="user123")

for mem in memories:
    recency_score = mem.calculate_recency_score(
        half_life_days=30  # From configuration
    )
    print(f"Memory: {mem.text[:50]}")
    print(f"  Age: {(datetime.now(timezone.utc) - mem.created_at).days} days")
    print(f"  Recency score: {recency_score:.3f}")

# Manually update importance to prevent decay
client.update_memory(
    memory_id=mem.id,
    importance=10.0  # Max importance
)
```

### 7. Semantic Clustering

```python
from hippocampai.pipeline.semantic_clustering import SemanticClusterer

# Initialize clusterer
clusterer = SemanticClusterer(
    embedder=client.embedder,
    llm=llm
)

# Cluster memories by topic
memories = client.get_memories(user_id="user123")

clusters = clusterer.cluster_memories(
    memories=memories,
    n_clusters=5,  # Auto-detect if None
    min_cluster_size=3
)

for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}: {cluster['label']}")
    print(f"Size: {len(cluster['memories'])}")
    print(f"Representative: {cluster['representative_text']}")
    print("Memories:")
    for mem in cluster['memories'][:3]:
        print(f"  - {mem.text[:60]}...")

# Generate cluster summaries
for cluster in clusters:
    summary = clusterer.generate_cluster_summary(
        memories=cluster['memories'],
        llm=llm
    )
    print(f"\nCluster: {cluster['label']}")
    print(f"Summary: {summary}")
```

### 8. Knowledge Graph Integration

```python
from hippocampai.graph.knowledge_graph import KnowledgeGraph

# Initialize knowledge graph
kg = KnowledgeGraph(
    qdrant_store=client.qdrant,
    embedder=client.embedder
)

# Build graph from memories
memories = client.get_memories(user_id="user123")

kg.build_from_memories(
    memories=memories,
    user_id="user123"
)

# Query graph
# Find all entities related to a topic
entities = kg.get_entities(
    user_id="user123",
    entity_type="PERSON"
)

for entity in entities:
    print(f"Entity: {entity['text']}")
    print(f"Type: {entity['type']}")
    print(f"Connections: {entity['connection_count']}")

# Get entity relationships
relationships = kg.get_entity_relationships(
    user_id="user123",
    entity_id="ent_abc123"
)

for rel in relationships:
    print(f"{rel['source_name']} --[{rel['type']}]--> {rel['target_name']}")

# Find connection path between entities
path = kg.find_path(
    user_id="user123",
    source_entity="John",
    target_entity="TechCorp",
    max_depth=3
)

print("Connection path:")
for step in path:
    print(f"  {step['entity']} --[{step['relation']}]-->")
```

### 9. Memory Versioning

```python
from hippocampai.versioning.memory_versioning import MemoryVersioning

# Initialize versioning
versioning = MemoryVersioning(
    qdrant_store=client.qdrant,
    redis_store=client.redis
)

# Create initial memory
memory = client.remember(
    text="Paris is the capital of France",
    user_id="user123"
)

# Update creates a new version
updated = client.update_memory(
    memory_id=memory.id,
    text="Paris is the capital and largest city of France",
    track_version=True
)

# View version history
versions = versioning.get_version_history(
    memory_id=memory.id
)

for version in versions:
    print(f"Version {version['version_number']}")
    print(f"  Text: {version['text']}")
    print(f"  Modified: {version['modified_at']}")
    print(f"  Modified by: {version['modified_by']}")

# Rollback to previous version
versioning.rollback_to_version(
    memory_id=memory.id,
    version_number=1
)

# Compare versions
diff = versioning.compare_versions(
    memory_id=memory.id,
    version1=1,
    version2=2
)

print(f"Changes: {diff['changes']}")
```

### 10. Search and Suggestions

```python
from hippocampai.search.saved_searches import SavedSearchManager
from hippocampai.search.suggestions import SearchSuggestionEngine

# Initialize search manager
search_manager = SavedSearchManager(redis_store=client.redis)

# Save a search
search = search_manager.save_search(
    user_id="user123",
    query="my dietary preferences",
    filters={"tags": "food"},
    name="Food Preferences"
)

# List saved searches
saved = search_manager.get_user_searches(user_id="user123")

for s in saved:
    print(f"Search: {s['name']}")
    print(f"  Query: {s['query']}")
    print(f"  Used {s['use_count']} times")

# Execute saved search
results = search_manager.execute_saved_search(
    user_id="user123",
    search_id=search['id'],
    client=client
)

# Initialize suggestion engine
suggestion_engine = SearchSuggestionEngine(
    client=client,
    embedder=client.embedder
)

# Get search suggestions
suggestions = suggestion_engine.generate_suggestions(
    user_id="user123",
    partial_query="coff"
)

for suggestion in suggestions:
    print(f"Suggestion: {suggestion['text']}")
    print(f"  Score: {suggestion['score']}")
    print(f"  Type: {suggestion['type']}")  # recent, popular, semantic
```

---

## API Reference

### REST API Endpoints

#### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "qdrant": "connected",
    "redis": "connected",
    "llm": "available"
  }
}
```

#### Create Memory

```bash
POST /v1/memories

Request:
{
  "text": "User prefers dark mode interface",
  "user_id": "user123",
  "session_id": "session_abc",
  "type": "preference",
  "importance": 7.5,
  "tags": ["ui", "preferences"],
  "metadata": {"source": "settings"},
  "ttl_days": 365
}

Response:
{
  "id": "mem_xyz789",
  "text": "User prefers dark mode interface",
  "user_id": "user123",
  "type": "preference",
  "importance": 7.5,
  "created_at": "2025-01-15T10:30:00Z",
  "tags": ["ui", "preferences"]
}
```

#### Recall Memories

```bash
POST /v1/memories/recall

Request:
{
  "query": "What are my UI preferences?",
  "user_id": "user123",
  "k": 5,
  "filters": {
    "tags": "ui"
  },
  "custom_weights": {
    "sim": 0.4,
    "importance": 0.3,
    "recency": 0.2,
    "rerank": 0.1
  }
}

Response:
{
  "results": [
    {
      "memory": {
        "id": "mem_xyz789",
        "text": "User prefers dark mode interface",
        "type": "preference"
      },
      "score": 0.892,
      "breakdown": {
        "sim": 0.85,
        "rerank": 0.91,
        "recency": 0.95,
        "importance": 0.75
      }
    }
  ],
  "total": 1
}
```

#### Get Memories

```bash
GET /v1/memories?user_id=user123&type=preference&limit=20

Response:
{
  "memories": [...],
  "total": 15,
  "limit": 20,
  "offset": 0
}
```

#### Update Memory

```bash
PUT /v1/memories/{memory_id}

Request:
{
  "text": "User strongly prefers dark mode interface",
  "importance": 9.0,
  "tags": ["ui", "preferences", "accessibility"]
}

Response: (updated memory object)
```

#### Delete Memory

```bash
DELETE /v1/memories/{memory_id}?user_id=user123

Response:
{
  "deleted": true,
  "memory_id": "mem_xyz789"
}
```

#### Intelligence Features

```bash
# Extract Facts
POST /v1/intelligence/extract-facts

Request:
{
  "text": "I'm a software engineer living in NYC...",
  "user_id": "user123"
}

# Extract Entities
POST /v1/intelligence/extract-entities

# Map Relationships
POST /v1/intelligence/map-relationships

# Generate Summary
POST /v1/intelligence/summarize

# Cluster Memories
POST /v1/intelligence/cluster
```

#### Background Tasks

```bash
# Trigger Deduplication
POST /v1/background/deduplicate
{
  "user_id": "user123",
  "similarity_threshold": 0.85
}

# Trigger Consolidation
POST /v1/background/consolidate

# Expire Old Memories
POST /v1/background/expire
```

### Python Client API

#### MemoryClient

```python
class MemoryClient:
    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
        check_duplicate: bool = True
    ) -> Memory:
        """Store a memory."""

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[RecallResult]:
        """Recall memories using semantic search."""

    def get_memories(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        importance_min: Optional[float] = None,
        search_text: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """Get memories with filtering."""

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> Optional[Memory]:
        """Update a memory."""

    def delete_memory(
        self,
        memory_id: str,
        user_id: str
    ) -> bool:
        """Delete a memory."""

    def batch_create_memories(
        self,
        memories: List[Dict[str, Any]],
        check_duplicates: bool = False
    ) -> List[Memory]:
        """Batch create memories."""

    def deduplicate_user_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Deduplicate memories."""

    def consolidate_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.75,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Consolidate similar memories."""
```

---

## Best Practices

### 1. Memory Organization

```python
# Use meaningful tags for organization
client.remember(
    text="User completed Python course",
    user_id="user123",
    type="event",
    tags=["education", "programming", "python", "achievement"],
    importance=8.0
)

# Set appropriate TTLs
client.remember(
    text="Temporary access code: ABC123",
    user_id="user123",
    type="context",
    ttl_days=1  # Expires after 1 day
)

# Use importance scores strategically
client.remember(
    text="User is allergic to peanuts",
    user_id="user123",
    type="fact",
    importance=10.0,  # Critical health information
    tags=["health", "allergies", "critical"]
)
```

### 2. Search Optimization

```python
# Use specific queries
# Good:
results = client.recall(
    query="user's coffee preferences in the morning",
    user_id="user123"
)

# Less effective:
results = client.recall(
    query="preferences",
    user_id="user123"
)

# Combine filters for precision
results = client.recall(
    query="recent health information",
    user_id="user123",
    filters={
        "type": "fact",
        "tags": "health"
    },
    custom_weights={
        "importance": 0.4,  # Emphasize importance for health data
        "recency": 0.3
    }
)
```

### 3. Session Management

```python
# Always use sessions for conversational contexts
session = session_manager.create_session(user_id="user123")

# Store context in session
client.remember(
    text="User asked about pricing for enterprise plan",
    user_id="user123",
    session_id=session.id,
    type="context",
    ttl_days=7  # Context expires after 7 days
)

# Recall with session context
results = client.recall(
    query="pricing discussion",
    user_id="user123",
    session_id=session.id  # Prioritizes session memories
)

# End session when conversation concludes
session_manager.end_session(session_id=session.id)
```

### 4. Performance Optimization

```python
# Use batch operations for bulk inserts
memories = [
    {"text": f"Log entry {i}", "user_id": "user123"}
    for i in range(100)
]
client.batch_create_memories(memories)

# Leverage caching
# First call - hits database
results1 = client.recall(query="coffee", user_id="user123")

# Second call within cache TTL - uses cache
results2 = client.recall(query="coffee", user_id="user123")

# Paginate large result sets
page_size = 50
offset = 0
while True:
    memories = client.get_memories(
        user_id="user123",
        limit=page_size,
        offset=offset
    )
    if not memories:
        break
    process_memories(memories)
    offset += page_size
```

### 5. Error Handling

```python
from hippocampai.exceptions import (
    MemoryNotFoundError,
    DuplicateMemoryError,
    ValidationError
)

try:
    memory = client.remember(
        text="Important fact",
        user_id="user123",
        check_duplicate=True
    )
except DuplicateMemoryError as e:
    print(f"Memory already exists: {e.duplicate_id}")
    # Handle duplicate appropriately

except ValidationError as e:
    print(f"Invalid input: {e}")
    # Fix validation issues

except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and handle gracefully
```

### 6. Monitoring and Observability

```python
# Enable telemetry
from hippocampai.telemetry import TelemetryManager

telemetry = TelemetryManager(client=client)

# Track operations
with telemetry.track_operation("recall"):
    results = client.recall(query="test", user_id="user123")

# Get metrics
metrics = client.get_telemetry_metrics()
print(f"Average recall time: {metrics['recall']['avg']}ms")
print(f"Cache hit rate: {metrics['cache']['hit_rate']}")

# Monitor recent operations
operations = client.get_recent_operations(limit=10)
for op in operations:
    if op.status == "error":
        print(f"Error in {op.operation}: {op.error_message}")
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Errors

```python
# Issue: Cannot connect to Qdrant
# Solution: Check Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant

# Check connection
curl http://localhost:6333/health
```

#### 2. Memory Not Found

```python
# Issue: get_memory returns None
# Possible causes:
# 1. Memory was deleted
# 2. Wrong user_id
# 3. Memory expired (TTL)

# Debug:
memory = client.get_memory(memory_id="mem_123")
if memory is None:
    # Check if it exists with include_expired
    memories = client.get_memories(
        user_id="user123",
        filters={"include_expired": True}
    )
    # Search for the ID
```

#### 3. Slow Retrieval

```python
# Issue: Recall taking too long
# Solutions:

# 1. Reduce search space
results = client.recall(
    query="test",
    user_id="user123",
    filters={"type": "fact"},  # Narrows search
    k=5  # Reduce result count
)

# 2. Check cache configuration
# Ensure Redis is running and configured properly

# 3. Optimize Qdrant parameters
# In .env:
# TOP_K_QDRANT=100  # Reduce from 200
# EF_SEARCH=64      # Reduce from 128
```

#### 4. Out of Memory

```python
# Issue: Application crashes with OOM
# Solutions:

# 1. Reduce batch sizes
EMBED_BATCH_SIZE=16  # Default is 32

# 2. Enable memory quantization
EMBED_QUANTIZED=true

# 3. Limit concurrent operations
# Use connection pooling:
REDIS_MAX_CONNECTIONS=50
```

#### 5. LLM Errors

```python
# Issue: LLM generation fails
# Solutions:

# 1. Check LLM service is running
curl http://localhost:11434/api/tags  # For Ollama

# 2. Verify API keys (for cloud providers)
# In .env:
GROQ_API_KEY=gsk_your_key_here
ALLOW_CLOUD=true

# 3. Fallback gracefully
try:
    facts = fact_extractor.extract_facts(text, user_id)
except Exception as e:
    # Use rule-based extraction as fallback
    logger.warning(f"LLM failed, using fallback: {e}")
    facts = rule_based_extraction(text)
```

### Debug Mode

```python
# Enable debug logging
import logging

logging.basicConfig(level=logging.DEBUG)

# Or in .env:
LOG_LEVEL=DEBUG

# Check component status
health = client.get_health_status()
print(f"Qdrant: {health['qdrant']}")
print(f"Redis: {health['redis']}")
print(f"LLM: {health['llm']}")
```

### Performance Profiling

```python
import time

# Profile operations
start = time.time()
memory = client.remember(text="test", user_id="user123")
create_time = time.time() - start

start = time.time()
results = client.recall(query="test", user_id="user123")
recall_time = time.time() - start

print(f"Create: {create_time*1000:.2f}ms")
print(f"Recall: {recall_time*1000:.2f}ms")

# Check cache stats
stats = await client.redis.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total keys: {stats['total_keys']}")
```

---

## Appendix

### A. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `LLM_PROVIDER` | `ollama` | LLM provider (ollama/openai/groq) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `EMBED_DIMENSION` | `384` | Embedding vector dimension |
| `TOP_K_QDRANT` | `200` | Initial retrieval size |
| `TOP_K_FINAL` | `20` | Final result size |
| `WEIGHT_SIM` | `0.35` | Semantic similarity weight |
| `WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL in seconds |

### B. Model Recommendations

#### Embedding Models

| Model | Dimension | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | Good | Fast | General purpose |
| `BAAI/bge-base-en-v1.5` | 768 | Better | Medium | High accuracy |
| `BAAI/bge-large-en-v1.5` | 1024 | Best | Slow | Maximum quality |

#### LLM Models

**Ollama (Local)**

- `qwen2.5:7b-instruct` - Balanced quality/speed
- `llama3.1:8b` - Fast, good quality
- `mixtral:8x7b` - High quality, slower

**Groq (Cloud)**

- `llama-3.3-70b-versatile` - Best quality
- `llama-3.1-8b-instant` - Fastest

**OpenAI (Cloud)**

- `gpt-4o-mini` - Cost-effective
- `gpt-4o` - Highest quality

### C. Architecture Diagrams

[See repository for detailed diagrams]

### D. Migration Guide

#### From Mem0

```python
# Mem0 code
from mem0 import Memory
mem = Memory()
mem.add("User likes coffee", user_id="user123")

# HippocampAI equivalent
from hippocampai import MemoryClient
client = MemoryClient()
client.remember(
    text="User likes coffee",
    user_id="user123"
)
```

### E. API Rate Limits

| Endpoint | Rate Limit | Burst |
|----------|------------|-------|
| `/v1/memories` (POST) | 100/min | 150 |
| `/v1/memories/recall` (POST) | 200/min | 300 |
| `/v1/memories` (GET) | 500/min | 750 |
| `/v1/intelligence/*` | 50/min | 75 |

---

## Support and Resources

- **Documentation**: [https://docs.hippocampai.com](https://docs.hippocampai.com)
- **GitHub**: [https://github.com/rexdivakar/HippocampAI](https://github.com/rexdivakar/HippocampAI)
- **Issues**: [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rexdivakar/HippocampAI/discussions)
- **Email**: <support@hippocampai.com>

---

**Copyright**: © 2025 HippocampAI Contributors

**Version**: 1.0.0
