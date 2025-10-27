# Memory Management API - Setup Guide

Complete setup instructions for the HippocampAI Memory Management API.

## Prerequisites

- Python 3.9+
- Docker (for Redis and Qdrant)
- Git

## Quick Start

### 1. Install Dependencies

```bash
# Clone repository (if needed)
git clone https://github.com/your-org/HippocampAI.git
cd HippocampAI

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Required Services

#### Using Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  qdrant_data:
```

Start services:

```bash
docker-compose up -d
```

#### Using Docker Manually

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:latest

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_CACHE_TTL=300

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
COLLECTION_FACTS=hippocampai_facts
COLLECTION_PREFS=hippocampai_prefs

# Embedding Configuration
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_QUANTIZED=false
EMBED_BATCH_SIZE=32
EMBED_DIMENSION=384

# Reranker Configuration
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# LLM Configuration (optional, for extraction)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
ALLOW_CLOUD=false

# Scoring Weights
WEIGHT_SIM=0.55
WEIGHT_RERANK=0.20
WEIGHT_RECENCY=0.15
WEIGHT_IMPORTANCE=0.10

# Half-lives (days)
HALF_LIFE_PREFS=90
HALF_LIFE_FACTS=30
HALF_LIFE_EVENTS=14
```

### 4. Initialize Vector Collections

Run this script to initialize Qdrant collections:

```python
# scripts/init_collections.py
from hippocampai.config import get_config
from hippocampai.vector.qdrant_store import QdrantStore

config = get_config()
store = QdrantStore(
    url=config.qdrant_url,
    collection_facts=config.collection_facts,
    collection_prefs=config.collection_prefs,
)

# Create collections
store.ensure_collection(
    collection_name=config.collection_facts,
    vector_size=config.embed_dimension,
    distance="Cosine",
)
store.ensure_collection(
    collection_name=config.collection_prefs,
    vector_size=config.embed_dimension,
    distance="Cosine",
)

print("Collections initialized successfully!")
```

Run it:

```bash
python scripts/init_collections.py
```

### 5. Start the API Server

#### Development Mode

```bash
# Using Python directly
python -m hippocampai.api.async_app

# Or with uvicorn
uvicorn hippocampai.api.async_app:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode

```bash
# With multiple workers
uvicorn hippocampai.api.async_app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### 6. Verify Installation

```bash
# Health check
curl http://localhost:8000/healthz

# Expected response:
# {"status":"ok","service":"hippocampai","version":"2.0.0"}

# Check Redis stats
curl http://localhost:8000/stats
```

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/test_memory_management_api.py -v

# Run specific test
pytest tests/test_memory_management_api.py::test_create_memory -v
```

### Run Example Demo

```bash
python examples/10_memory_management_api.py
```

## API Documentation

### Interactive API Docs

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Manual Testing with cURL

```bash
# Create a memory
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paris is the capital of France",
    "user_id": "test_user",
    "type": "fact",
    "importance": 8.0,
    "tags": ["geography"]
  }'

# Recall memories
curl -X POST http://localhost:8000/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "capital of France",
    "user_id": "test_user",
    "k": 5
  }'
```

## Troubleshooting

### Redis Connection Issues

```bash
# Check if Redis is running
docker ps | grep redis

# Test Redis connection
redis-cli ping
# Expected: PONG

# Check Redis logs
docker logs redis
```

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Test Qdrant connection
curl http://localhost:6333/collections
# Expected: {"result":{"collections":[]}}

# Check Qdrant logs
docker logs qdrant
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade

# Verify installation
python -c "import redis; import fastapi; import qdrant_client; print('All imports OK')"
```

### Model Download Issues

If embedding or reranking models fail to download:

```bash
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5')
SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Models downloaded successfully')
"
```

## Performance Tuning

### Redis Configuration

For high-traffic scenarios:

```bash
# Increase Redis memory
docker run -d --name redis -p 6379:6379 \
  redis:latest \
  --maxmemory 2gb \
  --maxmemory-policy allkeys-lru
```

### Qdrant Configuration

For better performance:

```bash
# Increase Qdrant memory and optimize
docker run -d --name qdrant -p 6333:6333 \
  -e QDRANT__SERVICE__GRPC_PORT=6334 \
  qdrant/qdrant:latest
```

### API Server Configuration

```bash
# Production deployment with optimal workers
uvicorn hippocampai.api.async_app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers $((2 * $(nproc) + 1)) \
  --limit-concurrency 1000 \
  --timeout-keep-alive 5
```

## Monitoring

### Check Server Logs

```bash
# With standard Python logging
tail -f hippocampai.log

# With systemd
journalctl -u hippocampai -f
```

### Monitor Redis

```bash
# Redis CLI
redis-cli

# Inside Redis CLI:
INFO memory
INFO stats
DBSIZE
```

### Monitor Qdrant

```bash
# Collection info
curl http://localhost:6333/collections/hippocampai_facts

# Cluster info
curl http://localhost:6333/cluster
```

## Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "hippocampai.api.async_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t hippocampai-api .
docker run -d -p 8000:8000 --env-file .env hippocampai-api
```

### Using Kubernetes

See `kubernetes/` directory for deployment manifests.

### Using systemd

Create `/etc/systemd/system/hippocampai.service`:

```ini
[Unit]
Description=HippocampAI Memory Management API
After=network.target

[Service]
Type=simple
User=hippocampai
WorkingDirectory=/opt/hippocampai
Environment="PATH=/opt/hippocampai/venv/bin"
ExecStart=/opt/hippocampai/venv/bin/uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable hippocampai
sudo systemctl start hippocampai
sudo systemctl status hippocampai
```

## Security Considerations

1. **Authentication**: Add authentication middleware for production
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Input Validation**: All inputs are validated via Pydantic models
4. **Redis Security**: Use Redis AUTH and encrypted connections
5. **CORS**: Configure CORS appropriately for your frontend
6. **HTTPS**: Use reverse proxy (nginx, Caddy) with SSL/TLS

## Backup & Recovery

### Backup Redis

```bash
# Create backup
redis-cli SAVE
cp /var/lib/redis/dump.rdb /backup/redis-$(date +%Y%m%d).rdb

# Or use RDB snapshots
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

### Backup Qdrant

```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/hippocampai_facts/snapshots

# Download snapshot
curl http://localhost:6333/collections/hippocampai_facts/snapshots/{snapshot_name} \
  --output snapshot.tar
```

## Next Steps

1. Review the [API Documentation](MEMORY_MANAGEMENT_API.md)
2. Run the example scripts in `examples/`
3. Customize configuration in `.env`
4. Implement authentication for production
5. Set up monitoring and alerting
6. Configure backups

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/HippocampAI/issues
- Documentation: https://docs.hippocampai.io
- Email: support@hippocampai.io

## License

MIT License - See LICENSE file for details
