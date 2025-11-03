# HippocampAI Troubleshooting & FAQ

Complete troubleshooting guide and frequently asked questions for HippocampAI.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Connection Issues](#connection-issues)
- [Performance Issues](#performance-issues)
- [Memory Issues](#memory-issues)
- [LLM Provider Issues](#llm-provider-issues)
- [Docker & Deployment Issues](#docker--deployment-issues)
- [Data & Storage Issues](#data--storage-issues)
- [API & Integration Issues](#api--integration-issues)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Getting Help](#getting-help)

---

## Installation Issues

### Problem: `pip install hippocampai` fails

**Symptom:**
```bash
ERROR: Could not find a version that satisfies the requirement hippocampai
```

**Solutions:**

1. **Check Python version:**
```bash
python --version  # Should be 3.9+
```

If version is too old:
```bash
# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

2. **Upgrade pip:**
```bash
pip install --upgrade pip setuptools wheel
pip install hippocampai
```

3. **Install from source:**
```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .
```

---

### Problem: Dependency conflicts with sentence-transformers

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions:**

1. **Use a clean virtual environment:**
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or: fresh_env\Scripts\activate  # Windows
pip install hippocampai[all]
```

2. **Install specific versions:**
```bash
pip install sentence-transformers==2.2.2
pip install transformers==4.35.0
pip install hippocampai
```

3. **Use conda (alternative):**
```bash
conda create -n hippocampai python=3.11
conda activate hippocampai
pip install hippocampai[all]
```

---

### Problem: CUDA/GPU issues with sentence-transformers

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **CPU/GPU is auto-detected** by sentence-transformers. To force CPU-only, set environment variable before starting:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only
python your_app.py
```

2. **Reduce batch size:**
```bash
# .env
EMBED_BATCH_SIZE=8  # Reduce from default 32
```

3. **Use quantized models:**
```bash
# .env
EMBED_QUANTIZED=true
```

4. **Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

### Problem: Apple Silicon (M1/M2) compatibility issues

**Symptom:**
```
ERROR: Could not build wheels for hnswlib
```

**Solutions:**

1. **Install Xcode command line tools:**
```bash
xcode-select --install
```

2. **Use Rosetta (if needed):**
```bash
arch -x86_64 /bin/bash
pip install hippocampai
```

3. **Install dependencies separately:**
```bash
pip install --no-binary hnswlib hnswlib
pip install hippocampai
```

---

### Problem: Windows installation errors

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solutions:**

1. **Install Visual C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

2. **Use WSL2 (recommended):**
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3.11 python3.11-venv
python3.11 -m venv venv
source venv/bin/activate
pip install hippocampai
```

---

## Connection Issues

### Problem: Cannot connect to Qdrant

**Symptom:**
```
qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 404 (Not Found)
```

**Solutions:**

1. **Verify Qdrant is running:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

2. **Check connection URL:**
```python
from hippocampai import MemoryClient

# Test connection
client = MemoryClient(qdrant_url="http://localhost:6333")
try:
    stats = client.get_memory_statistics(user_id="test")
    print("✅ Connected successfully")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

3. **Verify network accessibility:**
```bash
# Test port accessibility
curl http://localhost:6333/collections

# Check firewall
sudo ufw status
```

4. **Check logs:**
```bash
docker logs $(docker ps -q --filter ancestor=qdrant/qdrant)
```

---

### Problem: Cannot connect to Redis

**Symptom:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions:**

1. **Verify Redis is running:**
```bash
# Check if Redis is running
redis-cli ping  # Should return PONG

# If not running
docker run -d -p 6379:6379 redis:7-alpine

# Or on Ubuntu
sudo systemctl start redis-server
```

2. **Check Redis configuration:**
```bash
# .env
REDIS_URL=redis://localhost:6379
```

3. **Test Redis connection:**
```python
import redis

try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print("✅ Redis connected")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

4. **Check Redis password:**
```bash
# If Redis has requirepass
REDIS_URL=redis://:password@localhost:6379
```

---

### Problem: API server not responding

**Symptom:**
```
requests.exceptions.ConnectionError: Connection refused
```

**Solutions:**

1. **Verify API is running:**
```bash
# Check running processes
ps aux | grep uvicorn

# Check Docker containers
docker ps | grep hippocampai
```

2. **Start API server:**
```bash
# Development
cd HippocampAI
python -m uvicorn api.async_app:app --host 0.0.0.0 --port 8000

# Production with Docker
docker-compose up -d
```

3. **Check API health:**
```bash
curl http://localhost:8000/health
```

4. **Check logs:**
```bash
# Docker logs
docker logs hippocampai_hippocampai_1

# Or check log files
tail -f logs/api.log
```

---

## Performance Issues

### Problem: Slow recall/search queries

**Symptom:**
```
Recall takes 5+ seconds for simple queries
```

**Solutions:**

1. **Redis caching is enabled by default** if Redis is available. Configure TTL:
```bash
# .env
REDIS_CACHE_TTL=3600  # Cache TTL in seconds (default: 300)
```

2. **Reduce number of candidates:**
```bash
# .env
TOP_K_QDRANT=50  # Reduce from default 200
TOP_K_FINAL=10    # Reduce from default 20
```

3. **Optimize HNSW parameters:**
```bash
# .env
EF_SEARCH=64  # Reduce from 128 for faster search
HNSW_M=32     # Reduce from 48
```

4. **Disable reranking for non-critical queries:**
```python
# Skip reranking for faster results
results = client.recall(
    query="test",
    user_id="alice",
    enable_rerank=False  # Faster but less accurate
)
```

5. **Use smaller embedding model:**
```bash
# .env
EMBED_MODEL=BAAI/bge-small-en-v1.5  # Default, 384 dim
# vs
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # 384 dim, faster
```

6. **Check system resources:**
```bash
# Monitor CPU/Memory
htop

# Check Qdrant memory usage
docker stats

# Check disk I/O
iostat -x 1
```

---

### Problem: High memory usage

**Symptom:**
```
Process using 8GB+ RAM
```

**Solutions:**

1. **Reduce embedding batch size:**
```bash
# .env
EMBED_BATCH_SIZE=16  # Reduce from 32
```

2. **Enable quantization:**
```bash
# .env
EMBED_QUANTIZED=true
```

3. **Limit concurrent requests:**
```python
# Use connection pooling limits
from hippocampai import MemoryClient, Config

config = Config(max_concurrent_requests=10)
client = MemoryClient(config=config)
```

4. **Clear cache periodically:**
```bash
# Clear Redis cache
redis-cli FLUSHDB

# Or set TTL for automatic expiration
CACHE_TTL_SECONDS=1800  # 30 minutes
```

5. **Restart services:**
```bash
docker-compose restart
```

---

### Problem: Slow memory consolidation

**Symptom:**
```
consolidate_all_memories() takes 10+ minutes
```

**Solutions:**

1. **Adjust consolidation threshold:**
```python
# Higher threshold = faster but fewer consolidations
count = client.consolidate_all_memories(similarity_threshold=0.90)
```

2. **Run consolidation during off-peak hours:**
```bash
# Schedule with cron
0 3 * * * /path/to/consolidate.sh  # Run at 3 AM
```

3. **Background consolidation is disabled by default:**
```bash
# .env
AUTO_CONSOLIDATION_ENABLED=false  # Default is false
```

4. **Run manual consolidation during off-peak:**
```python
# Manual consolidation (affects all users)
count = client.consolidate_all_memories(similarity_threshold=0.85)
print(f"Consolidated {count} memories")
```

---

## Memory Issues

### Problem: Memories not being stored

**Symptom:**
```
client.remember() succeeds but recall returns nothing
```

**Solutions:**

1. **Verify memory was created:**
```python
memory = client.remember("test", user_id="alice")
print(f"Memory ID: {memory.id}")

# Immediately try to recall
results = client.recall("test", user_id="alice")
print(f"Found {len(results)} results")
```

2. **Check collection exists:**
```python
from qdrant_client import QdrantClient

qdrant = QdrantClient(url="http://localhost:6333")
collections = qdrant.get_collections()
print(f"Collections: {collections}")
```

3. **Check user_id mismatch:**
```python
# Make sure user_id matches exactly
memory = client.remember("test", user_id="alice")
results = client.recall("test", user_id="alice")  # Same user_id

# Case sensitive!
# "Alice" != "alice"
```

4. **Check Qdrant storage:**
```bash
# Check if data is being persisted
ls -lh qdrant_storage/collections/
```

---

### Problem: Duplicate memories being created

**Symptom:**
```
Same memory appears multiple times in recall
```

**Solutions:**

1. **Automatic deduplication is enabled by default** via smart memory updates. The system automatically detects similar memories and skips duplicates.

2. **Check for multiple remember() calls:**
```python
# Don't do this
for i in range(10):
    client.remember("same text", user_id="alice")  # Creates 10 duplicates

# Do this instead
if not already_exists:
    client.remember("new text", user_id="alice")
```

3. **Use idempotency keys:**
```python
memory = client.remember(
    text="important fact",
    user_id="alice",
    metadata={"idempotency_key": "unique_key_123"}
)
```

---

### Problem: Memory size/token count issues

**Symptom:**
```
Memory rejected due to size limits
```

**Solutions:**

1. **There are no hard size limits** in the current implementation. Very large memories may impact performance.

2. **Split large memories for better performance:**
```python
def split_large_memory(text: str, max_size: int = 5000) -> list[str]:
    chunks = []
    for i in range(0, len(text), max_size):
        chunks.append(text[i:i+max_size])
    return chunks

large_text = "..." * 10000
for chunk in split_large_memory(large_text):
    client.remember(chunk, user_id="alice")
```

3. **Summarize before storing:**
```python
# Use LLM to summarize
summary = client.summarize_text(large_text)
client.remember(summary, user_id="alice")
```

---

## LLM Provider Issues

### Problem: Ollama connection failed

**Symptom:**
```
Failed to connect to Ollama at http://localhost:11434
```

**Solutions:**

1. **Verify Ollama is running:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama (macOS)
ollama serve

# Start Ollama (Linux systemd)
sudo systemctl start ollama
```

2. **Pull required model:**
```bash
ollama pull qwen2.5:7b-instruct
ollama list  # Verify model is downloaded
```

3. **Check Ollama configuration:**
```bash
# .env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b-instruct
```

4. **Test Ollama directly:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "Hello"
}'
```

---

### Problem: OpenAI API errors

**Symptom:**
```
openai.error.AuthenticationError: Invalid API key
```

**Solutions:**

1. **Verify API key:**
```python
import os
print(f"API Key: {os.getenv('OPENAI_API_KEY')[:10]}...")  # Don't print full key
```

2. **Check API key format:**
```bash
# .env
OPENAI_API_KEY=sk-proj-...  # Should start with sk-proj- or sk-
```

3. **Verify account has credits:**
   - Check: https://platform.openai.com/account/usage

4. **Test API key:**
```python
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}]
    )
    print("✅ OpenAI connected")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

### Problem: Groq rate limiting

**Symptom:**
```
groq.RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Implement retry logic:**
```python
from hippocampai import MemoryClient
from hippocampai.adapters import GroqLLM

# Retry is built-in
provider = GroqLLM(
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3,
    retry_delay=2
)
client = MemoryClient(llm_provider=provider)
```

2. **Use rate limiting:**
```python
import time

for text in texts:
    client.remember(text, user_id="alice")
    time.sleep(0.5)  # Delay between requests
```

3. **Switch to local Ollama for development:**
```bash
# .env
LLM_PROVIDER=ollama  # No rate limits
```

---

### Problem: Anthropic API errors

**Symptom:**
```
anthropic.APIError: Invalid request
```

**Solutions:**

1. **Check API key:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...  # Should start with sk-ant-
```

2. **Verify model name:**
```bash
# .env
LLM_MODEL=claude-3-5-sonnet-20241022  # Correct model name
```

3. **Check account limits:**
   - Console: https://console.anthropic.com/

---

## Docker & Deployment Issues

### Problem: Docker Compose fails to start

**Symptom:**
```
ERROR: Container failed to start
```

**Solutions:**

1. **Check Docker is running:**
```bash
docker info
# If error, start Docker daemon
sudo systemctl start docker
```

2. **Check ports availability:**
```bash
# Check if ports are in use
sudo lsof -i :6333  # Qdrant
sudo lsof -i :6379  # Redis
sudo lsof -i :8000  # API

# Kill processes using ports
sudo kill -9 <PID>
```

3. **Check Docker Compose file:**
```bash
# Validate docker-compose.yml
docker-compose config

# View logs
docker-compose logs
```

4. **Rebuild containers:**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

5. **Check disk space:**
```bash
df -h  # Check available disk space
docker system prune -a  # Clean up Docker
```

---

### Problem: Container permissions issues

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/qdrant/storage'
```

**Solutions:**

1. **Fix volume permissions:**
```bash
sudo chown -R 1000:1000 qdrant_storage/
sudo chown -R 1000:1000 data/
sudo chown -R 1000:1000 logs/
```

2. **Use named volumes:**
```yaml
# docker-compose.yml
volumes:
  qdrant_data:
  redis_data:

services:
  qdrant:
    volumes:
      - qdrant_data:/qdrant/storage
```

---

### Problem: Environment variables not loading

**Symptom:**
```
Configuration using defaults, not loading from .env
```

**Solutions:**

1. **Verify .env file location:**
```bash
ls -la .env  # Should be in project root
```

2. **Check .env syntax:**
```bash
# No spaces around =
QDRANT_URL=http://localhost:6333  # ✅ Correct
QDRANT_URL = http://localhost:6333  # ❌ Wrong
```

3. **Load .env explicitly:**
```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file
```

4. **Check Docker Compose env_file:**
```yaml
services:
  hippocampai:
    env_file:
      - .env  # Ensure this is present
```

---

## Data & Storage Issues

### Problem: Data not persisting after restart

**Symptom:**
```
All memories disappear after docker restart
```

**Solutions:**

1. **Check Docker volumes:**
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect hippocampai_qdrant_data
```

2. **Verify volume mounts:**
```yaml
# docker-compose.yml
services:
  qdrant:
    volumes:
      - ./qdrant_storage:/qdrant/storage  # Ensure this exists
```

3. **Check storage path:**
```bash
ls -lh qdrant_storage/collections/
```

4. **Create directories:**
```bash
mkdir -p qdrant_storage data logs backups
```

---

### Problem: Qdrant collection errors

**Symptom:**
```
Collection 'hippocampai_facts' does not exist
```

**Solutions:**

1. **Collections are auto-created:**
```python
# First remember() call creates collection
memory = client.remember("test", user_id="alice")
```

2. **Manually create collection:**
```python
from qdrant_client import QdrantClient

qdrant = QdrantClient(url="http://localhost:6333")
qdrant.create_collection(
    collection_name="hippocampai_facts",
    vectors_config={"size": 384, "distance": "Cosine"}
)
```

3. **Reset collections:**
```python
# Delete and recreate
qdrant.delete_collection("hippocampai_facts")
# Collections will be recreated on next use
```

---

## API & Integration Issues

### Problem: CORS errors in browser

**Symptom:**
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solutions:**

1. **Configure CORS:**
```python
# api/async_app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Use proxy in development:**
```javascript
// package.json (React)
{
  "proxy": "http://localhost:8000"
}
```

---

### Problem: Authentication errors

**Symptom:**
```
401 Unauthorized
```

**Solutions:**

1. **Check authentication is disabled:**
```bash
# .env
ENABLE_AUTH=false  # For development
```

2. **Provide valid credentials:**
```python
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(url, headers=headers, json=data)
```

---

## Frequently Asked Questions

### General Questions

**Q: How much does HippocampAI cost?**
A: HippocampAI is free and open-source (Apache 2.0). You only pay for:
- Cloud hosting (if not self-hosted)
- LLM API calls (OpenAI, Groq, Anthropic)
- Infrastructure (Qdrant Cloud, Redis Cloud)

Local deployment with Ollama is completely free.

---

**Q: Can I use HippocampAI commercially?**
A: Yes! Apache 2.0 license allows commercial use.

---

**Q: What's the difference between local and remote mode?**
A:
- **Local mode**: Direct connection to Qdrant/Redis (5-15ms latency)
- **Remote mode**: HTTP API connection (20-50ms latency)

Use local for single applications, remote for microservices.

---

**Q: How many memories can HippocampAI handle?**
A: Tested with 1M+ memories per user. Performance depends on:
- Hardware resources
- Qdrant configuration
- Caching enabled

---

**Q: Can I migrate from another memory system?**
A: Yes, use the import/export functionality:
```python
# Export from old system to JSON
client.export_graph_to_json("memories.json")

# Import to HippocampAI
client.import_graph_from_json("memories.json")
```

---

### Technical Questions

**Q: Which embedding model should I use?**
A: Default (`BAAI/bge-small-en-v1.5`) is recommended. For alternatives:
- **Faster**: `all-MiniLM-L6-v2` (smallest, fastest)
- **Better quality**: `BAAI/bge-base-en-v1.5` (larger, slower)
- **Multilingual**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

---

**Q: Should I enable reranking?**
A: Yes for production (better accuracy). No for development/testing (faster).

---

**Q: How often should I consolidate memories?**
A:
- High volume: Daily
- Medium volume: Weekly
- Low volume: Monthly

---

**Q: Can I use HippocampAI without Redis?**
A: Yes, but performance will be significantly slower (50-100x for repeated queries).

---

**Q: What's the minimum hardware requirement?**
A:
- **CPU**: 2 cores
- **RAM**: 4GB (8GB recommended)
- **Disk**: 10GB+ for storage
- **GPU**: Optional (for faster embeddings)

---

**Q: Is my data secure?**
A: See [SECURITY.md](SECURITY.md) for best practices. Key points:
- Enable authentication
- Use HTTPS/TLS
- Encrypt data at rest
- Implement access control

---

**Q: How do I backup my data?**
A: See [BACKUP_RECOVERY.md](BACKUP_RECOVERY.md). Quick backup:
```bash
# Backup Qdrant
docker exec qdrant sh -c 'tar czf - /qdrant/storage' > qdrant_backup.tar.gz

# Export memories
python -c "from hippocampai import MemoryClient; MemoryClient().export_graph_to_json('backup.json')"
```

---

**Q: Can I use multiple LLM providers?**
A: Yes, switch providers per operation:
```python
from hippocampai.adapters import OllamaLLM, OpenAILLM

# Use Ollama for most operations (free)
client = MemoryClient(llm_provider=OllamaLLM())

# Use OpenAI for critical operations
openai_provider = OpenAILLM(api_key=key)
facts = client.extract_facts(text, llm_provider=openai_provider)
```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues**: https://github.com/rexdivakar/HippocampAI/issues
3. **Review documentation**: [docs/](../docs/)
4. **Enable debug logging**:
```bash
# .env
LOG_LEVEL=DEBUG
```

### Where to Get Help

1. **GitHub Issues**: For bugs and feature requests
   - https://github.com/rexdivakar/HippocampAI/issues

2. **GitHub Discussions**: For questions and community support
   - https://github.com/rexdivakar/HippocampAI/discussions

3. **Discord**: For real-time chat
   - https://discord.gg/pPSNW9J7gB

### Reporting Bugs

Include this information:
```
**Environment:**
- OS: Ubuntu 22.04
- Python: 3.11.5
- HippocampAI: v0.2.5
- Docker: 24.0.5

**Steps to reproduce:**
1. ...
2. ...

**Expected behavior:**
...

**Actual behavior:**
...

**Logs:**
```
[paste relevant logs]
```

**Configuration:**
```
[paste .env (redact secrets)]
```
```

---

**Version:** v0.2.5
**Last Updated:** November 2025
