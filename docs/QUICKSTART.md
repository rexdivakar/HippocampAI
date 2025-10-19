# HippocampAI Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed
2. **Qdrant** running (Docker or standalone)
3. **Anthropic API key** (get from https://console.anthropic.com/)

## Step 1: Install Qdrant (if not already running)

### Option A: Docker (Recommended)

```bash
docker run -p 6334:6333 -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### Option B: Docker Compose

Create `docker-compose.qdrant.yaml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6334:6333"
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Run:
```bash
docker-compose -f docker-compose.qdrant.yaml up -d
```

### Option C: Standalone

Download from https://github.com/qdrant/qdrant/releases

## Step 2: Install HippocampAI

```bash
# Clone repository
git clone <your-repo-url>
cd HippocampAI

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure

```bash
# Run setup script
python setup.py
```

This creates:
- `.env` file from template
- `logs/`, `data/`, `backups/` directories
- Qdrant collections

**Edit `.env` and add your API key:**
```bash
# Open in your editor
nano .env  # or vim, code, etc.

# Add this:
ANTHROPIC_API_KEY=sk-ant-your-key-here

# If Qdrant is on localhost, change:
QDRANT_HOST=localhost
QDRANT_PORT=6334
```

## Step 4: Run Examples

### Example 1: Basic Memory Storage

```bash
python examples/memory_store_example.py
```

**What it does:**
- Stores sample memories
- Retrieves memories by ID
- Shows collection statistics

### Example 2: Memory Extraction from Conversation

```bash
python examples/memory_extractor_example.py
```

**What it does:**
- Takes a conversation
- Extracts memories using Claude AI
- Stores them in Qdrant

### Example 3: Smart Search & Retrieval

```bash
python examples/smart_retrieval_example.py
```

**What it does:**
- Demonstrates multi-factor ranking
- Shows session management
- Tests memory consolidation

### Example 4: Advanced Features

```bash
python examples/advanced_memory_example.py
```

**What it does:**
- Deduplication
- Conflict resolution
- Importance scoring
- Memory updates

## Step 5: Use in Your Code

### Basic Usage

```python
from src.qdrant_client import QdrantManager
from src.embedding_service import EmbeddingService
from src.memory_store import MemoryStore, MemoryType, Category

# Initialize
qdrant = QdrantManager(host="localhost", port=6334)
qdrant.create_collections()

embeddings = EmbeddingService()
store = MemoryStore(qdrant_manager=qdrant, embedding_service=embeddings)

# Store a memory
memory_id = store.store_memory(
    text="I prefer tea over coffee",
    memory_type=MemoryType.PREFERENCE.value,
    metadata={
        "user_id": "user_123",
        "importance": 7,
        "category": Category.PERSONAL.value,
        "session_id": "session_001",
        "confidence": 0.9
    }
)

print(f"Stored memory: {memory_id}")
```

### Search Memories

```python
from src.memory_retriever import MemoryRetriever

retriever = MemoryRetriever(qdrant_manager=qdrant, embedding_service=embeddings)

# Simple search
results = retriever.search_memories(
    query="What drinks do I like?",
    limit=5,
    filters={"user_id": "user_123"}
)

for result in results:
    print(f"[{result['similarity_score']:.2f}] {result['text']}")
```

### Smart Search with Ranking

```python
# Multi-factor ranking (similarity + importance + recency)
results = retriever.smart_search(
    query="What drinks do I like?",
    user_id="user_123",
    context_type="personal",
    limit=5
)

for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Breakdown: {result['score_breakdown']}")
```

### Extract Memories from Conversation

```python
from src.memory_extractor import MemoryExtractor

extractor = MemoryExtractor()

conversation = """
User: I really enjoy hiking on weekends
Assistant: That's great! What kind of trails do you prefer?
User: I like mountain trails, especially in the morning
"""

# Extract memories
memories = extractor.extract_memories(
    conversation_text=conversation,
    user_id="user_123",
    session_id="session_001"
)

# Store them
for memory in memories:
    store.store_memory(**memory)
```

### Session Management

```python
from src.session_manager import SessionManager

session_mgr = SessionManager(
    memory_store=store,
    retriever=retriever,
    embedding_service=embeddings
)

# Start session
session_id = session_mgr.start_session(user_id="user_123")

# Add messages
session_mgr.add_message(session_id, "user", "Hello!")
session_mgr.add_message(session_id, "assistant", "Hi! How can I help?")

# End session (auto-summarizes)
summary_id = session_mgr.end_session(session_id)
```

## Step 6: Interactive Python Session

```bash
python
```

```python
from src.settings import get_settings
from src.qdrant_client import QdrantManager
from src.embedding_service import EmbeddingService
from src.memory_store import MemoryStore, MemoryType, Category
from src.memory_retriever import MemoryRetriever

# Setup
settings = get_settings()
print(f"Connected to: {settings.qdrant.host}:{settings.qdrant.port}")

qdrant = QdrantManager()
qdrant.create_collections()

embeddings = EmbeddingService()
store = MemoryStore(qdrant, embeddings)
retriever = MemoryRetriever(qdrant, embeddings)

# Now use store and retriever interactively
store.store_memory(
    text="I work as a software engineer",
    memory_type="fact",
    metadata={
        "user_id": "me",
        "importance": 9,
        "category": "work",
        "session_id": "interactive",
        "confidence": 1.0
    }
)

# Search
results = retriever.search_memories("my job", filters={"user_id": "me"})
for r in results:
    print(r['text'])
```

## Common Issues

### "ANTHROPIC_API_KEY not set"
```bash
# Add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env
```

### "Qdrant connection failed"
```bash
# Check Qdrant is running
docker ps | grep qdrant

# Or check logs
docker logs <container-id>

# Make sure host/port in .env match:
QDRANT_HOST=localhost  # or 192.168.1.120
QDRANT_PORT=6334
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install individually
pip install qdrant-client sentence-transformers anthropic pyyaml python-json-logger
```

### "Permission denied" on logs/
```bash
# Create logs directory
mkdir -p logs
chmod 755 logs
```

## Next Steps

1. **Read the docs:**
   - [Configuration Guide](docs/CONFIGURATION.md)
   - [Architecture Overview](README.md)

2. **Customize:**
   - Edit `config.yaml` for application settings
   - Adjust `logging_config.yaml` for log verbosity
   - Modify `.env` for environment-specific settings

3. **Build your application:**
   - See examples for patterns
   - Combine components as needed
   - Add your custom logic

## Tips

- **Start with examples** to understand the flow
- **Use smart_search** for better relevance
- **Enable consolidation** to reduce memory bloat
- **Monitor logs/** directory for debugging
- **Backup qdrant_storage/** regularly

## Getting Help

- Check logs: `tail -f logs/hippocampai.log`
- Run setup again: `python setup.py`
- Validate config: `python -c "from src.settings import Settings; Settings()"`

Happy memory management! 🧠✨
