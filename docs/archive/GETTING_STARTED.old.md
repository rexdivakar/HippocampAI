# Getting Started with HippocampAI Package

## Prerequisites

1. **Qdrant** (vector database)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

2. **Ollama** (optional, for local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5:7b-instruct
```

## Installation

**Option A: Install from PyPI (Recommended)**

```bash
pip install hippocampai
```

View on PyPI: [https://pypi.org/project/hippocampai/](https://pypi.org/project/hippocampai/)

**Option B: Install from source**

```bash
cd /Users/rexdivakar/workspace/HippocampAI
pip install -e .
```

## Quick Test

### 1. Initialize

```bash
hippocampai init
```

Expected output:

```
✓ HippocampAI initialized successfully
  Qdrant: http://localhost:6333
  Collections: hippocampai_facts, hippocampai_prefs
```

### 2. Store Memories (CLI)

```bash
# Via CLI
hippocampai remember --user alice --text "I love pizza and sushi" --type preference
hippocampai remember --user alice --text "I live in San Francisco" --type fact

# Via stdin
echo "I prefer working in the afternoon" | hippocampai remember --user alice --type preference
```

### 3. Recall Memories (CLI)

```bash
hippocampai recall --user alice --query "what food does alice like?" -k 3
hippocampai recall --user alice --query "where does alice live?" -k 3
hippocampai recall --user alice --query "when does alice work?" -k 3
```

You should see a table with scores and breakdowns!

### 4. Python API

```python
from hippocampai import MemoryClient

# Initialize client
client = MemoryClient(
    llm_provider="ollama",
    embed_quantized=False  # Set to True for int8 quantization
)

# Store
client.remember(
    "I prefer dark mode in all my apps",
    user_id="bob",
    type="preference"
)

# Recall
results = client.recall(
    "what UI preferences does bob have?",
    user_id="bob",
    k=5
)

for r in results:
    print(f"[{r.score:.3f}] {r.memory.text}")
    print(f"  sim={r.breakdown['sim']:.2f}, rerank={r.breakdown['rerank']:.2f}")
```

### 5. FastAPI Server

```bash
# Terminal 1: Start server
hippocampai api --port 8000

# Terminal 2: Test API
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am allergic to peanuts",
    "user_id": "charlie",
    "type": "fact",
    "importance": 9.5
  }'

curl -X POST http://localhost:8000/v1/memories:recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what allergies does charlie have?",
    "user_id": "charlie",
    "k": 5
  }' | jq .
```

### 6. Extract from Conversation

```python
conversation = """
User: I'm planning a trip to Japan next month.
AI: That sounds exciting!
User: Yeah, I love Japanese food and culture.
User: I've been learning Japanese for 2 years.
"""

memories = client.extract_from_conversation(
    conversation=conversation,
    user_id="david",
    session_id="trip_planning"
)

for mem in memories:
    print(f"{mem.type}: {mem.text} (importance={mem.importance:.1f})")
```

## Running Tests

```bash
pytest -v tests/
```

Expected output:

```
test_retrieval.py::test_rrf_fusion PASSED
test_retrieval.py::test_normalize PASSED
test_retrieval.py::test_recency_score PASSED
test_retrieval.py::test_fuse_scores PASSED
```

## Configuration

Create `.env` file:

```env
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
EMBED_QUANTIZED=false
TOP_K_QDRANT=200
TOP_K_FINAL=20
```

## Troubleshooting

### Qdrant Connection Error

```bash
# Check if Qdrant is running
curl http://localhost:6333

# Restart Qdrant
docker restart <qdrant-container-id>
```

### Ollama Not Found

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull model
ollama pull qwen2.5:7b-instruct
```

### Module Not Found

```bash
# Reinstall in editable mode
pip install -e .
```

## What's Next?

1. Check `README.md` for full documentation
2. See `FEATURES.md` for complete feature guide
3. Explore `src/hippocampai/` for implementation
4. Read `CHANGELOG.md` for version history

Enjoy your new production-ready memory engine! 🚀
