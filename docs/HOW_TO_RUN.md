# How to Run HippocampAI - Step by Step Guide

## 📋 Prerequisites Checklist

Before you start, make sure you have:

- [ ] Python 3.8 or higher installed
- [ ] pip (Python package manager)
- [ ] Docker installed (for Qdrant) OR Qdrant running somewhere
- [ ] At least ONE of these API keys:
  - [ ] Anthropic API key (Claude) - https://console.anthropic.com/
  - [ ] OpenAI API key (GPT) - https://platform.openai.com/
  - [ ] Groq API key - https://console.groq.com/

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

**What this installs:**
- Qdrant client (vector database)
- Sentence transformers (embeddings)
- LLM providers (Anthropic, OpenAI, Groq)
- Configuration tools (YAML, logging)

### Step 2: Start Qdrant Database

**Option A: Docker (Recommended)**
```bash
docker run -d -p 6334:6333 -p 6333:6333 \
  --name qdrant \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option B: Docker Compose**
```bash
# Create docker-compose.yml with Qdrant
docker-compose up -d qdrant
```

**Option C: Use Existing Qdrant**
If you already have Qdrant running, just note the host and port.

**Verify Qdrant is running:**
```bash
curl http://localhost:6334/collections
# Should return: {"result":{"collections":[]}}
```

### Step 3: Run Setup Script

```bash
python setup_initial.py
```

**This will:**
- ✅ Check Python version
- ✅ Verify dependencies installed
- ✅ Create necessary directories (logs/, data/, backups/)
- ✅ Create `.env` from `.env`
- ✅ Test Qdrant connection
- ✅ Initialize vector collections

**Expected output:**
```
========================================
  HippocampAI Memory Assistant - Setup
========================================

[1/7] Checking Python version...
✓ Python 3.10.0

[2/7] Checking dependencies...
✓ qdrant_client installed
✓ sentence_transformers installed
...

✓ Setup completed successfully!
```

### Step 4: Configure API Key

**Edit the `.env` file:**
```bash
nano .env
# or
vim .env
# or
code .env
```

**Add your API key (choose ONE):**

```env
# Option 1: Use Anthropic (Claude) - Best Quality
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# Option 2: Use OpenAI (GPT) - Balanced
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Option 3: Use Groq - Fastest & Cheapest
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-actual-groq-key-here
```

**If Qdrant is not on localhost, also update:**
```env
QDRANT_HOST=your-qdrant-host
QDRANT_PORT=6334
```

### Step 5: Run Examples

**Option A: Interactive Menu (Best for Beginners)**
```bash
./run_example.sh
```

You'll see:
```
========================================
  HippocampAI Example Runner
========================================

Available examples:

  1) Basic Memory Storage
  2) Memory Extraction (requires API key)
  3) Memory Retrieval & Search
  4) Advanced Features (deduplication, updates)
  5) Smart Retrieval & Sessions (requires API key)
  6) All Examples (in sequence)
  7) Setup & Configuration Test
  0) Exit

Select example (0-7):
```

**Option B: Run Specific Example**
```bash
# Basic (no API key needed)
python examples/memory_store_example.py

# AI-powered (API key required)
python examples/memory_extractor_example.py
```

**Option C: Run All Examples**
```bash
./run_example.sh --all
```

---

## 📖 Detailed Usage

### Test Your Setup

**1. Verify Configuration:**
```bash
python -c "from hippocampai.settings import get_settings; s = get_settings(); print(f'Provider: {s.llm.provider}'); print(f'Qdrant: {s.qdrant.host}:{s.qdrant.port}')"
```

**2. Test LLM Provider:**
```bash
python examples/provider_comparison.py
```

**3. Check Qdrant Collections:**
```bash
python -c "from hippocampai.qdrant_client import QdrantManager; q = QdrantManager(); print(q.list_collections())"
```

### Basic Usage in Python

**Start Python interactive session:**
```bash
python
```

**Run this code:**
```python
from hippocampai.qdrant_client import QdrantManager
from hippocampai.embedding_service import EmbeddingService
from hippocampai.memory_store import MemoryStore, MemoryType, Category
from hippocampai.memory_retriever import MemoryRetriever

# 1. Setup
qdrant = QdrantManager()  # Uses settings from .env
qdrant.create_collections()

embeddings = EmbeddingService()
store = MemoryStore(qdrant, embeddings)
retriever = MemoryRetriever(qdrant, embeddings)

# 2. Store a memory
memory_id = store.store_memory(
    text="I prefer working in the morning",
    memory_type=MemoryType.PREFERENCE.value,
    metadata={
        "user_id": "test_user",
        "importance": 7,
        "category": Category.PERSONAL.value,
        "session_id": "test_session",
        "confidence": 0.9
    }
)

print(f"Stored: {memory_id}")

# 3. Search memories
results = retriever.search_memories(
    query="work preferences",
    filters={"user_id": "test_user"}
)

for r in results:
    print(f"[{r['similarity_score']:.2f}] {r['text']}")

# 4. Close connection
qdrant.close()
```

### Running Examples

**1. Embedding Service:**
```bash
python examples/embedding_example.py
```
- Tests embedding generation
- Shows caching in action
- No API key required

**2. Memory Storage:**
```bash
python examples/memory_store_example.py
```
- Store memories with metadata
- Batch storage
- Retrieve by ID
- No API key required

**3. Memory Retrieval:**
```bash
python examples/memory_retriever_example.py
```
- Vector similarity search
- Filtering by metadata
- Recent and important memories
- No API key required

**4. AI Memory Extraction:**
```bash
python examples/memory_extractor_example.py
```
- Extract memories from conversations
- AI classification
- Automatic importance scoring
- **Requires API key**

**5. Advanced Features:**
```bash
python examples/advanced_memory_example.py
```
- Deduplication
- Conflict resolution
- Memory updates
- Importance scoring
- **Requires API key**

**6. Smart Retrieval:**
```bash
python examples/smart_retrieval_example.py
```
- Multi-factor ranking
- Session management
- Memory consolidation
- **Requires API key**

**7. Provider Comparison:**
```bash
python examples/provider_comparison.py
```
- Test all providers
- Speed comparison
- Shows which API keys work

---

## 🔧 Troubleshooting

### Problem: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Qdrant connection failed"

**Check if Qdrant is running:**
```bash
docker ps | grep qdrant
```

**If not running, start it:**
```bash
docker run -d -p 6334:6333 --name qdrant qdrant/qdrant
```

**Check connection:**
```bash
curl http://localhost:6334/collections
```

**Update .env if Qdrant is elsewhere:**
```env
QDRANT_HOST=192.168.1.120  # Your Qdrant host
QDRANT_PORT=6334
```

### Problem: "API key required for provider"

**Check which provider is active:**
```bash
grep LLM_PROVIDER .env
```

**Make sure corresponding API key is set:**
```bash
# For anthropic
grep ANTHROPIC_API_KEY .env

# For openai
grep OPENAI_API_KEY .env

# For groq
grep GROQ_API_KEY .env
```

**Add the key to .env:**
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Problem: "./run_example.sh: Permission denied"

**Make it executable:**
```bash
chmod +x run_example.sh
./run_example.sh
```

### Problem: "Port 6334 already in use"

**Find what's using it:**
```bash
lsof -i :6334
```

**Kill the process or use different port:**
```env
QDRANT_PORT=6335
```

### Problem: Examples fail with API errors

**Check logs:**
```bash
tail -f logs/hippocampai.log
```

**Verify API key format:**
- Anthropic: `sk-ant-...`
- OpenAI: `sk-...`
- Groq: `gsk_...`

**Test provider directly:**
```python
from hippocampai.llm_provider import get_llm_client

client = get_llm_client()
response = client.generate("Say hello")
print(response)
```

---

## 📚 Next Steps

### 1. Customize Configuration

**Edit `config/config.yaml` for:**
- Memory decay rates
- Importance thresholds
- Search weights
- Collection settings

**Edit `.env` for:**
- Provider selection
- API keys
- Qdrant connection
- Log levels

### 2. Build Your Application

**Basic pattern:**
```python
from hippocampai.settings import get_settings
from hippocampai.qdrant_client import QdrantManager
from hippocampai.embedding_service import EmbeddingService
from hippocampai.memory_store import MemoryStore
from hippocampai.memory_retriever import MemoryRetriever
from hippocampai.memory_extractor import MemoryExtractor

# Initialize
settings = get_settings()
qdrant = QdrantManager()
embeddings = EmbeddingService()
store = MemoryStore(qdrant, embeddings)
retriever = MemoryRetriever(qdrant, embeddings)
extractor = MemoryExtractor()

# Your logic here...
```

### 3. Switch Providers

**Try different providers for different tasks:**

```python
# Use Claude for quality
from hippocampai.memory_extractor import MemoryExtractor
extractor = MemoryExtractor(provider="anthropic")

# Use Groq for speed
from hippocampai.importance_scorer import ImportanceScorer
scorer = ImportanceScorer(provider="groq")

# Use OpenAI for balance
from hippocampai.session_manager import SessionManager
session_mgr = SessionManager(
    memory_store=store,
    retriever=retriever,
    embedding_service=embeddings,
    provider="openai"
)
```

### 4. Read Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[CONFIGURATION.md](docs/CONFIGURATION.md)** - Configuration reference
- **[PROVIDERS.md](docs/PROVIDERS.md)** - LLM provider guide
- **[README.md](README.md)** - Full documentation

---

## 🎯 Common Use Cases

### Use Case 1: Extract Memories from Conversation

```bash
python examples/memory_extractor_example.py
```

### Use Case 2: Search User's Memories

```python
from hippocampai.memory_retriever import MemoryRetriever

retriever = MemoryRetriever(qdrant, embeddings)

results = retriever.smart_search(
    query="What does the user like?",
    user_id="user_123",
    context_type="personal"
)

for r in results:
    print(r['text'])
```

### Use Case 3: Session Management

```python
from hippocampai.session_manager import SessionManager

session_mgr = SessionManager(store, retriever, embeddings)

# Start session
session_id = session_mgr.start_session("user_123")

# Add messages
session_mgr.add_message(session_id, "user", "Hello")
session_mgr.add_message(session_id, "assistant", "Hi!")

# End and summarize
summary_id = session_mgr.end_session(session_id)
```

### Use Case 4: Prevent Duplicates

```python
from hippocampai.memory_deduplicator import MemoryDeduplicator

deduplicator = MemoryDeduplicator(retriever, embeddings)

result = deduplicator.process_new_memory(
    new_memory={"text": "I like coffee", "memory_type": "preference", "importance": 7},
    user_id="user_123",
    auto_decide=True
)

if result['action'] == 'skip':
    print("Duplicate found, skipping")
elif result['action'] == 'store':
    store.store_memory(...)
```

---

## ✅ Quick Checklist

Before running examples, verify:

- [ ] Python 3.8+ installed: `python --version`
- [ ] Dependencies installed: `pip list | grep qdrant`
- [ ] Qdrant running: `curl http://localhost:6334/collections`
- [ ] `.env` file exists: `ls -la .env`
- [ ] API key set: `grep API_KEY .env`
- [ ] Setup completed: `python setup_initial.py`
- [ ] Collections created: Check setup output

Then run:
```bash
./run_example.sh
```

---

## 🆘 Getting Help

**Check logs:**
```bash
tail -f logs/hippocampai.log
tail -f logs/hippocampai_error.log
```

**Validate setup:**
```bash
python setup_initial.py
```

**Test configuration:**
```bash
python -c "from hippocampai.settings import Settings; Settings()"
```

**Run diagnostics:**
```bash
./run_example.sh --check
```

**Still stuck?** Check the individual example files for detailed comments and explanations.

---

## 🎉 Success!

If you can run `./run_example.sh` and see the menu, **you're all set!**

Start with option `1) Basic Memory Storage` to verify everything works, then explore the AI-powered features.

Happy memory management! 🧠✨
