# Quick Start Guide

Get HippocampAI running in 5 minutes!

## Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- Git

## Step-by-Step Setup

### 1. Install HippocampAI

**Option A: Install from PyPI (Recommended)**

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install HippocampAI
pip install hippocampai
```

View on PyPI: [https://pypi.org/project/hippocampai/](https://pypi.org/project/hippocampai/)

**Option B: Install from source**

```bash
# Clone repository
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install HippocampAI
pip install -e .
```

### 2. Start Qdrant

```bash
# Using Docker
docker run -d -p 6333:6333 qdrant/qdrant

# Verify it's running
curl http://localhost:6333/health
```

### 3. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your favorite editor
```

**Minimal .env configuration:**

```bash
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
```

### 4. Initialize Collections

```bash
python -c "from hippocampai import MemoryClient; MemoryClient()"
```

You should see:

```
Connected to Qdrant at http://localhost:6333
Created collection: hippocampai_facts
Created collection: hippocampai_prefs
```

## Your First Memory

Create a file `test_memory.py`:

```python
from hippocampai import MemoryClient

# Initialize
client = MemoryClient()

# Store a memory
memory = client.remember(
    text="I love hiking in the mountains",
    user_id="test_user",
    type="preference",
    importance=8.0
)

print(f"✓ Stored memory: {memory.id}")

# Recall it
results = client.recall(
    query="What outdoor activities does the user enjoy?",
    user_id="test_user",
    k=3
)

print(f"\n✓ Found {len(results)} relevant memories:")
for result in results:
    print(f"  - {result.memory.text} (score: {result.score:.3f})")
```

Run it:

```bash
python test_memory.py
```

Expected output:

```
✓ Stored memory: f47ac10b-58cc-4372-a567-0e02b2c3d479
✓ Found 1 relevant memories:
  - I love hiking in the mountains (score: 0.852)
```

## Try the CLI

```bash
python cli_chat.py my_username
```

Type some messages and watch memories being created!

## Try the Web Interface

```bash
# Install web dependencies
pip install -e ".[web]"

# Start server
python web_chat.py
```

Open <http://localhost:5000> in your browser.

## Run Examples

```bash
# Run all examples interactively
./run_examples.sh

# Or run individually
python examples/01_basic_usage.py
python examples/02_conversation_extraction.py
python examples/03_hybrid_retrieval.py
```

## Next Steps

- **Configure LLM Provider**: See [PROVIDERS.md](PROVIDERS.md)
- **Customize Configuration**: See [CONFIGURATION.md](CONFIGURATION.md)
- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)
- **Complete Features Guide**: See [FEATURES.md](FEATURES.md)

## Troubleshooting

### Qdrant Connection Error

```
ERROR: Failed to connect to Qdrant
```

**Solution**: Make sure Qdrant is running:

```bash
docker ps | grep qdrant
```

### Import Error

```
ModuleNotFoundError: No module named 'qdrant_client'
```

**Solution**: Install dependencies:

```bash
pip install -e .
```

### LLM Provider Error

```
ERROR: LLM_PROVIDER not configured
```

**Solution**: Set up .env file with valid provider settings (see step 3)

## Getting Help

- **Discord**: <https://discord.gg/pPSNW9J7gB>
- **Issues**: <https://github.com/rexdivakar/HippocampAI/issues>

## What's Next?

Now that you have HippocampAI running:

1. **Explore Examples** - Run through all 5 examples to see different features
2. **Read Architecture** - Understand how hybrid retrieval works
3. **Try Telemetry** - Monitor your memory operations
4. **Customize** - Tune retrieval weights, decay settings, etc.
5. **Integrate** - Add HippocampAI to your AI agent!

Happy building! 🧠
