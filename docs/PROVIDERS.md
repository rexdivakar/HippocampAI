## LLM Provider Configuration

HippocampAI supports multiple LLM providers: **Anthropic (Claude)**, **OpenAI (GPT)**, and **Groq**.

## Quick Setup

### 1. Choose Your Provider

Edit `.env` and set your preferred provider:

```env
# Options: anthropic, openai, groq
LLM_PROVIDER=anthropic
```

### 2. Add API Key

Add the API key for your chosen provider:

**For Anthropic (Claude):**
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**For OpenAI (GPT):**
```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o
```

**For Groq (Fast Inference):**
```env
GROQ_API_KEY=gsk_your-groq-key-here
GROQ_MODEL=llama-3.1-70b-versatile
```

## Provider Details

### Anthropic (Claude)

**Best for:** High-quality reasoning, long context, safety

**Models:**
- `claude-3-5-sonnet-20241022` - Best balance (recommended)
- `claude-3-opus-20240229` - Most capable
- `claude-3-haiku-20240307` - Fastest, cheapest

**Configuration:**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_TEMPERATURE=0.0
```

**Get API Key:** https://console.anthropic.com/

### OpenAI (GPT)

**Best for:** General purpose, wide ecosystem support

**Models:**
- `gpt-4o` - Latest multimodal (recommended)
- `gpt-4-turbo` - High quality, faster
- `gpt-3.5-turbo` - Fast, economical

**Configuration:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.0
```

**Get API Key:** https://platform.openai.com/

### Groq (Fast Inference)

**Best for:** Speed, cost efficiency, open models

**Models:**
- `llama-3.1-70b-versatile` - Best quality (recommended)
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Large context

**Configuration:**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-70b-versatile
GROQ_MAX_TOKENS=8192
GROQ_TEMPERATURE=0.0
```

**Get API Key:** https://console.groq.com/

## Usage in Code

### Automatic Provider Selection (from .env)

```python
from src.llm_provider import get_llm_client

# Uses LLM_PROVIDER from .env
client = get_llm_client()

# Generate text
response = client.generate("What is the capital of France?")
print(response)
```

### Explicit Provider Selection

```python
# Use specific provider
client = get_llm_client(provider="openai")

# Use specific model
client = get_llm_client(
    provider="groq",
    model="llama-3.1-8b-instant"
)

# Full customization
client = get_llm_client(
    provider="anthropic",
    model="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=8000
)
```

### In Memory Components

All AI-powered components support provider selection:

```python
from src.memory_extractor import MemoryExtractor
from src.importance_scorer import ImportanceScorer

# Use default provider from .env
extractor = MemoryExtractor()

# Use specific provider
extractor = MemoryExtractor(
    provider="openai",
    model="gpt-4o"
)

scorer = ImportanceScorer(
    provider="groq",
    model="llama-3.1-70b-versatile"
)
```

## Multi-Provider Setup

You can use different providers for different tasks:

```python
# Use GPT for extraction (fast)
extractor = MemoryExtractor(provider="openai")

# Use Claude for scoring (thoughtful)
scorer = ImportanceScorer(provider="anthropic")

# Use Groq for consolidation (economical)
consolidator = MemoryConsolidator(
    retriever=retriever,
    updater=updater,
    embedding_service=embeddings,
    provider="groq"
)
```

## Provider Comparison

| Feature | Anthropic | OpenAI | Groq |
|---------|-----------|--------|------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | $$$ | $$$ | $ |
| **Context** | 200K | 128K | 32-128K |
| **Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Best Practices

### 1. Provider Selection

**Use Anthropic when:**
- Quality and reasoning are critical
- Working with long documents
- Safety is paramount

**Use OpenAI when:**
- You need broad ecosystem support
- Multimodal capabilities required
- Balanced performance/cost

**Use Groq when:**
- Speed is critical
- Cost optimization needed
- Using open-source models

### 2. Model Selection

**For Production:**
```env
# Anthropic
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI
OPENAI_MODEL=gpt-4o

# Groq
GROQ_MODEL=llama-3.1-70b-versatile
```

**For Development/Testing:**
```env
# Cheaper, faster options
ANTHROPIC_MODEL=claude-3-haiku-20240307
OPENAI_MODEL=gpt-3.5-turbo
GROQ_MODEL=llama-3.1-8b-instant
```

### 3. Environment-Specific Config

**Production (.env.production):**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-prod-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_TEMPERATURE=0.0
```

**Development (.env.development):**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_dev-...
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.3
```

## Troubleshooting

### "API key required for provider 'xyz'"

Add the API key to `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

### "Unsupported provider"

Check spelling in `.env`:
```env
LLM_PROVIDER=anthropic  # ✓
LLM_PROVIDER=claude     # ✗
```

Valid options: `anthropic`, `openai`, `groq`

### Rate Limit Errors

Adjust rate limiting:
```env
RATE_LIMIT_INTERVAL_MS=200  # Slower requests
MAX_RETRIES=3               # More retries
```

### Model Not Found

Check model name for your provider:

**Anthropic:**
- `claude-3-5-sonnet-20241022` ✓
- `claude-3.5-sonnet` ✗

**OpenAI:**
- `gpt-4o` ✓
- `gpt4` ✗

**Groq:**
- `llama-3.1-70b-versatile` ✓
- `llama3` ✗

## Testing Different Providers

```python
# Test script
from src.llm_provider import get_llm_client

providers = ["anthropic", "openai", "groq"]

for provider in providers:
    try:
        client = get_llm_client(provider=provider)
        response = client.generate("Say hello in French")
        print(f"{provider}: {response}")
    except Exception as e:
        print(f"{provider}: ERROR - {e}")
```

## Cost Optimization

**Mix providers based on task:**

```python
# Expensive but accurate: use Claude
importance_scorer = ImportanceScorer(provider="anthropic")

# Fast and cheap: use Groq
session_summarizer = SessionManager(
    memory_store=store,
    retriever=retriever,
    embedding_service=embeddings,
    provider="groq"
)
```

**Environment-based selection:**

```python
import os

provider = "anthropic" if os.getenv("ENV") == "production" else "groq"
client = get_llm_client(provider=provider)
```

## Advanced Configuration

### Custom Provider Settings

```python
from src.llm_provider import LLMClientFactory

client = LLMClientFactory.create_client(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.1
)
```

### Provider-Specific Features

**Anthropic - System Prompts:**
```python
# Anthropic supports system messages
client = get_llm_client(provider="anthropic")
response = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ]
)
```

**OpenAI - Function Calling:**
```python
client = get_llm_client(provider="openai")
# OpenAI supports function calling (not shown here)
```

## Reference

### Environment Variables

| Variable | Provider | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | All | `anthropic` | Active provider |
| `ANTHROPIC_API_KEY` | Anthropic | - | Claude API key |
| `ANTHROPIC_MODEL` | Anthropic | `claude-3-5-sonnet-20241022` | Model name |
| `OPENAI_API_KEY` | OpenAI | - | OpenAI API key |
| `OPENAI_MODEL` | OpenAI | `gpt-4o` | Model name |
| `GROQ_API_KEY` | Groq | - | Groq API key |
| `GROQ_MODEL` | Groq | `llama-3.1-70b-versatile` | Model name |

### Supported Models

See provider documentation:
- Anthropic: https://docs.anthropic.com/
- OpenAI: https://platform.openai.com/docs/models
- Groq: https://console.groq.com/docs/models
