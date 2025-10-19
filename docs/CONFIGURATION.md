## Configuration Guide

HippocampAI uses a multi-layered configuration system combining environment variables and YAML files for maximum flexibility.

## Configuration Files

### 1. `.env` - Environment Variables

Create from template:
```bash
cp .env .env
```

**Required Settings:**
```env
# Claude API (required for AI features)
ANTHROPIC_API_KEY=your_api_key_here

# Qdrant Database
QDRANT_HOST=192.168.1.120
QDRANT_PORT=6334
```

**Optional Settings:**
```env
# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_CACHE_SIZE=1000

# Claude Model
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_MAX_TOKENS=4096

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Memory Settings
SIMILARITY_THRESHOLD=0.88
CONSOLIDATION_THRESHOLD=0.85

# Performance
RATE_LIMIT_INTERVAL_MS=100
MAX_RETRIES=2
```

### 2. `config.yaml` - Application Settings

Detailed configuration for all components. Key sections:

**Collections:**
```yaml
collections:
  personal_facts:
    description: "Personal facts, preferences, and goals"
    vector_size: 384
    distance: "Cosine"
```

**Memory Management:**
```yaml
memory:
  importance:
    min_score: 1
    max_score: 10
    critical_threshold: 9

  decay_rates:
    preference: 0.001  # Very slow decay
    fact: 0.002        # Slow decay
    event: 0.01        # Very fast decay
```

**Smart Retrieval:**
```yaml
retrieval:
  smart_search:
    similarity_weight: 0.50
    importance_weight: 0.30
    recency_weight: 0.20
```

### 3. `logging_config.yaml` - Logging Setup

Controls log levels, formats, and outputs:

```yaml
handlers:
  console:
    level: INFO
    formatter: simple

  file:
    level: DEBUG
    filename: logs/hippocampai.log
    maxBytes: 10485760  # 10MB
```

## Configuration Hierarchy

Settings are loaded in this order (later overrides earlier):

1. Default values in code
2. `config.yaml` settings
3. `.env` file variables
4. System environment variables

## Using Settings in Code

```python
from src.settings import get_settings

# Get settings instance
settings = get_settings()

# Access settings
print(settings.qdrant.host)
print(settings.memory.similarity_threshold)
print(settings.retrieval.default_limit)

# Get all settings as dict
config_dict = settings.to_dict()
```

## Environment-Specific Configuration

### Development

```env
LOG_LEVEL=DEBUG
LOG_FORMAT=text
QDRANT_HOST=localhost
```

### Production

```env
LOG_LEVEL=INFO
LOG_FORMAT=json
QDRANT_HOST=qdrant.production.com
QDRANT_API_KEY=your_production_key
```

## Configuration Validation

Settings are validated on startup:

```python
from src.settings import Settings

try:
    settings = Settings()
    print("Configuration valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Common Configuration Scenarios

### 1. Use Cloud Qdrant

```env
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key
```

### 2. Adjust Memory Sensitivity

```env
# More aggressive deduplication
SIMILARITY_THRESHOLD=0.95

# More aggressive consolidation
CONSOLIDATION_THRESHOLD=0.90
```

### 3. Performance Tuning

```env
# Larger embedding cache
EMBEDDING_CACHE_SIZE=5000

# More search results
DEFAULT_SEARCH_LIMIT=20
MAX_SEARCH_LIMIT=100
```

### 4. Custom Logging

```env
LOG_LEVEL=DEBUG
LOG_FORMAT=json
LOG_FILE=custom_logs/app.log
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
Add your API key to `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
```

### "Qdrant connection failed"
Check:
1. Qdrant is running: `docker ps` or service status
2. Host/port correct in `.env`
3. Firewall allows connection

### "Configuration validation failed"
Run validation:
```bash
python -c "from src.settings import Settings; Settings()"
```

Check error messages for specific issues.

## Advanced Configuration

### Custom Config Files

```python
from src.settings import Settings

settings = Settings(
    env_file="custom.env",
    config_file="custom_config.yaml"
)
```

### Programmatic Configuration

```python
import os

os.environ["SIMILARITY_THRESHOLD"] = "0.90"
os.environ["LOG_LEVEL"] = "DEBUG"

from src.settings import reload_settings
settings = reload_settings()
```

### Multiple Environments

Use different `.env` files:

```bash
# Development
cp .env.development .env

# Production
cp .env.production .env
```

## Best Practices

1. **Never commit `.env`** - Use `.env` as template
2. **Use environment variables for secrets** - API keys, passwords
3. **Use config.yaml for application logic** - Thresholds, weights
4. **Validate after changes** - Run `python setup_initial.py`
5. **Document custom settings** - Add comments in config files

## Reference

### All Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ANTHROPIC_API_KEY` | string | - | Claude API key (required) |
| `QDRANT_HOST` | string | localhost | Qdrant server host |
| `QDRANT_PORT` | int | 6334 | Qdrant server port |
| `QDRANT_API_KEY` | string | - | Qdrant API key (cloud) |
| `EMBEDDING_MODEL` | string | all-MiniLM-L6-v2 | Sentence-transformers model |
| `EMBEDDING_CACHE_SIZE` | int | 1000 | Embedding cache size |
| `CLAUDE_MODEL` | string | claude-3-5-sonnet-20241022 | Claude model |
| `LOG_LEVEL` | string | INFO | Logging level |
| `SIMILARITY_THRESHOLD` | float | 0.88 | Duplicate detection threshold |
| `DEFAULT_SEARCH_LIMIT` | int | 10 | Default search results |

See `.env` for complete list.
