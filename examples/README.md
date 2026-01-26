# HippocampAI Demo Scripts

This directory contains comprehensive demonstration scripts showcasing all HippocampAI features.

## Groq + HippocampAI Chat Demo

**File:** `groq_llama_chat_demo.py`

A fully-featured interactive chatbot demonstrating all HippocampAI capabilities integrated with Groq's LLM API.

### Features Tested

#### 1. Core CRUD Operations
- ✅ `remember()` - Store memories with metadata, tags, importance
- ✅ `recall()` - Semantic search and retrieval with relevance scoring
- ✅ `get_memory()` - Retrieve single memory by ID
- ✅ `get_memories()` - List all memories for a user
- ✅ `update_memory()` - Update existing memories
- ✅ `delete_memory()` - Remove memories

#### 2. Batch Operations
- ✅ `batch_remember()` - Bulk memory creation
- ✅ `batch_get_memories()` - Retrieve multiple memories at once
- ✅ `batch_delete_memories()` - Bulk deletion

#### 3. Advanced Filtering & Search
- ✅ Tag-based filtering
- ✅ Importance-based filtering
- ✅ Minimum score threshold filtering
- ✅ Combined filters (tags + importance + score)
- ✅ Semantic hybrid search (BM25 + vector)

#### 4. Entity & Fact Extraction
- ✅ `extract_entities=True` - Extract people, places, organizations
- ✅ `extract_facts=True` - Extract factual statements
- ✅ `extract_relationships=True` - Extract entity relationships

#### 5. Memory Lifecycle Management
- ✅ Memory expiration with `expires_at` parameter
- ✅ TTL (Time-To-Live) support
- ✅ `cleanup_expired_memories()` - Remove expired memories
- ✅ Importance decay over time

#### 6. Memory Consolidation
- ✅ `consolidate_memories()` - Merge similar/duplicate memories
- ✅ Configurable similarity thresholds
- ✅ Lookback time windows

#### 7. Analytics & Monitoring
- ✅ `get_memory_analytics()` - Memory statistics per user
- ✅ `health_check()` - System health status
- ✅ Memory count by type
- ✅ Total memory count

#### 8. Session Management
- ✅ Session-based conversation tracking
- ✅ Session IDs for organizing conversations
- ✅ Session-specific memory retrieval

#### 9. Automatic Memory Type Detection
- ✅ Facts (identity, personal info)
- ✅ Preferences (likes, dislikes)
- ✅ Goals (intentions, plans)
- ✅ Habits (routines, regular activities)
- ✅ Events (specific occurrences)
- ✅ Context (general conversation)

### Setup

#### Prerequisites

```bash
# Install dependencies
pip install groq hippocampai rich

# Set required environment variables
export GROQ_API_KEY="your_groq_api_key"

# Optional: For remote mode
export HIPPOCAMPAI_API_KEY="your_hippocampai_api_key"
```

#### Running the Demo

**Local Mode (Direct Qdrant/Redis connection):**
```bash
# Default (localhost)
python groq_llama_chat_demo.py

# Custom Qdrant URL
python groq_llama_chat_demo.py --qdrant-url http://100.113.229.40:6333

# Custom Redis URL
python groq_llama_chat_demo.py --redis-url redis://localhost:6379
```

**Remote Mode (via HippocampAI API):**
```bash
python groq_llama_chat_demo.py --base-url http://localhost:8000
```

**Custom User ID:**
```bash
python groq_llama_chat_demo.py --user-id my-test-user-123
```

**Custom Session ID (for reproducible testing):**
```bash
# Start a session with specific ID
python groq_llama_chat_demo.py --session-id test-session-1

# Continue the same session later
python groq_llama_chat_demo.py --session-id test-session-1

# All memories will be associated with this session
```

**Full Configuration:**
```bash
python groq_llama_chat_demo.py \
  --user-id alice \
  --session-id team-planning-2024 \
  --qdrant-url http://localhost:6333 \
  --redis-url redis://localhost:6379
```

### Interactive Commands

Once the chat is running, use these commands:

| Command | Description |
|---------|-------------|
| `/test` | Run comprehensive feature tests (all 9 test suites) |
| `/analytics` | Show memory analytics and statistics |
| `/health` | Check system health status |
| `/memories` | Display stored memories in a table |
| `/search` | Interactive memory search |
| `/info` | Show session information |
| `/clear` | Clear conversation history |
| `/help` | Show available commands |
| `/quit` | Exit the demo |

### Feature Testing

Run the comprehensive test suite by typing `/test` in the chat:

```
🧪 Running Comprehensive Feature Tests

Test 1: Basic Memory CRUD Operations
  ✅ PASS: Create, Read, Update, Delete

Test 2: Batch Operations
  ✅ PASS: Created, retrieved, deleted 3 memories

Test 3: Advanced Filtering & Search
  ✅ PASS: Retrieved X filtered memories

Test 4: Entity & Fact Extraction
  ✅ PASS: Entity extraction enabled

Test 5: Memory Expiration (TTL)
  ✅ PASS: Memory with expiration created

Test 6: Memory Consolidation
  ✅ PASS: Memory consolidation executed

Test 7: Cleanup Expired Memories
  ✅ PASS: Cleaned up X expired memories

Test 8: Get All Memories
  ✅ PASS: Retrieved X total memories

Test 9: System Health Check
  ✅ PASS: System health: healthy

📊 Test Summary
Total: 9/9 tests passed
```

### Example Conversation

```
You: Hello! My name is Alice and I love pizza.