# HippocampAI Core Architecture & Memory Management

## Overview
HippocampAI is a production-ready memory management system that provides intelligent storage, retrieval, and organization of conversational and contextual memories. Here's a comprehensive breakdown of how it works internally.

## 🧠 Core Memory Architecture

### Memory Model Structure
```python
class Memory(BaseModel):
    id: str                          # Unique UUID for each memory
    text: str                        # The actual memory content
    user_id: str                     # User isolation/multi-tenancy
    session_id: Optional[str]        # Session-based grouping
    type: MemoryType                 # PREFERENCE, FACT, GOAL, HABIT, EVENT, CONTEXT
    importance: float                # 0.0-10.0 scoring for prioritization
    confidence: float                # 0.0-1.0 confidence in memory accuracy
    tags: list[str]                  # Semantic tags for categorization
    created_at: datetime             # Creation timestamp
    updated_at: datetime             # Last update timestamp
    expires_at: Optional[datetime]   # TTL support for temporary memories
    access_count: int                # Usage frequency tracking
    metadata: dict[str, Any]         # Flexible additional data
    
    # Advanced Features
    agent_id: Optional[str]          # Multi-agent support
    run_id: Optional[str]            # Execution context tracking
    visibility: Optional[str]        # Privacy controls
    entities: Optional[dict]         # Named entity extraction
    relationships: Optional[list]    # Graph-based connections
    embedding: Optional[list[float]] # Vector representation
```

## 🗄️ Storage Architecture

### Multi-Collection Storage Strategy
HippocampAI uses a **dual-collection approach** for optimized performance:

1. **Facts Collection** (`hippocampai_facts`)
   - Stores: FACT, EVENT, CONTEXT memories
   - Optimized for: Factual information, temporal events
   - Use case: "Paris is the capital of France", "Meeting at 3pm"

2. **Preferences Collection** (`hippocampai_prefs`)  
   - Stores: PREFERENCE, GOAL, HABIT memories
   - Optimized for: User preferences, behavioral patterns
   - Use case: "I love Italian food", "Exercise daily at 6am"

### Vector Storage (Qdrant)
```python
# Collection Configuration
VectorParams(
    size=384,                    # Embedding dimension (BGE-small-en-v1.5)
    distance=Distance.COSINE     # Similarity metric
)

# HNSW Optimization
HnswConfig(
    m=48,                       # Graph connectivity (higher = better recall)
    ef_construct=256,           # Build-time accuracy
    ef_search=128              # Search-time accuracy  
)

# Performance Indices (5-10x faster filtered queries)
PayloadIndex("user_id", KEYWORD)      # User isolation
PayloadIndex("type", KEYWORD)         # Memory type filtering
PayloadIndex("tags", KEYWORD)         # Tag-based search
PayloadIndex("importance", FLOAT)     # Importance range queries
PayloadIndex("created_at", DATETIME)  # Time-based filtering
```

## 🔍 Retrieval System (Hybrid Search)

### Multi-Stage Retrieval Pipeline

1. **Query Routing**
   ```python
   # Automatic collection selection based on query intent
   collections = router.route_query(query)
   # Returns: ["facts"], ["prefs"], or ["facts", "prefs"]
   ```

2. **Parallel Search Strategies**
   ```python
   # Vector Search (Semantic)
   vector_results = qdrant.search(
       vector=embedder.encode(query),
       limit=200,
       filters={"user_id": user_id}
   )
   
   # BM25 Search (Keyword)  
   bm25_results = bm25_retriever.search(query, k=200)
   ```

3. **Reciprocal Rank Fusion (RRF)**
   ```python
   # Combines vector + BM25 results
   fused_results = reciprocal_rank_fusion(
       [vector_results, bm25_results], 
       k=60  # RRF parameter
   )
   ```

4. **Cross-Encoder Reranking**
   ```python
   # Semantic reranking for final accuracy
   reranked = reranker.rerank(
       query=query,
       documents=[r.text for r in candidates],
       top_k=20
   )
   ```

5. **Score Fusion**
   ```python
   # Weighted combination of multiple signals
   final_score = (
       0.55 * similarity_score +      # Vector similarity
       0.20 * rerank_score +         # Cross-encoder score  
       0.15 * recency_score +        # Time-based decay
       0.10 * importance_score       # User-defined importance
   )
   ```

## 🔄 Duplicate Detection & Memory Management

### Automatic Deduplication System
```python
class MemoryDeduplicator:
    def check_duplicate(self, new_memory: Memory, user_id: str):
        # 1. Vector search for similar memories
        candidates = qdrant.search(
            vector=embedder.encode(new_memory.text),
            filters={"user_id": user_id},
            threshold=0.88  # Configurable similarity threshold
        )
        
        # 2. Decision logic
        if similarity > 0.95:
            return ("skip", duplicate_ids)      # Exact duplicate
        elif similarity > 0.88:
            return ("update", duplicate_ids)    # Similar - merge/update
        else:
            return ("store", [])                # New memory
```

### Smart Memory Updates
```python
class SmartMemoryUpdater:
    def should_update_memory(self, existing: Memory, new_text: str):
        similarity = calculate_similarity(existing.text, new_text)
        
        if similarity > 0.95:
            # Nearly identical - skip with confidence boost
            return UpdateDecision(
                action="skip",
                confidence_adjustment=0.1  # Reinforce existing memory
            )
        
        elif similarity > 0.85:
            if len(new_text) > len(existing.text) * 1.2:
                # New memory has significantly more content
                return UpdateDecision(action="merge")
            else:
                return UpdateDecision(action="skip")
        
        elif similarity > 0.6:
            # Related but different - check for conflicts
            if detect_conflict(existing.text, new_text):
                return resolve_conflict(existing, new_text)
            else:
                return UpdateDecision(action="keep_both")
```

## 🎯 Core Features Deep Dive

### 1. Session Management
```python
class SessionManager:
    # Automatic session boundary detection
    def detect_session_boundary(self, messages):
        # Uses LLM to analyze topic changes
        # Factors: time gaps, topic shifts, explicit markers
        
    # Smart session summarization  
    def summarize_session(self, session):
        # Extract key facts and entities
        # Generate concise summaries
        # Maintain conversation context
```

### 2. Semantic Clustering
```python
class SemanticCategorizer:
    def enrich_memory_with_categories(self, memory):
        # Automatic type classification
        memory.type = self.classify_memory_type(memory.text)
        
        # Tag generation
        memory.tags = self.suggest_tags(memory.text)
        
        # Entity extraction
        memory.entities = self.extract_entities(memory.text)
        
        return memory
```

### 3. Multi-User Isolation
```python
# Every operation includes user filtering
filters = {
    "user_id": user_id,           # Strict user isolation
    "session_id": session_id      # Optional session scoping
}

# Session isolation security check
def validate_session_access(user_id: str, session_id: str):
    # Ensures users can only access their own sessions
    session = get_session(session_id)
    if session.user_id != user_id:
        raise PermissionError("Access denied")
```

### 4. Temporal Reasoning
```python
# Time-based memory scoring
def calculate_recency_score(memory: Memory, half_life_days: int):
    age_days = (datetime.now() - memory.created_at).days
    return 2 ** (-age_days / half_life_days)

# Memory type specific decay rates
HALF_LIVES = {
    "preference": 90,  # User preferences decay slowly
    "goal": 90,        # Goals remain relevant longer  
    "fact": 30,        # Facts may become outdated
    "event": 14,       # Events become less relevant quickly
}
```

### 5. Graph-Based Relationships
```python
# Entity relationship extraction
memory.entities = {
    "people": ["Alice", "Bob"],
    "places": ["Paris", "France"], 
    "organizations": ["OpenAI"]
}

memory.relationships = [
    {"subject": "Alice", "predicate": "works_at", "object": "OpenAI"},
    {"subject": "Paris", "predicate": "capital_of", "object": "France"}
]
```

## 🚀 Unified Client Architecture

### Dual-Mode Operation
```python
class UnifiedMemoryClient:
    def __init__(self, mode: str = "local"):
        if mode == "local":
            # Direct connection to Qdrant/Redis/Ollama
            self._backend = LocalBackend()
            # 5-15ms latency, full feature set
            
        elif mode == "remote":
            # HTTP API connection to HippocampAI SaaS
            self._backend = RemoteBackend(api_url="http://localhost:8000")
            # 20-50ms latency, language-agnostic
    
    # Same API regardless of mode
    def remember(self, text: str, user_id: str) -> Memory:
        return self._backend.remember(text, user_id)
        
    def recall(self, query: str, user_id: str) -> List[Memory]:
        return self._backend.recall(query, user_id)
```

## 📊 Performance Characteristics

### Storage Performance
- **First memory**: ~500ms (model loading overhead)
- **Subsequent memories**: ~2.7s average
  - Semantic analysis: ~1.5s
  - Vector embedding: ~200ms  
  - Duplicate check: ~300ms
  - Storage: ~100ms
  - LLM enrichment: ~600ms

### Retrieval Performance  
- **Vector search**: ~15ms
- **BM25 search**: ~5ms
- **RRF fusion**: ~2ms
- **Cross-encoder rerank**: ~15ms
- **Total retrieval**: ~40ms average

### Scalability
- **Users**: Unlimited (user_id isolation)
- **Memories per user**: 10,000+ (tested)
- **Concurrent requests**: 100+ (async design)
- **Storage**: Limited by Qdrant capacity

## 🔐 Security & Privacy

### Data Isolation
- **User-level**: Strict filtering on all operations
- **Session-level**: Optional session scoping
- **Agent-level**: Multi-agent isolation support

### Access Control
- **Memory visibility**: private/shared/public
- **Session validation**: User ownership checks
- **API authentication**: Token-based access (remote mode)

### Data Retention
- **TTL support**: Automatic expiration
- **Manual deletion**: Full GDPR compliance
- **Audit logging**: All operations tracked

## 🛠️ Configuration & Tuning

### Performance Tuning
```python
Config(
    # Vector search optimization
    hnsw_m=64,              # Higher = better recall, slower build
    ef_construction=512,    # Higher = better index quality
    ef_search=256,         # Higher = better accuracy, slower search
    
    # Retrieval tuning  
    top_k_qdrant=500,      # More candidates = better recall
    top_k_final=50,        # Final result count
    
    # Scoring weights
    weight_sim=0.50,       # Vector similarity importance
    weight_rerank=0.30,    # Cross-encoder importance
    weight_recency=0.15,   # Time decay importance
    weight_importance=0.05 # User importance scoring
)
```

### Memory Management
```python
# Deduplication settings
SIMILARITY_THRESHOLD=0.88    # Duplicate detection sensitivity
AUTO_DEDUP_ENABLED=True     # Background deduplication
DEDUP_INTERVAL_HOURS=24     # Cleanup frequency

# Consolidation settings  
CONSOLIDATION_THRESHOLD=0.85 # Similar memory grouping
MAX_CONSOLIDATION_SIZE=5    # Max memories per group
```

This architecture provides **production-ready memory management** with enterprise-grade features like multi-user isolation, automatic deduplication, semantic search, and hybrid retrieval while maintaining high performance and scalability.