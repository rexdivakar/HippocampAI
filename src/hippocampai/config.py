"""Configuration with env var overrides."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    collection_facts: str = Field(default="hippocampai_facts", validation_alias="COLLECTION_FACTS")
    collection_prefs: str = Field(default="hippocampai_prefs", validation_alias="COLLECTION_PREFS")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_cache_ttl: int = Field(default=300, validation_alias="REDIS_CACHE_TTL")  # 5 minutes
    redis_max_connections: int = Field(default=100, validation_alias="REDIS_MAX_CONNECTIONS")
    redis_min_idle: int = Field(default=10, validation_alias="REDIS_MIN_IDLE")

    # PostgreSQL (for authentication)
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="hippocampai", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="hippocampai", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(
        default="hippocampai_secret", validation_alias="POSTGRES_PASSWORD"
    )

    # HNSW tuning
    hnsw_m: int = Field(default=48, validation_alias="HNSW_M")
    ef_construction: int = Field(default=256, validation_alias="EF_CONSTRUCTION")
    ef_search: int = Field(default=128, validation_alias="EF_SEARCH")

    # Embeddings
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="EMBED_MODEL")
    embed_quantized: bool = Field(default=False, validation_alias="EMBED_QUANTIZED")
    embed_batch_size: int = Field(default=32, validation_alias="EMBED_BATCH_SIZE")
    embed_dimension: int = Field(default=384, validation_alias="EMBED_DIMENSION")

    # Reranker
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="RERANKER_MODEL"
    )
    rerank_cache_ttl: int = Field(default=86400, validation_alias="RERANK_CACHE_TTL")  # 24h

    # BM25
    bm25_backend: str = Field(default="rank-bm25", validation_alias="BM25_BACKEND")

    # LLM
    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="LLM_MODEL")
    llm_base_url: str = Field(default="http://localhost:11434", validation_alias="LLM_BASE_URL")
    allow_cloud: bool = Field(default=False, validation_alias="ALLOW_CLOUD")

    # Retrieval
    top_k_qdrant: int = Field(default=200, validation_alias="TOP_K_QDRANT")
    top_k_final: int = Field(default=20, validation_alias="TOP_K_FINAL")
    rrf_k: int = Field(default=60, validation_alias="RRF_K")

    # Scoring weights (must sum to ~1.0)
    weight_sim: float = Field(default=0.55, validation_alias="WEIGHT_SIM")
    weight_rerank: float = Field(default=0.20, validation_alias="WEIGHT_RERANK")
    weight_recency: float = Field(default=0.15, validation_alias="WEIGHT_RECENCY")
    weight_importance: float = Field(default=0.10, validation_alias="WEIGHT_IMPORTANCE")

    # Half-lives (days)
    half_life_prefs: int = Field(default=90, validation_alias="HALF_LIFE_PREFS")
    half_life_facts: int = Field(default=30, validation_alias="HALF_LIFE_FACTS")
    half_life_events: int = Field(default=14, validation_alias="HALF_LIFE_EVENTS")

    # Jobs
    enable_scheduler: bool = Field(default=True, validation_alias="ENABLE_SCHEDULER")
    decay_cron: str = Field(default="0 2 * * *", validation_alias="DECAY_CRON")  # 2am daily
    consolidate_cron: str = Field(
        default="0 3 * * 0", validation_alias="CONSOLIDATE_CRON"
    )  # 3am Sunday
    snapshot_cron: str = Field(default="0 * * * *", validation_alias="SNAPSHOT_CRON")  # hourly

    # Background Tasks
    enable_background_tasks: bool = Field(default=True, validation_alias="ENABLE_BACKGROUND_TASKS")
    dedup_interval_hours: int = Field(default=24, validation_alias="DEDUP_INTERVAL_HOURS")
    consolidation_interval_hours: int = Field(
        default=168, validation_alias="CONSOLIDATION_INTERVAL_HOURS"
    )  # 7 days
    expiration_interval_hours: int = Field(default=1, validation_alias="EXPIRATION_INTERVAL_HOURS")
    auto_dedup_enabled: bool = Field(default=True, validation_alias="AUTO_DEDUP_ENABLED")
    auto_consolidation_enabled: bool = Field(
        default=False, validation_alias="AUTO_CONSOLIDATION_ENABLED"
    )
    dedup_threshold: float = Field(default=0.88, validation_alias="DEDUP_THRESHOLD")
    consolidation_threshold: float = Field(default=0.85, validation_alias="CONSOLIDATION_THRESHOLD")

    # Auto-Summarization Settings
    auto_summarization_enabled: bool = Field(
        default=True, validation_alias="AUTO_SUMMARIZATION_ENABLED"
    )
    hierarchical_summarization_enabled: bool = Field(
        default=True, validation_alias="HIERARCHICAL_SUMMARIZATION_ENABLED"
    )
    sliding_window_enabled: bool = Field(default=True, validation_alias="SLIDING_WINDOW_ENABLED")
    sliding_window_size: int = Field(default=10, validation_alias="SLIDING_WINDOW_SIZE")
    sliding_window_keep_recent: int = Field(
        default=5, validation_alias="SLIDING_WINDOW_KEEP_RECENT"
    )
    max_tokens_per_summary: int = Field(default=150, validation_alias="MAX_TOKENS_PER_SUMMARY")
    hierarchical_batch_size: int = Field(default=5, validation_alias="HIERARCHICAL_BATCH_SIZE")
    hierarchical_max_levels: int = Field(default=3, validation_alias="HIERARCHICAL_MAX_LEVELS")

    # Memory Tiering Settings
    hot_threshold_days: int = Field(default=7, validation_alias="HOT_THRESHOLD_DAYS")
    warm_threshold_days: int = Field(default=30, validation_alias="WARM_THRESHOLD_DAYS")
    cold_threshold_days: int = Field(default=90, validation_alias="COLD_THRESHOLD_DAYS")
    hot_access_count_threshold: int = Field(
        default=10, validation_alias="HOT_ACCESS_COUNT_THRESHOLD"
    )

    # Importance Decay Settings
    importance_decay_enabled: bool = Field(
        default=True, validation_alias="IMPORTANCE_DECAY_ENABLED"
    )
    decay_function: str = Field(
        default="exponential", validation_alias="DECAY_FUNCTION"
    )  # linear, exponential, logarithmic, step, hybrid
    decay_interval_hours: int = Field(default=24, validation_alias="DECAY_INTERVAL_HOURS")
    min_importance_threshold: float = Field(
        default=1.0, validation_alias="MIN_IMPORTANCE_THRESHOLD"
    )
    access_boost_factor: float = Field(default=0.5, validation_alias="ACCESS_BOOST_FACTOR")

    # Pruning Settings
    auto_pruning_enabled: bool = Field(default=False, validation_alias="AUTO_PRUNING_ENABLED")
    pruning_interval_hours: int = Field(
        default=168, validation_alias="PRUNING_INTERVAL_HOURS"
    )  # Weekly
    pruning_strategy: str = Field(
        default="comprehensive", validation_alias="PRUNING_STRATEGY"
    )  # importance_only, age_based, access_based, comprehensive, conservative
    min_health_threshold: float = Field(default=3.0, validation_alias="MIN_HEALTH_THRESHOLD")
    pruning_target_percentage: float = Field(
        default=0.1, validation_alias="PRUNING_TARGET_PERCENTAGE"
    )  # Target to prune 10% max

    # Conflict Resolution Settings
    enable_conflict_resolution: bool = Field(
        default=True, validation_alias="ENABLE_CONFLICT_RESOLUTION"
    )
    conflict_resolution_strategy: str = Field(
        default="temporal", validation_alias="CONFLICT_RESOLUTION_STRATEGY"
    )  # temporal, confidence, importance, user_review, auto_merge, keep_both
    conflict_similarity_threshold: float = Field(
        default=0.75, validation_alias="CONFLICT_SIMILARITY_THRESHOLD"
    )
    conflict_contradiction_threshold: float = Field(
        default=0.85, validation_alias="CONFLICT_CONTRADICTION_THRESHOLD"
    )
    auto_resolve_conflicts: bool = Field(default=True, validation_alias="AUTO_RESOLVE_CONFLICTS")
    conflict_check_llm: bool = Field(
        default=True, validation_alias="CONFLICT_CHECK_LLM"
    )  # Use LLM for deep contradiction analysis
    conflict_resolution_interval_hours: int = Field(
        default=24, validation_alias="CONFLICT_RESOLUTION_INTERVAL_HOURS"
    )

    # Feature 5: Real-Time Incremental Knowledge Graph
    enable_realtime_graph: bool = Field(default=True, validation_alias="ENABLE_REALTIME_GRAPH")
    graph_extraction_mode: str = Field(
        default="pattern", validation_alias="GRAPH_EXTRACTION_MODE"
    )  # "pattern" or "llm"
    graph_persistence_path: str = Field(
        default="data/knowledge_graph.json", validation_alias="GRAPH_PERSISTENCE_PATH"
    )
    graph_auto_save_interval: int = Field(
        default=300, validation_alias="GRAPH_AUTO_SAVE_INTERVAL"
    )  # seconds

    # Feature 3: Graph-Aware Retrieval
    enable_graph_retrieval: bool = Field(
        default=False, validation_alias="ENABLE_GRAPH_RETRIEVAL"
    )
    graph_retrieval_max_depth: int = Field(
        default=2, validation_alias="GRAPH_RETRIEVAL_MAX_DEPTH"
    )
    weight_graph: float = Field(default=0.0, validation_alias="WEIGHT_GRAPH")

    # Feature 1: Memory Relevance Feedback Loop
    weight_feedback: float = Field(default=0.1, validation_alias="WEIGHT_FEEDBACK")
    feedback_window_days: int = Field(default=90, validation_alias="FEEDBACK_WINDOW_DAYS")

    # Feature 2: Memory Triggers
    enable_triggers: bool = Field(default=True, validation_alias="ENABLE_TRIGGERS")
    trigger_webhook_timeout: int = Field(
        default=10, validation_alias="TRIGGER_WEBHOOK_TIMEOUT"
    )

    # Feature 4: Procedural Memory
    enable_procedural_memory: bool = Field(
        default=False, validation_alias="ENABLE_PROCEDURAL_MEMORY"
    )
    procedural_rule_max_count: int = Field(
        default=50, validation_alias="PROCEDURAL_RULE_MAX_COUNT"
    )
    half_life_procedural: int = Field(
        default=180, validation_alias="HALF_LIFE_PROCEDURAL"
    )

    # Feature 7: Prospective Memory
    enable_prospective_memory: bool = Field(
        default=False, validation_alias="ENABLE_PROSPECTIVE_MEMORY"
    )
    prospective_max_intents_per_user: int = Field(
        default=100, validation_alias="PROSPECTIVE_MAX_INTENTS_PER_USER"
    )
    prospective_eval_interval_seconds: int = Field(
        default=60, validation_alias="PROSPECTIVE_EVAL_INTERVAL_SECONDS"
    )
    half_life_prospective: int = Field(
        default=30, validation_alias="HALF_LIFE_PROSPECTIVE"
    )

    # Feature 6: Embedding Model Migration
    embed_model_version: str = Field(default="1", validation_alias="EMBED_MODEL_VERSION")

    def get_weights(self) -> dict[str, float]:
        return {
            "sim": self.weight_sim,
            "rerank": self.weight_rerank,
            "recency": self.weight_recency,
            "importance": self.weight_importance,
            "graph": self.weight_graph,
            "feedback": self.weight_feedback,
        }

    def get_half_lives(self) -> dict[str, int]:
        return {
            "preference": self.half_life_prefs,
            "goal": self.half_life_prefs,
            "fact": self.half_life_facts,
            "event": self.half_life_events,
            "context": self.half_life_facts,
            "habit": self.half_life_prefs,
            "procedural": self.half_life_procedural,
            "prospective": self.half_life_prospective,
        }


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
