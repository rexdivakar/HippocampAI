"""Configuration with env var overrides."""

from typing import Dict

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

    def get_weights(self) -> dict[str, float]:
        return {
            "sim": self.weight_sim,
            "rerank": self.weight_rerank,
            "recency": self.weight_recency,
            "importance": self.weight_importance,
        }

    def get_half_lives(self) -> dict[str, int]:
        return {
            "preference": self.half_life_prefs,
            "goal": self.half_life_prefs,
            "fact": self.half_life_facts,
            "event": self.half_life_events,
            "context": self.half_life_facts,
            "habit": self.half_life_prefs,
        }


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
