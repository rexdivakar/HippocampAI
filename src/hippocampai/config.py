"""Configuration with environment-aware overrides using Pydantic v2 settings."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def env_field(default, *env_vars: str, description: Optional[str] = None):
    """Helper to declare settings fields with explicit environment overrides."""
    aliases = AliasChoices(*env_vars) if env_vars else None
    return Field(default=default, validation_alias=aliases, description=description)


class Config(BaseSettings):
    """Central configuration model used across HippocampAI services."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Qdrant
    qdrant_url: str = env_field("http://localhost:6333", "QDRANT_URL")
    collection_facts: str = env_field("hippocampai_facts", "COLLECTION_FACTS")
    collection_prefs: str = env_field("hippocampai_prefs", "COLLECTION_PREFS")

    # HNSW tuning
    hnsw_m: int = env_field(48, "HNSW_M")
    ef_construction: int = env_field(256, "EF_CONSTRUCTION")
    ef_search: int = env_field(128, "EF_SEARCH")

    # Embeddings
    embed_model: str = env_field("BAAI/bge-small-en-v1.5", "EMBED_MODEL")
    embed_quantized: bool = env_field(False, "EMBED_QUANTIZED")
    embed_batch_size: int = env_field(32, "EMBED_BATCH_SIZE")
    embed_dimension: int = env_field(384, "EMBED_DIMENSION")

    # Reranker
    reranker_model: str = env_field("cross-encoder/ms-marco-MiniLM-L-6-v2", "RERANKER_MODEL")
    rerank_cache_ttl: int = env_field(86400, "RERANK_CACHE_TTL")  # 24h

    # BM25
    bm25_backend: str = env_field("rank-bm25", "BM25_BACKEND")

    # LLM
    llm_provider: str = env_field("ollama", "LLM_PROVIDER")
    llm_model: str = env_field("qwen2.5:7b-instruct", "LLM_MODEL")
    llm_base_url: str = env_field("http://localhost:11434", "LLM_BASE_URL")
    allow_cloud: bool = env_field(False, "ALLOW_CLOUD")

    # Retrieval
    top_k_qdrant: int = env_field(200, "TOP_K_QDRANT")
    top_k_final: int = env_field(20, "TOP_K_FINAL")
    rrf_k: int = env_field(60, "RRF_K")

    # Scoring weights (must sum to ~1.0)
    weight_sim: float = env_field(0.55, "WEIGHT_SIM")
    weight_rerank: float = env_field(0.20, "WEIGHT_RERANK")
    weight_recency: float = env_field(0.15, "WEIGHT_RECENCY")
    weight_importance: float = env_field(0.10, "WEIGHT_IMPORTANCE")

    # Half-lives (days)
    half_life_prefs: int = env_field(90, "HALF_LIFE_PREFS")
    half_life_facts: int = env_field(30, "HALF_LIFE_FACTS")
    half_life_events: int = env_field(14, "HALF_LIFE_EVENTS")

    # Jobs
    enable_scheduler: bool = env_field(True, "ENABLE_SCHEDULER")
    decay_cron: str = env_field("0 2 * * *", "DECAY_CRON")  # 2am daily
    consolidate_cron: str = env_field("0 3 * * 0", "CONSOLIDATE_CRON")  # 3am Sunday
    snapshot_cron: str = env_field("0 * * * *", "SNAPSHOT_CRON")  # hourly

    def get_weights(self) -> Dict[str, float]:
        return {
            "sim": self.weight_sim,
            "rerank": self.weight_rerank,
            "recency": self.weight_recency,
            "importance": self.weight_importance,
        }

    def get_half_lives(self) -> Dict[str, int]:
        return {
            "preference": self.half_life_prefs,
            "goal": self.half_life_prefs,
            "fact": self.half_life_facts,
            "event": self.half_life_events,
            "context": self.half_life_facts,
            "habit": self.half_life_prefs,
        }


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
