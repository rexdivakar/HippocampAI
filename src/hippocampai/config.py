"""Configuration with env var overrides."""

import os
from typing import Dict
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    collection_facts: str = Field(default="hippocampai_facts", env="COLLECTION_FACTS")
    collection_prefs: str = Field(default="hippocampai_prefs", env="COLLECTION_PREFS")

    # HNSW tuning
    hnsw_m: int = Field(default=48, env="HNSW_M")
    ef_construction: int = Field(default=256, env="EF_CONSTRUCTION")
    ef_search: int = Field(default=128, env="EF_SEARCH")

    # Embeddings
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", env="EMBED_MODEL")
    embed_quantized: bool = Field(default=False, env="EMBED_QUANTIZED")
    embed_batch_size: int = Field(default=32, env="EMBED_BATCH_SIZE")
    embed_dimension: int = Field(default=384, env="EMBED_DIMENSION")

    # Reranker
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL"
    )
    rerank_cache_ttl: int = Field(default=86400, env="RERANK_CACHE_TTL")  # 24h

    # BM25
    bm25_backend: str = Field(default="rank-bm25", env="BM25_BACKEND")

    # LLM
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:7b-instruct", env="LLM_MODEL")
    llm_base_url: str = Field(default="http://localhost:11434", env="LLM_BASE_URL")
    allow_cloud: bool = Field(default=False, env="ALLOW_CLOUD")

    # Retrieval
    top_k_qdrant: int = Field(default=200, env="TOP_K_QDRANT")
    top_k_final: int = Field(default=20, env="TOP_K_FINAL")
    rrf_k: int = Field(default=60, env="RRF_K")

    # Scoring weights (must sum to ~1.0)
    weight_sim: float = Field(default=0.55, env="WEIGHT_SIM")
    weight_rerank: float = Field(default=0.20, env="WEIGHT_RERANK")
    weight_recency: float = Field(default=0.15, env="WEIGHT_RECENCY")
    weight_importance: float = Field(default=0.10, env="WEIGHT_IMPORTANCE")

    # Half-lives (days)
    half_life_prefs: int = Field(default=90, env="HALF_LIFE_PREFS")
    half_life_facts: int = Field(default=30, env="HALF_LIFE_FACTS")
    half_life_events: int = Field(default=14, env="HALF_LIFE_EVENTS")

    # Jobs
    enable_scheduler: bool = Field(default=True, env="ENABLE_SCHEDULER")
    decay_cron: str = Field(default="0 2 * * *", env="DECAY_CRON")  # 2am daily
    consolidate_cron: str = Field(default="0 3 * * 0", env="CONSOLIDATE_CRON")  # 3am Sunday
    snapshot_cron: str = Field(default="0 * * * *", env="SNAPSHOT_CRON")  # hourly

    class Config:
        env_file = ".env"
        case_sensitive = False

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


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
