"""Platform (SaaS) configuration for HippocampAI.

This configuration extends the core configuration with SaaS-specific settings
for Redis, PostgreSQL, Celery, API server, and other platform components.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformConfig(BaseSettings):
    """Platform configuration for HippocampAI SaaS deployment.
    
    This configuration includes all settings needed for running HippocampAI
    as a SaaS platform, including:
    - Redis (caching, Celery broker)
    - PostgreSQL (authentication)
    - Celery (background tasks)
    - API server settings
    - Monitoring
    """
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ==================== Redis Configuration ====================
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_cache_ttl: int = Field(default=300, validation_alias="REDIS_CACHE_TTL")  # 5 minutes
    redis_max_connections: int = Field(default=100, validation_alias="REDIS_MAX_CONNECTIONS")
    redis_min_idle: int = Field(default=10, validation_alias="REDIS_MIN_IDLE")

    # ==================== PostgreSQL Configuration ====================
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="hippocampai", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="hippocampai", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(
        default="hippocampai_secret", validation_alias="POSTGRES_PASSWORD"
    )
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # ==================== Celery Configuration ====================
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", validation_alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", validation_alias="CELERY_RESULT_BACKEND"
    )
    celery_worker_concurrency: int = Field(
        default=4, validation_alias="CELERY_WORKER_CONCURRENCY"
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=4, validation_alias="CELERY_WORKER_PREFETCH_MULTIPLIER"
    )
    celery_task_acks_late: bool = Field(
        default=True, validation_alias="CELERY_TASK_ACKS_LATE"
    )
    celery_worker_max_tasks_per_child: int = Field(
        default=1000, validation_alias="CELERY_WORKER_MAX_TASKS_PER_CHILD"
    )

    # ==================== API Server Configuration ====================
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_workers: int = Field(default=1, validation_alias="API_WORKERS")
    api_reload: bool = Field(default=False, validation_alias="API_RELOAD")
    
    # CORS
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, validation_alias="CORS_ALLOW_CREDENTIALS")

    # ==================== Authentication Configuration ====================
    user_auth_enabled: bool = Field(default=False, validation_alias="USER_AUTH_ENABLED")
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production", validation_alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, validation_alias="JWT_EXPIRATION_HOURS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(
        default=60, validation_alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    rate_limit_burst: int = Field(default=10, validation_alias="RATE_LIMIT_BURST")

    # ==================== Background Tasks Configuration ====================
    enable_background_tasks: bool = Field(default=True, validation_alias="ENABLE_BACKGROUND_TASKS")
    auto_dedup_enabled: bool = Field(default=True, validation_alias="AUTO_DEDUP_ENABLED")
    auto_consolidation_enabled: bool = Field(
        default=False, validation_alias="AUTO_CONSOLIDATION_ENABLED"
    )
    dedup_interval_hours: int = Field(default=24, validation_alias="DEDUP_INTERVAL_HOURS")
    consolidation_interval_hours: int = Field(
        default=168, validation_alias="CONSOLIDATION_INTERVAL_HOURS"
    )  # 7 days
    expiration_interval_hours: int = Field(default=1, validation_alias="EXPIRATION_INTERVAL_HOURS")
    decay_interval_hours: int = Field(default=24, validation_alias="DECAY_INTERVAL_HOURS")

    # ==================== Scheduler Configuration ====================
    enable_scheduler: bool = Field(default=True, validation_alias="ENABLE_SCHEDULER")
    decay_cron: str = Field(default="0 2 * * *", validation_alias="DECAY_CRON")  # 2am daily
    consolidate_cron: str = Field(
        default="0 3 * * 0", validation_alias="CONSOLIDATE_CRON"
    )  # 3am Sunday
    snapshot_cron: str = Field(default="0 * * * *", validation_alias="SNAPSHOT_CRON")  # hourly

    # ==================== Sleep Phase / Consolidation ====================
    active_consolidation_enabled: bool = Field(
        default=False, validation_alias="ACTIVE_CONSOLIDATION_ENABLED"
    )
    consolidation_dry_run: bool = Field(default=False, validation_alias="CONSOLIDATION_DRY_RUN")
    consolidation_schedule_hour: int = Field(
        default=3, validation_alias="CONSOLIDATION_SCHEDULE_HOUR"
    )
    consolidation_lookback_hours: int = Field(
        default=24, validation_alias="CONSOLIDATION_LOOKBACK_HOURS"
    )
    consolidation_llm_provider: Optional[str] = Field(
        default=None, validation_alias="CONSOLIDATION_LLM_PROVIDER"
    )
    consolidation_llm_model: Optional[str] = Field(
        default=None, validation_alias="CONSOLIDATION_LLM_MODEL"
    )

    # ==================== Monitoring Configuration ====================
    prometheus_enabled: bool = Field(default=True, validation_alias="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, validation_alias="PROMETHEUS_PORT")
    
    # Flower (Celery monitoring)
    flower_port: int = Field(default=5555, validation_alias="FLOWER_PORT")
    flower_user: str = Field(default="admin", validation_alias="FLOWER_USER")
    flower_password: str = Field(default="admin", validation_alias="FLOWER_PASSWORD")
    
    # Grafana
    grafana_port: int = Field(default=3000, validation_alias="GRAFANA_PORT")
    grafana_admin_user: str = Field(default="admin", validation_alias="GRAFANA_ADMIN_USER")
    grafana_admin_password: str = Field(default="admin", validation_alias="GRAFANA_ADMIN_PASSWORD")

    # ==================== Pruning Configuration ====================
    auto_pruning_enabled: bool = Field(default=False, validation_alias="AUTO_PRUNING_ENABLED")
    pruning_interval_hours: int = Field(
        default=168, validation_alias="PRUNING_INTERVAL_HOURS"
    )  # Weekly
    pruning_strategy: str = Field(
        default="comprehensive", validation_alias="PRUNING_STRATEGY"
    )
    min_health_threshold: float = Field(default=3.0, validation_alias="MIN_HEALTH_THRESHOLD")
    pruning_target_percentage: float = Field(
        default=0.1, validation_alias="PRUNING_TARGET_PERCENTAGE"
    )

    # ==================== Conflict Resolution (Scheduled) ====================
    conflict_resolution_interval_hours: int = Field(
        default=24, validation_alias="CONFLICT_RESOLUTION_INTERVAL_HOURS"
    )


_platform_config: PlatformConfig | None = None


def get_platform_config() -> PlatformConfig:
    """Get the global platform configuration instance."""
    global _platform_config
    if _platform_config is None:
        _platform_config = PlatformConfig()
    return _platform_config


def reset_platform_config() -> None:
    """Reset the global platform configuration (useful for testing)."""
    global _platform_config
    _platform_config = None
