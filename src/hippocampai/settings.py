"""Settings and configuration management for HippocampAI."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QdrantSettings:
    """Qdrant database settings."""

    host: str = "localhost"
    port: int = 6334
    api_key: Optional[str] = None
    collection_prefix: str = "hippocampai"


@dataclass
class EmbeddingSettings:
    """Embedding model settings."""

    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    cache_size: int = 1000


@dataclass
class LLMSettings:
    """LLM provider settings."""

    provider: str = "anthropic"  # anthropic, openai, groq

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_max_tokens: int = 4096
    anthropic_temperature: float = 0.0

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.0

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-70b-versatile"
    groq_max_tokens: int = 8192
    groq_temperature: float = 0.0


@dataclass
class MemorySettings:
    """Memory management settings."""

    similarity_threshold: float = 0.88
    consolidation_threshold: float = 0.85
    min_importance: int = 1
    max_importance: int = 10


@dataclass
class RetrievalSettings:
    """Retrieval and search settings."""

    default_limit: int = 10
    max_limit: int = 50
    candidates_multiplier: int = 3
    similarity_weight: float = 0.50
    importance_weight: float = 0.30
    recency_weight: float = 0.20


@dataclass
class SessionSettings:
    """Session management settings."""

    timeout_minutes: int = 30
    max_messages: int = 100
    auto_summarize: bool = True


@dataclass
class PerformanceSettings:
    """Performance and rate limiting settings."""

    rate_limit_interval_ms: int = 100
    max_retries: int = 2
    request_timeout_seconds: int = 30


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: str = "logs/hippocampai.log"


class Settings:
    """
    Application settings manager.

    Loads configuration from:
    1. .env file (environment variables)
    2. config/config.yaml (application settings)
    3. Environment variables (override)

    Example:
        settings = Settings()
        print(settings.qdrant.host)
        print(settings.memory.similarity_threshold)
    """

    def __init__(self, env_file: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize settings.

        Args:
            env_file: Path to .env file (default: project_root/.env)
            config_file: Path to config.yaml (default: project_root/config/config.yaml)
        """
        self.project_root = Path(__file__).resolve().parents[2]

        # Load .env file
        self._load_env_file(env_file)

        # Load config.yaml
        self.config = self._load_config_file(config_file)

        # Initialize settings objects
        self.qdrant = self._load_qdrant_settings()
        self.embedding = self._load_embedding_settings()
        self.llm = self._load_llm_settings()
        self.memory = self._load_memory_settings()
        self.retrieval = self._load_retrieval_settings()
        self.session = self._load_session_settings()
        self.performance = self._load_performance_settings()
        self.logging_config = self._load_logging_settings()

        # Validate settings
        self._validate()

        logger.info("Settings loaded successfully")

    def _load_env_file(self, env_file: Optional[str] = None) -> None:
        """Load environment variables from .env file."""
        if env_file is None:
            env_file = self.project_root / ".env"
        else:
            env_file = Path(env_file)

        if not env_file.exists():
            logger.warning(f".env file not found: {env_file}")
            return

        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value

            logger.info(f"Loaded environment from {env_file}")
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")

    def _load_config_file(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_file is None:
            config_file = self.project_root / "config" / "config.yaml"
        else:
            config_file = Path(config_file)

        if not config_file.exists():
            logger.warning(f"config.yaml not found: {config_file}, using defaults")
            return {}

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config.yaml: {e}")
            return {}

    def _get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable with type conversion."""
        value = os.getenv(key)
        if value is None:
            return default

        # Type conversion based on default type
        if isinstance(default, bool):
            return value.lower() in ("true", "1", "yes", "on")
        elif isinstance(default, int):
            try:
                return int(value)
            except ValueError:
                return default
        elif isinstance(default, float):
            try:
                return float(value)
            except ValueError:
                return default
        return value

    def _load_qdrant_settings(self) -> QdrantSettings:
        """Load Qdrant settings."""
        return QdrantSettings(
            host=self._get_env("QDRANT_HOST", "localhost"),
            port=self._get_env("QDRANT_PORT", 6334),
            api_key=self._get_env("QDRANT_API_KEY"),
            collection_prefix=self._get_env("QDRANT_COLLECTION_PREFIX", "hippocampai"),
        )

    def _load_embedding_settings(self) -> EmbeddingSettings:
        """Load embedding settings."""
        return EmbeddingSettings(
            model=self._get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            dimension=self._get_env("EMBEDDING_DIMENSION", 384),
            cache_size=self._get_env("EMBEDDING_CACHE_SIZE", 1000),
        )

    def _load_llm_settings(self) -> LLMSettings:
        """Load LLM provider settings."""
        return LLMSettings(
            provider=self._get_env("LLM_PROVIDER", "anthropic"),
            # Anthropic
            anthropic_api_key=self._get_env("ANTHROPIC_API_KEY"),
            anthropic_model=self._get_env("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            anthropic_max_tokens=self._get_env("ANTHROPIC_MAX_TOKENS", 4096),
            anthropic_temperature=self._get_env("ANTHROPIC_TEMPERATURE", 0.0),
            # OpenAI
            openai_api_key=self._get_env("OPENAI_API_KEY"),
            openai_model=self._get_env("OPENAI_MODEL", "gpt-4o"),
            openai_max_tokens=self._get_env("OPENAI_MAX_TOKENS", 4096),
            openai_temperature=self._get_env("OPENAI_TEMPERATURE", 0.0),
            # Groq
            groq_api_key=self._get_env("GROQ_API_KEY"),
            groq_model=self._get_env("GROQ_MODEL", "llama-3.1-70b-versatile"),
            groq_max_tokens=self._get_env("GROQ_MAX_TOKENS", 8192),
            groq_temperature=self._get_env("GROQ_TEMPERATURE", 0.0),
        )

    def _load_memory_settings(self) -> MemorySettings:
        """Load memory management settings."""
        return MemorySettings(
            similarity_threshold=self._get_env("SIMILARITY_THRESHOLD", 0.88),
            consolidation_threshold=self._get_env("CONSOLIDATION_THRESHOLD", 0.85),
            min_importance=self._get_env("MIN_IMPORTANCE_SCORE", 1),
            max_importance=self._get_env("MAX_IMPORTANCE_SCORE", 10),
        )

    def _load_retrieval_settings(self) -> RetrievalSettings:
        """Load retrieval settings."""
        config = self.config.get("retrieval", {})
        smart_search = config.get("smart_search", {})

        return RetrievalSettings(
            default_limit=self._get_env("DEFAULT_SEARCH_LIMIT", 10),
            max_limit=self._get_env("MAX_SEARCH_LIMIT", 50),
            candidates_multiplier=self._get_env("SMART_SEARCH_CANDIDATES_MULTIPLIER", 3),
            similarity_weight=smart_search.get("similarity_weight", 0.50),
            importance_weight=smart_search.get("importance_weight", 0.30),
            recency_weight=smart_search.get("recency_weight", 0.20),
        )

    def _load_session_settings(self) -> SessionSettings:
        """Load session settings."""
        return SessionSettings(
            timeout_minutes=self._get_env("SESSION_TIMEOUT_MINUTES", 30),
            max_messages=self._get_env("MAX_SESSION_MESSAGES", 100),
            auto_summarize=True,
        )

    def _load_performance_settings(self) -> PerformanceSettings:
        """Load performance settings."""
        return PerformanceSettings(
            rate_limit_interval_ms=self._get_env("RATE_LIMIT_INTERVAL_MS", 100),
            max_retries=self._get_env("MAX_RETRIES", 2),
            request_timeout_seconds=self._get_env("REQUEST_TIMEOUT_SECONDS", 30),
        )

    def _load_logging_settings(self) -> LoggingSettings:
        """Load logging settings."""
        return LoggingSettings(
            level=self._get_env("LOG_LEVEL", "INFO"),
            format=self._get_env("LOG_FORMAT", "json"),
            file=self._get_env("LOG_FILE", "logs/hippocampai.log"),
        )

    def _validate(self) -> None:
        """Validate settings."""
        errors = []

        # Validate Qdrant settings
        if not self.qdrant.host:
            errors.append("QDRANT_HOST is required")
        if not 1 <= self.qdrant.port <= 65535:
            errors.append(f"Invalid QDRANT_PORT: {self.qdrant.port}")

        # Validate embedding settings
        if self.embedding.dimension <= 0:
            errors.append(f"Invalid EMBEDDING_DIMENSION: {self.embedding.dimension}")
        if self.embedding.cache_size < 0:
            errors.append(f"Invalid EMBEDDING_CACHE_SIZE: {self.embedding.cache_size}")

        # Validate LLM settings
        provider = self.llm.provider.lower()
        if provider == "anthropic" and not self.llm.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set - AI features will not work")
        elif provider == "openai" and not self.llm.openai_api_key:
            logger.warning("OPENAI_API_KEY not set - AI features will not work")
        elif provider == "groq" and not self.llm.groq_api_key:
            logger.warning("GROQ_API_KEY not set - AI features will not work")

        if provider not in ["anthropic", "openai", "groq"]:
            errors.append(f"Invalid LLM_PROVIDER: {provider} (must be anthropic, openai, or groq)")

        # Validate memory settings
        if not 0.0 <= self.memory.similarity_threshold <= 1.0:
            errors.append(f"Invalid SIMILARITY_THRESHOLD: {self.memory.similarity_threshold}")
        if not 0.0 <= self.memory.consolidation_threshold <= 1.0:
            errors.append(f"Invalid CONSOLIDATION_THRESHOLD: {self.memory.consolidation_threshold}")

        # Validate retrieval settings
        if self.retrieval.default_limit <= 0:
            errors.append(f"Invalid DEFAULT_SEARCH_LIMIT: {self.retrieval.default_limit}")
        if self.retrieval.max_limit < self.retrieval.default_limit:
            errors.append("MAX_SEARCH_LIMIT must be >= DEFAULT_SEARCH_LIMIT")

        # Raise if any errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def get_collection_names(self) -> list:
        """Get list of collection names."""
        collections = self.config.get("collections", {})
        if collections:
            return list(collections.keys())
        return ["personal_facts", "conversation_history", "knowledge_base"]

    def get_decay_rates(self) -> Dict[str, float]:
        """Get importance decay rates by memory type."""
        return self.config.get("memory", {}).get(
            "decay_rates",
            {
                "preference": 0.001,
                "fact": 0.002,
                "goal": 0.003,
                "habit": 0.004,
                "context": 0.008,
                "event": 0.01,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        llm_dict = self.llm.__dict__.copy()
        # Mask API keys
        if llm_dict.get("anthropic_api_key"):
            llm_dict["anthropic_api_key"] = "***"
        if llm_dict.get("openai_api_key"):
            llm_dict["openai_api_key"] = "***"
        if llm_dict.get("groq_api_key"):
            llm_dict["groq_api_key"] = "***"

        return {
            "qdrant": self.qdrant.__dict__,
            "embedding": self.embedding.__dict__,
            "llm": llm_dict,
            "memory": self.memory.__dict__,
            "retrieval": self.retrieval.__dict__,
            "session": self.session.__dict__,
            "performance": self.performance.__dict__,
            "logging": self.logging_config.__dict__,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Settings(qdrant={self.qdrant.host}:{self.qdrant.port}, "
            f"embedding={self.embedding.model})"
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton).

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from files.

    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings
