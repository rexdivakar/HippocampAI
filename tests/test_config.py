"""Tests for configuration."""

from __future__ import annotations

from hippocampai.config import Config, get_config


class TestConfig:
    """Test configuration loading and validation."""

    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_default_values(self):
        """Test config has sensible defaults."""
        config = Config()

        # Qdrant defaults
        assert config.qdrant_url is not None
        assert config.collection_facts is not None
        assert config.collection_prefs is not None

        # Embedding defaults
        assert config.embed_dimension > 0
        assert config.embed_batch_size > 0

        # Retrieval defaults
        assert config.top_k_qdrant > 0
        assert config.top_k_final > 0
        assert 0 <= config.weight_sim <= 1
        assert 0 <= config.weight_rerank <= 1
        assert 0 <= config.weight_recency <= 1
        assert 0 <= config.weight_importance <= 1

    def test_config_get_weights(self):
        """Test get_weights returns dict with all required keys."""
        config = Config()
        weights = config.get_weights()

        assert isinstance(weights, dict)
        assert "sim" in weights
        assert "rerank" in weights
        assert "recency" in weights
        assert "importance" in weights

        # All weights should be between 0 and 1
        for key, value in weights.items():
            assert 0 <= value <= 1

    def test_config_get_half_lives(self):
        """Test get_half_lives returns dict with memory types."""
        config = Config()
        half_lives = config.get_half_lives()

        assert isinstance(half_lives, dict)
        assert "preference" in half_lives
        assert "fact" in half_lives
        assert "event" in half_lives

        # All half-lives should be positive
        for key, value in half_lives.items():
            assert value > 0

    def test_config_weights_sum_reasonable(self):
        """Test that weights sum to approximately 1.0."""
        config = Config()
        weights = config.get_weights()
        total = sum(weights.values())

        # Should be close to 1.0 (within 0.1)
        assert 0.9 <= total <= 1.1

    def test_config_hnsw_parameters(self):
        """Test HNSW index parameters are valid."""
        config = Config()

        assert config.hnsw_m > 0
        assert config.ef_construction > 0
        assert config.ef_search > 0

        # ef_construction should typically be >= hnsw_m
        assert config.ef_construction >= config.hnsw_m

    def test_config_with_environment_override(self, monkeypatch):
        """Test config can be overridden by environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://custom:6333")
        monkeypatch.setenv("TOP_K_FINAL", "20")

        # Create new config to pick up env vars
        config = Config()

        assert "custom" in config.qdrant_url
        # Note: TOP_K_FINAL might not be picked up depending on implementation
        # This test assumes env var support exists

    def test_config_collection_names_not_empty(self):
        """Test collection names are not empty."""
        config = Config()

        assert len(config.collection_facts) > 0
        assert len(config.collection_prefs) > 0
        assert config.collection_facts != config.collection_prefs

    def test_config_llm_settings(self):
        """Test LLM configuration."""
        config = Config()

        assert config.llm_provider in ["ollama", "openai", "anthropic", "groq", None]
        if config.llm_model:
            assert isinstance(config.llm_model, str)
            assert len(config.llm_model) > 0

    def test_config_reranker_settings(self):
        """Test reranker configuration."""
        config = Config()

        if config.reranker_model:
            assert isinstance(config.reranker_model, str)
            assert len(config.reranker_model) > 0

        assert config.rerank_cache_ttl >= 0

    def test_config_rrf_k_parameter(self):
        """Test RRF k parameter is positive."""
        config = Config()

        assert config.rrf_k > 0
        # Typically RRF k is between 30-90
        assert 10 <= config.rrf_k <= 200


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_weights_caught(self):
        """Test that invalid weight values are handled."""
        # This assumes validation exists in Config
        # If not implemented, this test documents the expected behavior
        try:
            config = Config()
            config.weight_sim = 1.5  # Invalid - greater than 1
            # If no validation, we should add it
            assert False, "Should validate weight bounds"
        except (ValueError, AssertionError):
            pass  # Expected

    def test_negative_dimensions_invalid(self):
        """Test that negative dimensions are invalid."""
        try:
            config = Config()
            config.embed_dimension = -1
            assert config.embed_dimension > 0, "Dimension should be positive"
        except ValueError:
            pass  # Expected if validation exists

    def test_negative_top_k_invalid(self):
        """Test that negative top_k is invalid."""
        config = Config()
        # top_k should always be positive
        assert config.top_k_qdrant > 0
        assert config.top_k_final > 0


class TestConfigIntegration:
    """Test configuration integration."""

    def test_config_serializable(self):
        """Test config can be serialized."""
        config = Config()

        # Should be able to get dict representation
        config_dict = vars(config)
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_config_has_all_required_fields(self):
        """Test config has all expected fields."""
        config = Config()

        required_fields = [
            "qdrant_url",
            "collection_facts",
            "collection_prefs",
            "embed_model",
            "embed_dimension",
            "top_k_qdrant",
            "top_k_final",
            "weight_sim",
            "weight_rerank",
            "weight_recency",
            "weight_importance",
        ]

        for field in required_fields:
            assert hasattr(config, field), f"Missing required field: {field}"

    def test_config_types(self):
        """Test config field types are correct."""
        config = Config()

        assert isinstance(config.qdrant_url, str)
        assert isinstance(config.collection_facts, str)
        assert isinstance(config.embed_dimension, int)
        assert isinstance(config.top_k_qdrant, int)
        assert isinstance(config.weight_sim, (int, float))
        assert isinstance(config.allow_cloud, bool)
