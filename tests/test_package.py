"""Tests for package structure and imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _has_optional_deps() -> bool:
    """Check if optional dependencies are installed."""
    try:
        import cachetools  # noqa: F401
        import qdrant_client  # noqa: F401
        import rank_bm25  # noqa: F401
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _has_fastapi() -> bool:
    """Check if FastAPI is installed."""
    try:
        import fastapi  # noqa: F401

        return True
    except ImportError:
        return False


class TestPackageStructure:
    """Test package structure and organization."""

    def test_package_importable(self):
        """Test that hippocampai package can be imported."""
        import hippocampai

        assert hippocampai is not None

    def test_package_has_version(self):
        """Test that package has __version__ attribute."""
        import hippocampai

        assert hasattr(hippocampai, "__version__")
        assert isinstance(hippocampai.__version__, str)
        assert len(hippocampai.__version__) > 0

    def test_package_has_all_attribute(self):
        """Test that package has __all__ for public API."""
        import hippocampai

        assert hasattr(hippocampai, "__all__")
        assert isinstance(hippocampai.__all__, list)
        assert len(hippocampai.__all__) > 0

    def test_public_api_exports(self):
        """Test that public API exports are accessible."""
        import hippocampai

        # Core exports should be available
        assert "Memory" in hippocampai.__all__
        assert "MemoryType" in hippocampai.__all__
        assert "MemoryClient" in hippocampai.__all__

    def test_memory_and_memory_type_importable(self):
        """Test that Memory and MemoryType can be imported."""
        from hippocampai import Memory, MemoryType

        assert Memory is not None
        assert MemoryType is not None


class TestModuleStructure:
    """Test module structure."""

    def test_models_module_exists(self):
        """Test that models module exists."""
        from hippocampai import models

        assert models is not None

    def test_utils_module_exists(self):
        """Test that utils module exists."""
        from hippocampai import utils

        assert utils is not None

    def test_config_module_exists(self):
        """Test that config module exists."""
        from hippocampai import config

        assert config is not None

    def test_pipeline_module_exists(self):
        """Test that pipeline module exists."""
        from hippocampai import pipeline

        assert pipeline is not None

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_retrieval_module_exists(self):
        """Test that retrieval module exists."""
        from hippocampai import retrieval

        assert retrieval is not None

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_vector_module_exists(self):
        """Test that vector module exists."""
        from hippocampai import vector

        assert vector is not None


class TestCoreImports:
    """Test core functionality imports."""

    def test_memory_model_import(self):
        """Test Memory model import."""
        from hippocampai.models.memory import Memory

        assert Memory is not None

    def test_memory_type_enum_import(self):
        """Test MemoryType enum import."""
        from hippocampai.models.memory import MemoryType

        assert MemoryType is not None
        assert hasattr(MemoryType, "PREFERENCE")
        assert hasattr(MemoryType, "FACT")

    def test_config_import(self):
        """Test Config import."""
        from hippocampai.config import Config

        assert Config is not None

    def test_time_utils_import(self):
        """Test time utils import."""
        from hippocampai.utils.time import now_utc

        assert now_utc is not None
        assert callable(now_utc)

    def test_scoring_utils_import(self):
        """Test scoring utils import."""
        from hippocampai.utils.scoring import normalize

        assert normalize is not None
        assert callable(normalize)


class TestOptionalImports:
    """Test optional dependency imports."""

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_memory_client_import(self):
        """Test MemoryClient import with optional deps."""
        from hippocampai import MemoryClient

        assert MemoryClient is not None

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_embedder_import(self):
        """Test embedder import."""
        from hippocampai.embed.embedder import get_embedder

        assert get_embedder is not None

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_qdrant_store_import(self):
        """Test QdrantStore import."""
        from hippocampai.vector.qdrant_store import QdrantStore

        assert QdrantStore is not None

    def test_memory_client_import_without_deps_raises(self):
        """Test that MemoryClient import fails gracefully without deps."""
        if _has_optional_deps():
            pytest.skip("Optional dependencies are installed")

        with pytest.raises(ModuleNotFoundError) as exc_info:
            from hippocampai import MemoryClient

            # Force the import to execute
            _ = MemoryClient

        assert "hippocampai[core]" in str(exc_info.value)


class TestPipelineImports:
    """Test pipeline component imports."""

    def test_importance_scorer_import(self):
        """Test ImportanceScorer import."""
        from hippocampai.pipeline.importance import ImportanceScorer

        assert ImportanceScorer is not None

    def test_memory_extractor_import(self):
        """Test MemoryExtractor import."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        assert MemoryExtractor is not None

    def test_memory_consolidator_import(self):
        """Test MemoryConsolidator import."""
        from hippocampai.pipeline.consolidate import MemoryConsolidator

        assert MemoryConsolidator is not None

    @pytest.mark.skipif(not _has_optional_deps(), reason="Requires optional dependencies")
    def test_memory_deduplicator_import(self):
        """Test MemoryDeduplicator import."""
        from hippocampai.pipeline.dedup import MemoryDeduplicator

        assert MemoryDeduplicator is not None


class TestAdapterImports:
    """Test LLM adapter imports."""

    def test_ollama_adapter_import(self):
        """Test OllamaLLM import."""
        try:
            from hippocampai.adapters.provider_ollama import OllamaLLM

            assert OllamaLLM is not None
        except ImportError:
            pytest.skip("Ollama adapter not available")

    def test_openai_adapter_import(self):
        """Test OpenAILLM import."""
        try:
            from hippocampai.adapters.provider_openai import OpenAILLM

            assert OpenAILLM is not None
        except ImportError:
            pytest.skip("OpenAI adapter not available")


class TestAPIImports:
    """Test API module imports."""

    @pytest.mark.skipif(not _has_fastapi(), reason="Requires FastAPI")
    def test_api_app_import(self):
        """Test API app import."""
        try:
            from hippocampai.api import app

            assert app is not None
        except ImportError:
            pytest.skip("API module not available")


class TestPackageMetadata:
    """Test package metadata."""

    def test_package_name(self):
        """Test package has correct name."""
        import hippocampai

        assert hippocampai.__name__ == "hippocampai"

    def test_version_format(self):
        """Test version follows semantic versioning."""
        import hippocampai

        version = hippocampai.__version__
        parts = version.split(".")

        assert len(parts) >= 2  # At least major.minor
        assert all(part.isdigit() or part[0].isdigit() for part in parts)

    def test_package_file_exists(self):
        """Test that package __init__.py exists."""
        import hippocampai

        assert hasattr(hippocampai, "__file__")
        assert hippocampai.__file__ is not None

    def test_package_in_correct_location(self):
        """Test that package is in expected location."""
        import hippocampai

        package_path = Path(hippocampai.__file__).parent
        assert package_path.name == "hippocampai"


class TestImportPerformance:
    """Test import performance."""

    def test_top_level_import_speed(self, benchmark=None):
        """Test that top-level import is reasonably fast."""
        import time

        start = time.time()
        import hippocampai  # noqa: F401

        elapsed = time.time() - start

        # Import should be fast (< 1 second)
        assert elapsed < 1.0, f"Import took {elapsed}s"

    def test_no_heavy_imports_at_package_level(self):
        """Test that heavy libraries aren't imported at package level."""

        # Remove hippocampai if already imported
        modules_to_remove = [k for k in sys.modules if k.startswith("hippocampai")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Also remove heavy libraries
        heavy_libs = ["qdrant_client", "sentence_transformers", "transformers", "torch"]
        for lib in heavy_libs:
            if lib in sys.modules:
                del sys.modules[lib]

        # Import package
        import hippocampai  # noqa: F401

        # Heavy libraries should not be imported
        for lib in heavy_libs:
            assert lib not in sys.modules, f"{lib} was imported at package level"


class TestNoCircularImports:
    """Test for circular import issues."""

    def test_no_circular_import_in_models(self):
        """Test no circular imports in models."""
        from hippocampai.models import memory

        assert memory is not None

    def test_no_circular_import_in_utils(self):
        """Test no circular imports in utils."""
        from hippocampai.utils import time

        assert time is not None

    def test_no_circular_import_in_config(self):
        """Test no circular imports in config."""
        from hippocampai import config

        assert config is not None
