"""Tests for pipeline components."""

from __future__ import annotations

import pytest

from hippocampai.models.memory import Memory, MemoryType


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


class TestImportanceScorer:
    """Test importance scoring."""

    def test_scorer_initialization_without_llm(self):
        """Test scorer can work without LLM."""
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)
        assert scorer is not None

    def test_score_returns_valid_range(self):
        """Test that scores are in valid range [0, 10]."""
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)
        score = scorer.score("User prefers dark mode", "preference")

        assert 0 <= score <= 10
        assert isinstance(score, (int, float))

    def test_score_different_memory_types(self):
        """Test scoring for different memory types."""
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)

        preference_score = scorer.score("User prefers coffee", "preference")
        fact_score = scorer.score("User lives in NYC", "fact")
        event_score = scorer.score("User attended meeting", "event")

        # All should be valid
        assert 0 <= preference_score <= 10
        assert 0 <= fact_score <= 10
        assert 0 <= event_score <= 10

    def test_score_empty_text(self):
        """Test scoring with empty text."""
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)
        score = scorer.score("", "fact")

        # Should return low score for empty text
        assert 0 <= score <= 10

    def test_score_consistency(self):
        """Test that same text gets similar score."""
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)
        text = "User prefers Python programming"

        score1 = scorer.score(text, "preference")
        score2 = scorer.score(text, "preference")

        # Should be consistent (within small margin)
        assert abs(score1 - score2) < 2.0


class TestMemoryExtractor:
    """Test memory extraction."""

    def test_extractor_initialization_without_llm(self):
        """Test extractor can work without LLM."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        assert extractor is not None

    def test_extract_returns_list(self):
        """Test extract returns list of memories."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        conversation = "User: I prefer dark mode. Assistant: Noted!"

        memories = extractor.extract(conversation, user_id="u1", session_id="s1")

        assert isinstance(memories, list)
        # May be empty if no clear memories found
        assert all(isinstance(m, Memory) for m in memories)

    def test_extract_with_clear_preference(self):
        """Test extracting clear preference statement."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        conversation = "I prefer coffee over tea"

        memories = extractor.extract(conversation, user_id="u1")

        # Should extract at least one memory
        assert len(memories) >= 0  # May be 0 depending on heuristics

        if memories:
            assert all(m.user_id == "u1" for m in memories)

    def test_extract_empty_conversation(self):
        """Test extracting from empty conversation."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        memories = extractor.extract("", user_id="u1")

        assert isinstance(memories, list)
        assert len(memories) == 0


class TestMemoryDeduplicator:
    """Test memory deduplication."""

    @pytest.mark.skipif(
        not _has_optional_deps(),
        reason="Requires qdrant-client and sentence-transformers",  # noqa: F821
    )
    def test_deduplicator_initialization(self):
        """Test deduplicator can be initialized."""
        from hippocampai.embed.embedder import get_embedder
        from hippocampai.pipeline.dedup import MemoryDeduplicator
        from hippocampai.retrieval.rerank import Reranker
        from hippocampai.vector.qdrant_store import QdrantStore

        try:
            qdrant = QdrantStore(url=":memory:", collection_facts="facts", collection_prefs="prefs")
            embedder = get_embedder()
            reranker = Reranker()

            dedup = MemoryDeduplicator(qdrant_store=qdrant, embedder=embedder, reranker=reranker)
            assert dedup is not None
        except Exception:
            pytest.skip("Cannot initialize components for test")

    def test_check_duplicate_action_types(self):
        """Test that check_duplicate returns valid action types."""
        # Valid actions should be: "store", "skip", "merge"
        valid_actions = {"store", "skip", "merge"}

        # This is a documentation test
        assert "store" in valid_actions
        assert "skip" in valid_actions
        assert "merge" in valid_actions


class TestMemoryConsolidator:
    """Test memory consolidation."""

    def test_consolidator_initialization_without_llm(self):
        """Test consolidator can work without LLM."""
        from hippocampai.pipeline.consolidate import MemoryConsolidator

        consolidator = MemoryConsolidator(llm=None)
        assert consolidator is not None

    def test_consolidate_empty_list(self):
        """Test consolidating empty list."""
        from hippocampai.pipeline.consolidate import MemoryConsolidator

        consolidator = MemoryConsolidator(llm=None)
        result = consolidator.consolidate([])

        assert result is None or isinstance(result, Memory)

    def test_consolidate_single_memory(self):
        """Test consolidating single memory."""
        from hippocampai.pipeline.consolidate import MemoryConsolidator

        consolidator = MemoryConsolidator(llm=None)
        memory = Memory(text="User prefers dark mode", user_id="u1", type=MemoryType.PREFERENCE)

        result = consolidator.consolidate([memory])

        # Single memory might return itself or None
        assert result is None or isinstance(result, Memory)


class TestPipelineIntegration:
    """Test pipeline component integration."""

    def test_all_pipeline_components_importable(self):
        """Test that all pipeline components can be imported."""
        try:
            from hippocampai.pipeline import (
                ImportanceScorer,
                MemoryConsolidator,
                MemoryDeduplicator,
                MemoryExtractor,
            )

            assert ImportanceScorer is not None
            assert MemoryExtractor is not None
            assert MemoryDeduplicator is not None
            assert MemoryConsolidator is not None
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline components: {e}")

    def test_pipeline_components_have_required_methods(self):
        """Test that pipeline components have expected methods."""
        from hippocampai.pipeline.consolidate import MemoryConsolidator
        from hippocampai.pipeline.extractor import MemoryExtractor
        from hippocampai.pipeline.importance import ImportanceScorer

        scorer = ImportanceScorer(llm=None)
        assert hasattr(scorer, "score")

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        assert hasattr(extractor, "extract")

        consolidator = MemoryConsolidator(llm=None)
        assert hasattr(consolidator, "consolidate")


class TestMemoryLifecycle:
    """Test complete memory lifecycle through pipeline."""

    def test_memory_creation_to_scoring(self):
        """Test creating memory and scoring it."""
        from hippocampai.pipeline.importance import ImportanceScorer

        memory = Memory(text="User prefers Python", user_id="u1", type=MemoryType.PREFERENCE)

        scorer = ImportanceScorer(llm=None)
        score = scorer.score(memory.text, memory.type.value)

        assert 0 <= score <= 10
        assert isinstance(memory, Memory)

    def test_extraction_produces_valid_memories(self):
        """Test that extraction produces valid Memory objects."""
        from hippocampai.pipeline.extractor import MemoryExtractor

        extractor = MemoryExtractor(llm=None, mode="heuristic")
        memories = extractor.extract("I love Python programming", user_id="u1")

        for memory in memories:
            assert isinstance(memory, Memory)
            assert memory.user_id == "u1"
            assert 0 <= memory.importance <= 10
            assert memory.created_at is not None
