"""Tests for memory models."""

from __future__ import annotations

from datetime import timezone, datetime

import pytest
from pydantic import ValidationError

from hippocampai.models.memory import Memory, MemoryType, RetrievalQuery, RetrievalResult


class TestMemoryType:
    """Test MemoryType enum."""

    def test_all_memory_types_exist(self):
        """Verify all expected memory types are defined."""
        expected_types = {"preference", "fact", "goal", "habit", "event", "context"}
        actual_types = {mt.value for mt in MemoryType}
        assert actual_types == expected_types

    def test_memory_type_string_conversion(self):
        """Test that MemoryType can be created from strings."""
        assert MemoryType("preference") == MemoryType.PREFERENCE
        assert MemoryType("fact") == MemoryType.FACT


class TestMemory:
    """Test Memory model."""

    def test_create_memory_with_defaults(self):
        """Test creating memory with minimal required fields."""
        memory = Memory(
            text="User prefers dark mode",
            user_id="user123",
            type=MemoryType.PREFERENCE,
        )

        assert memory.text == "User prefers dark mode"
        assert memory.user_id == "user123"
        assert memory.type == MemoryType.PREFERENCE
        assert memory.id is not None
        assert 0 <= memory.importance <= 10
        assert 0 <= memory.confidence <= 1
        assert memory.tags == []
        assert memory.access_count == 0
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_memory_with_all_fields(self):
        """Test creating memory with all fields."""
        now = datetime.now(UTC)
        memory = Memory(
            id="custom-id",
            text="User lives in San Francisco",
            user_id="user456",
            session_id="session789",
            type=MemoryType.FACT,
            importance=8.5,
            confidence=0.95,
            tags=["location", "personal"],
            created_at=now,
            updated_at=now,
            access_count=5,
            metadata={"source": "onboarding"},
        )

        assert memory.id == "custom-id"
        assert memory.session_id == "session789"
        assert memory.importance == 8.5
        assert memory.confidence == 0.95
        assert memory.tags == ["location", "personal"]
        assert memory.access_count == 5
        assert memory.metadata == {"source": "onboarding"}

    def test_importance_validation(self):
        """Test importance score validation."""
        # Valid importance
        memory = Memory(text="test", user_id="u1", type=MemoryType.FACT, importance=5.0)
        assert memory.importance == 5.0

        # Invalid - too high
        with pytest.raises(ValidationError):
            Memory(text="test", user_id="u1", type=MemoryType.FACT, importance=11.0)

        # Invalid - negative
        with pytest.raises(ValidationError):
            Memory(text="test", user_id="u1", type=MemoryType.FACT, importance=-1.0)

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        memory = Memory(text="test", user_id="u1", type=MemoryType.FACT, confidence=0.8)
        assert memory.confidence == 0.8

        # Invalid - too high
        with pytest.raises(ValidationError):
            Memory(text="test", user_id="u1", type=MemoryType.FACT, confidence=1.5)

        # Invalid - negative
        with pytest.raises(ValidationError):
            Memory(text="test", user_id="u1", type=MemoryType.FACT, confidence=-0.1)

    def test_collection_name_routing_preferences(self):
        """Test that preferences route to correct collection."""
        pref = Memory(text="test", user_id="u1", type=MemoryType.PREFERENCE)
        assert pref.collection_name("facts", "prefs") == "prefs"

        goal = Memory(text="test", user_id="u1", type=MemoryType.GOAL)
        assert goal.collection_name("facts", "prefs") == "prefs"

        habit = Memory(text="test", user_id="u1", type=MemoryType.HABIT)
        assert habit.collection_name("facts", "prefs") == "prefs"

    def test_collection_name_routing_facts(self):
        """Test that facts/events route to correct collection."""
        fact = Memory(text="test", user_id="u1", type=MemoryType.FACT)
        assert fact.collection_name("facts", "prefs") == "facts"

        event = Memory(text="test", user_id="u1", type=MemoryType.EVENT)
        assert event.collection_name("facts", "prefs") == "facts"

        context = Memory(text="test", user_id="u1", type=MemoryType.CONTEXT)
        assert context.collection_name("facts", "prefs") == "facts"

    def test_memory_serialization(self):
        """Test memory can be serialized to dict."""
        memory = Memory(
            text="User prefers Python",
            user_id="u1",
            type=MemoryType.PREFERENCE,
            tags=["programming", "language"],
        )

        data = memory.model_dump()
        assert isinstance(data, dict)
        assert data["text"] == "User prefers Python"
        assert data["user_id"] == "u1"
        assert data["type"] == "preference"
        assert data["tags"] == ["programming", "language"]

    def test_memory_json_serialization(self):
        """Test memory can be serialized to JSON."""
        memory = Memory(text="test", user_id="u1", type=MemoryType.FACT)
        json_str = memory.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str
        assert "u1" in json_str


class TestRetrievalResult:
    """Test RetrievalResult model."""

    def test_create_retrieval_result(self):
        """Test creating retrieval result."""
        memory = Memory(text="test", user_id="u1", type=MemoryType.FACT)
        result = RetrievalResult(
            memory=memory,
            score=0.85,
            breakdown={"sim": 0.9, "rerank": 0.8, "recency": 0.7, "importance": 0.6},
        )

        assert result.memory == memory
        assert result.score == 0.85
        assert result.breakdown["sim"] == 0.9

    def test_retrieval_result_without_breakdown(self):
        """Test retrieval result with default breakdown."""
        memory = Memory(text="test", user_id="u1", type=MemoryType.FACT)
        result = RetrievalResult(memory=memory, score=0.75)

        assert result.score == 0.75
        assert result.breakdown == {}


class TestRetrievalQuery:
    """Test RetrievalQuery model."""

    def test_create_query_minimal(self):
        """Test creating query with minimal fields."""
        query = RetrievalQuery(query="What are user's preferences?", user_id="u1")

        assert query.query == "What are user's preferences?"
        assert query.user_id == "u1"
        assert query.session_id is None
        assert query.k == 5
        assert query.filters is None

    def test_create_query_full(self):
        """Test creating query with all fields."""
        filters = {"type": "preference"}
        query = RetrievalQuery(
            query="preferences",
            user_id="u1",
            session_id="s1",
            k=10,
            filters=filters,
        )

        assert query.k == 10
        assert query.session_id == "s1"
        assert query.filters == filters

    def test_query_k_validation(self):
        """Test that k must be positive."""
        # Valid k
        query = RetrievalQuery(query="test", user_id="u1", k=1)
        assert query.k == 1

        query = RetrievalQuery(query="test", user_id="u1", k=100)
        assert query.k == 100
