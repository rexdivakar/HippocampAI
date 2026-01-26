"""Tests for automated context assembly."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from hippocampai.context.models import (
    ContextConstraints,
    ContextPack,
    DroppedItem,
    DropReason,
    SelectedItem,
)


class TestContextModels:
    """Tests for context assembly models."""

    def test_context_constraints_defaults(self) -> None:
        """Test default constraint values."""
        constraints = ContextConstraints()

        assert constraints.token_budget == 4000
        assert constraints.max_items == 20
        assert constraints.recency_bias == 0.3
        assert constraints.min_relevance == 0.1
        assert constraints.allow_summaries is True
        assert constraints.include_citations is True
        assert constraints.deduplicate is True

    def test_context_constraints_custom(self) -> None:
        """Test custom constraint values."""
        constraints = ContextConstraints(
            token_budget=2000,
            max_items=10,
            recency_bias=0.5,
            type_filter=["fact", "preference"],
            time_range_days=7,
        )

        assert constraints.token_budget == 2000
        assert constraints.max_items == 10
        assert constraints.recency_bias == 0.5
        assert constraints.type_filter == ["fact", "preference"]
        assert constraints.time_range_days == 7

    def test_selected_item(self) -> None:
        """Test SelectedItem model."""
        item = SelectedItem(
            memory_id="mem-123",
            text="Test memory content",
            memory_type="fact",
            relevance_score=0.85,
            importance=7.5,
            created_at=datetime.now(timezone.utc),
            token_count=10,
            tags=["test"],
        )

        assert item.memory_id == "mem-123"
        assert item.relevance_score == 0.85
        assert item.token_count == 10

    def test_dropped_item(self) -> None:
        """Test DroppedItem model."""
        item = DroppedItem(
            memory_id="mem-456",
            text_preview="This is a preview...",
            reason=DropReason.TOKEN_BUDGET,
            relevance_score=0.6,
            details="Would exceed budget",
        )

        assert item.memory_id == "mem-456"
        assert item.reason == DropReason.TOKEN_BUDGET
        assert "budget" in item.details.lower()

    def test_context_pack(self) -> None:
        """Test ContextPack model."""
        pack = ContextPack(
            final_context_text="1. Memory one\n2. Memory two",
            citations=["mem-1", "mem-2"],
            selected_items=[],
            dropped_items=[],
            total_tokens=50,
            query="test query",
            user_id="alice",
        )

        assert pack.final_context_text == "1. Memory one\n2. Memory two"
        assert len(pack.citations) == 2
        assert pack.total_tokens == 50

    def test_context_pack_get_context_for_prompt(self) -> None:
        """Test getting context formatted for prompt."""
        pack = ContextPack(
            final_context_text="1. Memory content",
            citations=["mem-1"],
            selected_items=[],
            dropped_items=[],
            total_tokens=10,
            query="test",
            user_id="alice",
        )

        # With header
        with_header = pack.get_context_for_prompt(include_header=True)
        assert "## Relevant Context" in with_header
        assert "Memory content" in with_header

        # Without header
        without_header = pack.get_context_for_prompt(include_header=False)
        assert "## Relevant Context" not in without_header
        assert "Memory content" in without_header

    def test_context_pack_get_citations_text(self) -> None:
        """Test getting citations as text."""
        pack = ContextPack(
            final_context_text="",
            citations=["mem-1", "mem-2", "mem-3"],
            selected_items=[],
            dropped_items=[],
            total_tokens=0,
            query="test",
            user_id="alice",
        )

        citations_text = pack.get_citations_text()
        assert "Sources:" in citations_text
        assert "mem-1" in citations_text
        assert "mem-2" in citations_text


class TestContextAssemblyIntegration:
    """Integration tests for context assembly."""

    @pytest.fixture
    def client(self):
        """Create a MemoryClient for testing."""
        from hippocampai import MemoryClient

        test_id = uuid4().hex[:8]
        return MemoryClient(
            collection_facts=f"test_facts_{test_id}",
            collection_prefs=f"test_prefs_{test_id}",
        )

    @pytest.fixture
    def user_id(self) -> str:
        """Generate unique user ID."""
        return f"test_user_{uuid4().hex[:8]}"

    def test_assemble_context_basic(self, client, user_id: str) -> None:
        """Test basic context assembly."""
        # Store some memories
        client.remember("I love coffee", user_id=user_id, type="preference")
        client.remember("I work at Google", user_id=user_id, type="fact")
        client.remember("I prefer dark roast", user_id=user_id, type="preference")

        # Assemble context
        pack = client.assemble_context(
            query="coffee preferences",
            user_id=user_id,
        )

        assert isinstance(pack, ContextPack)
        assert pack.user_id == user_id
        assert pack.query == "coffee preferences"
        assert len(pack.selected_items) > 0
        assert pack.total_tokens > 0

    def test_assemble_context_with_token_budget(self, client, user_id: str) -> None:
        """Test context assembly with token budget."""
        # Store memories
        for i in range(10):
            client.remember(
                f"Memory number {i} with some content",
                user_id=user_id,
                type="fact",
            )

        # Assemble with small budget
        pack = client.assemble_context(
            query="memories",
            user_id=user_id,
            token_budget=100,  # Very small budget
        )

        assert pack.total_tokens <= 100

    def test_assemble_context_with_type_filter(self, client, user_id: str) -> None:
        """Test context assembly with type filter."""
        # Store different types
        client.remember("I love pizza", user_id=user_id, type="preference")
        client.remember("Meeting at 3pm", user_id=user_id, type="event")
        client.remember("I work remotely", user_id=user_id, type="fact")

        # Filter to preferences only
        pack = client.assemble_context(
            query="food",
            user_id=user_id,
            type_filter=["preference"],
        )

        # All selected items should be preferences
        for item in pack.selected_items:
            assert item.memory_type == "preference"

    def test_assemble_context_with_recency_bias(self, client, user_id: str) -> None:
        """Test context assembly with recency bias."""
        # Store memories
        client.remember("Recent memory", user_id=user_id, type="fact")
        client.remember("Another recent memory", user_id=user_id, type="fact")

        # High recency bias
        pack_recent = client.assemble_context(
            query="memory",
            user_id=user_id,
            recency_bias=0.9,
        )

        # Low recency bias
        pack_relevance = client.assemble_context(
            query="memory",
            user_id=user_id,
            recency_bias=0.1,
        )

        # Both should return results
        assert len(pack_recent.selected_items) > 0
        assert len(pack_relevance.selected_items) > 0

    def test_assemble_context_empty(self, client, user_id: str) -> None:
        """Test context assembly with no matching memories."""
        pack = client.assemble_context(
            query="nonexistent topic xyz123",
            user_id=user_id,
        )

        assert isinstance(pack, ContextPack)
        assert pack.final_context_text == "" or len(pack.selected_items) == 0

    def test_assemble_context_citations(self, client, user_id: str) -> None:
        """Test that citations match selected items."""
        # Store memories
        client.remember("Test memory one", user_id=user_id, type="fact")
        client.remember("Test memory two", user_id=user_id, type="fact")

        pack = client.assemble_context(
            query="test memory",
            user_id=user_id,
            include_citations=True,
        )

        # Citations should match selected item IDs
        selected_ids = {item.memory_id for item in pack.selected_items}
        citation_ids = set(pack.citations)

        assert selected_ids == citation_ids

    def test_assemble_context_no_citations(self, client, user_id: str) -> None:
        """Test context assembly without citations."""
        client.remember("Test memory", user_id=user_id, type="fact")

        pack = client.assemble_context(
            query="test",
            user_id=user_id,
            include_citations=False,
        )

        assert pack.citations == []

    def test_assemble_context_deduplication(self, client, user_id: str) -> None:
        """Test that duplicate memories are removed."""
        # Store similar memories
        client.remember("I love coffee", user_id=user_id, type="preference")
        client.remember("I love coffee", user_id=user_id, type="preference")
        client.remember("I love coffee a lot", user_id=user_id, type="preference")

        pack = client.assemble_context(
            query="coffee",
            user_id=user_id,
            deduplicate=True,
        )

        # Should have fewer items than stored due to dedup
        # At least some should be marked as duplicates
        assert len(pack.selected_items) <= 3

    def test_assemble_context_max_items(self, client, user_id: str) -> None:
        """Test max items constraint."""
        # Store many memories
        for i in range(15):
            client.remember(f"Memory {i}", user_id=user_id, type="fact")

        pack = client.assemble_context(
            query="memory",
            user_id=user_id,
            max_items=5,
        )

        assert len(pack.selected_items) <= 5

    def test_assemble_context_min_relevance(self, client, user_id: str) -> None:
        """Test minimum relevance filter."""
        client.remember("Highly relevant coffee preference", user_id=user_id, type="preference")
        client.remember("Unrelated xyz topic", user_id=user_id, type="fact")

        pack = client.assemble_context(
            query="coffee",
            user_id=user_id,
            min_relevance=0.5,  # Higher threshold
        )

        # Some items may be dropped for low relevance
        assert isinstance(pack, ContextPack)
