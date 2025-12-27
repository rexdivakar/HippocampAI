"""Tests for bi-temporal fact tracking."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from hippocampai.models.bitemporal import (
    BiTemporalFact,
    BiTemporalQuery,
    FactRevision,
    FactStatus,
)


class TestBiTemporalFact:
    """Tests for BiTemporalFact model."""

    def test_create_fact(self) -> None:
        """Test creating a bi-temporal fact."""
        fact = BiTemporalFact(
            text="Alice works at Google",
            user_id="alice",
            entity_id="alice",
            property_name="employer",
        )

        assert fact.text == "Alice works at Google"
        assert fact.user_id == "alice"
        assert fact.entity_id == "alice"
        assert fact.property_name == "employer"
        assert fact.status == FactStatus.ACTIVE
        assert fact.valid_to is None  # Currently valid
        assert fact.id is not None
        assert fact.fact_id is not None

    def test_is_valid_at(self) -> None:
        """Test point-in-time validity check."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=30)
        future = now + timedelta(days=30)

        # Fact valid from past, no end
        fact = BiTemporalFact(
            text="Test fact",
            user_id="test",
            valid_from=past,
            valid_to=None,
        )

        assert fact.is_valid_at(now) is True
        assert fact.is_valid_at(past) is True
        assert fact.is_valid_at(future) is True
        assert fact.is_valid_at(past - timedelta(days=1)) is False

    def test_is_valid_at_with_end(self) -> None:
        """Test validity check with end date."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)
        end = now - timedelta(days=10)

        fact = BiTemporalFact(
            text="Test fact",
            user_id="test",
            valid_from=start,
            valid_to=end,
        )

        assert fact.is_valid_at(start) is True
        assert fact.is_valid_at(start + timedelta(days=10)) is True
        assert fact.is_valid_at(end) is False  # End is exclusive
        assert fact.is_valid_at(now) is False

    def test_is_currently_valid(self) -> None:
        """Test current validity check."""
        now = datetime.now(timezone.utc)

        # Currently valid fact
        valid_fact = BiTemporalFact(
            text="Valid fact",
            user_id="test",
            valid_from=now - timedelta(days=1),
            valid_to=None,
        )
        assert valid_fact.is_currently_valid() is True

        # Expired fact
        expired_fact = BiTemporalFact(
            text="Expired fact",
            user_id="test",
            valid_from=now - timedelta(days=30),
            valid_to=now - timedelta(days=1),
        )
        assert expired_fact.is_currently_valid() is False

    def test_overlaps_interval(self) -> None:
        """Test interval overlap check."""
        now = datetime.now(timezone.utc)
        fact_start = now - timedelta(days=20)
        fact_end = now - timedelta(days=10)

        fact = BiTemporalFact(
            text="Test fact",
            user_id="test",
            valid_from=fact_start,
            valid_to=fact_end,
        )

        # Query interval overlaps
        assert fact.overlaps_interval(
            now - timedelta(days=25),
            now - timedelta(days=15),
        ) is True

        # Query interval contains fact
        assert fact.overlaps_interval(
            now - timedelta(days=30),
            now,
        ) is True

        # Query interval before fact
        assert fact.overlaps_interval(
            now - timedelta(days=40),
            now - timedelta(days=25),
        ) is False

        # Query interval after fact
        assert fact.overlaps_interval(
            now - timedelta(days=5),
            now,
        ) is False


class TestBiTemporalQuery:
    """Tests for BiTemporalQuery model."""

    def test_create_query(self) -> None:
        """Test creating a bi-temporal query."""
        query = BiTemporalQuery(
            user_id="alice",
            entity_id="alice",
            property_name="employer",
        )

        assert query.user_id == "alice"
        assert query.entity_id == "alice"
        assert query.property_name == "employer"
        assert query.include_superseded is False
        assert query.include_retracted is False

    def test_as_of_query(self) -> None:
        """Test as-of system time query."""
        past = datetime.now(timezone.utc) - timedelta(days=30)

        query = BiTemporalQuery(
            user_id="alice",
            as_of_system_time=past,
        )

        assert query.as_of_system_time == past

    def test_valid_time_range_query(self) -> None:
        """Test valid-time range query."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 4, 1, tzinfo=timezone.utc)

        query = BiTemporalQuery(
            user_id="alice",
            valid_from=start,
            valid_to=end,
        )

        assert query.valid_from == start
        assert query.valid_to == end


class TestFactRevision:
    """Tests for FactRevision model."""

    def test_create_revision(self) -> None:
        """Test creating a fact revision."""
        revision = FactRevision(
            original_fact_id="fact-123",
            new_text="Alice works at Microsoft",
            reason="job_change",
        )

        assert revision.original_fact_id == "fact-123"
        assert revision.new_text == "Alice works at Microsoft"
        assert revision.reason == "job_change"
        assert revision.confidence == 0.9


class TestBiTemporalIntegration:
    """Integration tests for bi-temporal functionality."""

    @pytest.fixture
    def client(self):
        """Create a MemoryClient for testing."""
        import os

        from hippocampai import MemoryClient

        test_id = uuid4().hex[:8]
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        return MemoryClient(
            collection_facts=f"test_facts_{test_id}",
            collection_prefs=f"test_prefs_{test_id}",
            qdrant_url=qdrant_url,
        )

    @pytest.fixture
    def user_id(self) -> str:
        """Generate unique user ID."""
        return f"test_user_{uuid4().hex[:8]}"

    def test_store_and_query_fact(self, client, user_id: str) -> None:
        """Test storing and querying a bi-temporal fact."""
        # Store a fact
        fact = client.store_bitemporal_fact(
            text="Alice works at Google",
            user_id=user_id,
            entity_id="alice",
            property_name="employer",
        )

        assert fact.text == "Alice works at Google"
        assert fact.status == FactStatus.ACTIVE

        # Query the fact
        result = client.query_bitemporal_facts(
            user_id=user_id,
            entity_id="alice",
            property_name="employer",
        )

        assert result.total_count >= 1
        assert any(f.text == "Alice works at Google" for f in result.facts)

    def test_revise_fact(self, client, user_id: str) -> None:
        """Test revising a bi-temporal fact."""
        # Store original fact
        original = client.store_bitemporal_fact(
            text="Alice works at Google",
            user_id=user_id,
            entity_id="alice",
            property_name="employer",
        )

        # Revise the fact
        revised = client.revise_bitemporal_fact(
            original_fact_id=original.id,
            new_text="Alice works at Microsoft",
            user_id=user_id,
            reason="job_change",
        )

        assert revised.text == "Alice works at Microsoft"
        assert revised.supersedes == original.id
        assert revised.fact_id == original.fact_id  # Same logical fact

        # Query should return only the revised fact by default
        result = client.query_bitemporal_facts(
            user_id=user_id,
            entity_id="alice",
            property_name="employer",
        )

        active_facts = [f for f in result.facts if f.status == FactStatus.ACTIVE]
        assert len(active_facts) >= 1
        assert any(f.text == "Alice works at Microsoft" for f in active_facts)

    def test_query_with_superseded(self, client, user_id: str) -> None:
        """Test querying with superseded facts included."""
        # Store and revise a fact
        original = client.store_bitemporal_fact(
            text="Alice is 25 years old",
            user_id=user_id,
            entity_id="alice",
            property_name="age",
        )

        client.revise_bitemporal_fact(
            original_fact_id=original.id,
            new_text="Alice is 26 years old",
            user_id=user_id,
            reason="birthday",
        )

        # Query with superseded included
        result = client.query_bitemporal_facts(
            user_id=user_id,
            entity_id="alice",
            property_name="age",
            include_superseded=True,
        )

        # Should have both versions
        texts = [f.text for f in result.facts]
        assert "Alice is 26 years old" in texts

    def test_retract_fact(self, client, user_id: str) -> None:
        """Test retracting a bi-temporal fact."""
        # Store a fact
        fact = client.store_bitemporal_fact(
            text="Alice likes coffee",
            user_id=user_id,
        )

        # Retract it
        success = client.retract_bitemporal_fact(
            fact_id=fact.id,
            reason="incorrect",
        )

        assert success is True

        # Query should not return retracted facts by default
        result = client.query_bitemporal_facts(user_id=user_id)
        active_ids = [f.id for f in result.facts if f.status == FactStatus.ACTIVE]
        assert fact.id not in active_ids

    def test_get_latest_valid_fact(self, client, user_id: str) -> None:
        """Test getting the latest valid fact."""
        # Store multiple revisions
        fact1 = client.store_bitemporal_fact(
            text="Alice lives in NYC",
            user_id=user_id,
            entity_id="alice",
            property_name="location",
        )

        client.revise_bitemporal_fact(
            original_fact_id=fact1.id,
            new_text="Alice lives in SF",
            user_id=user_id,
            reason="moved",
        )

        # Get latest
        latest = client.get_latest_valid_fact(
            user_id=user_id,
            entity_id="alice",
            property_name="location",
        )

        assert latest is not None
        assert latest.text == "Alice lives in SF"

    def test_get_fact_history(self, client, user_id: str) -> None:
        """Test getting fact history."""
        # Store and revise
        fact1 = client.store_bitemporal_fact(
            text="Version 1",
            user_id=user_id,
            entity_id="test",
            property_name="version",
        )

        client.revise_bitemporal_fact(
            original_fact_id=fact1.id,
            new_text="Version 2",
            user_id=user_id,
            reason="update",
        )

        # Get history
        history = client.get_bitemporal_fact_history(fact1.fact_id)

        assert len(history) >= 2
        # History should be sorted by system_time
        assert history[0].system_time <= history[-1].system_time
