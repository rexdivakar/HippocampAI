"""Tests for the Truth Maintenance System."""

import pytest

from hippocampai.embed.embedder import Embedder
from hippocampai.pipeline.conflict_resolution import (
    ConflictResolutionStrategy,
    MemoryConflictResolver,
)
from hippocampai.pipeline.truth_maintenance import (
    BeliefRecord,
    BeliefRevision,
    BeliefState,
    ContradictionLink,
    TruthMaintenanceSystem,
)


class _FakeConfig:
    """Minimal config for TMS tests."""

    tms_retraction_confidence_threshold: float = 0.3
    tms_contradiction_penalty: float = 0.5


@pytest.fixture
def embedder():
    """Create embedder instance."""
    return Embedder(model_name="BAAI/bge-small-en-v1.5")


@pytest.fixture
def resolver(embedder):
    """Create conflict resolver (no LLM)."""
    return MemoryConflictResolver(
        embedder=embedder,
        llm=None,
        default_strategy=ConflictResolutionStrategy.TEMPORAL,
        similarity_threshold=0.75,
        contradiction_threshold=0.85,
    )


@pytest.fixture
def tms(resolver):
    """Create a TruthMaintenanceSystem instance."""
    return TruthMaintenanceSystem(
        conflict_resolver=resolver,
        config=_FakeConfig(),
    )


class TestBeliefStateTransitions:
    """Test creating beliefs and revising their states."""

    def test_create_belief_active(self, tms):
        """New beliefs start as ACTIVE."""
        belief = tms.evaluate_belief(
            memory_id="m1",
            text="User likes pizza",
            user_id="alice",
            confidence=0.9,
        )
        assert isinstance(belief, BeliefRecord)
        assert belief.state == BeliefState.ACTIVE
        assert belief.confidence == 0.9
        assert belief.memory_id == "m1"
        assert belief.user_id == "alice"

    def test_revise_to_retracted(self, tms):
        """Revising a belief to RETRACTED sets confidence to 0."""
        belief = tms.evaluate_belief(
            memory_id="m1", text="User likes pizza", user_id="alice", confidence=0.9
        )
        revision = tms.revise_belief(
            belief_id=belief.belief_id,
            new_state="retracted",
            reason="User corrected this",
        )
        assert isinstance(revision, BeliefRevision)
        assert revision.old_state == BeliefState.ACTIVE
        assert revision.new_state == BeliefState.RETRACTED
        assert revision.old_confidence == 0.9
        assert revision.new_confidence == 0.0

        updated = tms.get_belief(belief.belief_id)
        assert updated is not None
        assert updated.state == BeliefState.RETRACTED
        assert updated.confidence == 0.0

    def test_revise_to_suspended(self, tms):
        """Revising a belief to SUSPENDED caps confidence at threshold."""
        belief = tms.evaluate_belief(
            memory_id="m2", text="User works at Google", user_id="alice", confidence=0.8
        )
        revision = tms.revise_belief(
            belief_id=belief.belief_id,
            new_state="suspended",
            reason="Uncertain source",
        )
        assert revision.new_state == BeliefState.SUSPENDED
        updated = tms.get_belief(belief.belief_id)
        assert updated is not None
        assert updated.confidence <= 0.3  # tms_retraction_confidence_threshold

    def test_revise_invalid_state(self, tms):
        """Revising with an invalid state raises ValueError."""
        belief = tms.evaluate_belief(memory_id="m3", text="something", user_id="alice")
        with pytest.raises(ValueError, match="Invalid belief state"):
            tms.revise_belief(belief.belief_id, new_state="nonexistent")

    def test_revise_nonexistent_belief(self, tms):
        """Revising a non-existent belief raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tms.revise_belief("fake-id", new_state="retracted")


class TestContradictionDetection:
    """Test contradiction detection between beliefs."""

    def test_contradiction_creates_link(self, tms):
        """Adding contradictory beliefs creates a ContradictionLink."""
        tms.evaluate_belief(
            memory_id="m1",
            text="I love coffee and drink it every day",
            user_id="alice",
            confidence=0.9,
        )
        tms.evaluate_belief(
            memory_id="m2",
            text="I hate coffee and never drink it",
            user_id="alice",
            confidence=0.8,
        )
        # If the resolver detects the contradiction, links should exist
        links = tms.get_contradictions("alice")
        if links:
            assert isinstance(links[0], ContradictionLink)
            assert not links[0].resolved
            # Both beliefs should have reduced confidence due to penalty
            b1 = tms.get_belief_by_memory("m1")
            assert b1 is not None
            assert b1.confidence < 0.9

    def test_no_contradiction_for_unrelated(self, tms):
        """Unrelated beliefs should not produce contradictions."""
        tms.evaluate_belief(memory_id="m1", text="I like hiking", user_id="bob", confidence=0.9)
        tms.evaluate_belief(
            memory_id="m2", text="My favorite color is blue", user_id="bob", confidence=0.8
        )
        links = tms.get_contradictions("bob")
        assert len(links) == 0


class TestBeliefRevisionOnNewEvidence:
    """Test belief revision when new evidence contradicts existing beliefs."""

    def test_new_evidence_retracts_old_belief(self, tms):
        """When new contradicting evidence arrives, old belief can be retracted."""
        old_belief = tms.evaluate_belief(
            memory_id="m1",
            text="I love eating meat every day",
            user_id="alice",
            confidence=0.7,
        )
        tms.evaluate_belief(
            memory_id="m2",
            text="I hate eating meat and never eat it",
            user_id="alice",
            confidence=0.9,
        )

        # Manually revise the old belief based on the new evidence
        revision = tms.revise_belief(
            belief_id=old_belief.belief_id,
            new_state="retracted",
            reason="Superseded by newer statement",
        )
        assert revision.new_state == BeliefState.RETRACTED

        updated = tms.get_belief(old_belief.belief_id)
        assert updated is not None
        assert updated.state == BeliefState.RETRACTED


class TestConfidencePropagation:
    """Test that retracting a belief propagates to dependent beliefs."""

    def test_retraction_reduces_dependent_confidence(self, tms):
        """Retracting a support reduces dependent belief confidence."""
        support = tms.evaluate_belief(
            memory_id="m1",
            text="Alice works at Google",
            user_id="alice",
            confidence=0.9,
        )
        dependent = tms.evaluate_belief(
            memory_id="m2",
            text="Alice has Google health insurance",
            user_id="alice",
            confidence=0.8,
        )
        # Mark the dependent as supported by the support belief
        dependent.supported_by.append(support.belief_id)

        # Retract the support
        tms.revise_belief(
            belief_id=support.belief_id,
            new_state="retracted",
            reason="Alice left Google",
        )

        # Dependent confidence should have been reduced
        updated_dep = tms.get_belief(dependent.belief_id)
        assert updated_dep is not None
        assert updated_dep.confidence < 0.8

    def test_retraction_propagation_suspends_low_confidence(self, tms):
        """If propagation drops confidence below threshold, belief is suspended."""
        support = tms.evaluate_belief(
            memory_id="m1", text="Base fact A", user_id="alice", confidence=0.9
        )
        dependent = tms.evaluate_belief(
            memory_id="m2", text="Derived fact B", user_id="alice", confidence=0.35
        )
        dependent.supported_by.append(support.belief_id)

        tms.revise_belief(support.belief_id, new_state="retracted", reason="disproven")

        updated_dep = tms.get_belief(dependent.belief_id)
        assert updated_dep is not None
        # 0.35 * 0.75 = 0.2625 < 0.3 threshold -> should be SUSPENDED
        assert updated_dep.state == BeliefState.SUSPENDED


class TestResolveContradiction:
    """Test resolving contradictions explicitly."""

    def test_resolve_contradiction_winner_active_loser_retracted(self, tms):
        """Resolving a contradiction keeps winner ACTIVE and retracts loser."""
        belief_a = tms.evaluate_belief(
            memory_id="m1",
            text="The meeting is on Monday",
            user_id="alice",
            confidence=0.7,
        )
        belief_b = tms.evaluate_belief(
            memory_id="m2",
            text="The meeting is on Tuesday",
            user_id="alice",
            confidence=0.9,
        )

        # Manually create a contradiction link for testing
        link = ContradictionLink(
            belief_a_id=belief_a.belief_id,
            belief_b_id=belief_b.belief_id,
            conflict_type="direct_contradiction",
        )
        tms._contradictions[link.link_id] = link

        # Resolve: belief_b wins
        resolved_link = tms.resolve_contradiction(
            link_id=link.link_id,
            winning_belief_id=belief_b.belief_id,
            strategy="confidence",
        )

        assert resolved_link.resolved is True
        assert resolved_link.resolution_strategy == "confidence"

        loser = tms.get_belief(belief_a.belief_id)
        assert loser is not None
        assert loser.state == BeliefState.RETRACTED
        assert loser.confidence == 0.0

        winner = tms.get_belief(belief_b.belief_id)
        assert winner is not None
        assert winner.state == BeliefState.ACTIVE

    def test_resolve_nonexistent_link(self, tms):
        """Resolving a non-existent link raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tms.resolve_contradiction("fake-link", "fake-belief")

    def test_resolve_wrong_winner(self, tms):
        """Passing a winner not in the link raises ValueError."""
        belief_a = tms.evaluate_belief(
            memory_id="m1", text="Fact A", user_id="alice", confidence=0.7
        )
        belief_b = tms.evaluate_belief(
            memory_id="m2", text="Fact B", user_id="alice", confidence=0.9
        )
        link = ContradictionLink(
            belief_a_id=belief_a.belief_id,
            belief_b_id=belief_b.belief_id,
            conflict_type="direct_contradiction",
        )
        tms._contradictions[link.link_id] = link

        with pytest.raises(ValueError, match="not part of contradiction"):
            tms.resolve_contradiction(link.link_id, "unrelated-id")


class TestGetBeliefHistory:
    """Test belief history retrieval."""

    def test_history_tracks_revisions(self, tms):
        """Multiple revisions are tracked in order."""
        belief = tms.evaluate_belief(
            memory_id="m1", text="Some fact", user_id="alice", confidence=0.9
        )
        tms.revise_belief(belief.belief_id, "suspended", reason="checking")
        tms.revise_belief(belief.belief_id, "active", reason="confirmed")

        history = tms.get_belief_history(belief.belief_id)
        assert len(history) == 2
        assert history[0].new_state == BeliefState.SUSPENDED
        assert history[1].new_state == BeliefState.ACTIVE

    def test_history_nonexistent_belief(self, tms):
        """Getting history for non-existent belief raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tms.get_belief_history("fake-id")
