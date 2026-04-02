"""Truth Maintenance System (TMS) for tracking belief states and evidence chains.

Manages belief records, contradiction detection, confidence propagation,
and automated belief revision. Integrates with the existing MemoryConflictResolver.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.pipeline.conflict_resolution import MemoryConflictResolver

logger = logging.getLogger(__name__)


class BeliefState(str, Enum):
    """State of a belief in the truth maintenance system."""

    ACTIVE = "active"
    RETRACTED = "retracted"
    SUSPENDED = "suspended"
    CONTRADICTED = "contradicted"


class JustificationType(str, Enum):
    """How a belief was justified."""

    DIRECT_OBSERVATION = "direct_observation"
    INFERENCE = "inference"
    USER_STATED = "user_stated"
    LLM_EXTRACTED = "llm_extracted"
    TEMPORAL_PATTERN = "temporal_pattern"


class Justification(BaseModel):
    """A justification for a belief."""

    type: JustificationType
    source_memory_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BeliefRevision(BaseModel):
    """A revision event for a belief."""

    revision_id: str = Field(default_factory=lambda: str(uuid4()))
    belief_id: str
    old_state: BeliefState
    new_state: BeliefState
    old_confidence: float
    new_confidence: float
    reason: str = ""
    triggered_by: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BeliefRecord(BaseModel):
    """A belief tracked by the truth maintenance system."""

    belief_id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str
    user_id: str
    text: str
    state: BeliefState = BeliefState.ACTIVE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    justifications: list[Justification] = Field(default_factory=list)
    revisions: list[BeliefRevision] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    supported_by: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContradictionLink(BaseModel):
    """A link between two contradicting beliefs."""

    link_id: str = Field(default_factory=lambda: str(uuid4()))
    belief_a_id: str
    belief_b_id: str
    conflict_type: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution_strategy: Optional[str] = None


class TruthMaintenanceSystem:
    """Tracks belief states, evidence chains, and performs automated belief revision.

    Integrates with the existing MemoryConflictResolver for contradiction detection.
    """

    def __init__(
        self,
        conflict_resolver: MemoryConflictResolver,
        config: object,
    ) -> None:
        self.conflict_resolver = conflict_resolver
        self.config = config

        # In-memory storage
        self._beliefs: dict[str, BeliefRecord] = {}
        self._beliefs_by_memory: dict[str, str] = {}
        self._contradictions: dict[str, ContradictionLink] = {}
        self._user_beliefs: dict[str, list[str]] = {}

        # Config values with defaults
        self._retraction_threshold: float = getattr(
            config, "tms_retraction_confidence_threshold", 0.3
        )
        self._contradiction_penalty: float = getattr(config, "tms_contradiction_penalty", 0.5)

    def evaluate_belief(
        self,
        memory_id: str,
        text: str,
        user_id: str,
        confidence: float = 1.0,
        justification_type: str = "direct_observation",
        source_memory_ids: Optional[list[str]] = None,
    ) -> BeliefRecord:
        """Evaluate and register a belief from a memory.

        Creates a new BeliefRecord, detects contradictions with existing beliefs,
        and returns the record.
        """
        # Parse justification type
        try:
            j_type = JustificationType(justification_type)
        except ValueError:
            j_type = JustificationType.DIRECT_OBSERVATION

        justification = Justification(
            type=j_type,
            source_memory_ids=source_memory_ids or [],
            confidence=confidence,
            reasoning=f"Belief from memory {memory_id}",
        )

        belief = BeliefRecord(
            memory_id=memory_id,
            user_id=user_id,
            text=text,
            confidence=min(max(confidence, 0.0), 1.0),
            justifications=[justification],
        )

        # Store the belief
        self._beliefs[belief.belief_id] = belief
        self._beliefs_by_memory[memory_id] = belief.belief_id

        if user_id not in self._user_beliefs:
            self._user_beliefs[user_id] = []
        self._user_beliefs[user_id].append(belief.belief_id)

        # Detect contradictions with existing beliefs for this user
        existing_beliefs = [
            self._beliefs[bid]
            for bid in self._user_beliefs.get(user_id, [])
            if bid != belief.belief_id and self._beliefs[bid].state == BeliefState.ACTIVE
        ]

        contradictions = self.detect_contradictions(belief, existing_beliefs)
        for link in contradictions:
            belief.contradicts.append(link.belief_a_id)
            other_belief = self._beliefs.get(link.belief_a_id)
            if other_belief:
                other_belief.contradicts.append(belief.belief_id)

        logger.info(
            f"Evaluated belief {belief.belief_id} for memory {memory_id}, "
            f"contradictions={len(contradictions)}"
        )
        return belief

    def get_belief(self, belief_id: str) -> Optional[BeliefRecord]:
        """Get a belief by its ID."""
        return self._beliefs.get(belief_id)

    def get_belief_by_memory(self, memory_id: str) -> Optional[BeliefRecord]:
        """Get a belief by its associated memory ID."""
        belief_id = self._beliefs_by_memory.get(memory_id)
        if belief_id:
            return self._beliefs.get(belief_id)
        return None

    def detect_contradictions(
        self,
        belief: BeliefRecord,
        existing_beliefs: list[BeliefRecord],
    ) -> list[ContradictionLink]:
        """Detect contradictions between a belief and existing beliefs.

        Delegates to the conflict_resolver's detect_conflicts for semantic analysis,
        using lightweight Memory-like objects.
        """
        if not existing_beliefs:
            return []

        links: list[ContradictionLink] = []

        # Use simple text overlap heuristic for contradiction detection
        # (avoids requiring full Memory objects with embeddings)
        from hippocampai.models.memory import Memory, MemoryType

        new_memory = Memory(
            text=belief.text,
            user_id=belief.user_id,
            type=MemoryType.FACT,
        )

        existing_memories = [
            Memory(
                text=eb.text,
                user_id=eb.user_id,
                type=MemoryType.FACT,
            )
            for eb in existing_beliefs
        ]

        try:
            conflicts = self.conflict_resolver.detect_conflicts(
                new_memory, existing_memories, check_llm=False
            )
        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")
            return links

        for conflict in conflicts:
            # Map conflict back to belief IDs
            # conflict.memory_1 is the existing memory that matched
            matching_idx = next(
                (j for j, em in enumerate(existing_memories) if em.id == conflict.memory_1.id),
                None,
            )
            if matching_idx is not None and matching_idx < len(existing_beliefs):
                other_belief = existing_beliefs[matching_idx]
                link = ContradictionLink(
                    belief_a_id=other_belief.belief_id,
                    belief_b_id=belief.belief_id,
                    conflict_type=conflict.conflict_type.value,
                )
                self._contradictions[link.link_id] = link
                links.append(link)

                # Apply contradiction penalty to both beliefs
                self._apply_contradiction_penalty(belief)
                self._apply_contradiction_penalty(other_belief)

        return links

    def _apply_contradiction_penalty(self, belief: BeliefRecord) -> None:
        """Apply a confidence penalty to a contradicted belief."""
        old_confidence = belief.confidence
        belief.confidence = max(0.0, belief.confidence * (1.0 - self._contradiction_penalty))
        if belief.confidence < self._retraction_threshold:
            if belief.state == BeliefState.ACTIVE:
                belief.state = BeliefState.CONTRADICTED
        belief.updated_at = datetime.now(timezone.utc)

        if old_confidence != belief.confidence:
            revision = BeliefRevision(
                belief_id=belief.belief_id,
                old_state=BeliefState.ACTIVE,
                new_state=belief.state,
                old_confidence=old_confidence,
                new_confidence=belief.confidence,
                reason="Contradiction detected",
            )
            belief.revisions.append(revision)

    def propagate_confidence(self, belief_id: str) -> None:
        """Propagate confidence changes when a supporting belief is retracted.

        Reduces confidence of beliefs that depend on this one.
        """
        belief = self._beliefs.get(belief_id)
        if belief is None:
            return

        # Find beliefs that list this belief as a support
        for other_id, other_belief in self._beliefs.items():
            if other_id == belief_id:
                continue
            if belief_id in other_belief.supported_by and other_belief.state == BeliefState.ACTIVE:
                old_confidence = other_belief.confidence
                # Reduce confidence proportionally
                reduction_factor = 1.0 - (self._contradiction_penalty * 0.5)
                other_belief.confidence = max(0.0, other_belief.confidence * reduction_factor)
                other_belief.updated_at = datetime.now(timezone.utc)

                revision = BeliefRevision(
                    belief_id=other_belief.belief_id,
                    old_state=other_belief.state,
                    new_state=other_belief.state,
                    old_confidence=old_confidence,
                    new_confidence=other_belief.confidence,
                    reason=f"Supporting belief {belief_id} was retracted",
                    triggered_by=belief_id,
                )
                other_belief.revisions.append(revision)

                # Check if confidence dropped below threshold
                if other_belief.confidence < self._retraction_threshold:
                    other_belief.state = BeliefState.SUSPENDED
                    revision.new_state = BeliefState.SUSPENDED

                logger.info(
                    f"Propagated confidence reduction to belief {other_id}: "
                    f"{old_confidence:.2f} -> {other_belief.confidence:.2f}"
                )

    def revise_belief(
        self,
        belief_id: str,
        new_state: str,
        reason: str = "",
        triggered_by: Optional[str] = None,
    ) -> BeliefRevision:
        """Revise a belief's state.

        Args:
            belief_id: ID of the belief to revise.
            new_state: New state value (active, retracted, suspended, contradicted).
            reason: Reason for the revision.
            triggered_by: Optional ID of the belief that triggered this revision.

        Returns:
            The BeliefRevision record.

        Raises:
            ValueError: If belief_id is not found or new_state is invalid.
        """
        belief = self._beliefs.get(belief_id)
        if belief is None:
            raise ValueError(f"Belief {belief_id} not found")

        try:
            parsed_state = BeliefState(new_state)
        except ValueError:
            raise ValueError(f"Invalid belief state: {new_state}")

        old_state = belief.state
        old_confidence = belief.confidence

        belief.state = parsed_state
        belief.updated_at = datetime.now(timezone.utc)

        # Adjust confidence based on state
        if parsed_state == BeliefState.RETRACTED:
            belief.confidence = 0.0
        elif parsed_state == BeliefState.SUSPENDED:
            belief.confidence = min(belief.confidence, self._retraction_threshold)

        revision = BeliefRevision(
            belief_id=belief_id,
            old_state=old_state,
            new_state=parsed_state,
            old_confidence=old_confidence,
            new_confidence=belief.confidence,
            reason=reason,
            triggered_by=triggered_by,
        )
        belief.revisions.append(revision)

        # Propagate changes if belief was retracted
        if parsed_state in (BeliefState.RETRACTED, BeliefState.SUSPENDED):
            self.propagate_confidence(belief_id)

        logger.info(f"Revised belief {belief_id}: {old_state.value} -> {parsed_state.value}")
        return revision

    def resolve_contradiction(
        self,
        link_id: str,
        winning_belief_id: str,
        strategy: str = "manual",
    ) -> ContradictionLink:
        """Resolve a contradiction by choosing a winning belief.

        The losing belief is retracted.

        Args:
            link_id: ID of the ContradictionLink to resolve.
            winning_belief_id: ID of the belief that should remain ACTIVE.
            strategy: Description of the resolution strategy used.

        Returns:
            The updated ContradictionLink.

        Raises:
            ValueError: If link_id or winning_belief_id is not found.
        """
        link = self._contradictions.get(link_id)
        if link is None:
            raise ValueError(f"Contradiction link {link_id} not found")

        if winning_belief_id not in (link.belief_a_id, link.belief_b_id):
            raise ValueError(
                f"Winning belief {winning_belief_id} is not part of contradiction {link_id}"
            )

        # Determine loser
        loser_id = link.belief_b_id if winning_belief_id == link.belief_a_id else link.belief_a_id

        # Retract the loser
        self.revise_belief(
            loser_id,
            new_state="retracted",
            reason=f"Lost contradiction resolution (strategy={strategy})",
            triggered_by=winning_belief_id,
        )

        # Restore winner confidence if it was penalized
        winner = self._beliefs.get(winning_belief_id)
        if winner and winner.state == BeliefState.CONTRADICTED:
            winner.state = BeliefState.ACTIVE
            winner.confidence = min(1.0, winner.confidence / (1.0 - self._contradiction_penalty))
            winner.updated_at = datetime.now(timezone.utc)

        link.resolved = True
        link.resolution_strategy = strategy
        return link

    def get_contradictions(
        self,
        user_id: str,
        include_resolved: bool = False,
    ) -> list[ContradictionLink]:
        """Get all contradictions for a user's beliefs.

        Args:
            user_id: The user whose contradictions to retrieve.
            include_resolved: Whether to include already-resolved contradictions.

        Returns:
            List of ContradictionLink objects.
        """
        user_belief_ids = set(self._user_beliefs.get(user_id, []))
        result: list[ContradictionLink] = []

        for link in self._contradictions.values():
            if link.belief_a_id in user_belief_ids or link.belief_b_id in user_belief_ids:
                if include_resolved or not link.resolved:
                    result.append(link)

        return result

    def get_belief_history(self, belief_id: str) -> list[BeliefRevision]:
        """Get the revision history for a belief.

        Args:
            belief_id: ID of the belief.

        Returns:
            List of BeliefRevision records in chronological order.

        Raises:
            ValueError: If belief_id is not found.
        """
        belief = self._beliefs.get(belief_id)
        if belief is None:
            raise ValueError(f"Belief {belief_id} not found")
        return list(belief.revisions)
