"""Bi-temporal storage adapter for Qdrant.

Provides bi-temporal query capabilities on top of the existing Qdrant store.
Facts are stored with temporal metadata and queried using payload filters.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from qdrant_client.models import Distance, VectorParams

from hippocampai.models.bitemporal import (
    BiTemporalFact,
    BiTemporalQuery,
    BiTemporalQueryResult,
    FactRevision,
    FactStatus,
)
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Collection name for bi-temporal facts
BITEMPORAL_COLLECTION = "hippocampai_bitemporal_facts"


class BiTemporalStore:
    """Storage adapter for bi-temporal facts.

    Wraps QdrantStore to provide bi-temporal query semantics.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        collection_name: str = BITEMPORAL_COLLECTION,
    ) -> None:
        """Initialize bi-temporal store.

        Args:
            qdrant_store: Underlying Qdrant store
            collection_name: Collection for bi-temporal facts
        """
        self.qdrant = qdrant_store
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the bi-temporal collection exists."""
        try:
            if not self.qdrant.client.collection_exists(self.collection_name):
                self.qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.qdrant.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created bi-temporal collection: {self.collection_name}")
        except ConnectionError as e:
            logger.warning(
                f"Could not connect to Qdrant to ensure bi-temporal collection: {e}. "
                "Collection will be created on first use."
            )
        except ValueError as e:
            logger.error(f"Invalid configuration for bi-temporal collection: {e}")
            raise
        except Exception as e:
            # Log full exception for debugging but allow initialization to continue
            logger.warning(
                f"Could not ensure bi-temporal collection '{self.collection_name}': {e}. "
                f"This may indicate Qdrant is not available yet.",
                exc_info=True,
            )

    def store_fact(
        self,
        fact: BiTemporalFact,
        vector: list[float],
    ) -> BiTemporalFact:
        """Store a bi-temporal fact.

        Args:
            fact: The fact to store
            vector: Embedding vector for the fact text

        Returns:
            The stored fact
        """
        payload = self._fact_to_payload(fact)

        self.qdrant.upsert(
            collection_name=self.collection_name,
            id=fact.id,
            vector=np.array(vector),
            payload=payload,
        )

        logger.debug(f"Stored bi-temporal fact: {fact.id} (fact_id={fact.fact_id})")
        return fact

    def revise_fact(
        self,
        revision: FactRevision,
        vector: list[float],
        user_id: str,
    ) -> BiTemporalFact:
        """Create a revision of an existing fact.

        This supersedes the original fact without deleting it.

        Args:
            revision: The revision details
            vector: Embedding vector for the new text
            user_id: User ID for the new fact

        Returns:
            The new fact version
        """
        # Get the original fact
        original = self.get_fact_by_id(revision.original_fact_id)
        if original is None:
            raise ValueError(f"Original fact not found: {revision.original_fact_id}")

        now = datetime.now(timezone.utc)

        # Create new fact version first (don't mutate original yet)
        new_fact = BiTemporalFact(
            fact_id=original.fact_id,  # Same logical fact
            text=revision.new_text,
            user_id=user_id,
            entity_id=original.entity_id,
            property_name=original.property_name,
            event_time=now,
            valid_from=revision.new_valid_from or now,
            valid_to=revision.new_valid_to,
            system_time=now,
            status=FactStatus.ACTIVE,
            supersedes=original.id,
            confidence=revision.confidence,
            source=f"revision:{revision.reason}",
            metadata={**original.metadata, **revision.metadata},
        )

        # Store the new fact first - this can exist independently
        self.store_fact(new_fact, vector)

        # Now create an updated copy of the original fact (don't mutate the input)
        # This ensures atomicity - if this fails, new_fact exists but original is unchanged
        updated_original_payload = self._fact_to_payload(original)
        updated_original_payload["status"] = FactStatus.SUPERSEDED.value
        updated_original_payload["superseded_by"] = new_fact.id
        if original.valid_to is None:
            updated_original_payload["valid_to"] = now.isoformat()

        # Update original fact in storage
        try:
            self.qdrant.update(
                collection_name=self.collection_name,
                id=original.id,
                payload=updated_original_payload,
            )
        except Exception as e:
            # Log error but don't fail - new_fact is stored, we can reconcile later
            logger.error(
                f"Failed to mark original fact {original.id} as superseded: {e}. "
                f"New fact {new_fact.id} was created successfully."
            )

        logger.info(
            f"Revised fact {original.id} -> {new_fact.id} "
            f"(fact_id={original.fact_id}, reason={revision.reason})"
        )

        return new_fact

    def retract_fact(self, fact_id: str, reason: str = "retracted") -> bool:
        """Retract a fact (mark as invalid without deleting).

        Args:
            fact_id: ID of the fact to retract
            reason: Reason for retraction

        Returns:
            True if retracted successfully
        """
        fact = self.get_fact_by_id(fact_id)
        if fact is None:
            return False

        now = datetime.now(timezone.utc)
        fact.status = FactStatus.RETRACTED
        fact.valid_to = now
        fact.metadata["retraction_reason"] = reason
        fact.metadata["retracted_at"] = now.isoformat()

        payload = self._fact_to_payload(fact)

        # Use update method to change payload without needing vector
        try:
            result = self.qdrant.update(self.collection_name, fact_id, payload)
            success = bool(result)
            if success:
                logger.info(f"Retracted fact: {fact_id} (reason={reason})")
            return success
        except Exception as e:
            logger.error(f"Failed to retract fact {fact_id}: {e}")
            return False

    def query(
        self,
        query: BiTemporalQuery,
        query_vector: Optional[list[float]] = None,
    ) -> BiTemporalQueryResult:
        """Query bi-temporal facts.

        Args:
            query: Query parameters
            query_vector: Optional embedding vector for semantic search

        Returns:
            Query results
        """
        # Build filters
        filters = self._build_query_filters(query)

        # Execute query
        if query_vector is not None:
            # Semantic search with filters
            results = self.qdrant.search(
                collection_name=self.collection_name,
                vector=np.array(query_vector),
                limit=query.limit,
                filters=filters,
            )
        else:
            # Filter-only query (scroll)
            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                filters=filters,
                limit=query.limit,
            )

        # Convert to facts
        facts = [self._payload_to_fact(r["payload"], r["id"]) for r in results]

        # Apply post-filters for complex temporal logic
        facts = self._apply_temporal_filters(facts, query)

        return BiTemporalQueryResult(
            facts=facts,
            total_count=len(facts),
            query=query,
            as_of_system_time=query.as_of_system_time,
        )

    def get_fact_by_id(self, fact_id: str) -> Optional[BiTemporalFact]:
        """Get a specific fact by ID."""
        result = self.qdrant.get(self.collection_name, fact_id)
        if result is None:
            return None
        return self._payload_to_fact(result["payload"], fact_id)

    def get_fact_history(self, fact_id: str) -> list[BiTemporalFact]:
        """Get all versions of a logical fact.

        Args:
            fact_id: The logical fact ID (not version ID)

        Returns:
            All versions sorted by system_time
        """
        results = self.qdrant.scroll(
            collection_name=self.collection_name,
            filters={"fact_id": fact_id},
            limit=1000,
        )

        facts = [self._payload_to_fact(r["payload"], r["id"]) for r in results]
        facts.sort(key=lambda f: f.system_time)
        return facts

    def get_latest_valid_fact(
        self,
        user_id: str,
        entity_id: Optional[str] = None,
        property_name: Optional[str] = None,
    ) -> Optional[BiTemporalFact]:
        """Get the latest valid fact for an entity/property.

        Args:
            user_id: User ID
            entity_id: Optional entity ID
            property_name: Optional property name

        Returns:
            Latest valid fact or None
        """
        filters: dict[str, Any] = {
            "user_id": user_id,
            "status": FactStatus.ACTIVE.value,
        }
        if entity_id:
            filters["entity_id"] = entity_id
        if property_name:
            filters["property_name"] = property_name

        results = self.qdrant.scroll(
            collection_name=self.collection_name,
            filters=filters,
            limit=100,
        )

        if not results:
            return None

        # Filter to currently valid and get latest
        now = datetime.now(timezone.utc)
        valid_facts = []
        for r in results:
            fact = self._payload_to_fact(r["payload"], r["id"])
            if fact.is_valid_at(now):
                valid_facts.append(fact)

        if not valid_facts:
            return None

        # Return most recent by system_time
        valid_facts.sort(key=lambda f: f.system_time, reverse=True)
        return valid_facts[0]

    def _build_query_filters(self, query: BiTemporalQuery) -> dict[str, Any]:
        """Build Qdrant filters from query parameters."""
        filters: dict[str, Any] = {"user_id": query.user_id}

        if query.entity_id:
            filters["entity_id"] = query.entity_id
        if query.property_name:
            filters["property_name"] = query.property_name

        # Status filters
        if not query.include_superseded and not query.include_retracted:
            filters["status"] = FactStatus.ACTIVE.value
        elif not query.include_superseded:
            # Exclude superseded only
            pass  # Complex filter, handle in post-processing
        elif not query.include_retracted:
            # Exclude retracted only
            pass  # Complex filter, handle in post-processing

        return filters

    def _apply_temporal_filters(
        self,
        facts: list[BiTemporalFact],
        query: BiTemporalQuery,
    ) -> list[BiTemporalFact]:
        """Apply complex temporal filters in post-processing."""
        filtered = facts

        # As-of system time filter
        if query.as_of_system_time:
            filtered = [f for f in filtered if f.system_time <= query.as_of_system_time]

        # Valid-at point-in-time filter
        if query.valid_at:
            filtered = [f for f in filtered if f.is_valid_at(query.valid_at)]

        # Valid-time range filter
        if query.valid_from and query.valid_to:
            filtered = [
                f for f in filtered if f.overlaps_interval(query.valid_from, query.valid_to)
            ]

        # Status filters (if not handled in Qdrant query)
        if not query.include_superseded:
            filtered = [f for f in filtered if f.status != FactStatus.SUPERSEDED]
        if not query.include_retracted:
            filtered = [f for f in filtered if f.status != FactStatus.RETRACTED]

        return filtered

    def _fact_to_payload(self, fact: BiTemporalFact) -> dict[str, Any]:
        """Convert fact to Qdrant payload."""
        return {
            "fact_id": fact.fact_id,
            "text": fact.text,
            "user_id": fact.user_id,
            "entity_id": fact.entity_id,
            "property_name": fact.property_name,
            "event_time": fact.event_time.isoformat(),
            "valid_from": fact.valid_from.isoformat(),
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            "system_time": fact.system_time.isoformat(),
            "status": fact.status.value,
            "superseded_by": fact.superseded_by,
            "supersedes": fact.supersedes,
            "confidence": fact.confidence,
            "source": fact.source,
            "metadata": fact.metadata,
        }

    def _payload_to_fact(self, payload: dict[str, Any], fact_version_id: str) -> BiTemporalFact:
        """Convert Qdrant payload to fact."""
        return BiTemporalFact(
            id=fact_version_id,
            fact_id=payload["fact_id"],
            text=payload["text"],
            user_id=payload["user_id"],
            entity_id=payload.get("entity_id"),
            property_name=payload.get("property_name"),
            event_time=datetime.fromisoformat(payload["event_time"]),
            valid_from=datetime.fromisoformat(payload["valid_from"]),
            valid_to=(
                datetime.fromisoformat(payload["valid_to"]) if payload.get("valid_to") else None
            ),
            system_time=datetime.fromisoformat(payload["system_time"]),
            status=FactStatus(payload["status"]),
            superseded_by=payload.get("superseded_by"),
            supersedes=payload.get("supersedes"),
            confidence=payload.get("confidence", 0.9),
            source=payload.get("source", "unknown"),
            metadata=payload.get("metadata", {}),
        )
