"""Integration tests for the Truth Maintenance System and bi-temporal store
exercised through ``MemoryClient`` against a real Qdrant.

Locks in two regressions:

1. ``client.remember`` used to pass ``importance`` (0-10 scale) as TMS confidence
   (0-1 scale), which the ``Justification`` pydantic validator rejected. The TMS
   call was swallowed by a try/except and beliefs were never stored.

2. ``BiTemporalStore.get_latest_valid_fact`` ignored historical state -- the
   headline time-travel feature returned the current version even when asked for
   a past moment.

Marked ``integration`` so they're skipped in unit-only runs; they require Qdrant
reachable at ``QDRANT_URL`` (default ``http://localhost:6333``).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable() -> bool:
    try:
        httpx.get(QDRANT_URL, timeout=2.0)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant is not reachable"),
]


@pytest.fixture(scope="module")
def tms_client():
    """A MemoryClient with TMS enabled. Cached for the module to amortize init."""
    os.environ["HIPPOCAMPAI_ENABLE_TMS"] = "true"
    from hippocampai import MemoryClient

    return MemoryClient()


def test_remember_records_belief_at_high_importance(tms_client) -> None:
    """Regression: importance=8 used to break TMS via the confidence validator."""
    assert tms_client.truth_maintenance is not None
    memory = tms_client.remember(
        "Alice works at Google.", user_id="tms_int_user_1", type="fact", importance=8
    )
    belief = tms_client.get_belief_state(memory.id)
    assert belief is not None, "belief was not recorded for importance=8"
    assert 0.0 <= belief.confidence <= 1.0


def test_revise_belief_records_history(tms_client) -> None:
    memory = tms_client.remember(
        "Bob lives in Berlin.", user_id="tms_int_user_2", type="fact", importance=7
    )
    belief = tms_client.get_belief_state(memory.id)
    assert belief is not None
    tms_client.revise_belief(belief.belief_id, new_state="retracted", reason="test")
    history = tms_client.get_belief_history(belief.belief_id)
    assert len(history) == 1


def test_bitemporal_time_travel(tms_client) -> None:
    """Regression: get_latest_valid_fact(as_of=...) returns the historical version."""
    from hippocampai.models.bitemporal import FactRevision

    uid = "tms_int_user_3"
    now = datetime.now(timezone.utc)
    fact = tms_client.store_bitemporal_fact(
        text="Carol's title is Engineer.",
        user_id=uid,
        entity_id="carol",
        property_name="title",
        valid_from=now - timedelta(days=365),
    )

    revision = FactRevision(
        original_fact_id=fact.id,
        new_text="Carol's title is Senior Engineer.",
        new_valid_from=now - timedelta(days=7),
        reason="promotion",
        confidence=0.95,
    )
    vec = tms_client.embedder.encode_single(revision.new_text).tolist()
    new_version = tms_client._get_bitemporal_store().revise_fact(
        revision=revision, vector=vec, user_id=uid
    )
    assert new_version.supersedes == fact.id

    store = tms_client._get_bitemporal_store()
    six_months_ago = store.get_latest_valid_fact(
        entity_id="carol", property_name="title", user_id=uid,
        as_of=now - timedelta(days=180),
    )
    current = store.get_latest_valid_fact(
        entity_id="carol", property_name="title", user_id=uid, as_of=now,
    )
    assert six_months_ago is not None and "Senior" not in six_months_ago.text, (
        "time-travel returned the wrong version for 6mo ago"
    )
    assert current is not None and "Senior" in current.text


def test_bitemporal_no_as_of_returns_current(tms_client) -> None:
    """Without as_of the convenience method returns the present-day valid fact."""
    uid = "tms_int_user_4"
    now = datetime.now(timezone.utc)
    tms_client.store_bitemporal_fact(
        text="Dave's role is Designer.",
        user_id=uid,
        entity_id="dave",
        property_name="role",
        valid_from=now - timedelta(days=30),
    )
    store = tms_client._get_bitemporal_store()
    current = store.get_latest_valid_fact(
        entity_id="dave", property_name="role", user_id=uid
    )
    assert current is not None
    assert "Designer" in current.text
