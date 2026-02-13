"""Tests for prospective memory (remembering to remember)."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.prospective.prospective_memory import (
    ProspectiveIntent,
    ProspectiveMemoryManager,
    ProspectiveStatus,
    ProspectiveTriggerType,
    RecurrencePattern,
)


@pytest.fixture
def manager() -> ProspectiveMemoryManager:
    return ProspectiveMemoryManager(max_intents_per_user=10)


# ---------- CRUD ----------


class TestCreateIntent:
    def test_create_basic(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(
            user_id="u1",
            intent_text="Follow up with John",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["john", "follow up"],
        )
        assert intent.status == ProspectiveStatus.PENDING
        assert intent.user_id == "u1"
        assert "john" in intent.context_keywords

    def test_create_time_based(self, manager: ProspectiveMemoryManager) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        intent = manager.create_intent(
            user_id="u1",
            intent_text="Remind me at 3pm",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=future,
        )
        assert intent.trigger_type == ProspectiveTriggerType.TIME_BASED
        assert intent.trigger_at == future

    def test_max_intents_enforced(self, manager: ProspectiveMemoryManager) -> None:
        for i in range(12):
            manager.create_intent(user_id="u1", intent_text=f"Intent {i}")
        assert len(manager.list_intents("u1")) == 10

    def test_create_from_natural_language(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent_from_natural_language(
            user_id="u1",
            text="When the user mentions budget, bring up Q3 cost overrun",
        )
        assert intent.trigger_type == ProspectiveTriggerType.EVENT_BASED
        assert len(intent.context_keywords) > 0


class TestGetListIntents:
    def test_get_intent(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(user_id="u1", intent_text="test")
        fetched = manager.get_intent(intent.id)
        assert fetched is not None
        assert fetched.id == intent.id

    def test_get_nonexistent(self, manager: ProspectiveMemoryManager) -> None:
        assert manager.get_intent("nonexistent") is None

    def test_list_intents_all(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(user_id="u1", intent_text="a")
        manager.create_intent(user_id="u1", intent_text="b")
        manager.create_intent(user_id="u2", intent_text="c")
        assert len(manager.list_intents("u1")) == 2
        assert len(manager.list_intents("u2")) == 1

    def test_list_intents_filtered(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(user_id="u1", intent_text="a")
        i2 = manager.create_intent(user_id="u1", intent_text="b")
        manager.cancel_intent(i2.id, "u1")
        assert len(manager.list_intents("u1", status=ProspectiveStatus.PENDING)) == 1
        assert len(manager.list_intents("u1", status=ProspectiveStatus.CANCELLED)) == 1


class TestCancelComplete:
    def test_cancel(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(user_id="u1", intent_text="test")
        cancelled = manager.cancel_intent(intent.id, "u1")
        assert cancelled is not None
        assert cancelled.status == ProspectiveStatus.CANCELLED

    def test_cancel_wrong_user(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(user_id="u1", intent_text="test")
        assert manager.cancel_intent(intent.id, "u2") is None

    def test_complete(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(user_id="u1", intent_text="test")
        completed = manager.complete_intent(intent.id, "u1")
        assert completed is not None
        assert completed.status == ProspectiveStatus.COMPLETED
        assert completed.completed_at is not None

    def test_cannot_cancel_completed(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent(user_id="u1", intent_text="test")
        manager.complete_intent(intent.id, "u1")
        assert manager.cancel_intent(intent.id, "u1") is None


# ---------- Evaluation ----------


class TestEvaluateContext:
    def test_keyword_match(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Bring up Q3 cost overrun",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["budget"],
            priority=8,
        )
        triggered = manager.evaluate_context("u1", "What's our budget status?")
        assert len(triggered) == 1
        assert triggered[0].status == ProspectiveStatus.TRIGGERED
        assert triggered[0].trigger_count == 1

    def test_no_match(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Bring up Q3 cost overrun",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["budget"],
        )
        triggered = manager.evaluate_context("u1", "Tell me about the weather")
        assert len(triggered) == 0

    def test_regex_match(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Flag security issues",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_pattern=r"security|vulnerability|CVE-\d+",
        )
        triggered = manager.evaluate_context("u1", "There is a new CVE-2024-1234")
        assert len(triggered) == 1

    def test_embedding_match(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Similar topic",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_embedding=[1.0, 0.0, 0.0],
            similarity_threshold=0.9,
        )
        # Very similar embedding
        triggered = manager.evaluate_context(
            "u1", "anything", context_embedding=[0.99, 0.1, 0.0]
        )
        assert len(triggered) == 1

    def test_embedding_no_match(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Similar topic",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_embedding=[1.0, 0.0, 0.0],
            similarity_threshold=0.9,
        )
        # Orthogonal embedding
        triggered = manager.evaluate_context(
            "u1", "anything", context_embedding=[0.0, 1.0, 0.0]
        )
        assert len(triggered) == 0

    def test_priority_ordering(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Low priority",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["test"],
            priority=2,
        )
        manager.create_intent(
            user_id="u1",
            intent_text="High priority",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["test"],
            priority=9,
        )
        triggered = manager.evaluate_context("u1", "test query")
        assert len(triggered) == 2
        assert triggered[0].priority > triggered[1].priority

    def test_already_triggered_not_reevaluated(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(
            user_id="u1",
            intent_text="Once only",
            trigger_type=ProspectiveTriggerType.EVENT_BASED,
            context_keywords=["fire"],
        )
        triggered1 = manager.evaluate_context("u1", "fire!")
        assert len(triggered1) == 1
        # Second evaluation should not retrigger (already TRIGGERED)
        triggered2 = manager.evaluate_context("u1", "fire again!")
        assert len(triggered2) == 0


class TestEvaluateTimeTriggers:
    def test_past_trigger_fires(self, manager: ProspectiveMemoryManager) -> None:
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        manager.create_intent(
            user_id="u1",
            intent_text="Should fire",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=past,
        )
        triggered = manager.evaluate_time_triggers()
        assert len(triggered) == 1

    def test_future_trigger_does_not_fire(self, manager: ProspectiveMemoryManager) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        manager.create_intent(
            user_id="u1",
            intent_text="Should not fire",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=future,
        )
        triggered = manager.evaluate_time_triggers()
        assert len(triggered) == 0

    def test_expired_window_skipped(self, manager: ProspectiveMemoryManager) -> None:
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        window_end = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.create_intent(
            user_id="u1",
            intent_text="Window expired",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=past,
            time_window_end=window_end,
        )
        triggered = manager.evaluate_time_triggers()
        assert len(triggered) == 0


# ---------- Recurrence ----------


class TestRecurrence:
    def test_recurring_resets_to_pending(self, manager: ProspectiveMemoryManager) -> None:
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        intent = manager.create_intent(
            user_id="u1",
            intent_text="Daily standup",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=past,
            recurrence=RecurrencePattern.DAILY,
        )
        triggered = manager.evaluate_time_triggers()
        assert len(triggered) == 1
        # After triggering, recurring intent should be PENDING again
        assert intent.status == ProspectiveStatus.PENDING
        assert intent.trigger_at is not None
        assert intent.trigger_at > datetime.now(timezone.utc)

    def test_remaining_occurrences_decremented(self, manager: ProspectiveMemoryManager) -> None:
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        intent = manager.create_intent(
            user_id="u1",
            intent_text="Limited repeat",
            trigger_type=ProspectiveTriggerType.TIME_BASED,
            trigger_at=past,
            recurrence=RecurrencePattern.DAILY,
            remaining_occurrences=1,
        )
        triggered = manager.evaluate_time_triggers()
        assert len(triggered) == 1
        # Last occurrence — should remain TRIGGERED (not reset)
        assert intent.status == ProspectiveStatus.TRIGGERED


# ---------- Lifecycle ----------


class TestLifecycle:
    def test_expire_stale(self, manager: ProspectiveMemoryManager) -> None:
        past = datetime.now(timezone.utc) - timedelta(days=1)
        manager.create_intent(
            user_id="u1",
            intent_text="Expired",
            expires_at=past,
        )
        manager.create_intent(user_id="u1", intent_text="Active")
        expired = manager.expire_stale_intents("u1")
        assert expired == 1
        pending = manager.list_intents("u1", status=ProspectiveStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].intent_text == "Active"

    def test_consolidate_simple(self, manager: ProspectiveMemoryManager) -> None:
        manager.create_intent(user_id="u1", intent_text="Do task A")
        manager.create_intent(user_id="u1", intent_text="Do task A")  # duplicate
        manager.create_intent(user_id="u1", intent_text="Do task B")
        consolidated = manager.consolidate_intents("u1")
        assert len(consolidated) == 2


# ---------- NL Parsing ----------


class TestNLParsing:
    def test_heuristic_event_based(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent_from_natural_language(
            user_id="u1",
            text="When the user mentions budget, bring up Q3 report",
        )
        assert intent.trigger_type == ProspectiveTriggerType.EVENT_BASED

    def test_heuristic_time_based(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent_from_natural_language(
            user_id="u1",
            text="Remind me tomorrow to send the report",
        )
        assert intent.trigger_type == ProspectiveTriggerType.TIME_BASED

    def test_heuristic_recurring(self, manager: ProspectiveMemoryManager) -> None:
        intent = manager.create_intent_from_natural_language(
            user_id="u1",
            text="Every week, summarize the key decisions",
        )
        assert intent.recurrence == RecurrencePattern.WEEKLY


# ---------- Models ----------


class TestModels:
    def test_prospective_intent_defaults(self) -> None:
        intent = ProspectiveIntent(
            user_id="u1",
            intent_text="test",
        )
        assert intent.status == ProspectiveStatus.PENDING
        assert intent.trigger_type == ProspectiveTriggerType.EVENT_BASED
        assert intent.recurrence == RecurrencePattern.NONE
        assert intent.priority == 5
        assert intent.trigger_count == 0

    def test_memory_type_prospective(self) -> None:
        from hippocampai.models.memory import Memory, MemoryType

        mem = Memory(text="test", user_id="u1", type=MemoryType.PROSPECTIVE)
        assert mem.type == MemoryType.PROSPECTIVE
        # Should route to prefs collection
        assert mem.collection_name("facts_col", "prefs_col") == "prefs_col"

    def test_config_has_prospective_fields(self) -> None:
        from hippocampai.config import Config

        config = Config()
        assert hasattr(config, "enable_prospective_memory")
        assert hasattr(config, "prospective_max_intents_per_user")
        assert hasattr(config, "prospective_eval_interval_seconds")
        assert hasattr(config, "half_life_prospective")
        half_lives = config.get_half_lives()
        assert "prospective" in half_lives

    def test_trigger_event_has_prospective(self) -> None:
        from hippocampai.triggers.trigger_manager import TriggerEvent

        assert hasattr(TriggerEvent, "ON_PROSPECTIVE_TRIGGER")
        assert TriggerEvent.ON_PROSPECTIVE_TRIGGER.value == "on_prospective_trigger"
