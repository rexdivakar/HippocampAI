"""Trigger manager for event-driven memory actions."""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TriggerEvent(str, Enum):
    """Events that can fire triggers."""

    ON_REMEMBER = "on_remember"
    ON_RECALL = "on_recall"
    ON_UPDATE = "on_update"
    ON_DELETE = "on_delete"
    ON_CONFLICT = "on_conflict"
    ON_EXPIRE = "on_expire"


class TriggerAction(str, Enum):
    """Actions a trigger can perform."""

    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    LOG = "log"


class TriggerCondition(BaseModel):
    """A single condition for trigger evaluation."""

    field: str  # e.g. "type", "importance", "tags", "text"
    operator: str  # "eq", "gt", "lt", "contains", "matches"
    value: Any


class TriggerFire(BaseModel):
    """Record of a trigger firing."""

    trigger_id: str
    fired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    memory_id: str
    event: TriggerEvent
    success: bool = True
    error: Optional[str] = None


class Trigger(BaseModel):
    """A registered trigger."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_id: str
    event: TriggerEvent
    conditions: list[TriggerCondition] = Field(default_factory=list)
    action: TriggerAction = TriggerAction.LOG
    action_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fired_count: int = 0


class TriggerManager:
    """Manages triggers and evaluates them on memory events."""

    def __init__(self, webhook_timeout: int = 10) -> None:
        self.webhook_timeout = webhook_timeout
        self._triggers: dict[str, Trigger] = {}  # trigger_id -> Trigger
        self._fire_history: list[TriggerFire] = []

    def register_trigger(self, trigger: Trigger) -> Trigger:
        """Register a new trigger.

        Args:
            trigger: Trigger to register.

        Returns:
            The registered Trigger.
        """
        self._triggers[trigger.id] = trigger
        logger.info(
            f"Registered trigger '{trigger.name}' (id={trigger.id}) "
            f"for event={trigger.event.value}, user={trigger.user_id}"
        )
        return trigger

    def remove_trigger(self, trigger_id: str, user_id: str) -> bool:
        """Remove a trigger.

        Args:
            trigger_id: ID of the trigger to remove.
            user_id: Must match trigger owner.

        Returns:
            True if removed, False if not found or unauthorized.
        """
        trigger = self._triggers.get(trigger_id)
        if trigger is None:
            return False
        if trigger.user_id != user_id:
            logger.warning(
                f"User {user_id} tried to remove trigger {trigger_id} "
                f"owned by {trigger.user_id}"
            )
            return False
        del self._triggers[trigger_id]
        logger.info(f"Removed trigger {trigger_id}")
        return True

    def list_triggers(self, user_id: str) -> list[Trigger]:
        """List all triggers for a user.

        Args:
            user_id: Filter by user.

        Returns:
            List of Trigger objects.
        """
        return [t for t in self._triggers.values() if t.user_id == user_id]

    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get a trigger by ID."""
        return self._triggers.get(trigger_id)

    def get_fire_history(
        self, trigger_id: str, limit: int = 50
    ) -> list[TriggerFire]:
        """Get fire history for a trigger."""
        history = [f for f in self._fire_history if f.trigger_id == trigger_id]
        history.sort(key=lambda f: f.fired_at, reverse=True)
        return history[:limit]

    def evaluate_triggers(
        self,
        event: TriggerEvent,
        memory_data: dict[str, Any],
        user_id: str,
    ) -> list[TriggerFire]:
        """Evaluate all triggers for a user on a given event.

        Args:
            event: The event type that occurred.
            memory_data: Dict of memory fields (id, text, type, importance, tags, etc.)
            user_id: The user whose triggers to evaluate.

        Returns:
            List of TriggerFire records for triggers that matched and fired.
        """
        fires: list[TriggerFire] = []
        user_triggers = [
            t for t in self._triggers.values()
            if t.user_id == user_id and t.enabled and t.event == event
        ]

        for trigger in user_triggers:
            if self._check_conditions(trigger.conditions, memory_data):
                fire = self._fire_trigger(trigger, event, memory_data)
                fires.append(fire)

        return fires

    def _check_conditions(
        self, conditions: list[TriggerCondition], memory_data: dict[str, Any]
    ) -> bool:
        """Check if all conditions match the memory data."""
        if not conditions:
            return True  # No conditions = always match

        for condition in conditions:
            field_value = memory_data.get(condition.field)
            if not self._evaluate_condition(field_value, condition.operator, condition.value):
                return False

        return True

    @staticmethod
    def _evaluate_condition(field_value: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single condition."""
        if field_value is None:
            return False

        if operator == "eq":
            return field_value == expected
        if operator == "gt":
            return float(field_value) > float(expected)
        if operator == "lt":
            return float(field_value) < float(expected)
        if operator == "contains":
            if isinstance(field_value, list):
                return expected in field_value
            return str(expected) in str(field_value)
        if operator == "matches":
            return bool(re.search(str(expected), str(field_value)))

        logger.warning(f"Unknown operator: {operator}")
        return False

    def _fire_trigger(
        self,
        trigger: Trigger,
        event: TriggerEvent,
        memory_data: dict[str, Any],
    ) -> TriggerFire:
        """Execute the trigger action."""
        memory_id = str(memory_data.get("id", "unknown"))
        success = True
        error: Optional[str] = None

        try:
            if trigger.action == TriggerAction.LOG:
                self._fire_log(trigger, event, memory_data)
            elif trigger.action == TriggerAction.WEBHOOK:
                self._fire_webhook(trigger, event, memory_data)
            elif trigger.action == TriggerAction.WEBSOCKET:
                self._fire_websocket(trigger, event, memory_data)
        except Exception as e:
            success = False
            error = str(e)
            logger.error(f"Trigger {trigger.id} fire failed: {e}")

        trigger.fired_count += 1
        fire = TriggerFire(
            trigger_id=trigger.id,
            memory_id=memory_id,
            event=event,
            success=success,
            error=error,
        )
        self._fire_history.append(fire)
        return fire

    @staticmethod
    def _fire_log(
        trigger: Trigger,
        event: TriggerEvent,
        memory_data: dict[str, Any],
    ) -> None:
        """Log trigger firing."""
        logger.info(
            f"TRIGGER FIRED: '{trigger.name}' on {event.value} "
            f"for memory {memory_data.get('id', 'unknown')}"
        )

    def _fire_webhook(
        self,
        trigger: Trigger,
        event: TriggerEvent,
        memory_data: dict[str, Any],
    ) -> None:
        """POST to a webhook URL."""
        url = trigger.action_config.get("url")
        if not url:
            raise ValueError("Webhook trigger missing 'url' in action_config")

        try:
            import httpx

            payload = {
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
                "event": event.value,
                "memory": memory_data,
                "fired_at": datetime.now(timezone.utc).isoformat(),
            }
            with httpx.Client(timeout=self.webhook_timeout) as http_client:
                response = http_client.post(url, json=payload)
                response.raise_for_status()
            logger.info(f"Webhook fired to {url}, status={response.status_code}")
        except ImportError:
            logger.warning("httpx not installed, webhook trigger skipped")
        except Exception as e:
            raise RuntimeError(f"Webhook POST to {url} failed: {e}") from e

    @staticmethod
    def _fire_websocket(
        trigger: Trigger,
        event: TriggerEvent,
        memory_data: dict[str, Any],
    ) -> None:
        """Emit a websocket event. Non-blocking best-effort."""
        try:
            import asyncio

            from hippocampai.api.websocket import sio

            payload = {
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
                "event": event.value,
                "memory": memory_data,
                "fired_at": datetime.now(timezone.utc).isoformat(),
            }

            loop: Optional[asyncio.AbstractEventLoop] = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                loop.create_task(
                    sio.emit(
                        "trigger:fired",
                        payload,
                        room=f"user:{trigger.user_id}",
                    )
                )
            else:
                logger.debug("No running event loop, websocket trigger skipped")
        except Exception as e:
            logger.warning(f"Websocket trigger fire failed: {e}")
