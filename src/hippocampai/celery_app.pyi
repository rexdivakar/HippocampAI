"""Type stubs for celery_app module."""

from typing import Any, Callable, Generic, TypeVar

from celery import Celery
from celery.result import AsyncResult

P = TypeVar("P")
R = TypeVar("R")

class Task(Generic[P, R]):
    """Celery Task wrapper with proper type support."""
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...
    def delay(self, *args: Any, **kwargs: Any) -> AsyncResult: ...
    def apply_async(self, *args: Any, **kwargs: Any) -> AsyncResult: ...

class _TypedCelery(Celery):
    """Type-aware Celery wrapper for proper task decorator typing."""

    def task(
        self,
        *args: Any,
        bind: bool = ...,
        name: str | None = ...,
        max_retries: int | None = ...,
        **kwargs: Any,
    ) -> Callable[[Callable[..., R]], Task[Any, R]]: ...

celery_app: _TypedCelery

def get_beat_schedule() -> dict[str, dict[str, Any]]: ...
