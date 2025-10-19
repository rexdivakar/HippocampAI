"""Time utilities."""

from datetime import UTC, datetime
from typing import Optional


def now_utc() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(tz=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Coerce a datetime into UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert Unix timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


def parse_iso_datetime(value: str) -> datetime:
    """Parse ISO 8601 string into a timezone-aware UTC datetime."""
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return ensure_utc(dt)


def isoformat_utc(dt: Optional[datetime] = None) -> str:
    """Return ISO 8601 representation in UTC."""
    target = ensure_utc(dt) if dt else now_utc()
    return target.isoformat()
