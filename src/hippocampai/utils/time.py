"""Time utilities."""

import sys
from datetime import datetime, timezone
from typing import Optional

# Handle Python 3.11+ datetime.UTC compatibility
if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc


def now_utc() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(UTC)


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
