"""Time utilities."""

from datetime import datetime


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.utcfromtimestamp(ts)
