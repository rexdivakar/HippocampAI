"""Error types for memory pipeline."""

from typing import Optional


class MemoryError(Exception):
    """Base class for memory-related exceptions."""

    pass


class ConflictResolutionError(MemoryError):
    """Error raised during conflict resolution."""

    def __init__(self, message: str, conflict_id: Optional[str] = None):
        """Initialize error with optional conflict ID."""
        super().__init__(message)
        self.conflict_id = conflict_id


class MemoryNotFoundError(MemoryError):
    """Raised when a memory cannot be found."""

    pass
