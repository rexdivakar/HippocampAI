"""Models for tiered storage."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StorageTier(str, Enum):
    """Storage tier levels."""

    HOT = "hot"  # Frequently accessed, full fidelity
    WARM = "warm"  # Less frequent, compressed
    COLD = "cold"  # Archived, highly compressed
    FROZEN = "frozen"  # Long-term archive, offline


@dataclass
class TierConfig:
    """Configuration for storage tiers."""

    # Days before moving to next tier (from last access)
    hot_to_warm_days: int = 30
    warm_to_cold_days: int = 90
    cold_to_frozen_days: int = 365

    # Compression settings
    warm_compression: str = "summary"  # summary, truncate, semantic
    cold_compression: str = "heavy_summary"
    frozen_compression: str = "metadata_only"

    # Retention
    max_hot_memories: int = 10000
    max_warm_memories: int = 50000
    max_cold_memories: int = 200000

    # Auto-migration
    auto_migrate: bool = True
    migrate_batch_size: int = 100


@dataclass
class TierStats:
    """Statistics for a storage tier."""

    tier: StorageTier
    memory_count: int = 0
    storage_bytes: int = 0
    avg_access_count: float = 0.0
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None


@dataclass
class MigrationStats:
    """Statistics from tier migration."""

    hot_to_warm: int = 0
    warm_to_cold: int = 0
    cold_to_frozen: int = 0
    promoted: int = 0  # Moved to hotter tier due to access
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    bytes_saved: int = 0


@dataclass
class CompressedMemory:
    """A compressed memory representation."""

    id: str
    original_id: str
    user_id: str
    tier: StorageTier
    summary: str  # Compressed text
    original_length: int
    compressed_length: int
    compression_ratio: float
    memory_type: str
    importance: float
    tags: list[str]
    created_at: str
    last_accessed: str
    access_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # For restoration
    can_restore: bool = True
    original_hash: Optional[str] = None
