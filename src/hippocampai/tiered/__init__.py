"""Tiered Storage for memory compression and archival.

Implements hot/warm/cold storage tiers for efficient memory management:
- Hot: Frequently accessed, full fidelity, fast retrieval
- Warm: Less frequent, compressed, moderate retrieval speed
- Cold: Archived, highly compressed, slow retrieval

Example:
    >>> from hippocampai.tiered import TieredStorageManager, StorageTier
    >>>
    >>> manager = TieredStorageManager(client)
    >>> manager.configure_tiers(
    ...     hot_days=30,      # Keep in hot for 30 days
    ...     warm_days=90,     # Move to warm after 30 days
    ...     cold_days=365,    # Move to cold after 90 days
    ... )
    >>>
    >>> # Run tier migration
    >>> stats = manager.migrate_tiers(user_id="alice")
"""

from hippocampai.tiered.manager import TieredStorageManager
from hippocampai.tiered.models import (
    MigrationStats,
    StorageTier,
    TierConfig,
    TierStats,
)

__all__ = [
    "TieredStorageManager",
    "StorageTier",
    "TierConfig",
    "TierStats",
    "MigrationStats",
]
