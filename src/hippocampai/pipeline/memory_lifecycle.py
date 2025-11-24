"""
Intelligent Memory Lifecycle Management.

This module implements tiered storage with automatic memory migration based on access patterns:
- HOT: Frequently accessed, stored in Redis cache + vector DB
- WARM: Occasionally accessed, stored in vector DB
- COLD: Rarely accessed, stored in vector DB with lower priority
- ARCHIVED: Very old/unused, compressed in vector DB
- HIBERNATED: Extremely old, highly compressed storage

Features:
- Temperature scoring based on access frequency, recency, and importance
- Automatic tier migration based on configurable thresholds
- Memory compression for archived/hibernated memories
- Tiered search with automatic decompression
"""

import gzip
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryTier(str, Enum):
    """Memory storage tiers based on access patterns."""

    HOT = "hot"  # Frequently accessed (< 7 days old, > 10 accesses)
    WARM = "warm"  # Occasionally accessed (< 30 days old, 3-10 accesses)
    COLD = "cold"  # Rarely accessed (< 90 days old, < 3 accesses)
    ARCHIVED = "archived"  # Very old (> 90 days old)
    HIBERNATED = "hibernated"  # Extremely old (> 365 days old)


class MemoryTemperature(BaseModel):
    """Temperature metrics for a memory."""

    memory_id: str
    tier: MemoryTier
    temperature_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="0=coldest, 100=hottest"
    )
    access_frequency: float = Field(ge=0.0, description="Accesses per day since creation")
    recency_score: float = Field(ge=0.0, le=1.0, description="How recently was it accessed")
    importance_weight: float = Field(ge=0.0, le=1.0, description="Memory importance factor")
    last_access: Optional[datetime] = None
    created_at: datetime
    access_count: int = 0
    days_since_creation: float = 0.0
    days_since_last_access: float = 0.0

    def calculate_score(self) -> float:
        """
        Calculate overall temperature score (0-100).

        Formula:
        - 40% access frequency (normalized)
        - 30% recency (exponential decay)
        - 30% importance weight
        """
        # Normalize access frequency (cap at 1 access/day = 100)
        freq_score = min(self.access_frequency * 100, 100)

        # Recency with exponential decay (half-life = 30 days)
        recency_score = self.recency_score * 100

        # Importance contribution
        importance_score = self.importance_weight * 100

        # Weighted average
        self.temperature_score = 0.4 * freq_score + 0.3 * recency_score + 0.3 * importance_score
        return self.temperature_score

    def recommend_tier(self) -> MemoryTier:
        """Recommend tier based on temperature score and age."""
        # Priority rules based on age and access patterns
        if self.days_since_creation > 365:
            if self.access_count < 2 or self.temperature_score < 10:
                return MemoryTier.HIBERNATED
            elif self.temperature_score < 30:
                return MemoryTier.ARCHIVED

        if self.days_since_creation > 90:
            if self.access_count < 3 or self.temperature_score < 20:
                return MemoryTier.ARCHIVED
            elif self.temperature_score < 40:
                return MemoryTier.COLD

        if self.days_since_creation > 30:
            if self.temperature_score < 30:
                return MemoryTier.COLD
            elif self.temperature_score < 60:
                return MemoryTier.WARM

        if self.days_since_creation <= 7:
            if self.access_count > 10 or self.temperature_score > 70:
                return MemoryTier.HOT

        # Default tiers based on temperature
        if self.temperature_score >= 70:
            return MemoryTier.HOT
        elif self.temperature_score >= 50:
            return MemoryTier.WARM
        elif self.temperature_score >= 30:
            return MemoryTier.COLD
        else:
            return MemoryTier.ARCHIVED


class LifecycleConfig(BaseModel):
    """Configuration for memory lifecycle management."""

    # Tier thresholds (days)
    hot_max_age_days: int = 7
    warm_max_age_days: int = 30
    cold_max_age_days: int = 90
    archived_max_age_days: int = 365

    # Access frequency thresholds
    hot_min_accesses: int = 10
    warm_min_accesses: int = 3
    cold_max_accesses: int = 3

    # Temperature thresholds
    hot_min_temperature: float = 70.0
    warm_min_temperature: float = 50.0
    cold_min_temperature: float = 30.0

    # Migration settings
    auto_migrate: bool = True
    migration_interval_hours: int = 24

    # Compression settings
    compress_archived: bool = True
    compress_hibernated: bool = True
    compression_level: int = 6  # gzip compression level (1-9)

    # Recency decay settings
    recency_half_life_days: int = 30  # Half-life for recency score decay


class MemoryLifecycleManager:
    """Manages memory lifecycle and tiered storage."""

    def __init__(self, config: Optional[LifecycleConfig] = None):
        """Initialize lifecycle manager."""
        self.config = config or LifecycleConfig()

    def calculate_temperature(
        self,
        memory_id: str,
        created_at: datetime,
        access_count: int,
        last_access: Optional[datetime] = None,
        importance: float = 5.0,
    ) -> MemoryTemperature:
        """
        Calculate temperature metrics for a memory.

        Args:
            memory_id: Memory identifier
            created_at: When memory was created
            access_count: Number of times accessed
            last_access: Last access timestamp
            importance: Memory importance (0-10)

        Returns:
            MemoryTemperature with calculated metrics
        """
        now = datetime.now(timezone.utc)

        # Ensure datetime objects are timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # Calculate time deltas
        age = now - created_at
        days_since_creation = max(age.total_seconds() / 86400, 0.001)  # Avoid div by 0

        if last_access:
            if last_access.tzinfo is None:
                last_access = last_access.replace(tzinfo=timezone.utc)
            days_since_last_access = (now - last_access).total_seconds() / 86400
        else:
            days_since_last_access = days_since_creation
            last_access = created_at

        # Calculate access frequency (accesses per day)
        access_frequency = access_count / days_since_creation

        # Calculate recency score with exponential decay
        # Score = 0.5^(days_since_last_access / half_life)
        half_life = self.config.recency_half_life_days
        recency_score = 0.5 ** (days_since_last_access / half_life)

        # Normalize importance (0-10 -> 0-1)
        importance_weight = min(importance / 10.0, 1.0)

        # Create temperature object
        temp = MemoryTemperature(
            memory_id=memory_id,
            tier=MemoryTier.WARM,  # Temporary, will be calculated
            access_frequency=access_frequency,
            recency_score=recency_score,
            importance_weight=importance_weight,
            last_access=last_access,
            created_at=created_at,
            access_count=access_count,
            days_since_creation=days_since_creation,
            days_since_last_access=days_since_last_access,
        )

        # Calculate temperature score
        temp.calculate_score()

        # Determine recommended tier
        temp.tier = temp.recommend_tier()

        return temp

    def should_migrate(self, current_tier: MemoryTier, recommended_tier: MemoryTier) -> bool:
        """
        Check if memory should be migrated to a different tier.

        Args:
            current_tier: Current storage tier
            recommended_tier: Recommended tier based on temperature

        Returns:
            True if migration is recommended
        """
        if not self.config.auto_migrate:
            return False

        # Tier ordering
        tier_order = {
            MemoryTier.HOT: 0,
            MemoryTier.WARM: 1,
            MemoryTier.COLD: 2,
            MemoryTier.ARCHIVED: 3,
            MemoryTier.HIBERNATED: 4,
        }

        # Allow migration if tier changes
        return tier_order.get(current_tier, 1) != tier_order.get(recommended_tier, 1)

    def compress_memory(self, memory_data: dict[str, Any]) -> bytes:
        """
        Compress memory data for archived/hibernated storage.

        Args:
            memory_data: Memory data dictionary

        Returns:
            Compressed bytes
        """
        # Convert to JSON
        json_str = json.dumps(memory_data, default=str)

        # Compress with gzip
        compressed = gzip.compress(
            json_str.encode("utf-8"), compresslevel=self.config.compression_level
        )

        logger.debug(
            f"Compressed memory {memory_data.get('id')}: "
            f"{len(json_str)} -> {len(compressed)} bytes "
            f"({100 * len(compressed) / len(json_str):.1f}% of original)"
        )

        return compressed

    def decompress_memory(self, compressed_data: bytes) -> dict[str, Any]:
        """
        Decompress memory data.

        Args:
            compressed_data: Compressed bytes

        Returns:
            Decompressed memory data dictionary
        """
        # Decompress
        json_str = gzip.decompress(compressed_data).decode("utf-8")

        # Parse JSON
        memory_data = json.loads(json_str)

        logger.debug(f"Decompressed memory {memory_data.get('id')}")

        return cast(dict[str, Any], memory_data)

    def get_tier_metadata(self, tier: MemoryTier) -> dict[str, Any]:
        """
        Get metadata about a storage tier.

        Args:
            tier: Memory tier

        Returns:
            Tier metadata
        """
        tier_info = {
            MemoryTier.HOT: {
                "description": "Frequently accessed memories",
                "storage": "Redis cache + Vector DB",
                "compressed": False,
                "search_priority": "highest",
                "max_age_days": self.config.hot_max_age_days,
            },
            MemoryTier.WARM: {
                "description": "Occasionally accessed memories",
                "storage": "Vector DB",
                "compressed": False,
                "search_priority": "high",
                "max_age_days": self.config.warm_max_age_days,
            },
            MemoryTier.COLD: {
                "description": "Rarely accessed memories",
                "storage": "Vector DB",
                "compressed": False,
                "search_priority": "medium",
                "max_age_days": self.config.cold_max_age_days,
            },
            MemoryTier.ARCHIVED: {
                "description": "Very old or unused memories",
                "storage": "Vector DB (compressed)",
                "compressed": self.config.compress_archived,
                "search_priority": "low",
                "max_age_days": self.config.archived_max_age_days,
            },
            MemoryTier.HIBERNATED: {
                "description": "Extremely old memories",
                "storage": "Vector DB (highly compressed)",
                "compressed": self.config.compress_hibernated,
                "search_priority": "lowest",
                "max_age_days": None,
            },
        }

        return tier_info.get(tier, {})

    def update_access_metadata(
        self, memory_data: dict[str, Any], temperature: MemoryTemperature
    ) -> dict[str, Any]:
        """
        Update memory metadata with lifecycle information.

        Args:
            memory_data: Memory data dictionary
            temperature: Calculated temperature metrics

        Returns:
            Updated memory data with lifecycle metadata
        """
        if "metadata" not in memory_data:
            memory_data["metadata"] = {}

        # Add lifecycle metadata
        memory_data["metadata"]["lifecycle"] = {
            "tier": temperature.tier.value,
            "temperature_score": temperature.temperature_score,
            "access_frequency": temperature.access_frequency,
            "recency_score": temperature.recency_score,
            "days_since_creation": temperature.days_since_creation,
            "days_since_last_access": temperature.days_since_last_access,
            "last_tier_update": datetime.now(timezone.utc).isoformat(),
        }

        return memory_data
