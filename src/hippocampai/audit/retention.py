"""Audit log retention policies for compliance."""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetentionPeriod(str, Enum):
    """Standard retention periods."""

    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    DAYS_180 = "180_days"
    YEAR_1 = "1_year"
    YEARS_3 = "3_years"
    YEARS_7 = "7_years"  # Common compliance requirement
    FOREVER = "forever"


class AuditRetentionPolicy(BaseModel):
    """Retention policy configuration."""

    policy_id: str = Field(
        default_factory=lambda: f"policy_{int(datetime.now(timezone.utc).timestamp())}"
    )
    tenant_id: Optional[str] = None  # None = global policy

    # Retention periods by severity
    debug_retention: RetentionPeriod = RetentionPeriod.DAYS_30
    info_retention: RetentionPeriod = RetentionPeriod.DAYS_90
    warning_retention: RetentionPeriod = RetentionPeriod.YEAR_1
    error_retention: RetentionPeriod = RetentionPeriod.YEARS_3
    critical_retention: RetentionPeriod = RetentionPeriod.YEARS_7

    # Special retention for compliance-critical actions
    auth_events_retention: RetentionPeriod = RetentionPeriod.YEARS_3
    admin_events_retention: RetentionPeriod = RetentionPeriod.YEARS_7
    data_export_retention: RetentionPeriod = RetentionPeriod.YEARS_7

    # Settings
    enabled: bool = True
    archive_before_delete: bool = True
    archive_location: Optional[str] = None  # S3 bucket, etc.

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetentionManager:
    """Manages audit log retention and cleanup."""

    PERIOD_DAYS = {
        RetentionPeriod.DAYS_30: 30,
        RetentionPeriod.DAYS_90: 90,
        RetentionPeriod.DAYS_180: 180,
        RetentionPeriod.YEAR_1: 365,
        RetentionPeriod.YEARS_3: 365 * 3,
        RetentionPeriod.YEARS_7: 365 * 7,
        RetentionPeriod.FOREVER: None,
    }

    def __init__(
        self,
        db_pool: Any = None,
        default_policy: Optional[AuditRetentionPolicy] = None,
    ) -> None:
        """
        Initialize retention manager.

        Args:
            db_pool: Database connection pool
            default_policy: Default retention policy
        """
        self.db_pool = db_pool
        self.default_policy = default_policy or AuditRetentionPolicy()
        self.policies: dict[str, AuditRetentionPolicy] = {}

        logger.info("RetentionManager initialized")

    def set_policy(self, policy: AuditRetentionPolicy) -> None:
        """Set retention policy for a tenant."""
        key = policy.tenant_id or "global"
        policy.updated_at = datetime.now(timezone.utc)
        self.policies[key] = policy
        logger.info(f"Retention policy set for {key}")

    def get_policy(self, tenant_id: Optional[str] = None) -> AuditRetentionPolicy:
        """Get retention policy for a tenant."""
        return self.policies.get(tenant_id or "global", self.default_policy)

    def get_cutoff_date(
        self,
        period: RetentionPeriod,
        from_date: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """
        Calculate cutoff date for a retention period.

        Args:
            period: Retention period
            from_date: Reference date (default: now)

        Returns:
            Cutoff datetime, or None for FOREVER
        """
        days = self.PERIOD_DAYS.get(period)
        if days is None:
            return None

        reference = from_date or datetime.now(timezone.utc)
        return reference - timedelta(days=days)

    async def cleanup(
        self,
        tenant_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run retention cleanup.

        Args:
            tenant_id: Tenant to clean up (None = all)
            dry_run: If True, only report what would be deleted

        Returns:
            Cleanup statistics
        """
        policy = self.get_policy(tenant_id)

        if not policy.enabled:
            return {"status": "skipped", "reason": "policy disabled"}

        stats = {
            "deleted_count": 0,
            "archived_count": 0,
            "by_severity": {},
            "dry_run": dry_run,
        }

        if not self.db_pool:
            logger.warning("No database pool configured, skipping cleanup")
            return {"status": "skipped", "reason": "no database"}

        now = datetime.now(timezone.utc)

        # Define cleanup rules
        cleanup_rules = [
            ("debug", policy.debug_retention),
            ("info", policy.info_retention),
            ("warning", policy.warning_retention),
            ("error", policy.error_retention),
            ("critical", policy.critical_retention),
        ]

        async with self.db_pool.acquire() as conn:
            for severity, period in cleanup_rules:
                cutoff = self.get_cutoff_date(period, now)
                if cutoff is None:
                    continue  # FOREVER retention

                # Build query
                conditions = ["severity = $1", "timestamp < $2"]
                params: list[Any] = [severity, cutoff]

                if tenant_id:
                    conditions.append("tenant_id = $3")
                    params.append(tenant_id)

                where_clause = " AND ".join(conditions)

                if dry_run:
                    # Just count
                    row = await conn.fetchrow(
                        f"SELECT COUNT(*) as count FROM audit_logs WHERE {where_clause}",
                        *params,
                    )
                    count = row["count"]
                else:
                    # Archive if configured
                    if policy.archive_before_delete and policy.archive_location:
                        # In production, this would export to S3/GCS
                        logger.info(f"Would archive {severity} logs to {policy.archive_location}")

                    # Delete
                    result = await conn.execute(
                        f"DELETE FROM audit_logs WHERE {where_clause}",
                        *params,
                    )
                    count = int(result.split()[-1])

                stats["by_severity"][severity] = count
                stats["deleted_count"] += count

        logger.info(
            f"Retention cleanup {'(dry run) ' if dry_run else ''}"
            f"completed: {stats['deleted_count']} events"
        )

        return stats

    async def get_retention_report(
        self,
        tenant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate retention status report.

        Args:
            tenant_id: Tenant to report on

        Returns:
            Report with counts by age bucket
        """
        if not self.db_pool:
            return {"status": "error", "reason": "no database"}

        policy = self.get_policy(tenant_id)
        now = datetime.now(timezone.utc)

        buckets = [
            ("last_24h", timedelta(hours=24)),
            ("last_7d", timedelta(days=7)),
            ("last_30d", timedelta(days=30)),
            ("last_90d", timedelta(days=90)),
            ("last_year", timedelta(days=365)),
            ("older", None),
        ]

        report = {
            "tenant_id": tenant_id,
            "policy": policy.model_dump(),
            "counts": {},
            "generated_at": now.isoformat(),
        }

        async with self.db_pool.acquire() as conn:
            tenant_condition = "tenant_id = $1" if tenant_id else "1=1"

            for bucket_name, delta in buckets:
                if delta:
                    cutoff = now - delta
                    if tenant_id:
                        row = await conn.fetchrow(
                            f"""
                            SELECT COUNT(*) as count FROM audit_logs
                            WHERE {tenant_condition} AND timestamp >= $2
                            """,
                            tenant_id,
                            cutoff,
                        )
                    else:
                        row = await conn.fetchrow(
                            "SELECT COUNT(*) as count FROM audit_logs WHERE timestamp >= $1",
                            cutoff,
                        )
                else:
                    # Older than 1 year
                    cutoff = now - timedelta(days=365)
                    if tenant_id:
                        row = await conn.fetchrow(
                            f"""
                            SELECT COUNT(*) as count FROM audit_logs
                            WHERE {tenant_condition} AND timestamp < $2
                            """,
                            tenant_id,
                            cutoff,
                        )
                    else:
                        row = await conn.fetchrow(
                            "SELECT COUNT(*) as count FROM audit_logs WHERE timestamp < $1",
                            cutoff,
                        )

                report["counts"][bucket_name] = row["count"]

        return report
