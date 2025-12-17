"""Audit logger for tracking all system events."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from hippocampai.audit.models import (
    AuditAction,
    AuditEvent,
    AuditLog,
    AuditQuery,
    AuditSeverity,
    AuditStats,
)

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Centralized audit logger for compliance tracking.

    Supports multiple backends:
    - In-memory (default, for testing)
    - PostgreSQL (production)
    - Redis (high-throughput)
    """

    # Severity ordering for filtering
    SEVERITY_ORDER = {
        AuditSeverity.DEBUG: 0,
        AuditSeverity.INFO: 1,
        AuditSeverity.WARNING: 2,
        AuditSeverity.ERROR: 3,
        AuditSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        db_pool: Any = None,
        redis_client: Any = None,
        max_memory_events: int = 10000,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            db_pool: asyncpg connection pool for PostgreSQL storage
            redis_client: Redis client for high-throughput logging
            max_memory_events: Max events to keep in memory (for in-memory backend)
        """
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.max_memory_events = max_memory_events

        # In-memory storage (fallback/testing)
        self._events: list[AuditEvent] = []

        logger.info("AuditLogger initialized")

    async def log(
        self,
        action: AuditAction,
        description: str,
        user_id: Optional[UUID] = None,
        user_email: Optional[str] = None,
        api_key_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        old_value: Optional[dict[str, Any]] = None,
        new_value: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            action: Type of action being logged
            description: Human-readable description
            user_id: User performing the action
            user_email: User's email
            api_key_id: API key used (if applicable)
            tenant_id: Tenant identifier
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            ip_address: Client IP address
            user_agent: Client user agent
            endpoint: API endpoint called
            method: HTTP method
            severity: Event severity level
            old_value: Previous value (for updates)
            new_value: New value (for updates)
            metadata: Additional metadata
            success: Whether the action succeeded
            error_message: Error message if failed

        Returns:
            Created audit event
        """
        event = AuditEvent(
            action=action,
            description=description,
            user_id=user_id,
            user_email=user_email,
            api_key_id=api_key_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            severity=severity,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
            success=success,
            error_message=error_message,
        )

        # Store event
        await self._store_event(event)

        # Log to standard logger as well
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)

        logger.log(
            log_level,
            f"AUDIT: {action.value} - {description} "
            f"[user={user_id}, resource={resource_type}:{resource_id}]",
        )

        return event

    async def _store_event(self, event: AuditEvent) -> None:
        """Store event to configured backend."""
        if self.db_pool:
            await self._store_to_postgres(event)
        elif self.redis_client:
            await self._store_to_redis(event)
        else:
            self._store_to_memory(event)

    async def _store_to_postgres(self, event: AuditEvent) -> None:
        """Store event to PostgreSQL."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_logs (
                    id, timestamp, user_id, user_email, api_key_id, tenant_id,
                    action, severity, resource_type, resource_id,
                    ip_address, user_agent, endpoint, method,
                    description, old_value, new_value, metadata,
                    success, error_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
                )
                """,
                event.id,
                event.timestamp,
                event.user_id,
                event.user_email,
                event.api_key_id,
                event.tenant_id,
                event.action.value,
                event.severity.value,
                event.resource_type,
                event.resource_id,
                event.ip_address,
                event.user_agent,
                event.endpoint,
                event.method,
                event.description,
                json.dumps(event.old_value) if event.old_value else None,
                json.dumps(event.new_value) if event.new_value else None,
                json.dumps(event.metadata),
                event.success,
                event.error_message,
            )

    async def _store_to_redis(self, event: AuditEvent) -> None:
        """Store event to Redis (for high-throughput scenarios)."""
        key = f"audit:{event.timestamp.strftime('%Y%m%d')}:{event.id}"
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days TTL
            event.model_dump_json(),
        )

        # Also add to sorted set for time-based queries
        await self.redis_client.zadd(
            f"audit:timeline:{event.tenant_id or 'global'}",
            {str(event.id): event.timestamp.timestamp()},
        )

    def _store_to_memory(self, event: AuditEvent) -> None:
        """Store event to in-memory list."""
        self._events.append(event)

        # Trim if over limit
        if len(self._events) > self.max_memory_events:
            self._events = self._events[-self.max_memory_events :]

    async def query(self, query: AuditQuery) -> AuditLog:
        """
        Query audit logs with filters.

        Args:
            query: Query parameters

        Returns:
            Paginated audit log results
        """
        if self.db_pool:
            return await self._query_postgres(query)
        else:
            return self._query_memory(query)

    async def _query_postgres(self, query: AuditQuery) -> AuditLog:
        """Query PostgreSQL for audit events."""
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if query.user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(query.user_id)
            param_idx += 1

        if query.tenant_id:
            conditions.append(f"tenant_id = ${param_idx}")
            params.append(query.tenant_id)
            param_idx += 1

        if query.action:
            conditions.append(f"action = ${param_idx}")
            params.append(query.action.value)
            param_idx += 1

        if query.actions:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.actions)))
            conditions.append(f"action IN ({placeholders})")
            params.extend(a.value for a in query.actions)
            param_idx += len(query.actions)

        if query.severity:
            conditions.append(f"severity = ${param_idx}")
            params.append(query.severity.value)
            param_idx += 1

        if query.resource_type:
            conditions.append(f"resource_type = ${param_idx}")
            params.append(query.resource_type)
            param_idx += 1

        if query.resource_id:
            conditions.append(f"resource_id = ${param_idx}")
            params.append(query.resource_id)
            param_idx += 1

        if query.start_time:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(query.start_time)
            param_idx += 1

        if query.end_time:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(query.end_time)
            param_idx += 1

        if query.success is not None:
            conditions.append(f"success = ${param_idx}")
            params.append(query.success)
            param_idx += 1

        if query.search_text:
            conditions.append(f"description ILIKE ${param_idx}")
            params.append(f"%{query.search_text}%")
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        async with self.db_pool.acquire() as conn:
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM audit_logs WHERE {where_clause}",
                *params,
            )
            total_count = count_row["count"]

            # Get paginated results
            offset = (query.page - 1) * query.page_size
            params.extend([query.page_size, offset])

            rows = await conn.fetch(
                f"""
                SELECT * FROM audit_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
            )

            events = []
            for row in rows:
                row_dict = dict(row)
                row_dict["action"] = AuditAction(row_dict["action"])
                row_dict["severity"] = AuditSeverity(row_dict["severity"])
                if row_dict.get("old_value"):
                    row_dict["old_value"] = json.loads(row_dict["old_value"])
                if row_dict.get("new_value"):
                    row_dict["new_value"] = json.loads(row_dict["new_value"])
                if row_dict.get("metadata"):
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                events.append(AuditEvent(**row_dict))

            return AuditLog(
                events=events,
                total_count=total_count,
                page=query.page,
                page_size=query.page_size,
                has_more=(offset + len(events)) < total_count,
            )

    def _query_memory(self, query: AuditQuery) -> AuditLog:
        """Query in-memory events."""
        filtered = self._events.copy()

        if query.user_id:
            filtered = [e for e in filtered if e.user_id == query.user_id]

        if query.tenant_id:
            filtered = [e for e in filtered if e.tenant_id == query.tenant_id]

        if query.action:
            filtered = [e for e in filtered if e.action == query.action]

        if query.actions:
            filtered = [e for e in filtered if e.action in query.actions]

        if query.severity:
            filtered = [e for e in filtered if e.severity == query.severity]

        if query.min_severity:
            min_order = self.SEVERITY_ORDER[query.min_severity]
            filtered = [
                e for e in filtered if self.SEVERITY_ORDER[e.severity] >= min_order
            ]

        if query.resource_type:
            filtered = [e for e in filtered if e.resource_type == query.resource_type]

        if query.resource_id:
            filtered = [e for e in filtered if e.resource_id == query.resource_id]

        if query.start_time:
            filtered = [e for e in filtered if e.timestamp >= query.start_time]

        if query.end_time:
            filtered = [e for e in filtered if e.timestamp <= query.end_time]

        if query.success is not None:
            filtered = [e for e in filtered if e.success == query.success]

        if query.search_text:
            search_lower = query.search_text.lower()
            filtered = [e for e in filtered if search_lower in e.description.lower()]

        # Sort by timestamp descending
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        total_count = len(filtered)
        offset = (query.page - 1) * query.page_size
        paginated = filtered[offset : offset + query.page_size]

        return AuditLog(
            events=paginated,
            total_count=total_count,
            page=query.page,
            page_size=query.page_size,
            has_more=(offset + len(paginated)) < total_count,
        )

    async def get_stats(
        self,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AuditStats:
        """
        Get audit statistics.

        Args:
            tenant_id: Filter by tenant
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Audit statistics
        """
        query = AuditQuery(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            page_size=10000,  # Get all for stats
        )

        log = await self.query(query)

        events_by_action: dict[str, int] = defaultdict(int)
        events_by_severity: dict[str, int] = defaultdict(int)
        events_by_user: dict[str, int] = defaultdict(int)
        failed_count = 0

        for event in log.events:
            events_by_action[event.action.value] += 1
            events_by_severity[event.severity.value] += 1
            if event.user_email:
                events_by_user[event.user_email] += 1
            if not event.success:
                failed_count += 1

        success_rate = (
            ((log.total_count - failed_count) / log.total_count * 100)
            if log.total_count > 0
            else 100.0
        )

        return AuditStats(
            total_events=log.total_count,
            events_by_action=dict(events_by_action),
            events_by_severity=dict(events_by_severity),
            events_by_user=dict(events_by_user),
            failed_events=failed_count,
            success_rate=success_rate,
            time_range_start=start_time,
            time_range_end=end_time,
        )

    # Convenience methods for common audit events

    async def log_login(
        self,
        user_id: UUID,
        user_email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """Log a login attempt."""
        action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING

        return await self.log(
            action=action,
            description=f"User login {'successful' if success else 'failed'}: {user_email}",
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=severity,
            success=success,
            error_message=error_message,
        )

    async def log_api_key_action(
        self,
        action: AuditAction,
        user_id: UUID,
        api_key_id: UUID,
        key_name: str,
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log an API key action (create, revoke, rotate, delete)."""
        descriptions = {
            AuditAction.API_KEY_CREATE: f"API key created: {key_name}",
            AuditAction.API_KEY_REVOKE: f"API key revoked: {key_name}",
            AuditAction.API_KEY_ROTATE: f"API key rotated: {key_name}",
            AuditAction.API_KEY_DELETE: f"API key deleted: {key_name}",
        }

        return await self.log(
            action=action,
            description=descriptions.get(action, f"API key action: {key_name}"),
            user_id=user_id,
            api_key_id=api_key_id,
            resource_type="api_key",
            resource_id=str(api_key_id),
            ip_address=ip_address,
            severity=AuditSeverity.INFO,
        )

    async def log_memory_action(
        self,
        action: AuditAction,
        user_id: UUID,
        memory_id: str,
        description: str,
        ip_address: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a memory operation."""
        return await self.log(
            action=action,
            description=description,
            user_id=user_id,
            resource_type="memory",
            resource_id=memory_id,
            ip_address=ip_address,
            metadata=metadata,
        )

    async def log_unauthorized_access(
        self,
        endpoint: str,
        method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        reason: str = "Unauthorized access attempt",
    ) -> AuditEvent:
        """Log an unauthorized access attempt."""
        return await self.log(
            action=AuditAction.UNAUTHORIZED_ACCESS,
            description=f"Unauthorized access to {method} {endpoint}: {reason}",
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=AuditSeverity.WARNING,
            success=False,
            error_message=reason,
        )
