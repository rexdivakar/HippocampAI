"""API routes for audit log access and management."""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from hippocampai.api.middleware import require_admin
from hippocampai.audit import (
    AuditAction,
    AuditLog,
    AuditLogger,
    AuditQuery,
    AuditRetentionPolicy,
    AuditSeverity,
    AuditStats,
    RetentionManager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])


# ============================================
# HELPER FUNCTIONS
# ============================================


def get_audit_logger(request: Request) -> AuditLogger:
    """Get audit logger from app state."""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if not audit_logger:
        # Create in-memory logger if not configured
        audit_logger = AuditLogger()
        request.app.state.audit_logger = audit_logger
    return audit_logger


def get_retention_manager(request: Request) -> RetentionManager:
    """Get retention manager from app state."""
    manager = getattr(request.app.state, "retention_manager", None)
    if not manager:
        db_pool = getattr(request.app.state, "db_pool", None)
        manager = RetentionManager(db_pool=db_pool)
        request.app.state.retention_manager = manager
    return manager


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================


class AuditQueryRequest(BaseModel):
    """Request model for audit log queries."""

    user_id: Optional[UUID] = None
    tenant_id: Optional[str] = None
    action: Optional[str] = None
    actions: Optional[list[str]] = None
    severity: Optional[str] = None
    min_severity: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: Optional[bool] = None
    search_text: Optional[str] = None
    page: int = 1
    page_size: int = 100


class RetentionPolicyRequest(BaseModel):
    """Request model for setting retention policy."""

    tenant_id: Optional[str] = None
    debug_retention: str = "30_days"
    info_retention: str = "90_days"
    warning_retention: str = "1_year"
    error_retention: str = "3_years"
    critical_retention: str = "7_years"
    enabled: bool = True
    archive_before_delete: bool = True
    archive_location: Optional[str] = None


# ============================================
# API ENDPOINTS
# ============================================


@router.get("/logs", dependencies=[Depends(require_admin)])
async def query_audit_logs(
    request: Request,
    user_id: Optional[UUID] = None,
    tenant_id: Optional[str] = None,
    action: Optional[str] = None,
    severity: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    success: Optional[bool] = None,
    search: Optional[str] = Query(None, description="Search in description"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
) -> AuditLog:
    """
    Query audit logs with filters (admin only).

    Returns paginated audit events matching the specified criteria.
    """
    try:
        audit_logger = get_audit_logger(request)

        # Build query
        query = AuditQuery(
            user_id=user_id,
            tenant_id=tenant_id,
            action=AuditAction(action) if action else None,
            severity=AuditSeverity(severity) if severity else None,
            resource_type=resource_type,
            resource_id=resource_id,
            start_time=start_time,
            end_time=end_time,
            success=success,
            search_text=search,
            page=page,
            page_size=page_size,
        )

        return await audit_logger.query(query)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to query audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query logs: {str(e)}")


@router.post("/logs/search", dependencies=[Depends(require_admin)])
async def search_audit_logs(
    query_request: AuditQueryRequest,
    request: Request,
) -> AuditLog:
    """
    Advanced audit log search with POST body (admin only).

    Allows more complex queries than the GET endpoint.
    """
    try:
        audit_logger = get_audit_logger(request)

        # Convert string enums to actual enums
        action = AuditAction(query_request.action) if query_request.action else None
        actions = [AuditAction(a) for a in query_request.actions] if query_request.actions else None
        severity = AuditSeverity(query_request.severity) if query_request.severity else None
        min_severity = (
            AuditSeverity(query_request.min_severity) if query_request.min_severity else None
        )

        query = AuditQuery(
            user_id=query_request.user_id,
            tenant_id=query_request.tenant_id,
            action=action,
            actions=actions,
            severity=severity,
            min_severity=min_severity,
            resource_type=query_request.resource_type,
            resource_id=query_request.resource_id,
            ip_address=query_request.ip_address,
            start_time=query_request.start_time,
            end_time=query_request.end_time,
            success=query_request.success,
            search_text=query_request.search_text,
            page=query_request.page,
            page_size=query_request.page_size,
        )

        return await audit_logger.query(query)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to search audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")


@router.get("/logs/{event_id}", dependencies=[Depends(require_admin)])
async def get_audit_event(
    event_id: UUID,
    request: Request,
) -> dict[str, Any]:
    """
    Get a specific audit event by ID (admin only).
    """
    try:
        audit_logger = get_audit_logger(request)

        # Query for specific event
        query = AuditQuery(page_size=10000)
        log = await audit_logger.query(query)

        for event in log.events:
            if event.id == event_id:
                return event.model_dump()

        raise HTTPException(status_code=404, detail="Audit event not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get audit event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get event: {str(e)}")


@router.get("/stats", dependencies=[Depends(require_admin)])
async def get_audit_stats(
    request: Request,
    tenant_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> AuditStats:
    """
    Get audit log statistics (admin only).

    Returns aggregate statistics about audit events.
    """
    try:
        audit_logger = get_audit_logger(request)
        return await audit_logger.get_stats(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
        )

    except Exception as e:
        logger.exception(f"Failed to get audit stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/actions")
async def list_audit_actions() -> dict[str, list[str]]:
    """
    List all available audit action types.

    Returns categorized list of audit actions.
    """
    return {
        "authentication": [
            AuditAction.LOGIN.value,
            AuditAction.LOGOUT.value,
            AuditAction.LOGIN_FAILED.value,
            AuditAction.PASSWORD_CHANGE.value,
        ],
        "api_keys": [
            AuditAction.API_KEY_CREATE.value,
            AuditAction.API_KEY_REVOKE.value,
            AuditAction.API_KEY_ROTATE.value,
            AuditAction.API_KEY_DELETE.value,
        ],
        "users": [
            AuditAction.USER_CREATE.value,
            AuditAction.USER_UPDATE.value,
            AuditAction.USER_DELETE.value,
            AuditAction.USER_TIER_CHANGE.value,
        ],
        "memories": [
            AuditAction.MEMORY_CREATE.value,
            AuditAction.MEMORY_UPDATE.value,
            AuditAction.MEMORY_DELETE.value,
            AuditAction.MEMORY_BULK_DELETE.value,
            AuditAction.MEMORY_EXPORT.value,
            AuditAction.MEMORY_IMPORT.value,
        ],
        "admin": [
            AuditAction.ADMIN_ACCESS.value,
            AuditAction.SETTINGS_CHANGE.value,
            AuditAction.RATE_LIMIT_OVERRIDE.value,
        ],
        "system": [
            AuditAction.SYSTEM_ERROR.value,
            AuditAction.RATE_LIMIT_EXCEEDED.value,
            AuditAction.UNAUTHORIZED_ACCESS.value,
        ],
    }


@router.get("/severities")
async def list_audit_severities() -> list[str]:
    """List all audit severity levels."""
    return [s.value for s in AuditSeverity]


# ============================================
# RETENTION POLICY ENDPOINTS
# ============================================


@router.get("/retention/policy", dependencies=[Depends(require_admin)])
async def get_retention_policy(
    request: Request,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get current retention policy (admin only).
    """
    manager = get_retention_manager(request)
    policy = manager.get_policy(tenant_id)
    return policy.model_dump()


@router.put("/retention/policy", dependencies=[Depends(require_admin)])
async def set_retention_policy(
    policy_request: RetentionPolicyRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Set retention policy (admin only).
    """
    try:
        from hippocampai.audit.retention import RetentionPeriod

        manager = get_retention_manager(request)

        policy = AuditRetentionPolicy(
            tenant_id=policy_request.tenant_id,
            debug_retention=RetentionPeriod(policy_request.debug_retention),
            info_retention=RetentionPeriod(policy_request.info_retention),
            warning_retention=RetentionPeriod(policy_request.warning_retention),
            error_retention=RetentionPeriod(policy_request.error_retention),
            critical_retention=RetentionPeriod(policy_request.critical_retention),
            enabled=policy_request.enabled,
            archive_before_delete=policy_request.archive_before_delete,
            archive_location=policy_request.archive_location,
        )

        manager.set_policy(policy)
        return {"status": "success", "policy": policy.model_dump()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to set retention policy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set policy: {str(e)}")


@router.post("/retention/cleanup", dependencies=[Depends(require_admin)])
async def run_retention_cleanup(
    request: Request,
    tenant_id: Optional[str] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Run retention cleanup (admin only).

    Args:
        tenant_id: Tenant to clean up (None = all)
        dry_run: If True, only report what would be deleted
    """
    try:
        manager = get_retention_manager(request)
        return await manager.cleanup(tenant_id=tenant_id, dry_run=dry_run)

    except Exception as e:
        logger.exception(f"Failed to run cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run cleanup: {str(e)}")


@router.get("/retention/report", dependencies=[Depends(require_admin)])
async def get_retention_report(
    request: Request,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get retention status report (admin only).

    Shows audit log counts by age bucket.
    """
    try:
        manager = get_retention_manager(request)
        return await manager.get_retention_report(tenant_id=tenant_id)

    except Exception as e:
        logger.exception(f"Failed to get retention report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")
