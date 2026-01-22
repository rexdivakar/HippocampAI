"""Tests for SaaS features: API key rotation, audit logging, usage analytics."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

# Check if bcrypt is available
try:
    import bcrypt  # noqa: F401

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


class TestAPIKeyRotation:
    """Tests for API key rotation functionality."""

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_auth_service_has_rotate_method(self):
        """Test that AuthService has rotate_api_key method."""
        from hippocampai.auth.auth_service import AuthService

        assert hasattr(AuthService, "rotate_api_key")

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_generate_api_key(self):
        """Test API key generation format."""
        from hippocampai.auth.auth_service import AuthService

        full_key, key_hash = AuthService.generate_api_key()

        # Check format
        assert full_key.startswith("hc_live_")
        assert len(full_key) > 20
        assert key_hash != full_key  # Hash should be different


class TestAuditModels:
    """Tests for audit logging models."""

    def test_audit_action_enum(self):
        """Test AuditAction enum values."""
        from hippocampai.audit import AuditAction

        assert AuditAction.LOGIN.value == "login"
        assert AuditAction.API_KEY_CREATE.value == "api_key_create"
        assert AuditAction.MEMORY_DELETE.value == "memory_delete"
        assert AuditAction.UNAUTHORIZED_ACCESS.value == "unauthorized_access"

    def test_audit_severity_enum(self):
        """Test AuditSeverity enum values."""
        from hippocampai.audit import AuditSeverity

        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_audit_event_creation(self):
        """Test creating an AuditEvent."""
        from hippocampai.audit import AuditAction, AuditEvent, AuditSeverity

        event = AuditEvent(
            action=AuditAction.LOGIN,
            description="User logged in",
            user_id=uuid4(),
            user_email="test@example.com",
            ip_address="192.168.1.1",
            severity=AuditSeverity.INFO,
        )

        assert event.action == AuditAction.LOGIN
        assert event.description == "User logged in"
        assert event.user_email == "test@example.com"
        assert event.success is True
        assert event.timestamp is not None

    def test_audit_query_defaults(self):
        """Test AuditQuery default values."""
        from hippocampai.audit import AuditQuery

        query = AuditQuery()

        assert query.page == 1
        assert query.page_size == 100
        assert query.user_id is None
        assert query.action is None

    def test_audit_stats_model(self):
        """Test AuditStats model."""
        from hippocampai.audit import AuditStats

        stats = AuditStats(
            total_events=100,
            events_by_action={"login": 50, "logout": 30},
            failed_events=5,
            success_rate=95.0,
        )

        assert stats.total_events == 100
        assert stats.events_by_action["login"] == 50
        assert stats.success_rate == 95.0


class TestAuditLogger:
    """Tests for AuditLogger functionality."""

    @pytest.mark.asyncio
    async def test_audit_logger_creation(self):
        """Test creating an AuditLogger."""
        from hippocampai.audit import AuditLogger

        logger = AuditLogger()
        assert logger is not None

    @pytest.mark.asyncio
    async def test_log_event(self):
        """Test logging an audit event."""
        from hippocampai.audit import AuditAction, AuditLogger, AuditSeverity

        logger = AuditLogger()

        event = await logger.log(
            action=AuditAction.LOGIN,
            description="Test login",
            user_id=uuid4(),
            user_email="test@example.com",
            severity=AuditSeverity.INFO,
        )

        assert event.action == AuditAction.LOGIN
        assert event.description == "Test login"

    @pytest.mark.asyncio
    async def test_query_events(self):
        """Test querying audit events."""
        from hippocampai.audit import AuditAction, AuditLogger, AuditQuery

        logger = AuditLogger()
        user_id = uuid4()

        # Log some events
        await logger.log(
            action=AuditAction.LOGIN,
            description="Login 1",
            user_id=user_id,
        )
        await logger.log(
            action=AuditAction.LOGOUT,
            description="Logout 1",
            user_id=user_id,
        )

        # Query all
        query = AuditQuery()
        log = await logger.query(query)

        assert log.total_count >= 2
        assert len(log.events) >= 2

    @pytest.mark.asyncio
    async def test_query_by_action(self):
        """Test filtering by action."""
        from hippocampai.audit import AuditAction, AuditLogger, AuditQuery

        logger = AuditLogger()

        await logger.log(action=AuditAction.LOGIN, description="Login")
        await logger.log(action=AuditAction.LOGOUT, description="Logout")
        await logger.log(action=AuditAction.LOGIN, description="Login 2")

        query = AuditQuery(action=AuditAction.LOGIN)
        log = await logger.query(query)

        assert all(e.action == AuditAction.LOGIN for e in log.events)

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting audit statistics."""
        from hippocampai.audit import AuditAction, AuditLogger

        logger = AuditLogger()

        await logger.log(action=AuditAction.LOGIN, description="Login")
        await logger.log(action=AuditAction.LOGIN, description="Login 2")
        await logger.log(action=AuditAction.LOGOUT, description="Logout")

        stats = await logger.get_stats()

        assert stats.total_events >= 3
        assert "login" in stats.events_by_action

    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """Test convenience logging methods."""
        from hippocampai.audit import AuditAction, AuditLogger

        logger = AuditLogger()
        user_id = uuid4()

        # Test log_login
        event = await logger.log_login(
            user_id=user_id,
            user_email="test@example.com",
            ip_address="192.168.1.1",
            success=True,
        )
        assert event.action == AuditAction.LOGIN

        # Test log_login failed
        event = await logger.log_login(
            user_id=user_id,
            user_email="test@example.com",
            success=False,
            error_message="Invalid password",
        )
        assert event.action == AuditAction.LOGIN_FAILED
        assert event.success is False

        # Test log_api_key_action
        event = await logger.log_api_key_action(
            action=AuditAction.API_KEY_CREATE,
            user_id=user_id,
            api_key_id=uuid4(),
            key_name="Test Key",
        )
        assert event.action == AuditAction.API_KEY_CREATE
        assert event.resource_type == "api_key"


class TestRetentionPolicy:
    """Tests for audit retention policies."""

    def test_retention_period_enum(self):
        """Test RetentionPeriod enum."""
        from hippocampai.audit.retention import RetentionPeriod

        assert RetentionPeriod.DAYS_30.value == "30_days"
        assert RetentionPeriod.YEAR_1.value == "1_year"
        assert RetentionPeriod.YEARS_7.value == "7_years"
        assert RetentionPeriod.FOREVER.value == "forever"

    def test_retention_policy_defaults(self):
        """Test AuditRetentionPolicy default values."""
        from hippocampai.audit import AuditRetentionPolicy
        from hippocampai.audit.retention import RetentionPeriod

        policy = AuditRetentionPolicy()

        assert policy.debug_retention == RetentionPeriod.DAYS_30
        assert policy.info_retention == RetentionPeriod.DAYS_90
        assert policy.critical_retention == RetentionPeriod.YEARS_7
        assert policy.enabled is True

    def test_retention_manager_creation(self):
        """Test creating a RetentionManager."""
        from hippocampai.audit import RetentionManager

        manager = RetentionManager()
        assert manager is not None

    def test_get_cutoff_date(self):
        """Test calculating cutoff dates."""
        from hippocampai.audit import RetentionManager
        from hippocampai.audit.retention import RetentionPeriod

        manager = RetentionManager()
        now = datetime.now(timezone.utc)

        # 30 days
        cutoff = manager.get_cutoff_date(RetentionPeriod.DAYS_30, now)
        assert cutoff is not None
        assert (now - cutoff).days == 30

        # Forever should return None
        cutoff = manager.get_cutoff_date(RetentionPeriod.FOREVER, now)
        assert cutoff is None


class TestUsageAnalytics:
    """Tests for usage analytics models."""

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_tier_limits_defined(self):
        """Test that tier limits are defined."""
        from hippocampai.api.usage_routes import TIER_LIMITS

        assert "free" in TIER_LIMITS
        assert "pro" in TIER_LIMITS
        assert "enterprise" in TIER_LIMITS

        # Check free tier limits
        free = TIER_LIMITS["free"]
        assert free["memories"] == 1000
        assert free["api_calls_day"] == 100

        # Pro should have higher limits
        pro = TIER_LIMITS["pro"]
        assert pro["memories"] > free["memories"]
        assert pro["api_calls_day"] > free["api_calls_day"]

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_usage_period_model(self):
        """Test UsagePeriod model."""
        from hippocampai.api.usage_routes import UsagePeriod

        now = datetime.now(timezone.utc)
        period = UsagePeriod(
            period_start=now,
            period_end=now,
            api_calls=100,
            memories_created=50,
        )

        assert period.api_calls == 100
        assert period.memories_created == 50

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_tenant_usage_model(self):
        """Test TenantUsage model."""
        from hippocampai.api.usage_routes import TenantUsage

        usage = TenantUsage(
            user_id=uuid4(),
            tier="pro",
            total_memories=5000,
            memory_limit=100000,
        )

        assert usage.tier == "pro"
        assert usage.total_memories == 5000
        assert usage.memory_limit == 100000


class TestAuditRoutes:
    """Tests for audit API routes."""

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_routes_exist(self):
        """Test that audit routes are defined."""
        from hippocampai.api.audit_routes import router

        # Check router has routes
        routes = [r.path for r in router.routes]

        assert "/logs" in routes or any("/logs" in r for r in routes)

    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not installed")
    def test_list_audit_actions(self):
        """Test list_audit_actions returns categorized actions."""
        import asyncio

        from hippocampai.api.audit_routes import list_audit_actions

        result = asyncio.get_event_loop().run_until_complete(list_audit_actions())

        assert "authentication" in result
        assert "api_keys" in result
        assert "memories" in result
        assert "admin" in result
        assert "system" in result

        assert "login" in result["authentication"]
        assert "api_key_create" in result["api_keys"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
