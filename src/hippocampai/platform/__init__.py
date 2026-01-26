"""HippocampAI Platform (SaaS) Components.

This module contains the SaaS platform components that extend the core
memory engine with production infrastructure features:

- FastAPI REST API server
- Authentication and authorization
- Rate limiting
- Background task processing (Celery)
- Scheduled jobs
- Monitoring and metrics (Prometheus)
- Multi-tenant support

These components require additional infrastructure:
- Redis (caching, Celery broker)
- PostgreSQL (authentication, user management)
- Celery workers (background tasks)

For the core memory engine without SaaS dependencies,
use `hippocampai.core`.

Example:
    >>> # Start the API server
    >>> from hippocampai.platform import run_api_server
    >>> run_api_server(host="0.0.0.0", port=8000)

    >>> # Use automation controller
    >>> from hippocampai.platform import AutomationController
    >>> controller = AutomationController()
    >>> controller.start()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.3.0"

__all__ = [
    # API
    "create_app",
    "run_api_server",
    # Authentication
    "AuthService",
    "RateLimiter",
    # Automation
    "AutomationController",
    "AutomationPolicy",
    "AutomationSchedule",
    "PolicyType",
    # Tasks
    "TaskManager",
    "TaskPriority",
    "TaskStatus",
    "BackgroundTask",
    # Celery
    "celery_app",
    # Configuration
    "PlatformConfig",
    "get_platform_config",
]

if TYPE_CHECKING:
    from hippocampai.api.app import app as create_app
    from hippocampai.api.app import run_server as run_api_server
    from hippocampai.auth.auth_service import AuthService
    from hippocampai.auth.rate_limiter import RateLimiter
    from hippocampai.celery_app import celery_app
    from hippocampai.platform.config import PlatformConfig, get_platform_config
    from hippocampai.saas.automation import (
        AutomationController,
        AutomationPolicy,
        AutomationSchedule,
        PolicyType,
    )
    from hippocampai.saas.tasks import (
        BackgroundTask,
        TaskManager,
        TaskPriority,
        TaskStatus,
    )


def __getattr__(name: str):
    """Lazy loading for platform components."""

    # API
    if name == "create_app":
        from hippocampai.api.app import app

        return app

    if name == "run_api_server":
        from hippocampai.api.app import run_server

        return run_server

    # Authentication
    if name == "AuthService":
        from hippocampai.auth.auth_service import AuthService

        return AuthService

    if name == "RateLimiter":
        from hippocampai.auth.rate_limiter import RateLimiter

        return RateLimiter

    # Automation
    if name == "AutomationController":
        from hippocampai.saas.automation import AutomationController

        return AutomationController

    if name == "AutomationPolicy":
        from hippocampai.saas.automation import AutomationPolicy

        return AutomationPolicy

    if name == "AutomationSchedule":
        from hippocampai.saas.automation import AutomationSchedule

        return AutomationSchedule

    if name == "PolicyType":
        from hippocampai.saas.automation import PolicyType

        return PolicyType

    # Tasks
    if name == "TaskManager":
        from hippocampai.saas.tasks import TaskManager

        return TaskManager

    if name == "TaskPriority":
        from hippocampai.saas.tasks import TaskPriority

        return TaskPriority

    if name == "TaskStatus":
        from hippocampai.saas.tasks import TaskStatus

        return TaskStatus

    if name == "BackgroundTask":
        from hippocampai.saas.tasks import BackgroundTask

        return BackgroundTask

    # Celery
    if name == "celery_app":
        from hippocampai.celery_app import celery_app

        return celery_app

    # Configuration
    if name == "PlatformConfig":
        from hippocampai.platform.config import PlatformConfig

        return PlatformConfig

    if name == "get_platform_config":
        from hippocampai.platform.config import get_platform_config

        return get_platform_config

    raise AttributeError(f"module 'hippocampai.platform' has no attribute {name!r}")
