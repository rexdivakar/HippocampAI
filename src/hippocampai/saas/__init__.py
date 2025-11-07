"""SaaS automation and control layer for HippocampAI.

This module provides automated background processing and control interfaces
that work both for library users and SaaS deployments.
"""

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

__all__ = [
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
]
