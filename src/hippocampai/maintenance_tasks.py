"""Celery tasks for scheduled memory maintenance."""

import logging
from typing import Any

from celery.schedules import crontab

from hippocampai.celery_app import celery_app
from hippocampai.models.healing import AutoHealingConfig

logger = logging.getLogger(__name__)


@celery_app.task(name="maintenance.run_health_check", bind=True)
def run_health_check_task(self: Any, user_id: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Run scheduled health check for a user.

    Args:
        user_id: User ID
        config_dict: Auto-healing configuration as dict

    Returns:
        Task result dictionary
    """
    try:
        from hippocampai.client import MemoryClient

        # Initialize client
        client = MemoryClient(user_id=user_id)

        # Load configuration
        config = AutoHealingConfig(**config_dict)

        # Get all memories
        memories = client.get_memories(include_expired=False)

        # Run health check
        report = client.auto_healing_engine.run_full_health_check(
            user_id=user_id, memories=memories, config=config, dry_run=not config.enabled
        )

        logger.info(
            f"Health check completed for user {user_id}: {report.health_after:.1f}/100 "
            f"({len(report.actions_applied)} actions applied)"
        )

        return {
            "user_id": user_id,
            "status": "success",
            "health_score": report.health_after,
            "actions_applied": len(report.actions_applied),
            "actions_recommended": len(report.actions_recommended),
            "duration_seconds": report.duration_seconds,
        }

    except Exception as e:
        logger.error(f"Health check failed for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="maintenance.run_cleanup", bind=True)
def run_cleanup_task(self: Any, user_id: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Run scheduled cleanup for a user.

    Args:
        user_id: User ID
        config_dict: Auto-healing configuration as dict

    Returns:
        Task result dictionary
    """
    try:
        from hippocampai.client import MemoryClient

        # Initialize client
        client = MemoryClient(user_id=user_id)

        # Load configuration
        config = AutoHealingConfig(**config_dict)

        # Get all memories
        memories = client.get_memories(include_expired=False)

        # Run cleanup
        report = client.auto_healing_engine.auto_cleanup(
            user_id=user_id, memories=memories, config=config, dry_run=not config.enabled
        )

        logger.info(
            f"Cleanup completed for user {user_id}: {len(report.actions_applied)} actions applied"
        )

        return {
            "user_id": user_id,
            "status": "success",
            "actions_applied": len(report.actions_applied),
            "memories_affected": report.memories_affected,
            "duration_seconds": report.duration_seconds,
        }

    except Exception as e:
        logger.error(f"Cleanup failed for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="maintenance.run_deduplication", bind=True)
def run_deduplication_task(self: Any, user_id: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Run scheduled deduplication for a user.

    Args:
        user_id: User ID
        config_dict: Auto-healing configuration as dict

    Returns:
        Task result dictionary
    """
    try:
        from hippocampai.client import MemoryClient

        # Initialize client
        client = MemoryClient(user_id=user_id)

        # Load configuration
        config = AutoHealingConfig(**config_dict)

        # Get all memories
        memories = client.get_memories(include_expired=False)

        # Run deduplication
        report = client.auto_healing_engine.auto_consolidate(
            user_id=user_id, memories=memories, config=config, dry_run=not config.enabled
        )

        logger.info(
            f"Deduplication completed for user {user_id}: {len(report.actions_applied)} merges"
        )

        return {
            "user_id": user_id,
            "status": "success",
            "actions_applied": len(report.actions_applied),
            "clusters_found": len(report.actions_recommended),
            "duration_seconds": report.duration_seconds,
        }

    except Exception as e:
        logger.error(f"Deduplication failed for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="maintenance.generate_predictions", bind=True)
def generate_predictions_task(self: Any, user_id: str) -> dict[str, Any]:
    """
    Generate predictive insights for a user.

    Args:
        user_id: User ID

    Returns:
        Task result dictionary
    """
    try:
        from hippocampai.client import MemoryClient

        # Initialize client
        client = MemoryClient(user_id=user_id)

        # Get all memories
        memories = client.get_memories(include_expired=False)

        # Generate predictions
        predictions = client.predictive_engine.generate_predictive_insights(
            user_id=user_id, memories=memories
        )

        # Generate recommendations
        recommendations = client.predictive_engine.generate_recommendations(
            user_id=user_id, memories=memories
        )

        logger.info(
            f"Generated {len(predictions)} predictions and {len(recommendations)} "
            f"recommendations for user {user_id}"
        )

        return {
            "user_id": user_id,
            "status": "success",
            "predictions_count": len(predictions),
            "recommendations_count": len(recommendations),
        }

    except Exception as e:
        logger.error(f"Prediction generation failed for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="maintenance.detect_anomalies", bind=True)
def detect_anomalies_task(self: Any, user_id: str, lookback_days: int = 30) -> dict[str, Any]:
    """
    Detect anomalies in user's memory patterns.

    Args:
        user_id: User ID
        lookback_days: Days to look back

    Returns:
        Task result dictionary
    """
    try:
        from hippocampai.client import MemoryClient

        # Initialize client
        client = MemoryClient(user_id=user_id)

        # Get all memories
        memories = client.get_memories(include_expired=False)

        # Detect anomalies
        anomalies = client.predictive_engine.detect_anomalies(
            user_id=user_id, memories=memories, lookback_days=lookback_days
        )

        logger.info(f"Detected {len(anomalies)} anomalies for user {user_id}")

        return {
            "user_id": user_id,
            "status": "success",
            "anomalies_count": len(anomalies),
            "anomaly_types": [a.anomaly_type.value for a in anomalies],
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


# Schedule periodic tasks
@celery_app.on_after_finalize.connect
def setup_periodic_tasks(sender: Any, **kwargs: Any) -> None:
    """Set up periodic tasks for maintenance."""
    # Daily health check at 3 AM
    sender.add_periodic_task(
        crontab(hour=3, minute=0),
        run_health_check_task.s(),
        name="daily_health_check",
    )

    # Weekly cleanup on Sunday at 2 AM
    sender.add_periodic_task(
        crontab(hour=2, minute=0, day_of_week=0),
        run_cleanup_task.s(),
        name="weekly_cleanup",
    )

    # Daily anomaly detection at 4 AM
    sender.add_periodic_task(
        crontab(hour=4, minute=0),
        detect_anomalies_task.s(),
        name="daily_anomaly_detection",
    )

    # Weekly predictions on Monday at 6 AM
    sender.add_periodic_task(
        crontab(hour=6, minute=0, day_of_week=1),
        generate_predictions_task.s(),
        name="weekly_predictions",
    )
