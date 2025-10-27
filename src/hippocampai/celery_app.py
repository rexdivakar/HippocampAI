"""Celery application configuration for distributed task processing."""

import os

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

# Initialize Celery app
celery_app = Celery(
    "hippocampai",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=["hippocampai.tasks"],
)

# Celery Configuration
celery_app.conf.update(
    # Task routing
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("memory_ops", Exchange("memory_ops"), routing_key="memory.#"),
        Queue("background", Exchange("background"), routing_key="background.#"),
        Queue("scheduled", Exchange("scheduled"), routing_key="scheduled.#"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Task execution settings
    task_acks_late=os.getenv("CELERY_TASK_ACKS_LATE", "true").lower() == "true",
    worker_prefetch_multiplier=int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "4")),
    worker_max_tasks_per_child=int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "1000")),

    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    result_compression="gzip",

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task result format
    result_extended=True,

    # Worker settings
    worker_disable_rate_limits=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",

    # Beat scheduler settings (for periodic tasks)
    beat_scheduler="celery.beat:PersistentScheduler",
    beat_schedule_filename="/app/logs/celerybeat-schedule.db",
)

# Scheduled tasks configuration
def get_beat_schedule():
    """Get the Celery Beat schedule based on environment configuration."""
    schedule = {}

    # Automatic deduplication
    if os.getenv("AUTO_DEDUP_ENABLED", "true").lower() == "true":
        dedup_interval = int(os.getenv("DEDUP_INTERVAL_HOURS", "24"))
        schedule["auto-deduplicate-memories"] = {
            "task": "hippocampai.tasks.deduplicate_all_memories",
            "schedule": crontab(hour=f"*/{dedup_interval}"),
            "options": {"queue": "scheduled"},
        }

    # Memory consolidation
    if os.getenv("AUTO_CONSOLIDATION_ENABLED", "false").lower() == "true":
        consolidation_interval = int(os.getenv("CONSOLIDATION_INTERVAL_HOURS", "168"))
        schedule["auto-consolidate-memories"] = {
            "task": "hippocampai.tasks.consolidate_all_memories",
            "schedule": crontab(hour=f"*/{consolidation_interval}"),
            "options": {"queue": "scheduled"},
        }

    # Memory expiration cleanup
    expiration_interval = int(os.getenv("EXPIRATION_INTERVAL_HOURS", "1"))
    schedule["cleanup-expired-memories"] = {
        "task": "hippocampai.tasks.cleanup_expired_memories",
        "schedule": crontab(hour=f"*/{expiration_interval}"),
        "options": {"queue": "scheduled"},
    }

    # Importance decay (daily at 2am)
    schedule["decay-memory-importance"] = {
        "task": "hippocampai.tasks.decay_memory_importance",
        "schedule": crontab(hour=2, minute=0),
        "options": {"queue": "scheduled"},
    }

    # Create snapshots (hourly)
    schedule["create-snapshots"] = {
        "task": "hippocampai.tasks.create_collection_snapshots",
        "schedule": crontab(minute=0),  # Every hour
        "options": {"queue": "scheduled"},
    }

    # Health check (every 5 minutes)
    schedule["health-check"] = {
        "task": "hippocampai.tasks.health_check_task",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "background"},
    }

    return schedule

# Update beat schedule
celery_app.conf.beat_schedule = get_beat_schedule()

# Task routing
celery_app.conf.task_routes = {
    "hippocampai.tasks.create_memory_task": {"queue": "memory_ops"},
    "hippocampai.tasks.batch_create_memories_task": {"queue": "memory_ops"},
    "hippocampai.tasks.recall_memories_task": {"queue": "memory_ops"},
    "hippocampai.tasks.update_memory_task": {"queue": "memory_ops"},
    "hippocampai.tasks.delete_memory_task": {"queue": "memory_ops"},
    "hippocampai.tasks.deduplicate_all_memories": {"queue": "scheduled"},
    "hippocampai.tasks.consolidate_all_memories": {"queue": "scheduled"},
    "hippocampai.tasks.cleanup_expired_memories": {"queue": "scheduled"},
    "hippocampai.tasks.decay_memory_importance": {"queue": "scheduled"},
    "hippocampai.tasks.create_collection_snapshots": {"queue": "scheduled"},
    "hippocampai.tasks.health_check_task": {"queue": "background"},
}

if __name__ == "__main__":
    celery_app.start()
