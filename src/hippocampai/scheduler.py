"""Background scheduler for memory maintenance tasks.

This module provides a scheduler for periodic memory consolidation
and other maintenance tasks.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SchedulerWrapper:
    """Wrapper for APScheduler to provide better type safety."""

    def __init__(self) -> None:
        """Initialize the scheduler wrapper."""
        try:
            # Import APScheduler components at runtime
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            self._scheduler = BackgroundScheduler()
            self._cron_trigger_class = CronTrigger
            self._available = True
        except ImportError:
            logger.error("APScheduler not available. Scheduler functionality disabled.")
            self._scheduler = None
            self._cron_trigger_class = None
            self._available = False

    def is_available(self) -> bool:
        """Check if scheduler is available."""
        return self._available

    def add_job(
        self,
        func: Callable[[], None],
        cron_expression: str,
        job_id: str,
        name: str,
        replace_existing: bool = True,
    ) -> bool:
        """Add a job to the scheduler."""
        if not self._available or not self._scheduler or not self._cron_trigger_class:
            logger.warning(f"Cannot add job '{name}': APScheduler not available")
            return False

        try:
            trigger = self._cron_trigger_class.from_crontab(cron_expression)
            self._scheduler.add_job(
                func,
                trigger=trigger,
                id=job_id,
                name=name,
                replace_existing=replace_existing,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add job '{name}': {e}")
            return False

    def start(self) -> bool:
        """Start the scheduler."""
        if not self._available or not self._scheduler:
            logger.warning("Cannot start scheduler: APScheduler not available")
            return False

        try:
            self._scheduler.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        if self._scheduler:
            try:
                self._scheduler.shutdown()
            except Exception as e:
                logger.error(f"Error during scheduler shutdown: {e}")

    def get_jobs(self) -> list[dict[str, Any]]:
        """Get list of jobs with their information."""
        if not self._available or not self._scheduler:
            return []

        try:
            jobs = []
            for job in self._scheduler.get_jobs():
                next_run = job.next_run_time
                jobs.append(
                    {
                        "id": job.id,
                        "name": job.name,
                        "next_run": next_run.isoformat() if next_run else None,
                        "trigger": str(job.trigger),
                    }
                )
            return jobs
        except Exception as e:
            logger.error(f"Failed to get jobs: {e}")
            return []


logger = logging.getLogger(__name__)


class MemoryScheduler:
    """Background scheduler for memory maintenance tasks."""

    def __init__(self, client: Optional[Any] = None, config: Optional[Any] = None) -> None:
        """Initialize scheduler.

        Args:
            client: MemoryClient instance
            config: Config instance with scheduler settings
        """
        self.client = client
        self.config = config
        self.scheduler = SchedulerWrapper()
        self._running = False

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not self.scheduler.is_available():
            logger.error("APScheduler not available. Cannot start scheduler.")
            return

        if not self.config or not self.config.enable_scheduler:
            logger.info("Scheduler disabled in configuration")
            return

        # Schedule consolidation job
        if hasattr(self.config, "consolidate_cron"):
            success = self.scheduler.add_job(
                func=self._run_consolidation,
                cron_expression=self.config.consolidate_cron,
                job_id="consolidate_memories",
                name="Memory Consolidation",
                replace_existing=True,
            )
            if success:
                logger.info(f"Scheduled consolidation: {self.config.consolidate_cron}")

        # Schedule decay job
        if hasattr(self.config, "decay_cron"):
            success = self.scheduler.add_job(
                func=self._run_decay,
                cron_expression=self.config.decay_cron,
                job_id="decay_importance",
                name="Importance Decay",
                replace_existing=True,
            )
            if success:
                logger.info(f"Scheduled importance decay: {self.config.decay_cron}")

        # Schedule snapshot job
        if hasattr(self.config, "snapshot_cron"):
            success = self.scheduler.add_job(
                func=self._run_snapshot,
                cron_expression=self.config.snapshot_cron,
                job_id="create_snapshots",
                name="Snapshot Creation",
                replace_existing=True,
            )
            if success:
                logger.info(f"Scheduled snapshots: {self.config.snapshot_cron}")

        if self.scheduler.start():
            self._running = True
            logger.info("Memory scheduler started")
        else:
            logger.error("Failed to start scheduler")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown()
        self._running = False
        logger.info("Memory scheduler stopped")

    def _run_consolidation(self) -> None:
        """Run memory consolidation job."""
        if not self.client:
            logger.warning("No client configured for consolidation")
            return

        try:
            logger.info("Starting scheduled memory consolidation")
            start_time = datetime.now()

            # Get all users (this is a simplified approach)
            # In production, you'd want to batch this or have a user registry
            consolidated_count = 0

            # For now, we'll implement a method to consolidate all memories
            # This should be implemented in the client
            if hasattr(self.client, "consolidate_all_memories"):
                consolidated_count = self.client.consolidate_all_memories()
            else:
                logger.warning("consolidate_all_memories method not available on client")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Consolidation completed: {consolidated_count} memories consolidated "
                f"in {duration:.2f}s"
            )

        except Exception as e:
            logger.error(f"Consolidation job failed: {e}", exc_info=True)

    def _run_decay(self) -> None:
        """Run importance decay job."""
        if not self.client:
            logger.warning("No client configured for decay")
            return

        try:
            logger.info("Starting scheduled importance decay")
            start_time = datetime.now()

            # Apply decay to all memories
            # This should be implemented in the client
            if hasattr(self.client, "apply_importance_decay"):
                decayed_count = self.client.apply_importance_decay()
            else:
                logger.warning("apply_importance_decay method not available on client")
                decayed_count = 0

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Importance decay completed: {decayed_count} memories updated in {duration:.2f}s"
            )

        except Exception as e:
            logger.error(f"Decay job failed: {e}", exc_info=True)

    def _run_snapshot(self) -> None:
        """Run snapshot creation job."""
        if not self.client:
            logger.warning("No client configured for snapshots")
            return

        try:
            logger.info("Starting scheduled snapshot creation")

            # Create snapshots for both collections
            snapshots = []
            for collection in ["facts", "prefs"]:
                snapshot_name = self.client.create_snapshot(collection)
                snapshots.append(snapshot_name)
                logger.info(f"Created snapshot: {snapshot_name}")

            logger.info(f"Snapshot creation completed: {len(snapshots)} snapshots created")

        except Exception as e:
            logger.error(f"Snapshot job failed: {e}", exc_info=True)

    def get_job_status(self) -> Dict[str, Any]:
        """Get the status of all scheduled jobs."""
        jobs = self.scheduler.get_jobs()
        job_info = []

        for job in jobs:
            # The SchedulerWrapper returns dicts, not job objects
            next_run = job.get("next_run_time")
            job_info.append(
                {
                    "id": job.get("id"),
                    "name": job.get("name"),
                    "next_run": next_run.isoformat() if next_run else None,
                    "trigger": job.get("trigger", "cron"),
                }
            )

        status = "running" if self._running else "stopped"
        return {"status": status, "jobs": job_info}

    def trigger_consolidation_now(self) -> None:
        """Manually trigger consolidation job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering consolidation job")
        self._run_consolidation()

    def trigger_decay_now(self) -> None:
        """Manually trigger decay job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering decay job")
        self._run_decay()

    def trigger_snapshot_now(self) -> None:
        """Manually trigger snapshot job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering snapshot job")
        self._run_snapshot()
