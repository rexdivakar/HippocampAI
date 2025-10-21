"""Background scheduler for memory maintenance tasks.

This module provides a scheduler for periodic memory consolidation
and other maintenance tasks.
"""

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class MemoryScheduler:
    """Background scheduler for memory maintenance tasks."""

    def __init__(self, client=None, config=None):
        """Initialize scheduler.

        Args:
            client: MemoryClient instance
            config: Config instance with scheduler settings
        """
        self.client = client
        self.config = config
        self.scheduler = BackgroundScheduler()
        self._running = False

    def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not self.config or not self.config.enable_scheduler:
            logger.info("Scheduler disabled in configuration")
            return

        # Schedule consolidation job
        if hasattr(self.config, "consolidate_cron"):
            self.scheduler.add_job(
                self._run_consolidation,
                trigger=CronTrigger.from_crontab(self.config.consolidate_cron),
                id="consolidate_memories",
                name="Memory Consolidation",
                replace_existing=True,
            )
            logger.info(f"Scheduled consolidation: {self.config.consolidate_cron}")

        # Schedule decay job
        if hasattr(self.config, "decay_cron"):
            self.scheduler.add_job(
                self._run_decay,
                trigger=CronTrigger.from_crontab(self.config.decay_cron),
                id="decay_importance",
                name="Importance Decay",
                replace_existing=True,
            )
            logger.info(f"Scheduled importance decay: {self.config.decay_cron}")

        # Schedule snapshot job
        if hasattr(self.config, "snapshot_cron"):
            self.scheduler.add_job(
                self._run_snapshot,
                trigger=CronTrigger.from_crontab(self.config.snapshot_cron),
                id="create_snapshots",
                name="Snapshot Creation",
                replace_existing=True,
            )
            logger.info(f"Scheduled snapshots: {self.config.snapshot_cron}")

        self.scheduler.start()
        self._running = True
        logger.info("Memory scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown()
        self._running = False
        logger.info("Memory scheduler stopped")

    def _run_consolidation(self):
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

    def _run_decay(self):
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

    def _run_snapshot(self):
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

    def get_job_status(self) -> dict:
        """Get status of scheduled jobs.

        Returns:
            Dictionary with job information
        """
        if not self._running:
            return {"status": "stopped", "jobs": []}

        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger),
                }
            )

        return {"status": "running", "jobs": jobs}

    def trigger_consolidation_now(self):
        """Manually trigger consolidation job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering consolidation job")
        self._run_consolidation()

    def trigger_decay_now(self):
        """Manually trigger decay job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering decay job")
        self._run_decay()

    def trigger_snapshot_now(self):
        """Manually trigger snapshot job immediately."""
        if not self._running:
            logger.warning("Scheduler not running")
            return

        logger.info("Manually triggering snapshot job")
        self._run_snapshot()
