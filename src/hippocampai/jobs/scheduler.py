"""Background jobs scheduler."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from hippocampai.client import MemoryClient
from hippocampai.config import get_config

logger = logging.getLogger(__name__)


class JobScheduler:
    """Background job scheduler for maintenance tasks."""

    def __init__(self, client: MemoryClient):
        self.client = client
        self.config = get_config()
        self.scheduler = BackgroundScheduler()

    def start(self):
        """Start scheduler."""
        if not self.config.enable_scheduler:
            logger.info("Scheduler disabled")
            return

        # Decay job
        self.scheduler.add_job(
            self.decay_importance, trigger="cron", hour=2, minute=0, id="decay_importance"
        )

        # Consolidate job
        self.scheduler.add_job(
            self.consolidate_memories,
            trigger="cron",
            day_of_week="sun",
            hour=3,
            minute=0,
            id="consolidate_memories",
        )

        # Snapshot job
        self.scheduler.add_job(
            self.create_snapshots, trigger="cron", minute=0, id="create_snapshots"
        )

        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop scheduler."""
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")

    def decay_importance(self):
        """Apply importance decay (daily)."""
        logger.info("Running importance decay job")
        # Placeholder: implement decay logic

    def consolidate_memories(self):
        """Consolidate similar memories (weekly)."""
        logger.info("Running consolidation job")
        # Placeholder: implement consolidation logic

    def create_snapshots(self):
        """Create Qdrant snapshots (hourly)."""
        logger.info("Creating snapshots")
        try:
            facts_snap = self.client.qdrant.create_snapshot(self.client.config.collection_facts)
            prefs_snap = self.client.qdrant.create_snapshot(self.client.config.collection_prefs)
            logger.info(f"Snapshots created: {facts_snap}, {prefs_snap}")
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
