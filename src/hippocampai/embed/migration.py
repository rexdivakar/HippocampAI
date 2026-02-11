"""Embedding model migration: detect changes and re-embed all memories."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Status of an embedding migration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmbeddingMigration(BaseModel):
    """Represents an embedding model migration job."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    old_model: str = ""
    new_model: str
    new_dimension: int
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_memories: int = 0
    migrated_count: int = 0
    failed_count: int = 0
    error: Optional[str] = None


class EmbeddingMigrationManager:
    """Manages embedding model migrations.

    - Detects model changes by comparing payload ``embed_model`` vs config.
    - Starts a migration that re-encodes all memories with the new model.
    - Tracks progress and supports cancellation.
    """

    def __init__(
        self,
        qdrant_store: Any,
        embedder: Any,
        config: Any,
    ) -> None:
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.config = config
        self._migrations: dict[str, EmbeddingMigration] = {}

    def detect_model_change(self) -> bool:
        """Check if the configured embed model differs from stored memories.

        Scrolls one point from the facts collection and compares the
        ``embed_model`` payload field with the current config value.

        Returns:
            True if a model change is detected.
        """
        try:
            results = self.qdrant.scroll(
                collection_name=self.qdrant.collection_facts,
                limit=1,
            )
            if not results:
                return False

            stored_model = results[0].get("payload", {}).get("embed_model", "")
            current_model = self.config.embed_model
            if stored_model and stored_model != current_model:
                logger.info(
                    f"Embedding model change detected: '{stored_model}' -> '{current_model}'"
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"Model change detection failed: {e}")
            return False

    def start_migration(
        self,
        new_model: str,
        new_dimension: int,
    ) -> EmbeddingMigration:
        """Create a new migration job.

        Args:
            new_model: New embedding model name.
            new_dimension: Dimension of the new embeddings.

        Returns:
            The created EmbeddingMigration.
        """
        migration = EmbeddingMigration(
            old_model=self.config.embed_model,
            new_model=new_model,
            new_dimension=new_dimension,
        )
        self._migrations[migration.id] = migration
        logger.info(
            f"Migration {migration.id} created: "
            f"'{migration.old_model}' -> '{migration.new_model}'"
        )
        return migration

    def get_migration(self, migration_id: str) -> Optional[EmbeddingMigration]:
        """Get a migration by ID."""
        return self._migrations.get(migration_id)

    def cancel_migration(self, migration_id: str) -> bool:
        """Cancel a running migration.

        Returns:
            True if cancelled, False if not found or already completed.
        """
        migration = self._migrations.get(migration_id)
        if migration is None:
            return False
        if migration.status in (MigrationStatus.COMPLETED, MigrationStatus.CANCELLED):
            return False
        migration.status = MigrationStatus.CANCELLED
        logger.info(f"Migration {migration_id} cancelled")
        return True

    def run_migration(self, migration_id: str) -> EmbeddingMigration:
        """Run the migration: re-embed all memories with the new model.

        This is designed to be called from a Celery task.

        Args:
            migration_id: ID of the migration to run.

        Returns:
            Updated EmbeddingMigration with final status.
        """
        migration = self._migrations.get(migration_id)
        if migration is None:
            raise ValueError(f"Migration {migration_id} not found")

        migration.status = MigrationStatus.IN_PROGRESS
        migration.started_at = datetime.now(timezone.utc)

        try:
            from hippocampai.embed.embedder import Embedder

            new_embedder = Embedder(
                model_name=migration.new_model,
                dimension=migration.new_dimension,
            )

            for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
                self._migrate_collection(migration, new_embedder, collection)

                if migration.status == MigrationStatus.CANCELLED:
                    logger.info(f"Migration {migration_id} was cancelled during processing")
                    return migration

            migration.status = MigrationStatus.COMPLETED
            migration.completed_at = datetime.now(timezone.utc)
            logger.info(
                f"Migration {migration_id} completed: "
                f"{migration.migrated_count} migrated, {migration.failed_count} failed"
            )
        except Exception as e:
            migration.status = MigrationStatus.FAILED
            migration.error = str(e)
            migration.completed_at = datetime.now(timezone.utc)
            logger.error(f"Migration {migration_id} failed: {e}")

        return migration

    def _migrate_collection(
        self,
        migration: EmbeddingMigration,
        new_embedder: Any,
        collection: str,
    ) -> None:
        """Migrate a single collection's embeddings."""
        batch_size = 32
        offset = 0

        while True:
            if migration.status == MigrationStatus.CANCELLED:
                break

            results = self.qdrant.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset if offset > 0 else None,
            )

            if not results:
                break

            migration.total_memories += len(results)

            texts = []
            ids = []
            payloads = []

            for result in results:
                payload = result.get("payload", {})
                text = payload.get("text", "")
                if text:
                    texts.append(text)
                    ids.append(result["id"])
                    payload["embed_model"] = migration.new_model
                    payloads.append(payload)

            if texts:
                try:
                    new_vectors = new_embedder.encode(texts)
                    for i, (doc_id, payload) in enumerate(zip(ids, payloads)):
                        self.qdrant.upsert(
                            collection_name=collection,
                            id=doc_id,
                            vector=new_vectors[i],
                            payload=payload,
                        )
                    migration.migrated_count += len(texts)
                except Exception as e:
                    migration.failed_count += len(texts)
                    logger.error(
                        f"Batch migration failed for collection {collection}: {e}"
                    )

            if len(results) < batch_size:
                break
            offset += batch_size
