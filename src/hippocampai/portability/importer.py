"""Memory importer for restore and migration."""

import gzip
import json
import logging
import time
from typing import TYPE_CHECKING, Iterator, Optional
from uuid import uuid4

from hippocampai.portability.formats import (
    ExportFormat,
    ImportOptions,
    ImportStats,
    MemoryRecord,
)

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)


class MemoryImporter:
    """Import memories from portable formats.

    Supports JSON, JSONL, Parquet, and CSV formats with merge
    strategies and validation.

    Example:
        >>> importer = MemoryImporter(client)
        >>> stats = importer.import_json("backup.json", user_id="alice")
        >>> print(f"Imported {stats.imported_memories} memories")
    """

    def __init__(self, client: "MemoryClient"):
        self.client = client

    def import_json(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        options: Optional[ImportOptions] = None,
    ) -> ImportStats:
        """Import memories from JSON file."""
        options = options or ImportOptions(format=ExportFormat.JSON)
        return self._import(file_path, user_id, options)

    def import_jsonl(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        options: Optional[ImportOptions] = None,
    ) -> ImportStats:
        """Import memories from JSON Lines file."""
        options = options or ImportOptions(format=ExportFormat.JSONL)
        options.format = ExportFormat.JSONL
        return self._import(file_path, user_id, options)

    def import_parquet(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        options: Optional[ImportOptions] = None,
    ) -> ImportStats:
        """Import memories from Parquet file."""
        options = options or ImportOptions(format=ExportFormat.PARQUET)
        options.format = ExportFormat.PARQUET
        return self._import(file_path, user_id, options)

    def import_csv(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        options: Optional[ImportOptions] = None,
    ) -> ImportStats:
        """Import memories from CSV file."""
        options = options or ImportOptions(format=ExportFormat.CSV)
        options.format = ExportFormat.CSV
        return self._import(file_path, user_id, options)

    def _import(
        self,
        file_path: str,
        user_id: Optional[str],
        options: ImportOptions,
    ) -> ImportStats:
        """Internal import implementation."""
        start_time = time.time()
        stats = ImportStats(
            format=options.format,
            file_path=file_path,
            dry_run=options.dry_run,
        )

        try:
            # Read records
            records = list(self._read_records(file_path, options))
            stats.total_records = len(records)

            # Process in batches
            batch = []
            for record in records:
                # Remap user if specified
                if options.remap_user_id:
                    record.user_id = options.remap_user_id
                elif user_id:
                    record.user_id = user_id

                # Remap namespace if specified
                if options.remap_namespace:
                    record.namespace = options.remap_namespace

                # Generate new ID if not preserving
                if not options.preserve_ids:
                    record.id = str(uuid4())

                batch.append(record)

                if len(batch) >= options.batch_size:
                    self._import_batch(batch, options, stats)
                    batch = []

            # Import remaining
            if batch:
                self._import_batch(batch, options, stats)

            stats.duration_seconds = time.time() - start_time
            logger.info(
                f"Imported {stats.imported_memories} memories from {file_path} "
                f"in {stats.duration_seconds:.2f}s"
            )

        except Exception as e:
            stats.errors.append(f"Import failed: {e}")
            logger.error(f"Import failed: {e}")

        return stats

    def _read_records(
        self,
        file_path: str,
        options: ImportOptions,
    ) -> Iterator[MemoryRecord]:
        """Read records from file."""
        if options.format == ExportFormat.JSON:
            yield from self._read_json(file_path)
        elif options.format == ExportFormat.JSONL:
            yield from self._read_jsonl(file_path)
        elif options.format == ExportFormat.PARQUET:
            yield from self._read_parquet(file_path)
        elif options.format == ExportFormat.CSV:
            yield from self._read_csv(file_path)

    def _read_json(self, file_path: str) -> Iterator[MemoryRecord]:
        """Read records from JSON file."""
        is_gzip = file_path.endswith(".gz")
        opener = gzip.open if is_gzip else open

        with opener(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        # Validate schema
        if data.get("schema") != "hippocampai-memory-export":
            logger.warning(f"Unknown schema: {data.get('schema')}")

        for memory_data in data.get("memories", []):
            yield MemoryRecord.from_dict(memory_data)

    def _read_jsonl(self, file_path: str) -> Iterator[MemoryRecord]:
        """Read records from JSON Lines file."""
        is_gzip = file_path.endswith(".gz")
        opener = gzip.open if is_gzip else open

        with opener(file_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Skip header line
                if i == 0 and "schema" in data:
                    continue
                yield MemoryRecord.from_dict(data)

    def _read_parquet(self, file_path: str) -> Iterator[MemoryRecord]:
        """Read records from Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow required for Parquet import: pip install pyarrow")

        table = pq.read_table(file_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield MemoryRecord(
                id=row["id"],
                text=row["text"],
                user_id=row["user_id"],
                type=row["type"],
                importance=float(row["importance"]),
                confidence=float(row["confidence"]),
                tags=json.loads(row["tags"]) if row["tags"] else [],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                session_id=row.get("session_id"),
                namespace=row.get("namespace"),
                metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            )

    def _read_csv(self, file_path: str) -> Iterator[MemoryRecord]:
        """Read records from CSV file."""
        import csv

        is_gzip = file_path.endswith(".gz")
        opener = gzip.open if is_gzip else open

        with opener(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield MemoryRecord(
                    id=row["id"],
                    text=row["text"],
                    user_id=row["user_id"],
                    type=row["type"],
                    importance=float(row["importance"]),
                    confidence=float(row["confidence"]),
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    session_id=row.get("session_id") or None,
                    namespace=row.get("namespace") or None,
                )

    def _import_batch(
        self,
        records: list[MemoryRecord],
        options: ImportOptions,
        stats: ImportStats,
    ) -> None:
        """Import a batch of records."""
        for record in records:
            try:
                result = self._import_record(record, options, stats)
                if result == "imported":
                    stats.imported_memories += 1
                elif result == "skipped":
                    stats.skipped_memories += 1
                elif result == "merged":
                    stats.merged_memories += 1
                elif result == "replaced":
                    stats.replaced_memories += 1
            except Exception as e:
                stats.failed_memories += 1
                stats.errors.append(f"Failed to import {record.id}: {e}")

    def _import_record(
        self,
        record: MemoryRecord,
        options: ImportOptions,
        stats: ImportStats,
    ) -> str:
        """Import a single record.

        Returns:
            Action taken: imported, skipped, merged, replaced
        """
        if options.dry_run:
            return "imported"

        # Check for existing memory with same ID
        existing = None
        if options.preserve_ids:
            try:
                existing = self.client.get_memory(record.id)
            except Exception:
                pass

        if existing:
            if options.merge_strategy == "skip":
                return "skipped"
            elif options.merge_strategy == "replace":
                self.client.delete_memory(record.id, record.user_id)
            elif options.merge_strategy == "merge":
                # Merge metadata and update
                merged_metadata = {**existing.metadata, **record.metadata}
                self.client.update_memory(
                    memory_id=record.id,
                    text=record.text,
                    importance=max(existing.importance, record.importance),
                    tags=list(set(existing.tags + record.tags)),
                    metadata=merged_metadata,
                )
                return "merged"

        # Check for duplicates by content
        if options.skip_duplicates:
            results = self.client.recall(record.text, user_id=record.user_id, k=1)
            if results and results[0].score > 0.95:
                return "skipped"

        # Import the memory
        self.client.remember(
            text=record.text,
            user_id=record.user_id,
            session_id=record.session_id,
            type=record.type,
            importance=record.importance,
            tags=record.tags,
        )

        return "imported" if not existing else "replaced"
