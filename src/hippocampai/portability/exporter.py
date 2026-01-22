"""Memory exporter for backup and migration."""

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from hippocampai.portability.formats import (
    ExportFormat,
    ExportOptions,
    ExportStats,
    MemoryRecord,
)

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)


class MemoryExporter:
    """Export memories to portable formats.

    Supports JSON, JSONL, Parquet, and CSV formats with optional
    compression, filtering, and encryption.

    Example:
        >>> exporter = MemoryExporter(client)
        >>> stats = exporter.export_json("backup.json", user_id="alice")
        >>> print(f"Exported {stats.exported_memories} memories")
    """

    # Export file header/metadata
    EXPORT_VERSION = "1.0"
    EXPORT_SCHEMA = "hippocampai-memory-export"

    def __init__(self, client: "MemoryClient"):
        self.client = client

    def export_json(
        self,
        file_path: str,
        user_id: str,
        options: Optional[ExportOptions] = None,
    ) -> ExportStats:
        """Export memories to JSON file.

        Args:
            file_path: Output file path
            user_id: User ID to export
            options: Export options

        Returns:
            Export statistics
        """
        options = options or ExportOptions(format=ExportFormat.JSON)
        return self._export(file_path, user_id, options)

    def export_jsonl(
        self,
        file_path: str,
        user_id: str,
        options: Optional[ExportOptions] = None,
    ) -> ExportStats:
        """Export memories to JSON Lines file (streaming format)."""
        options = options or ExportOptions(format=ExportFormat.JSONL)
        options.format = ExportFormat.JSONL
        return self._export(file_path, user_id, options)

    def export_parquet(
        self,
        file_path: str,
        user_id: str,
        options: Optional[ExportOptions] = None,
    ) -> ExportStats:
        """Export memories to Parquet file."""
        options = options or ExportOptions(format=ExportFormat.PARQUET)
        options.format = ExportFormat.PARQUET
        return self._export(file_path, user_id, options)

    def export_csv(
        self,
        file_path: str,
        user_id: str,
        options: Optional[ExportOptions] = None,
    ) -> ExportStats:
        """Export memories to CSV file."""
        options = options or ExportOptions(format=ExportFormat.CSV)
        options.format = ExportFormat.CSV
        return self._export(file_path, user_id, options)

    def _export(
        self,
        file_path: str,
        user_id: str,
        options: ExportOptions,
    ) -> ExportStats:
        """Internal export implementation."""
        start_time = time.time()
        stats = ExportStats(format=options.format, file_path=file_path)

        try:
            # Fetch memories
            memories = list(self._fetch_memories(user_id, options))
            stats.total_memories = len(memories)

            # Convert to records
            records = []
            for memory in memories:
                try:
                    record = self._memory_to_record(memory, options)
                    if record:
                        records.append(record)
                        stats.exported_memories += 1
                    else:
                        stats.skipped_memories += 1
                except Exception as e:
                    stats.errors.append(f"Failed to convert memory {memory.id}: {e}")
                    stats.skipped_memories += 1

            # Write to file
            if options.format == ExportFormat.JSON:
                self._write_json(file_path, records, options, stats)
            elif options.format == ExportFormat.JSONL:
                self._write_jsonl(file_path, records, options, stats)
            elif options.format == ExportFormat.PARQUET:
                self._write_parquet(file_path, records, options, stats)
            elif options.format == ExportFormat.CSV:
                self._write_csv(file_path, records, options, stats)

            stats.duration_seconds = time.time() - start_time
            logger.info(
                f"Exported {stats.exported_memories} memories to {file_path} "
                f"in {stats.duration_seconds:.2f}s"
            )

        except Exception as e:
            stats.errors.append(f"Export failed: {e}")
            logger.error(f"Export failed: {e}")

        return stats

    def _fetch_memories(
        self,
        user_id: str,
        options: ExportOptions,
    ) -> Iterator[Any]:
        """Fetch memories with filtering."""
        # Get all memories for user
        memories = self.client.get_memories(user_id, limit=100000)

        for memory in memories:
            # Apply filters
            if options.filter_types and memory.type.value not in options.filter_types:
                continue
            if options.filter_tags:
                if not any(tag in memory.tags for tag in options.filter_tags):
                    continue
            if options.date_from and memory.created_at < options.date_from:
                continue
            if options.date_to and memory.created_at > options.date_to:
                continue
            if options.namespace:
                mem_ns = memory.metadata.get("namespace")
                if mem_ns != options.namespace:
                    continue

            yield memory

    def _memory_to_record(
        self,
        memory: Any,
        options: ExportOptions,
    ) -> Optional[MemoryRecord]:
        """Convert Memory to exportable record."""
        record = MemoryRecord(
            id=memory.id,
            text=memory.text if not options.anonymize else self._anonymize(memory.text),
            user_id=memory.user_id,
            type=memory.type.value if hasattr(memory.type, "value") else str(memory.type),
            importance=memory.importance,
            confidence=memory.confidence,
            tags=memory.tags,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
            session_id=memory.session_id,
            namespace=memory.metadata.get("namespace"),
            agent_id=memory.agent_id,
            metadata=memory.metadata if options.include_metadata else {},
        )

        # Include vectors if requested
        if options.include_vectors:
            try:
                vector = self.client.embedder.encode_single(memory.text)
                record.vector = vector.tolist()
            except Exception as e:
                logger.warning(f"Failed to get vector for {memory.id}: {e}")

        return record

    def _anonymize(self, text: str) -> str:
        """Basic PII anonymization."""
        import re

        # Email
        text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
        # Phone
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        # SSN
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        return text

    def _write_json(
        self,
        file_path: str,
        records: list[MemoryRecord],
        options: ExportOptions,
        stats: ExportStats,
    ) -> None:
        """Write records to JSON file."""
        export_data = {
            "schema": self.EXPORT_SCHEMA,
            "version": self.EXPORT_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "memories": [r.to_dict() for r in records],
        }

        json_str = json.dumps(export_data, indent=2, default=str)
        stats.total_bytes = len(json_str.encode())

        if options.compress:
            file_path = file_path if file_path.endswith(".gz") else f"{file_path}.gz"
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                f.write(json_str)
            stats.compressed_bytes = Path(file_path).stat().st_size
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            stats.compressed_bytes = stats.total_bytes

        stats.file_path = file_path

    def _write_jsonl(
        self,
        file_path: str,
        records: list[MemoryRecord],
        options: ExportOptions,
        stats: ExportStats,
    ) -> None:
        """Write records to JSON Lines file."""
        if options.compress:
            file_path = file_path if file_path.endswith(".gz") else f"{file_path}.gz"

        total_bytes = 0
        if options.compress:
            f = gzip.open(file_path, "wt", encoding="utf-8")
        else:
            f = open(file_path, "w", encoding="utf-8")

        try:
            # Write header
            header = {
                "schema": self.EXPORT_SCHEMA,
                "version": self.EXPORT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            line = json.dumps(header, default=str) + "\n"
            f.write(line)
            total_bytes += len(line.encode())

            # Write records
            for record in records:
                line = json.dumps(record.to_dict(), default=str) + "\n"
                f.write(line)
                total_bytes += len(line.encode())
        finally:
            f.close()

        stats.total_bytes = total_bytes
        stats.compressed_bytes = Path(file_path).stat().st_size
        stats.file_path = file_path

    def _write_parquet(
        self,
        file_path: str,
        records: list[MemoryRecord],
        options: ExportOptions,
        stats: ExportStats,
    ) -> None:
        """Write records to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow required for Parquet export: pip install pyarrow")

        # Convert to columnar format
        data = {
            "id": [r.id for r in records],
            "text": [r.text for r in records],
            "user_id": [r.user_id for r in records],
            "type": [r.type for r in records],
            "importance": [r.importance for r in records],
            "confidence": [r.confidence for r in records],
            "tags": [json.dumps(r.tags) for r in records],
            "created_at": [r.created_at for r in records],
            "updated_at": [r.updated_at for r in records],
            "session_id": [r.session_id for r in records],
            "namespace": [r.namespace for r in records],
            "metadata": [json.dumps(r.metadata) for r in records],
        }

        table = pa.Table.from_pydict(data)
        compression = "snappy" if options.compress else None
        pq.write_table(table, file_path, compression=compression)

        stats.total_bytes = sum(len(str(v).encode()) for v in data.values())
        stats.compressed_bytes = Path(file_path).stat().st_size
        stats.file_path = file_path

    def _write_csv(
        self,
        file_path: str,
        records: list[MemoryRecord],
        options: ExportOptions,
        stats: ExportStats,
    ) -> None:
        """Write records to CSV file."""
        import csv

        fieldnames = [
            "id",
            "text",
            "user_id",
            "type",
            "importance",
            "confidence",
            "tags",
            "created_at",
            "updated_at",
            "session_id",
            "namespace",
        ]

        if options.compress:
            file_path = file_path if file_path.endswith(".gz") else f"{file_path}.gz"
            f = gzip.open(file_path, "wt", encoding="utf-8", newline="")
        else:
            f = open(file_path, "w", encoding="utf-8", newline="")

        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {
                    "id": record.id,
                    "text": record.text,
                    "user_id": record.user_id,
                    "type": record.type,
                    "importance": record.importance,
                    "confidence": record.confidence,
                    "tags": json.dumps(record.tags),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "session_id": record.session_id or "",
                    "namespace": record.namespace or "",
                }
                writer.writerow(row)
        finally:
            f.close()

        stats.compressed_bytes = Path(file_path).stat().st_size
        stats.file_path = file_path
