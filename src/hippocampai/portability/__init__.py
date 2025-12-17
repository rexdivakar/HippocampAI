"""Memory Export/Import for backup and migration.

Provides portable memory formats (JSON, Parquet) for:
- Backup and restore
- Migration between instances
- Data portability compliance (GDPR)
- Sharing memory collections

Example:
    >>> from hippocampai.portability import MemoryExporter, MemoryImporter
    >>>
    >>> # Export user's memories
    >>> exporter = MemoryExporter(client)
    >>> exporter.export_json("backup.json", user_id="alice")
    >>> exporter.export_parquet("backup.parquet", user_id="alice")
    >>>
    >>> # Import memories
    >>> importer = MemoryImporter(client)
    >>> stats = importer.import_json("backup.json", user_id="alice")
"""

from hippocampai.portability.exporter import MemoryExporter
from hippocampai.portability.formats import (
    ExportFormat,
    ExportOptions,
    ExportStats,
    ImportOptions,
    ImportStats,
)
from hippocampai.portability.importer import MemoryImporter

__all__ = [
    "MemoryExporter",
    "MemoryImporter",
    "ExportFormat",
    "ExportOptions",
    "ImportOptions",
    "ExportStats",
    "ImportStats",
]
