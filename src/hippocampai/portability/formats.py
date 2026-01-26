"""Export/Import format definitions and options."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    JSONL = "jsonl"  # JSON Lines (streaming)
    PARQUET = "parquet"
    CSV = "csv"


@dataclass
class ExportOptions:
    """Options for memory export."""

    format: ExportFormat = ExportFormat.JSON
    include_vectors: bool = False  # Include embedding vectors
    include_metadata: bool = True
    include_versions: bool = False  # Include version history
    include_relationships: bool = False  # Include graph relationships
    compress: bool = True  # Compress output (gzip for JSON, snappy for Parquet)
    chunk_size: int = 1000  # Records per chunk for streaming
    filter_types: Optional[list[str]] = None  # Only export these types
    filter_tags: Optional[list[str]] = None  # Only export with these tags
    date_from: Optional[datetime] = None  # Export from this date
    date_to: Optional[datetime] = None  # Export until this date
    namespace: Optional[str] = None  # Export from specific namespace
    anonymize: bool = False  # Remove PII before export
    encryption_key: Optional[str] = None  # Encrypt export file


@dataclass
class ImportOptions:
    """Options for memory import."""

    format: ExportFormat = ExportFormat.JSON
    merge_strategy: str = "skip"  # skip, replace, merge
    preserve_ids: bool = False  # Keep original memory IDs
    preserve_timestamps: bool = True  # Keep original timestamps
    remap_user_id: Optional[str] = None  # Map all to this user
    remap_namespace: Optional[str] = None  # Map all to this namespace
    validate_schema: bool = True  # Validate before import
    dry_run: bool = False  # Preview without importing
    batch_size: int = 100  # Import batch size
    skip_duplicates: bool = True  # Skip exact duplicates
    recompute_embeddings: bool = False  # Recompute vectors on import
    decryption_key: Optional[str] = None  # Decrypt import file


@dataclass
class ExportStats:
    """Statistics from export operation."""

    total_memories: int = 0
    exported_memories: int = 0
    skipped_memories: int = 0
    total_bytes: int = 0
    compressed_bytes: int = 0
    duration_seconds: float = 0.0
    format: ExportFormat = ExportFormat.JSON
    file_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportStats:
    """Statistics from import operation."""

    total_records: int = 0
    imported_memories: int = 0
    skipped_memories: int = 0
    merged_memories: int = 0
    replaced_memories: int = 0
    failed_memories: int = 0
    duration_seconds: float = 0.0
    format: ExportFormat = ExportFormat.JSON
    file_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class MemoryRecord:
    """Portable memory record for export/import."""

    id: str
    text: str
    user_id: str
    type: str
    importance: float
    confidence: float
    tags: list[str]
    created_at: str  # ISO format
    updated_at: str
    session_id: Optional[str] = None
    namespace: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Optional[list[float]] = None
    versions: Optional[list[dict]] = None
    relationships: Optional[list[dict]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "text": self.text,
            "user_id": self.user_id,
            "type": self.type,
            "importance": self.importance,
            "confidence": self.confidence,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        if self.session_id:
            data["session_id"] = self.session_id
        if self.namespace:
            data["namespace"] = self.namespace
        if self.agent_id:
            data["agent_id"] = self.agent_id
        if self.vector:
            data["vector"] = self.vector
        if self.versions:
            data["versions"] = self.versions
        if self.relationships:
            data["relationships"] = self.relationships
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            user_id=data["user_id"],
            type=data.get("type", "fact"),
            importance=data.get("importance", 5.0),
            confidence=data.get("confidence", 0.9),
            tags=data.get("tags", []),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
            session_id=data.get("session_id"),
            namespace=data.get("namespace"),
            agent_id=data.get("agent_id"),
            metadata=data.get("metadata", {}),
            vector=data.get("vector"),
            versions=data.get("versions"),
            relationships=data.get("relationships"),
        )
