import gzip
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.portability.exporter import MemoryExporter
from hippocampai.portability.formats import ExportFormat, ExportOptions


class FakeEmbedder:
    def encode_single(self, text: str):
        class V:
            def __init__(self):
                self._v = [0.1, 0.2, 0.3]

            def tolist(self):
                return list(self._v)

        return V()


class FakeClient:
    def __init__(self, memories):
        self._memories = memories
        self.embedder = FakeEmbedder()

    def get_memories(self, user_id: str, limit: int = 100000):
        # Return only memories for the requested user
        return [m for m in self._memories if m.user_id == user_id]


@pytest.fixture()
def sample_memories():
    now = datetime.now(timezone.utc)
    return [
        Memory(
            id="m1",
            text="Email me at alice@example.com",
            user_id="alice",
            type=MemoryType.FACT,
            importance=5.0,
            confidence=0.9,
            tags=["t1", "work"],
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=1),
            metadata={"namespace": "work"},
        ),
        Memory(
            id="m2",
            text="I love pizza",
            user_id="alice",
            type=MemoryType.PREFERENCE,
            importance=6.0,
            confidence=0.95,
            tags=["t2"],
            created_at=now - timedelta(days=1, hours=1),
            updated_at=now - timedelta(days=1),
            metadata={"namespace": "personal"},
        ),
        Memory(
            id="m3",
            text="Project X kickoff happened",
            user_id="bob",
            type=MemoryType.EVENT,
            importance=4.0,
            confidence=0.8,
            tags=["work"],
            created_at=now - timedelta(days=1),
            updated_at=now - timedelta(hours=12),
            metadata={"namespace": "work"},
        ),
    ]


# Behaviors to validate
# 1. _fetch_memories applies type, tag, date, and namespace filters correctly
# 2. _memory_to_record respects anonymize and include_metadata flags
# 3. _memory_to_record includes vectors when include_vectors is True, using client's embedder
# 4. export_json writes gzip when compress=True and plain JSON when compress=False and updates stats
# 5. export_jsonl writes a header first line and records subsequently; gzip respected


def test_fetch_memories_filters(sample_memories):
    client = FakeClient(sample_memories)
    exporter = MemoryExporter(client)

    options = ExportOptions(
        format=ExportFormat.JSON,
        filter_types=[MemoryType.FACT.value],
        filter_tags=["work"],
        date_from=sample_memories[0].created_at - timedelta(days=1),
        date_to=sample_memories[0].created_at + timedelta(days=3),
        namespace="work",
    )

    fetched = list(exporter._fetch_memories(user_id="alice", options=options))
    # Only m1 should match: alice, FACT, has tag work, within dates, namespace work
    assert [m.id for m in fetched] == ["m1"]


def test_memory_to_record_anonymize_and_metadata(sample_memories):
    client = FakeClient(sample_memories)
    exporter = MemoryExporter(client)

    mem = sample_memories[0]  # contains an email

    # Without anonymize and without metadata
    opts_no_anon = ExportOptions(format=ExportFormat.JSON, anonymize=False, include_metadata=False)
    rec1 = exporter._memory_to_record(mem, opts_no_anon)
    assert rec1 is not None
    assert rec1.text == mem.text
    assert rec1.metadata == {}

    # With anonymize and with metadata
    opts_anon = ExportOptions(format=ExportFormat.JSON, anonymize=True, include_metadata=True)
    rec2 = exporter._memory_to_record(mem, opts_anon)
    assert rec2 is not None
    assert "[EMAIL]" in rec2.text and "@" not in rec2.text
    assert rec2.metadata.get("namespace") == "work"


def test_memory_to_record_includes_vectors(sample_memories):
    client = FakeClient(sample_memories)
    exporter = MemoryExporter(client)

    mem = sample_memories[1]

    opts = ExportOptions(format=ExportFormat.JSON, include_vectors=True)
    rec = exporter._memory_to_record(mem, opts)
    assert rec is not None
    assert isinstance(rec.vector, list)
    assert rec.vector == [0.1, 0.2, 0.3]


def test_export_json_writes_gzip_and_plain(sample_memories):
    client = FakeClient(sample_memories)
    exporter = MemoryExporter(client)

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "export.json")

        # Plain JSON (compress False)
        opts_plain = ExportOptions(format=ExportFormat.JSON, compress=False)
        stats_plain = exporter.export_json(json_path, user_id="alice", options=opts_plain)
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema"] == exporter.EXPORT_SCHEMA
        assert data["version"] == exporter.EXPORT_VERSION
        assert data["total_records"] == len(data["memories"])
        assert stats_plain.file_path == json_path
        assert stats_plain.total_bytes > 0
        assert stats_plain.compressed_bytes == stats_plain.total_bytes

        # Gzip JSON (compress True)
        gz_path = os.path.join(tmp, "export_gz.json")
        opts_gz = ExportOptions(format=ExportFormat.JSON, compress=True)
        stats_gz = exporter.export_json(gz_path, user_id="alice", options=opts_gz)
        expected_gz = gz_path + ".gz" if not gz_path.endswith(".gz") else gz_path
        assert os.path.exists(expected_gz)
        with gzip.open(expected_gz, "rt", encoding="utf-8") as f:
            data_gz = json.load(f)
        assert data_gz["schema"] == exporter.EXPORT_SCHEMA
        assert stats_gz.file_path == expected_gz
        assert stats_gz.compressed_bytes == os.path.getsize(expected_gz)


def test_export_jsonl_header_and_records(sample_memories):
    client = FakeClient(sample_memories)
    exporter = MemoryExporter(client)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "export.jsonl")

        # Plain JSONL
        opts = ExportOptions(format=ExportFormat.JSONL, compress=False)
        stats = exporter.export_jsonl(path, user_id="alice", options=opts)
        assert os.path.exists(path)
        with open(path, "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert len(lines) >= 2  # header + at least one record
        header = json.loads(lines[0])
        assert header["schema"] == exporter.EXPORT_SCHEMA
        record = json.loads(lines[1])
        assert "id" in record and "text" in record
        assert stats.file_path == path

        # Gzipped JSONL
        path_gz = os.path.join(tmp, "export2.jsonl")
        opts_gz = ExportOptions(format=ExportFormat.JSONL, compress=True)
        stats_gz = exporter.export_jsonl(path_gz, user_id="alice", options=opts_gz)
        expected_gz = path_gz + ".gz"
        assert os.path.exists(expected_gz)
        with gzip.open(expected_gz, "rt", encoding="utf-8") as f:
            gz_lines = f.read().splitlines()
        assert len(gz_lines) >= 2
        header_gz = json.loads(gz_lines[0])
        assert header_gz["version"] == exporter.EXPORT_VERSION
        assert stats_gz.file_path == expected_gz
