"""
Tests for RemoteBackend endpoint URL correctness.

Verifies that the RemoteBackend constructs correct API URLs
matching the actual HippocampAI API route patterns (colon-style custom methods).

These tests use mocking so they can run without a live API server.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hippocampai.backends.remote import RemoteBackend


@pytest.fixture
def backend() -> RemoteBackend:
    return RemoteBackend(api_url="http://testserver:8000")


def _mock_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _memory_dict(memory_id: str = "mem-123", user_id: str = "user-1") -> dict[str, Any]:
    """Create a sample memory dict matching API response format."""
    return {
        "id": memory_id,
        "text": "Test memory",
        "user_id": user_id,
        "type": "fact",
        "metadata": {},
        "tags": [],
        "importance": 0.5,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


class TestRemoteBackendEndpoints:
    """Verify RemoteBackend uses correct colon-style API endpoints."""

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_remember_uses_colon_endpoint(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """POST /v1/memories:remember - not /v1/memories/remember."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(json_data=_memory_dict())

        backend.remember(text="hello", user_id="user-1")

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "/v1/memories:remember" in url, f"Expected colon-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_recall_uses_colon_endpoint(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """POST /v1/memories:recall - not /v1/memories/recall."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(json_data=[])

        backend.recall(query="test", user_id="user-1")

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "/v1/memories:recall" in url, f"Expected colon-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_recall_sends_k_not_limit(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """recall() should send 'k' parameter, not 'limit', matching API schema."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(json_data=[])

        backend.recall(query="test", user_id="user-1", limit=10)

        call_args = mock_client.post.call_args
        sent_json = call_args[1]["json"]
        assert "k" in sent_json, f"Expected 'k' in payload, got keys: {list(sent_json.keys())}"
        assert sent_json["k"] == 10
        assert "limit" not in sent_json, "Should not send 'limit' to recall API"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_get_memories_uses_colon_endpoint(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """POST /v1/memories:get - not /v1/memories/query."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(json_data=[])

        backend.get_memories(user_id="user-1")

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "/v1/memories:get" in url, f"Expected colon-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_get_memory_uses_path_param(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """GET /v1/memories/{memory_id}."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(json_data=_memory_dict())

        backend.get_memory("mem-123")

        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "/v1/memories/mem-123" in url, f"Expected path-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_update_memory_uses_colon_endpoint(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """PATCH /v1/memories:update - not /v1/memories/{id}."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.patch.return_value = _mock_response(json_data=_memory_dict())

        backend.update_memory(memory_id="mem-123", text="updated")

        call_args = mock_client.patch.call_args
        url = call_args[0][0]
        assert "/v1/memories:update" in url, f"Expected colon-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_delete_memory_uses_colon_endpoint(self, mock_client_cls: MagicMock, backend: RemoteBackend) -> None:
        """DELETE /v1/memories:delete with JSON body - not DELETE /v1/memories/{id}."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.request.return_value = _mock_response(json_data={"success": True})

        backend.delete_memory("mem-123", user_id="user-1")

        call_args = mock_client.request.call_args
        method = call_args[0][0]
        url = call_args[0][1]
        assert method == "DELETE"
        assert "/v1/memories:delete" in url, f"Expected colon-style URL, got {url}"

    @patch("hippocampai.backends.remote.httpx.Client")
    def test_extract_from_conversation_uses_colon_endpoint(
        self, mock_client_cls: MagicMock, backend: RemoteBackend
    ) -> None:
        """POST /v1/memories:extract - not /v1/memories/extract."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(json_data=[])

        backend.extract_from_conversation(
            conversation=[{"role": "user", "content": "hello"}],
            user_id="user-1",
        )

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "/v1/memories:extract" in url, f"Expected colon-style URL, got {url}"


class TestConsolidationDBInlineFallback:
    """Verify consolidation DB creates tables when schema file is missing."""

    def test_inline_table_creation(self, tmp_path: Any) -> None:
        """When schema_sqlite.sql is absent, tables should be created inline."""
        from hippocampai.consolidation.db import ConsolidationDatabase

        db_path = str(tmp_path / "test_consolidation.db")
        # This should succeed even without the schema file present
        db = ConsolidationDatabase(db_type="sqlite", db_path=db_path)

        # Verify the table was created by querying it
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        assert "consolidation_runs" in tables
        assert "consolidation_run_details" in tables
        assert "consolidation_clusters" in tables
