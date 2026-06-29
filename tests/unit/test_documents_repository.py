"""
tests/unit/test_documents_repository.py

Unit-тесты для DocumentsRepository.
Supabase полностью замокан — тесты не ходят в реальную БД.

Проверяем:
  • Корректность SQL-запросов (какие методы builder вызываются)
  • Правильность Pydantic-валидации на выходе (DocumentRow)
  • Обработку ошибок (RepositoryError при пустом ответе)
  • update_status со всеми комбинациями аргументов
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import DocumentRow, DocumentUpdate
from src.db.repositories.base import RepositoryError


# ── Хелперы ───────────────────────────────────────────────────

def make_doc_row(**kwargs) -> dict[str, Any]:
    """Возвращает валидный словарь для DocumentRow с возможностью override."""
    base = {
        "id": "doc-uuid-test",
        "filename": "test.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "status": "pending",
        "chunk_count": 0,
        "chroma_ids": None,
        "error_msg": None,
        "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
    }
    base.update(kwargs)
    return base


# ── Фикстура репозитория с замоканным клиентом ────────────────

@pytest.fixture
def repo_with_mock(mock_supabase_client):
    """Возвращает (repo, builder, mock_result) с подменённым async-клиентом."""
    from src.db.repositories.documents import DocumentsRepository

    client, builder, mock_result = mock_supabase_client
    repo = DocumentsRepository()

    with patch("src.db.repositories.documents.get_async_supabase", AsyncMock(return_value=client)):
        yield repo, builder, mock_result, client


# ══════════════════════════════════════════════════════════════
# create()
# ══════════════════════════════════════════════════════════════

class TestDocumentsRepositoryCreate:

    @pytest.mark.asyncio
    async def test_create_returns_document_row(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        row_data = make_doc_row()
        mock_result.data = [row_data]

        with patch("src.db.repositories.documents.get_async_supabase",
                   AsyncMock(return_value=MagicMock(table=MagicMock(return_value=builder)))):
            result = await repo.create(
                doc_id="doc-uuid-test",
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
            )

        assert isinstance(result, DocumentRow)
        assert result.id == "doc-uuid-test"
        assert result.filename == "test.pdf"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_create_inserts_with_pending_status(self, repo_with_mock):
        """Новый документ всегда создаётся со статусом pending."""
        repo, builder, mock_result, client = repo_with_mock
        row_data = make_doc_row(status="pending")
        mock_result.data = [row_data]

        await repo.create(
            doc_id="doc-uuid-test",
            filename="test.pdf",
            file_type="pdf",
        )

        # Проверяем, что insert() был вызван
        builder.insert.assert_called_once()
        insert_payload = builder.insert.call_args[0][0]
        assert insert_payload["status"] == "pending"
        assert insert_payload["id"] == "doc-uuid-test"

    @pytest.mark.asyncio
    async def test_create_raises_on_empty_response(self, repo_with_mock):
        """RepositoryError если Supabase вернул пустой data."""
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        with pytest.raises(RepositoryError, match="Failed to create document"):
            await repo.create(
                doc_id="doc-uuid-test",
                filename="test.pdf",
                file_type="pdf",
            )


# ══════════════════════════════════════════════════════════════
# get_by_id()
# ══════════════════════════════════════════════════════════════

class TestDocumentsRepositoryGetById:

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row(status="done", chunk_count=42)]

        result = await repo.get_by_id("doc-uuid-test")

        assert result is not None
        assert result.status == "done"
        assert result.chunk_count == 42
        # eq() вызван с правильным значением
        builder.eq.assert_called_with("id", "doc-uuid-test")

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        result = await repo.get_by_id("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_returns_document_row_type(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row()]

        result = await repo.get_by_id("doc-uuid-test")

        assert isinstance(result, DocumentRow)


# ══════════════════════════════════════════════════════════════
# update_status()
# ══════════════════════════════════════════════════════════════

class TestDocumentsRepositoryUpdateStatus:

    @pytest.mark.asyncio
    async def test_update_status_to_processing(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row(status="processing")]

        result = await repo.update_status("doc-uuid-test", status="processing")

        assert result.status == "processing"
        update_payload = builder.update.call_args[0][0]
        assert update_payload["status"] == "processing"

    @pytest.mark.asyncio
    async def test_update_status_to_done_with_chunks(self, repo_with_mock):
        """При done передаём chunk_count и chroma_ids."""
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row(status="done", chunk_count=55)]

        result = await repo.update_status(
            "doc-uuid-test",
            status="done",
            chunk_count=55,
            chroma_ids=["c1", "c2", "c3"],
        )

        assert result.status == "done"
        update_payload = builder.update.call_args[0][0]
        assert update_payload["chunk_count"] == 55
        assert update_payload["chroma_ids"] == ["c1", "c2", "c3"]
        assert "error_msg" not in update_payload   # None-поля не попадают в payload

    @pytest.mark.asyncio
    async def test_update_status_to_error(self, repo_with_mock):
        """При error передаём error_msg."""
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row(status="error", error_msg="Connection refused")]

        result = await repo.update_status(
            "doc-uuid-test",
            status="error",
            error_msg="Connection refused",
        )

        assert result.status == "error"
        update_payload = builder.update.call_args[0][0]
        assert update_payload["error_msg"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_update_status_raises_if_not_found(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        with pytest.raises(RepositoryError, match="Document not found"):
            await repo.update_status("nonexistent", status="done")


# ══════════════════════════════════════════════════════════════
# list_all() и list_by_status()
# ══════════════════════════════════════════════════════════════

class TestDocumentsRepositoryList:

    @pytest.mark.asyncio
    async def test_list_all_returns_list_of_document_rows(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [
            make_doc_row(id="id-1", filename="a.pdf"),
            make_doc_row(id="id-2", filename="b.pdf"),
        ]

        result = await repo.list_all()

        assert len(result) == 2
        assert all(isinstance(r, DocumentRow) for r in result)
        assert result[0].filename == "a.pdf"

    @pytest.mark.asyncio
    async def test_list_all_applies_limit_and_offset(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        await repo.list_all(limit=10, offset=20)

        builder.limit.assert_called_with(10)
        builder.offset.assert_called_with(20)

    @pytest.mark.asyncio
    async def test_list_by_status_filters_correctly(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [make_doc_row(status="done")]

        result = await repo.list_by_status("done")

        assert len(result) == 1
        builder.eq.assert_called_with("status", "done")

    @pytest.mark.asyncio
    async def test_list_all_empty_returns_empty_list(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        result = await repo.list_all()

        assert result == []


# ══════════════════════════════════════════════════════════════
# delete()
# ══════════════════════════════════════════════════════════════

class TestDocumentsRepositoryDelete:

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_found(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [{"id": "doc-uuid-test"}]

        result = await repo.delete("doc-uuid-test")

        assert result is True
        builder.delete.assert_called_once()
        builder.eq.assert_called_with("id", "doc-uuid-test")

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        result = await repo.delete("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_true(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = [{"id": "doc-uuid-test"}]

        result = await repo.exists("doc-uuid-test")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, repo_with_mock):
        repo, builder, mock_result, _ = repo_with_mock
        mock_result.data = []

        result = await repo.exists("missing-id")

        assert result is False
