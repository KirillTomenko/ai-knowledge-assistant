"""
tests/unit/test_notes_repository.py

Unit-тесты для NotesRepository.

Проверяем:
  • create() — INSERT с правильными полями
  • get_by_user() — порядок, лимит, offset
  • search() — дедупликация при совпадении в content и title
  • update() — ownership check (user_id в eq())
  • delete() — ownership check
  • delete_all_by_user() — возврат количества
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.db.models import NoteRow, NoteUpdate
from src.db.repositories.base import RepositoryError


def make_note(idx: int = 1, user_id: int = 111222333, **kwargs) -> dict:
    base = {
        "id": f"note-{idx}",
        "user_id": user_id,
        "title": None,
        "content": f"Заметка {idx}",
        "source_document": None,
        "created_at": datetime(2025, 6, 1, 10, idx, 0, tzinfo=timezone.utc),
    }
    base.update(kwargs)
    return base


@pytest.fixture
def repo_with_mock(mock_supabase_client):
    from src.db.repositories.notes import NotesRepository

    client, builder, mock_result = mock_supabase_client
    repo = NotesRepository()

    import src.db.repositories.notes as notes_module

    async def fake_get_client():
        return client

    notes_module.get_async_supabase = fake_get_client
    yield repo, builder, mock_result


# ══════════════════════════════════════════════════════════════
# create()
# ══════════════════════════════════════════════════════════════

class TestNotesRepositoryCreate:

    @pytest.mark.asyncio
    async def test_create_returns_note_row(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [make_note()]

        result = await repo.create(user_id=111222333, content="Тестовая заметка")

        assert isinstance(result, NoteRow)
        assert result.user_id == 111222333
        assert result.content == "Заметка 1"

    @pytest.mark.asyncio
    async def test_create_inserts_correct_fields(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [make_note(title="Заголовок", source_document="doc.pdf")]

        await repo.create(
            user_id=111222333,
            content="Текст",
            title="Заголовок",
            source_document="doc.pdf",
        )

        payload = builder.insert.call_args[0][0]
        assert payload["user_id"] == 111222333
        assert payload["content"] == "Текст"
        assert payload["title"] == "Заголовок"
        assert payload["source_document"] == "doc.pdf"

    @pytest.mark.asyncio
    async def test_create_without_optional_fields(self, repo_with_mock):
        """title и source_document опциональны — None по умолчанию."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [make_note()]

        await repo.create(user_id=111222333, content="Просто текст")

        payload = builder.insert.call_args[0][0]
        assert payload["title"] is None
        assert payload["source_document"] is None

    @pytest.mark.asyncio
    async def test_create_raises_on_empty_response(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        with pytest.raises(RepositoryError, match="Failed to create note"):
            await repo.create(user_id=111222333, content="Текст")


# ══════════════════════════════════════════════════════════════
# get_by_user()
# ══════════════════════════════════════════════════════════════

class TestNotesRepositoryGetByUser:

    @pytest.mark.asyncio
    async def test_get_by_user_returns_note_rows(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [make_note(1), make_note(2)]

        result = await repo.get_by_user(user_id=111222333)

        assert len(result) == 2
        assert all(isinstance(r, NoteRow) for r in result)

    @pytest.mark.asyncio
    async def test_get_by_user_filters_by_user_id(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        await repo.get_by_user(user_id=999888)

        builder.eq.assert_called_with("user_id", 999888)

    @pytest.mark.asyncio
    async def test_get_by_user_applies_limit_offset(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        await repo.get_by_user(user_id=111222333, limit=5, offset=10)

        builder.limit.assert_called_with(5)
        builder.offset.assert_called_with(10)

    @pytest.mark.asyncio
    async def test_get_by_user_orders_desc(self, repo_with_mock):
        """Заметки должны быть от новых к старым."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        await repo.get_by_user(user_id=111222333)

        builder.order.assert_called_with("created_at", desc=True)


# ══════════════════════════════════════════════════════════════
# search()
# ══════════════════════════════════════════════════════════════

class TestNotesRepositorySearch:

    @pytest.mark.asyncio
    async def test_search_returns_deduplicated_results(self, repo_with_mock):
        """
        Одна и та же заметка может совпасть по content И по title.
        Результат должен быть дедуплицирован.
        """
        repo, builder, mock_result = repo_with_mock
        # Один и тот же note-1 в обоих запросах
        duplicate_note = make_note(1, content="удалёнка", title="удалёнка")
        mock_result.data = [duplicate_note]

        result = await repo.search(user_id=111222333, query="удалёнка")

        # Дедупликация: только 1 результат
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_uses_ilike_pattern(self, repo_with_mock):
        """Паттерн должен быть %query%."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        await repo.search(user_id=111222333, query="работа")

        builder.ilike.assert_called_with("title", "%работа%")

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty_list(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        result = await repo.search(user_id=111222333, query="несуществующее")

        assert result == []


# ══════════════════════════════════════════════════════════════
# update()
# ══════════════════════════════════════════════════════════════

class TestNotesRepositoryUpdate:

    @pytest.mark.asyncio
    async def test_update_checks_ownership(self, repo_with_mock):
        """Запрос включает .eq("user_id", ...) — нельзя обновить чужую заметку."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [make_note(content="Обновлено")]

        await repo.update(
            note_id="note-1",
            user_id=111222333,
            patch=NoteUpdate(content="Обновлено"),
        )

        # Проверяем, что eq вызывался с user_id
        eq_calls = [str(c) for c in builder.eq.call_args_list]
        assert any("user_id" in c for c in eq_calls)

    @pytest.mark.asyncio
    async def test_update_raises_on_not_found(self, repo_with_mock):
        """Если запись не найдена (или чужая) — RepositoryError."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        with pytest.raises(RepositoryError, match="Note not found or access denied"):
            await repo.update(
                note_id="wrong-id",
                user_id=111222333,
                patch=NoteUpdate(content="Изменение"),
            )


# ══════════════════════════════════════════════════════════════
# delete()
# ══════════════════════════════════════════════════════════════

class TestNotesRepositoryDelete:

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_deleted(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [{"id": "note-1"}]

        result = await repo.delete(note_id="note-1", user_id=111222333)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        result = await repo.delete(note_id="missing", user_id=111222333)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_all_by_user_returns_count(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [{"id": "n1"}, {"id": "n2"}]

        count = await repo.delete_all_by_user(user_id=111222333)

        assert count == 2

    @pytest.mark.asyncio
    async def test_count_by_user_returns_integer(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.count = 7

        count = await repo.count_by_user(user_id=111222333)

        assert count == 7
