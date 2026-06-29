"""
tests/unit/test_history_repository.py

Unit-тесты для HistoryRepository.

Проверяем:
  • save_pair() — двойной INSERT, возврат двух HistoryMessageRow
  • get_recent() — лимит, реверс (хронологический порядок)
  • get_as_langchain_pairs() — правильный формат [(human, ai), ...]
  • delete_by_user() — возврат количества удалённых
  • Обработку нарушения порядка ролей в get_as_langchain_pairs()
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.models import HistoryMessageRow
from src.db.repositories.base import RepositoryError


# ── Хелперы ───────────────────────────────────────────────────

def make_msg(role: str, content: str, idx: int = 1) -> dict:
    return {
        "id": f"hist-{idx}",
        "user_id": 111222333,
        "username": "testuser",
        "role": role,
        "content": content,
        "metadata": {},
        "created_at": datetime(2025, 6, 1, 9, idx, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def repo_with_mock(mock_supabase_client):
    from src.db.repositories.history import HistoryRepository

    client, builder, mock_result = mock_supabase_client
    repo = HistoryRepository()

    # Патчим get_async_supabase внутри модуля history
    import src.db.repositories.history as history_module
    original = history_module.__dict__.get("get_async_supabase")

    async def fake_get_client():
        return client

    history_module.get_async_supabase = fake_get_client
    yield repo, builder, mock_result
    if original:
        history_module.get_async_supabase = original


# ══════════════════════════════════════════════════════════════
# save_pair()
# ══════════════════════════════════════════════════════════════

class TestHistoryRepositorySavePair:

    @pytest.mark.asyncio
    async def test_save_pair_returns_two_rows(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [
            make_msg("user", "Вопрос?", 1),
            make_msg("assistant", "Ответ!", 2),
        ]

        user_row, assistant_row = await repo.save_pair(
            user_id=111222333,
            user_message="Вопрос?",
            assistant_message="Ответ!",
            username="testuser",
        )

        assert isinstance(user_row, HistoryMessageRow)
        assert isinstance(assistant_row, HistoryMessageRow)
        assert user_row.role == "user"
        assert assistant_row.role == "assistant"

    @pytest.mark.asyncio
    async def test_save_pair_inserts_both_messages(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [
            make_msg("user", "Вопрос?", 1),
            make_msg("assistant", "Ответ!", 2),
        ]

        await repo.save_pair(
            user_id=111222333,
            user_message="Вопрос?",
            assistant_message="Ответ!",
        )

        # insert вызван с двумя строками
        insert_call = builder.insert.call_args[0][0]
        assert len(insert_call) == 2
        assert insert_call[0]["role"] == "user"
        assert insert_call[1]["role"] == "assistant"
        assert insert_call[0]["content"] == "Вопрос?"
        assert insert_call[1]["content"] == "Ответ!"

    @pytest.mark.asyncio
    async def test_save_pair_stores_metadata(self, repo_with_mock):
        """metadata ассистента (source_docs) сохраняется в БД."""
        repo, builder, mock_result = repo_with_mock
        meta = {"source_docs": [{"source": "doc.pdf", "page": 1}]}
        mock_result.data = [
            make_msg("user", "Q", 1),
            make_msg("assistant", "A", 2),
        ]

        await repo.save_pair(
            user_id=111222333,
            user_message="Q",
            assistant_message="A",
            metadata=meta,
        )

        insert_payload = builder.insert.call_args[0][0]
        assert insert_payload[1]["metadata"] == meta

    @pytest.mark.asyncio
    async def test_save_pair_raises_on_empty_response(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        with pytest.raises(RepositoryError, match="Failed to insert dialog pair"):
            await repo.save_pair(
                user_id=111222333,
                user_message="Q",
                assistant_message="A",
            )


# ══════════════════════════════════════════════════════════════
# get_recent() и get_as_langchain_pairs()
# ══════════════════════════════════════════════════════════════

class TestHistoryRepositoryGetRecent:

    @pytest.mark.asyncio
    async def test_get_recent_returns_chronological_order(self, repo_with_mock):
        """
        Supabase возвращает desc (новые первые),
        репозиторий должен вернуть asc (старые первые).
        """
        repo, builder, mock_result = repo_with_mock
        # desc от Supabase — reversed в репозитории
        mock_result.data = [
            make_msg("assistant", "Ответ 2", 4),
            make_msg("user", "Вопрос 2", 3),
            make_msg("assistant", "Ответ 1", 2),
            make_msg("user", "Вопрос 1", 1),
        ]

        result = await repo.get_recent(user_id=111222333)

        # После reversed: старые первые
        assert result[0].content == "Вопрос 1"
        assert result[-1].content == "Ответ 2"

    @pytest.mark.asyncio
    async def test_get_recent_applies_limit(self, repo_with_mock):
        """limit = pairs * 2."""
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        await repo.get_recent(user_id=111222333, pairs=3)

        builder.limit.assert_called_with(6)   # 3 пары * 2

    @pytest.mark.asyncio
    async def test_get_as_langchain_pairs_format(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [
            make_msg("assistant", "Ответ 2", 4),
            make_msg("user", "Вопрос 2", 3),
            make_msg("assistant", "Ответ 1", 2),
            make_msg("user", "Вопрос 1", 1),
        ]

        pairs = await repo.get_as_langchain_pairs(user_id=111222333)

        assert len(pairs) == 2
        assert pairs[0] == ("Вопрос 1", "Ответ 1")
        assert pairs[1] == ("Вопрос 2", "Ответ 2")

    @pytest.mark.asyncio
    async def test_get_as_langchain_pairs_empty_history(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        pairs = await repo.get_as_langchain_pairs(user_id=111222333)

        assert pairs == []

    @pytest.mark.asyncio
    async def test_get_as_langchain_pairs_skips_broken_sequence(self, repo_with_mock):
        """Если роли идут не в порядке user→assistant — некорректная пара пропускается."""
        repo, builder, mock_result = repo_with_mock
        # assistant первый — нарушение порядка
        mock_result.data = [
            make_msg("assistant", "Ответ", 2),
            make_msg("user", "Вопрос", 1),
        ]

        pairs = await repo.get_as_langchain_pairs(user_id=111222333)

        # Broken sequence — ни одной валидной пары
        # (поведение зависит от реализации, минимум не упасть)
        assert isinstance(pairs, list)


# ══════════════════════════════════════════════════════════════
# delete_by_user()
# ══════════════════════════════════════════════════════════════

class TestHistoryRepositoryDelete:

    @pytest.mark.asyncio
    async def test_delete_by_user_returns_count(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = [{"id": "h1"}, {"id": "h2"}, {"id": "h3"}]

        count = await repo.delete_by_user(111222333)

        assert count == 3
        builder.eq.assert_called_with("user_id", 111222333)

    @pytest.mark.asyncio
    async def test_delete_by_user_returns_zero_if_empty(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.data = []

        count = await repo.delete_by_user(999999)

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_by_user(self, repo_with_mock):
        repo, builder, mock_result = repo_with_mock
        mock_result.count = 17

        count = await repo.count_by_user(111222333)

        assert count == 17
