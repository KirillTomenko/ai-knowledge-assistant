"""
src/db/repositories/history.py

Репозиторий для таблицы `dialog_history`.

Хранит хронологическую историю диалогов пользователей с ботом.
Каждое сообщение — отдельная строка с role: "user" | "assistant".

Полный список методов:
    save_pair()             — сохранить пару (вопрос пользователя + ответ бота)
    get_recent()            — последние N пар для конкретного пользователя
    get_as_langchain_pairs()— то же, в формате [(human, ai), ...] для LangChain
    get_by_user()           — полная история пользователя (с пагинацией)
    count_by_user()         — сколько сообщений у пользователя
    delete_by_user()        — удалить всю историю пользователя (GDPR / /clear)
    delete_older_than()     — очистка старых записей (для cron)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from loguru import logger

from src.core.config import settings
from src.db.models import HistoryMessageCreate, HistoryMessageRow
from src.db.repositories.base import BaseRepository, RepositoryError


class HistoryRepository(BaseRepository):
    TABLE = "dialog_history"

    # ── CREATE ────────────────────────────────────────────────

    async def save_pair(
        self,
        user_id: int,
        user_message: str,
        assistant_message: str,
        username: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[HistoryMessageRow, HistoryMessageRow]:
        """
        Сохраняет пару сообщений (user + assistant) одним INSERT.

        Args:
            user_id:           Telegram user ID
            user_message:      Вопрос пользователя
            assistant_message: Ответ бота
            username:          @username (опционально, для аналитики)
            metadata:          Доп. данные для ответа ассистента
                               (например, source_docs из RAG)

        Returns:
            Кортеж (user_row, assistant_row)
        """
        rows_payload = [
            HistoryMessageCreate(
                user_id=user_id,
                username=username,
                role="user",
                content=user_message,
                metadata={},
            ).model_dump(),
            HistoryMessageCreate(
                user_id=user_id,
                username=username,
                role="assistant",
                content=assistant_message,
                metadata=metadata or {},
            ).model_dump(),
        ]

        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE).insert(rows_payload).select("*")
        )

        if len(rows) < 2:
            raise RepositoryError("Failed to insert dialog pair")

        logger.debug(f"[history] Saved pair for user_id={user_id}")
        return HistoryMessageRow(**rows[0]), HistoryMessageRow(**rows[1])

    async def save_single(
        self,
        user_id: int,
        role: str,
        content: str,
        username: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HistoryMessageRow:
        """
        Сохранить одно сообщение.
        Используй save_pair() для стандартного диалога.
        Этот метод — для edge-кейсов (системные сообщения, ошибки).
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'")

        payload = HistoryMessageCreate(
            user_id=user_id,
            username=username,
            role=role,
            content=content,
            metadata=metadata or {},
        ).model_dump()

        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).insert(payload).select("*")
        )
        if row is None:
            raise RepositoryError("Failed to insert message")

        return HistoryMessageRow(**row)

    # ── READ ──────────────────────────────────────────────────

    async def get_recent(
        self,
        user_id: int,
        pairs: int | None = None,
    ) -> list[HistoryMessageRow]:
        """
        Последние N пар (user + assistant) в хронологическом порядке.

        Args:
            user_id: Telegram user ID
            pairs:   Количество пар (по умолчанию settings.dialog_history_limit)

        Returns:
            Список сообщений [user, assistant, user, assistant, ...]
            в порядке от старых к новым.
        """
        limit = (pairs or settings.dialog_history_limit) * 2
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)   # берём последние N...
            .limit(limit)
        )
        # ...и разворачиваем в хронологический порядок
        rows_sorted = list(reversed(rows))
        return [HistoryMessageRow(**r) for r in rows_sorted]

    async def get_as_langchain_pairs(
        self,
        user_id: int,
        pairs: int | None = None,
    ) -> list[tuple[str, str]]:
        """
        Возвращает историю в формате, который ожидает LangChain
        ConversationalRetrievalChain:

            [(human_msg_1, ai_msg_1), (human_msg_2, ai_msg_2), ...]

        Если пары не ровные (нечётное количество) — игнорирует хвост.
        """
        messages = await self.get_recent(user_id, pairs=pairs)
        result: list[tuple[str, str]] = []

        i = 0
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i + 1]
            if current.role == "user" and next_msg.role == "assistant":
                result.append((current.content, next_msg.content))
                i += 2
            else:
                # Нарушение порядка — пропускаем одно сообщение
                logger.warning(
                    f"[history] Unexpected role sequence at index {i}: "
                    f"{current.role} → {next_msg.role}"
                )
                i += 1

        return result

    async def get_by_user(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> list[HistoryMessageRow]:
        """
        Полная история пользователя с пагинацией.
        Полезно для экспорта / просмотра в админке.
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .offset(offset)
        )
        return [HistoryMessageRow(**r) for r in rows]

    async def count_by_user(self, user_id: int) -> int:
        """Количество сообщений у пользователя (для статистики)."""
        client = await self._client()
        result = await (
            client.table(self.TABLE)
            .select("*", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        return result.count or 0

    async def get_last_message(self, user_id: int) -> HistoryMessageRow | None:
        """Последнее сообщение пользователя (любой роли)."""
        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
        )
        return HistoryMessageRow(**row) if row else None

    # ── DELETE ────────────────────────────────────────────────

    async def delete_by_user(self, user_id: int) -> int:
        """
        Удалить всю историю пользователя.
        Используется при команде /clear или по запросу GDPR.

        Returns:
            Количество удалённых записей.
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .delete()
            .eq("user_id", user_id)
            .select("id")
        )
        count = len(rows)
        logger.info(f"[history] Deleted {count} messages for user_id={user_id}")
        return count

    async def delete_older_than(self, days: int) -> int:
        """
        Удалить записи старше N дней.
        Полезно для cron-задачи очистки (APScheduler / Celery Beat).

        Returns:
            Количество удалённых записей.
        """
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .delete()
            .lt("created_at", cutoff)
            .select("id")
        )
        count = len(rows)
        logger.info(f"[history] Cleaned {count} records older than {days} days")
        return count


# Глобальный синглтон
history_repo = HistoryRepository()
