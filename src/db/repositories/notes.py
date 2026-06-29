"""
src/db/repositories/notes.py

Репозиторий для таблицы `user_notes`.

Персональные заметки пользователей — произвольный текст, который они
сохраняют вручную (/note ...) или из ответа бота.

Полный список методов:
    create()            — создать заметку
    get_by_id()         — получить по UUID
    get_by_user()       — все заметки пользователя (с пагинацией)
    search()            — полнотекстовый поиск по заметкам пользователя
    count_by_user()     — количество заметок
    update()            — изменить content / title
    delete()            — удалить одну заметку (с проверкой ownership)
    delete_all_by_user()— удалить все заметки пользователя (/clear_notes)
"""

from __future__ import annotations

from loguru import logger

from src.db.models import NoteCreate, NoteRow, NoteUpdate
from src.db.repositories.base import BaseRepository, RepositoryError


class NotesRepository(BaseRepository):
    TABLE = "user_notes"

    # ── CREATE ────────────────────────────────────────────────

    async def create(
        self,
        user_id: int,
        content: str,
        title: str | None = None,
        source_document: str | None = None,
    ) -> NoteRow:
        """
        Создаёт заметку.

        Args:
            user_id:          Telegram user ID
            content:          Текст заметки
            title:            Заголовок (опционально)
            source_document:  Имя файла-источника (если заметка создана из ответа бота)

        Returns:
            NoteRow — созданная запись
        """
        payload = NoteCreate(
            user_id=user_id,
            content=content,
            title=title,
            source_document=source_document,
        ).model_dump()

        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).insert(payload).select("*")
        )
        if row is None:
            raise RepositoryError("Failed to create note")

        logger.debug(f"[notes] Created note for user_id={user_id}")
        return NoteRow(**row)

    # ── READ ──────────────────────────────────────────────────

    async def get_by_id(self, note_id: str) -> NoteRow | None:
        """Возвращает заметку по UUID или None."""
        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).select("*").eq("id", note_id)
        )
        return NoteRow(**row) if row else None

    async def get_by_user(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> list[NoteRow]:
        """
        Заметки пользователя, от новых к старым. Поддерживает пагинацию.

        Args:
            user_id: Telegram user ID
            limit:   Максимум записей (по умолчанию 20)
            offset:  Смещение (для кнопок "следующая страница")
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .offset(offset)
        )
        return [NoteRow(**r) for r in rows]

    async def search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
    ) -> list[NoteRow]:
        """
        Поиск по заметкам пользователя через ILIKE (case-insensitive substring).

        Supabase PostgREST поддерживает ilike() нативно.
        Ищем по content и title одновременно.

        Args:
            user_id: Telegram user ID
            query:   Строка поиска
            limit:   Максимум результатов
        """
        pattern = f"%{query}%"
        client = await self._client()

        # Поиск по content
        content_rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .ilike("content", pattern)
            .limit(limit)
        )

        # Поиск по title (можно совместить через OR, но PostgREST OR-фильтры
        # требуют синтаксиса or_(), поэтому делаем два запроса и мержим)
        title_rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .ilike("title", pattern)
            .limit(limit)
        )

        # Дедупликация по id
        seen: set[str] = set()
        result: list[NoteRow] = []
        for raw in content_rows + title_rows:
            if raw["id"] not in seen:
                seen.add(raw["id"])
                result.append(NoteRow(**raw))

        return result[:limit]

    async def count_by_user(self, user_id: int) -> int:
        """Количество заметок у пользователя."""
        client = await self._client()
        result = await (
            client.table(self.TABLE)
            .select("*", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        return result.count or 0

    # ── UPDATE ────────────────────────────────────────────────

    async def update(
        self,
        note_id: str,
        user_id: int,
        patch: NoteUpdate,
    ) -> NoteRow:
        """
        Обновляет заметку.
        Проверяет ownership: нельзя обновить чужую заметку.

        Args:
            note_id: UUID заметки
            user_id: Telegram user ID (для проверки ownership)
            patch:   Поля для обновления (NoteUpdate, все Optional)
        """
        payload = patch.model_dump(exclude_none=True)
        if not payload:
            note = await self.get_by_id(note_id)
            if note is None or note.user_id != user_id:
                raise RepositoryError(f"Note not found or access denied: {note_id}")
            return note

        client = await self._client()
        # .eq("user_id", user_id) — ownership check на уровне запроса
        row = await self._execute_one(
            client.table(self.TABLE)
            .update(payload)
            .eq("id", note_id)
            .eq("user_id", user_id)
            .select("*")
        )
        if row is None:
            raise RepositoryError(f"Note not found or access denied: {note_id}")

        logger.debug(f"[notes] Updated note {note_id} for user_id={user_id}")
        return NoteRow(**row)

    # ── DELETE ────────────────────────────────────────────────

    async def delete(self, note_id: str, user_id: int) -> bool:
        """
        Удаляет заметку с проверкой ownership.
        Возвращает True если заметка была удалена, False — если не найдена.
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .delete()
            .eq("id", note_id)
            .eq("user_id", user_id)   # нельзя удалить чужую заметку
            .select("id")
        )
        deleted = len(rows) > 0
        if deleted:
            logger.debug(f"[notes] Deleted note {note_id} for user_id={user_id}")
        return deleted

    async def delete_all_by_user(self, user_id: int) -> int:
        """
        Удаляет все заметки пользователя.
        Используется при команде /clear_notes или запросе GDPR.

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
        logger.info(f"[notes] Deleted {count} notes for user_id={user_id}")
        return count


# Глобальный синглтон
notes_repo = NotesRepository()
