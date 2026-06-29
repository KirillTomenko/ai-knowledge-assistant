"""
src/db/repositories/documents.py

Репозиторий для таблицы `documents`.

Таблица хранит метаданные загруженных файлов и статус их индексации:
    pending → processing → done | error

Полный список методов:
    create()            — создать запись при загрузке файла
    get_by_id()         — получить по UUID
    get_by_filename()   — получить по имени файла (для проверки дублей)
    list_all()          — все документы, сортировка по дате (новые вверху)
    list_by_status()    — фильтрация по статусу (например, все "done")
    update_status()     — обновить статус + chunk_count / chroma_ids / error_msg
    update()            — произвольное частичное обновление
    delete()            — удалить запись
    count()             — общее количество документов
    exists()            — проверить существование по ID
"""

from __future__ import annotations

from loguru import logger

from src.db.models import DocumentCreate, DocumentRow, DocumentUpdate
from src.db.repositories.base import BaseRepository, RepositoryError


class DocumentsRepository(BaseRepository):
    TABLE = "documents"

    # ── CREATE ────────────────────────────────────────────────

    async def create(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size: int | None = None,
    ) -> DocumentRow:
        """
        Создаёт запись документа со статусом 'pending'.
        Вызывается сразу после сохранения файла на диск.
        """
        payload = DocumentCreate(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            status="pending",
        ).model_dump()

        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).insert(payload)
        )
        if row is None:
            raise RepositoryError(f"Failed to create document record for {filename}")

        logger.info(f"[documents] Created: {doc_id} ({filename})")
        return DocumentRow(**row)

    # ── READ ──────────────────────────────────────────────────

    async def get_by_id(self, doc_id: str) -> DocumentRow | None:
        """Возвращает документ по UUID или None."""
        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).select("*").eq("id", doc_id)
        )
        return DocumentRow(**row) if row else None

    async def get_by_filename(self, filename: str) -> DocumentRow | None:
        """Поиск по имени файла — для проверки дублей перед загрузкой."""
        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).select("*").eq("filename", filename)
        )
        return DocumentRow(**row) if row else None

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[DocumentRow]:
        """Все документы, пагинация через limit/offset."""
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .offset(offset)
        )
        return [DocumentRow(**r) for r in rows]

    async def list_by_status(self, status: str) -> list[DocumentRow]:
        """
        Документы с конкретным статусом.
        Пример: list_by_status("done") — все проиндексированные.
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE)
            .select("*")
            .eq("status", status)
            .order("created_at", desc=True)
        )
        return [DocumentRow(**r) for r in rows]

    async def exists(self, doc_id: str) -> bool:
        """Быстрая проверка существования записи — выбираем только id."""
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE).select("id").eq("id", doc_id)
        )
        return len(rows) > 0

    async def count(self) -> int:
        """Общее количество документов в таблице."""
        client = await self._client()
        # Supabase поддерживает count через head=True + count="exact"
        result = await client.table(self.TABLE).select("*", count="exact").execute()
        return result.count or 0

    # ── UPDATE ────────────────────────────────────────────────

    async def update_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        chroma_ids: list[str] | None = None,
        error_msg: str | None = None,
    ) -> DocumentRow:
        """
        Обновляет статус индексации и сопутствующие поля.

        Вызывается из фоновой задачи по ходу индексации:
            pending  → update_status(id, "processing")
            done     → update_status(id, "done", chunk_count=N, chroma_ids=[...])
            error    → update_status(id, "error", error_msg="...")
        """
        patch = DocumentUpdate(
            status=status,
            chunk_count=chunk_count,
            chroma_ids=chroma_ids,
            error_msg=error_msg,
        )
        # Исключаем поля, которые не переданы (None → не обновляем)
        payload = patch.model_dump(exclude_none=True)

        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).update(payload).eq("id", doc_id).select("*")
        )
        if row is None:
            raise RepositoryError(f"Document not found: {doc_id}")

        logger.info(f"[documents] Status updated: {doc_id} → {status}")
        return DocumentRow(**row)

    async def update(self, doc_id: str, patch: DocumentUpdate) -> DocumentRow:
        """Произвольное частичное обновление через DocumentUpdate модель."""
        payload = patch.model_dump(exclude_none=True)
        if not payload:
            # Нечего обновлять — просто вернём текущую запись
            doc = await self.get_by_id(doc_id)
            if doc is None:
                raise RepositoryError(f"Document not found: {doc_id}")
            return doc

        client = await self._client()
        row = await self._execute_one(
            client.table(self.TABLE).update(payload).eq("id", doc_id).select("*")
        )
        if row is None:
            raise RepositoryError(f"Document not found: {doc_id}")

        return DocumentRow(**row)

    # ── DELETE ────────────────────────────────────────────────

    async def delete(self, doc_id: str) -> bool:
        """
        Удаляет запись документа.
        Возвращает True если запись была — False если уже не существовала.
        """
        client = await self._client()
        rows = await self._execute(
            client.table(self.TABLE).delete().eq("id", doc_id).select("id")
        )
        deleted = len(rows) > 0
        if deleted:
            logger.info(f"[documents] Deleted: {doc_id}")
        return deleted


# Глобальный синглтон
documents_repo = DocumentsRepository()
