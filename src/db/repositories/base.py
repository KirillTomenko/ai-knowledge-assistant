"""
src/db/repositories/base.py

Базовый класс для всех репозиториев.

Содержит:
  • Общую обработку ошибок Supabase
  • Утилиту _safe_execute() — оборачивает любой запрос в try/except
  • Логирование каждой операции
  • Тип-алиасы для удобства

Все дочерние репозитории наследуются от BaseRepository и используют
async-клиент через await self._client().
"""

from __future__ import annotations

from typing import Any
from loguru import logger
from postgrest.exceptions import APIError

from src.db.supabase_client import get_async_supabase


class RepositoryError(Exception):
    """Доменная ошибка слоя репозиториев — скрывает детали Supabase от вызывающего кода."""
    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class BaseRepository:
    """
    Базовый репозиторий.

    Использование в подклассах:
        client = await self._client()
        result = await client.table("my_table").select("*").execute()
    """

    TABLE: str  # обязателен в подклассах

    async def _client(self):
        """Возвращает async Supabase-клиент."""
        return await get_async_supabase()

    async def _execute(self, query_builder) -> list[dict[str, Any]]:
        """
        Выполняет запрос и возвращает data.
        Преобразует APIError в RepositoryError с понятным сообщением.
        """
        try:
            result = await query_builder.execute()
            return result.data or []
        except APIError as e:
            logger.error(f"[{self.__class__.__name__}] Supabase APIError: {e.message}")
            raise RepositoryError(f"Database error: {e.message}", original=e)
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Unexpected error: {e}")
            raise RepositoryError(f"Unexpected database error: {e}", original=e)

    async def _execute_one(self, query_builder) -> dict[str, Any] | None:
        """Как _execute(), но возвращает первую запись или None."""
        rows = await self._execute(query_builder)
        return rows[0] if rows else None
