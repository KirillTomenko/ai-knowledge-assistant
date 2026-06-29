"""
tests/conftest.py

Корневые фикстуры для всей тестовой suite.

Стратегия изоляции:
  • Supabase  — мокается через AsyncMock (не ходим в реальную БД)
  • ChromaDB  — мокается через MagicMock (синхронный SDK)
  • OpenAI    — мокается через respx (перехват httpx-запросов)
  • FastAPI   — AsyncClient через httpx для integration тестов

Переменные окружения подставляются через monkeypatch / pytest-env,
чтобы Settings() не падал при отсутствии реального .env.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Фиктивные переменные окружения ───────────────────────────

@pytest.fixture(autouse=True, scope="session")
def mock_env(monkeypatch=None):
    """
    Подставляет фиктивные переменные окружения для Settings().
    autouse=True — применяется ко всем тестам автоматически.
    """
    import os
    env_vars = {
        "OPENAI_API_KEY": "sk-test-key-000",
        "OPENAI_BASE_URL": "https://api.proxyapi.ru/openai/v1",
        "TELEGRAM_BOT_TOKEN": "123456789:AABBCCDDEEFFaabbccddeeff-test",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-anon-key",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8001",
        "LOG_LEVEL": "DEBUG",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "testpassword",
        "API_SECRET_KEY": "test-secret-key-32-chars-minimum!",
    }
    for k, v in env_vars.items():
        os.environ.setdefault(k, v)


# ── Общие тестовые данные ─────────────────────────────────────

@pytest.fixture
def sample_document_row() -> dict[str, Any]:
    """Эталонная строка таблицы documents."""
    return {
        "id": "doc-uuid-1234",
        "filename": "employee_handbook.pdf",
        "file_type": "pdf",
        "file_size": 204800,
        "status": "done",
        "chunk_count": 42,
        "chroma_ids": ["chroma-1", "chroma-2"],
        "error_msg": None,
        "created_at": datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 6, 1, 10, 5, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_history_rows() -> list[dict[str, Any]]:
    """Эталонные строки таблицы dialog_history (2 пары)."""
    base_time = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "id": "hist-1",
            "user_id": 111222333,
            "username": "testuser",
            "role": "user",
            "content": "Какой режим работы?",
            "metadata": {},
            "created_at": base_time,
        },
        {
            "id": "hist-2",
            "user_id": 111222333,
            "username": "testuser",
            "role": "assistant",
            "content": "Стандартный рабочий день с 09:00 до 18:00.",
            "metadata": {"source_docs": [{"source": "handbook.pdf", "page": 2}]},
            "created_at": base_time,
        },
        {
            "id": "hist-3",
            "user_id": 111222333,
            "username": "testuser",
            "role": "user",
            "content": "А удалёнка?",
            "metadata": {},
            "created_at": base_time,
        },
        {
            "id": "hist-4",
            "user_id": 111222333,
            "username": "testuser",
            "role": "assistant",
            "content": "Удалённая работа допускается по согласованию с руководителем.",
            "metadata": {"source_docs": [{"source": "handbook.pdf", "page": 3}]},
            "created_at": base_time,
        },
    ]


@pytest.fixture
def sample_note_rows() -> list[dict[str, Any]]:
    """Эталонные строки таблицы user_notes."""
    return [
        {
            "id": "note-uuid-1",
            "user_id": 111222333,
            "title": None,
            "content": "Режим работы: пн-пт, 09:00-18:00",
            "source_document": "handbook.pdf",
            "created_at": datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        },
        {
            "id": "note-uuid-2",
            "user_id": 111222333,
            "title": "Важно",
            "content": "Удалёнка по согласованию",
            "source_document": None,
            "created_at": datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc),
        },
    ]


# ── Моки Supabase ─────────────────────────────────────────────

@pytest.fixture
def mock_supabase_client():
    """
    AsyncMock для Supabase AsyncClient.
    Цепочка: client.table("x").select("*").eq(...).execute()
    """
    client = MagicMock()

    # builder — объект, возвращаемый table(), select(), eq() и т.д.
    builder = MagicMock()
    builder.select = MagicMock(return_value=builder)
    builder.insert = MagicMock(return_value=builder)
    builder.update = MagicMock(return_value=builder)
    builder.delete = MagicMock(return_value=builder)
    builder.eq = MagicMock(return_value=builder)
    builder.neq = MagicMock(return_value=builder)
    builder.lt = MagicMock(return_value=builder)
    builder.lte = MagicMock(return_value=builder)
    builder.gt = MagicMock(return_value=builder)
    builder.order = MagicMock(return_value=builder)
    builder.limit = MagicMock(return_value=builder)
    builder.offset = MagicMock(return_value=builder)
    builder.ilike = MagicMock(return_value=builder)

    # execute() — async
    mock_result = MagicMock()
    mock_result.data = []
    mock_result.count = 0
    builder.execute = AsyncMock(return_value=mock_result)

    client.table = MagicMock(return_value=builder)

    return client, builder, mock_result


# ── Моки ChromaDB ─────────────────────────────────────────────

@pytest.fixture
def mock_chroma_collection():
    """Мок chromadb.Collection."""
    col = MagicMock()
    col.count = MagicMock(return_value=10)
    col.add = MagicMock(return_value=None)
    col.query = MagicMock(return_value={
        "ids": [["id-1", "id-2"]],
        "documents": [["текст чанка 1", "текст чанка 2"]],
        "metadatas": [[
            {"source": "handbook.pdf", "page": 1, "doc_id": "doc-uuid-1234"},
            {"source": "handbook.pdf", "page": 2, "doc_id": "doc-uuid-1234"},
        ]],
        "distances": [[0.12, 0.25]],
    })
    col.get = MagicMock(return_value={"ids": ["id-1", "id-2"], "metadatas": [], "documents": []})
    col.delete = MagicMock(return_value=None)
    col.peek = MagicMock(return_value={
        "ids": ["id-1"],
        "documents": ["текст"],
        "metadatas": [{"source": "handbook.pdf", "page": 1}],
    })
    return col


@pytest.fixture
def mock_chroma_http_client(mock_chroma_collection):
    """Мок chromadb.HttpClient."""
    client = MagicMock()
    client.get_version = MagicMock(return_value="0.5.20")
    client.list_collections = MagicMock(return_value=[
        MagicMock(name="knowledge_base", metadata={})
    ])
    client.get_collection = MagicMock(return_value=mock_chroma_collection)
    client.get_or_create_collection = MagicMock(return_value=mock_chroma_collection)
    client.delete_collection = MagicMock(return_value=None)
    return client


# ── FastAPI TestClient ─────────────────────────────────────────

@pytest_asyncio.fixture
async def api_client():
    """
    AsyncClient для тестов FastAPI эндпоинтов.
    Использует ASGITransport — реальные запросы без сервера.
    """
    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
