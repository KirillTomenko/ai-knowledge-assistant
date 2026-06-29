"""
src/db/supabase_client.py

Supabase клиент — два варианта:
  • sync_client()  — обычный Client (supabase-py), используется в sync-контексте
  • async_client() — AsyncClient (supabase-py >= 2.x), для async FastAPI / bot

Supabase Python SDK v2 поддерживает AsyncClient из коробки.
Оба клиента — синглтоны, создаются один раз при первом обращении.
"""

from __future__ import annotations

from supabase import create_client, Client
from supabase._async.client import AsyncClient, create_async_client
from loguru import logger

from src.core.config import settings

# ── Sync singleton ────────────────────────────────────────────
_sync_client: Client | None = None

def get_supabase() -> Client:
    """Возвращает синхронный Supabase-клиент (для фоновых задач, скриптов)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = create_client(settings.supabase_url, settings.supabase_key)
        logger.debug("Supabase sync client initialized")
    return _sync_client


# ── Async singleton ───────────────────────────────────────────
_async_client: AsyncClient | None = None

async def get_async_supabase() -> AsyncClient:
    """Возвращает асинхронный Supabase-клиент (для FastAPI, aiogram)."""
    global _async_client
    if _async_client is None:
        _async_client = await create_async_client(
            settings.supabase_url, settings.supabase_key
        )
        logger.debug("Supabase async client initialized")
    return _async_client
