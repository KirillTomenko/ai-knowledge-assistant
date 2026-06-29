"""
src/core/config.py
Централизованная конфигурация через pydantic-settings.
Читает переменные из .env файла.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI / ProxyAPI ─────────────────────────────────────
    openai_api_key: str
    openai_base_url: str = "https://api.proxyapi.ru/openai/v1"
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # ── Прокси ───────────────────────────────────────────────
    http_proxy: str | None = None
    https_proxy: str | None = None

    # ── Telegram ─────────────────────────────────────────────
    telegram_bot_token: str

    # ── Supabase ─────────────────────────────────────────────
    supabase_url: str
    supabase_key: str
    database_url: str | None = None

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "knowledge_base"

    # ── FastAPI ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change_me"
    admin_username: str = "admin"
    admin_password: str = "admin"

    # ── RAG параметры ────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    dialog_history_limit: int = 5   # кол-во последних сообщений для контекста

    # ── Логирование ──────────────────────────────────────────
    log_level: str = "INFO"


# Синглтон настроек
settings = Settings()
