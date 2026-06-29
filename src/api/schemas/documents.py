"""src/api/schemas/documents.py — Pydantic модели для HTTP слоя."""

from datetime import datetime
from pydantic import BaseModel


class DocumentStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int | None = None
    status: str
    chunk_count: int = 0
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IndexingStatusResponse(BaseModel):
    """Облегчённый ответ для polling статуса индексации."""
    id: str
    filename: str
    status: str
    chunk_count: int = 0
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CollectionInfoResponse(BaseModel):
    name: str
    count: int
    sample_sources: list[str] = []


class HealthResponse(BaseModel):
    status: str                         # "ok" | "degraded" | "error"
    version: str = "1.0.0"
    services: dict[str, dict] = {}      # статус каждого сервиса
