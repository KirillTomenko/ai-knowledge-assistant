"""
src/db/models.py

Pydantic-модели, отражающие схему таблиц Supabase.
Используются для валидации данных на входе/выходе репозиториев.

Принцип именования:
  • <Table>Row   — то, что приходит из БД (все поля, включая id и timestamps)
  • <Table>Create — то, что передаётся при создании записи (без id, без timestamps)
  • <Table>Update — частичное обновление (все поля Optional)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# documents
# ═══════════════════════════════════════════════════════════════

class DocumentCreate(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int | None = None
    status: str = "pending"


class DocumentUpdate(BaseModel):
    status: str | None = None
    chunk_count: int | None = None
    chroma_ids: list[str] | None = None
    error_msg: str | None = None


class DocumentRow(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int | None = None
    status: str
    chunk_count: int = 0
    chroma_ids: list[str] | None = None
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════
# dialog_history
# ═══════════════════════════════════════════════════════════════

class HistoryMessageCreate(BaseModel):
    user_id: int
    username: str | None = None
    role: str                        # "user" | "assistant"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryMessageRow(BaseModel):
    id: str
    user_id: int
    username: str | None = None
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════
# user_notes
# ═══════════════════════════════════════════════════════════════

class NoteCreate(BaseModel):
    user_id: int
    content: str
    title: str | None = None
    source_document: str | None = None


class NoteUpdate(BaseModel):
    content: str | None = None
    title: str | None = None


class NoteRow(BaseModel):
    id: str
    user_id: int
    title: str | None = None
    content: str
    source_document: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
