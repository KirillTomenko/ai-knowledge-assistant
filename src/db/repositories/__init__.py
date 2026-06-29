"""
src/db/repositories/__init__.py

Удобный импорт всех репозиториев из одного места:

    from src.db.repositories import documents_repo, history_repo, notes_repo
"""

from src.db.repositories.documents import DocumentsRepository, documents_repo
from src.db.repositories.history import HistoryRepository, history_repo
from src.db.repositories.notes import NotesRepository, notes_repo

__all__ = [
    "DocumentsRepository",
    "HistoryRepository",
    "NotesRepository",
    "documents_repo",
    "history_repo",
    "notes_repo",
]
