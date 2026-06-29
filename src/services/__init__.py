"""
src/services/__init__.py

Удобный импорт всех сервисов:

    from src.services import rag_service, chroma_service
    from src.services import RAGResult, SearchResult, IndexingProgress
"""

from src.services.chroma_service import (
    ChromaService,
    CollectionInfo,
    IndexingProgress,
    SearchResult,
    chroma_service,
)
from src.services.rag_service import (
    RAGResult,
    RAGService,
    SourceReference,
    rag_service,
)

__all__ = [
    # Сервисы (синглтоны)
    "chroma_service",
    "rag_service",
    # Классы сервисов
    "ChromaService",
    "RAGService",
    # Typed результаты
    "RAGResult",
    "SearchResult",
    "SourceReference",
    "IndexingProgress",
    "CollectionInfo",
]
