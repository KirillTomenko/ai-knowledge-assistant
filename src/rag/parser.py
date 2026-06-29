"""
src/rag/parser.py
Парсинг документов и разбивка на чанки.

Поддерживает: PDF (до 500 стр.), DOCX, DOC, TXT
Использует RecursiveCharacterTextSplitter с overlap для сохранения контекста.
"""

import asyncio
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from src.core.config import settings


class DocumentParser:
    """
    Парсит файлы разных форматов и разбивает на семантические чанки.

    Стратегия чанкинга:
    - Разделители в порядке приоритета: абзац → строка → предложение → слово
    - Overlap 200 символов гарантирует сохранение контекста на границах
    - Метаданные чанка: filename, page, chunk_index
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def parse_file(self, file_path: str | Path) -> list[Document]:
        """
        Асинхронно парсит файл и возвращает список Document-чанков.
        Запускает синхронный парсер в executor, чтобы не блокировать event loop.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        logger.info(f"Parsing file: {path.name} ({suffix})")

        # Запускаем тяжёлый IO в thread executor
        loop = asyncio.get_event_loop()
        raw_docs = await loop.run_in_executor(
            None, self._parse_sync, path, suffix
        )

        chunks = self.splitter.split_documents(raw_docs)

        # Добавляем chunk_index в метаданные
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source"] = path.name

        logger.info(f"Parsed {path.name}: {len(chunks)} chunks")
        return chunks

    def _parse_sync(self, path: Path, suffix: str) -> list[Document]:
        """Синхронный парсер — вызывается из executor."""
        if suffix == ".pdf":
            return self._parse_pdf(path)
        elif suffix in (".docx", ".doc"):
            return self._parse_docx(path)
        elif suffix == ".txt":
            return self._parse_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _parse_pdf(self, path: Path) -> list[Document]:
        """Парсинг PDF через pypdf. Сохраняет номер страницы в метаданных."""
        from pypdf import PdfReader  # lazy import — не грузим если не нужно

        reader = PdfReader(str(path))
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": path.name},
                ))
        return docs

    def _parse_docx(self, path: Path) -> list[Document]:
        """Парсинг DOCX через python-docx."""
        from docx import Document as DocxDocument  # lazy import

        doc = DocxDocument(str(path))
        full_text = "\n\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )
        return [Document(
            page_content=full_text,
            metadata={"page": 1, "source": path.name},
        )]

    def _parse_txt(self, path: Path) -> list[Document]:
        """Парсинг plain text."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(
            page_content=text,
            metadata={"page": 1, "source": path.name},
        )]


# Глобальный инстанс
document_parser = DocumentParser()
