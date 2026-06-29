"""
src/api/routers/documents.py

FastAPI роутер для управления документами.
Бизнес-логика вынесена в сервисы — роутер только HTTP-слой:
  • валидация входных данных
  • вызов сервисов
  • HTTP-ответы и коды статусов

Фоновая индексация: FastAPI BackgroundTasks (не блокирует ответ API).
Статус отслеживается через polling: GET /{doc_id}/status.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from loguru import logger

from src.api.schemas.documents import DocumentResponse, DocumentStatus, IndexingStatusResponse
from src.db.repositories import documents_repo
from src.rag.parser import document_parser
from src.services import chroma_service

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".doc", ".txt"})
MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024   # 200 MB


# ── Эндпоинты ─────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Загрузить документ и запустить индексацию",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentResponse:
    """
    Загружает файл и ставит задачу индексации в очередь.

    Возвращает `document_id` — используй его для отслеживания статуса
    через `GET /{doc_id}/status`.

    Статусы документа:
    - `pending`    — файл принят, задача в очереди
    - `processing` — идёт парсинг и индексация
    - `done`       — документ проиндексирован, доступен для поиска
    - `error`      — ошибка индексации (см. поле `error_msg`)
    """
    # Валидация расширения
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип файла: '{suffix}'. Допустимые: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Проверка дубля по имени файла
    existing = await documents_repo.get_by_filename(file.filename or "")
    if existing and existing.status == "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Документ '{file.filename}' уже проиндексирован (id={existing.id}). "
                   f"Удали существующий перед повторной загрузкой.",
        )

    # Сохраняем файл на диск
    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка сохранения файла: {e}",
        ) from e

    file_size = file_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой: {file_size // 1024 // 1024} MB. Максимум: 200 MB.",
        )

    # Создаём запись в Supabase (status=pending)
    doc = await documents_repo.create(
        doc_id=doc_id,
        filename=file.filename or file_path.name,
        file_type=suffix.lstrip("."),
        file_size=file_size,
    )

    # Ставим индексацию в фон
    background_tasks.add_task(_run_indexing, doc_id, file_path)
    logger.info(f"[upload] Document queued: {doc_id} ({file.filename}, {file_size // 1024} KB)")

    return DocumentResponse.model_validate(doc.model_dump())


@router.get(
    "/{doc_id}/status",
    response_model=IndexingStatusResponse,
    summary="Статус индексации документа",
)
async def get_document_status(doc_id: str) -> IndexingStatusResponse:
    """
    Возвращает текущий статус индексации документа.
    Используй polling каждые 3-5 секунд пока status != 'done' | 'error'.
    """
    doc = await documents_repo.get_by_id(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Документ не найден")

    return IndexingStatusResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        error_msg=doc.error_msg,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="Список всех документов",
)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    status_filter: str | None = None,
) -> list[DocumentResponse]:
    """
    Список документов с пагинацией и опциональной фильтрацией по статусу.

    Параметры:
    - `limit`         — максимум записей (по умолчанию 50)
    - `offset`        — смещение для пагинации
    - `status_filter` — фильтр: pending | processing | done | error
    """
    if status_filter:
        docs = await documents_repo.list_by_status(status_filter)
    else:
        docs = await documents_repo.list_all(limit=limit, offset=offset)

    return [DocumentResponse.model_validate(d.model_dump()) for d in docs]


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить документ и его векторы",
)
async def delete_document(doc_id: str) -> None:
    """
    Удаляет документ из Supabase и все его векторы из ChromaDB.

    Если документ был проиндексирован (status=done), векторы удаляются
    по doc_id из метаданных ChromaDB.
    Если документ в процессе индексации (processing) — сначала дождись
    завершения или ошибки.
    """
    doc = await documents_repo.get_by_id(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Документ не найден")

    if doc.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Документ в процессе индексации. Дождись завершения.",
        )

    # Удаляем векторы из ChromaDB (по doc_id в метаданных)
    deleted_vectors = await chroma_service.delete_by_doc_id(doc_id)
    logger.info(f"[delete] Removed {deleted_vectors} vectors for doc_id={doc_id}")

    # Удаляем файл с диска
    file_path = UPLOAD_DIR / f"{doc_id}.{doc.file_type}"
    if file_path.exists():
        file_path.unlink()
        logger.debug(f"[delete] File removed: {file_path}")

    # Удаляем запись из Supabase
    await documents_repo.delete(doc_id)
    logger.info(f"[delete] Document {doc_id} ({doc.filename}) fully removed")


# ── Фоновая задача индексации ─────────────────────────────────

async def _run_indexing(doc_id: str, file_path: Path) -> None:
    """
    Фоновая задача: парсинг → чанкинг → embeddings → ChromaDB.

    Обновляет статус документа в Supabase на каждом этапе.
    Прогресс логируется каждые 10% через progress_callback.
    """
    logger.info(f"[indexing:{doc_id}] Started: {file_path.name}")

    try:
        # ── pending → processing ──────────────────────────────
        await documents_repo.update_status(doc_id, status="processing")

        # ── Шаг 1: Парсинг и чанкинг ─────────────────────────
        logger.debug(f"[indexing:{doc_id}] Parsing file...")
        chunks = await document_parser.parse_file(file_path)
        logger.info(f"[indexing:{doc_id}] Parsed: {len(chunks)} chunks")

        if not chunks:
            raise ValueError("Документ пустой или не удалось извлечь текст.")

        # ── Шаг 2: Embeddings + ChromaDB ─────────────────────
        logger.debug(f"[indexing:{doc_id}] Indexing {len(chunks)} chunks...")

        last_logged_pct = -1

        def _progress_callback(progress) -> None:
            nonlocal last_logged_pct
            # Логируем каждые 10%
            pct_bucket = int(progress.percent // 10) * 10
            if pct_bucket > last_logged_pct:
                last_logged_pct = pct_bucket
                logger.info(
                    f"[indexing:{doc_id}] Progress: {progress.percent}% "
                    f"({progress.indexed_chunks}/{progress.total_chunks} chunks)"
                )

        chroma_ids = await chroma_service.add_documents(
            documents=chunks,
            doc_id=doc_id,
            progress_callback=_progress_callback,
        )

        # ── processing → done ─────────────────────────────────
        await documents_repo.update_status(
            doc_id,
            status="done",
            chunk_count=len(chunks),
            chroma_ids=chroma_ids,
        )
        logger.info(
            f"[indexing:{doc_id}] Done: {len(chunks)} chunks, "
            f"{len(chroma_ids)} vectors"
        )

    except Exception as e:
        logger.error(f"[indexing:{doc_id}] Failed: {e}", exc_info=True)
        await documents_repo.update_status(
            doc_id,
            status="error",
            error_msg=str(e)[:500],   # Supabase TEXT, но обрезаем на всякий случай
        )
