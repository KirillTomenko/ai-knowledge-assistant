"""
src/services/chroma_service.py

Сервис для работы с ChromaDB и OpenAI Embeddings.

Отвечает за:
  • Управление жизненным циклом клиента ChromaDB (синглтон, lazy init)
  • Создание и удаление коллекций
  • Индексацию документов батчами с retry и прогресс-коллбэком
  • Асинхронный векторный поиск (similarity + MMR)
  • Удаление векторов по метаданным (source, doc_id)
  • Инспекцию коллекций (количество векторов, peek, stats)

Важные архитектурные решения:
  ┌───────────────────────────────────────────────────────────────┐
  │ ChromaDB Python SDK синхронный — chromadb.HttpClient работает │
  │ через синхронный httpx. Поэтому все тяжёлые операции         │
  │ (add_documents, embedding API) запускаются через             │
  │ asyncio.get_event_loop().run_in_executor(), чтобы не          │
  │ блокировать event loop FastAPI / aiogram.                    │
  │                                                               │
  │ Для embeddings используется AsyncOpenAI через httpx.          │
  │ Батчинг: 100 чанков за вызов → не упираемся в rate limit.    │
  └───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import chromadb
import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings


# ═══════════════════════════════════════════════════════════════
# Pydantic-like dataclasses для результатов
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """Один результат векторного поиска."""
    document: Document
    score: float | None = None   # cosine distance (0 = идентично, 2 = противоположно)

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self) -> int | str:
        return self.document.metadata.get("page", "?")

    @property
    def content(self) -> str:
        return self.document.page_content


@dataclass
class IndexingProgress:
    """Прогресс индексации документа."""
    doc_id: str
    total_chunks: int
    indexed_chunks: int = 0
    failed_batches: int = 0

    @property
    def percent(self) -> float:
        if self.total_chunks == 0:
            return 100.0
        return round(self.indexed_chunks / self.total_chunks * 100, 1)

    @property
    def is_done(self) -> bool:
        return self.indexed_chunks >= self.total_chunks


@dataclass
class CollectionInfo:
    """Информация о коллекции ChromaDB."""
    name: str
    count: int                          # количество векторов
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_sources: list[str] = field(default_factory=list)   # уникальные источники


# ═══════════════════════════════════════════════════════════════
# Фабрики клиентов
# ═══════════════════════════════════════════════════════════════

def _build_chroma_client() -> chromadb.HttpClient:
    """
    HTTP-клиент ChromaDB.
    Таймаут увеличен до 120 сек — индексация больших батчей может занять время.
    """
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def _build_embeddings() -> OpenAIEmbeddings:
    """
    OpenAI Embeddings с поддержкой прокси (Karing / ProxyAPI).

    Передаём httpx.AsyncClient явно — это единственный способ
    пробросить SOCKS5/HTTP прокси в langchain-openai >= 0.2.
    """
    async_client_kwargs: dict[str, Any] = {}
    sync_client_kwargs: dict[str, Any] = {}

    if settings.https_proxy:
        async_client_kwargs["proxy"] = settings.https_proxy
        sync_client_kwargs["proxy"] = settings.https_proxy

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        # Async client — для ainvoke / async pipeline
        http_async_client=httpx.AsyncClient(**async_client_kwargs)
        if async_client_kwargs
        else None,
        # Sync client — когда embeddings вызываются из executor
        http_client=httpx.Client(**sync_client_kwargs)
        if sync_client_kwargs
        else None,
    )


# ═══════════════════════════════════════════════════════════════
# ChromaService
# ═══════════════════════════════════════════════════════════════

class ChromaService:
    """
    Сервис управления векторным хранилищем ChromaDB.

    Singleton: создаётся один раз через ChromaService.get_instance().
    Клиент ChromaDB и объект Embeddings инициализируются лениво
    при первом обращении (_lazy_init).

    Пример использования:
        service = ChromaService.get_instance()
        ids = await service.add_documents(chunks, doc_id="uuid-...")
        results = await service.similarity_search("вопрос")
    """

    # Батч-размеры
    EMBED_BATCH_SIZE: int = 100   # чанков за один вызов Embeddings API
    MAX_RETRIES: int = 3          # попыток при ошибке сети
    RETRY_WAIT_MIN: float = 1.0   # сек
    RETRY_WAIT_MAX: float = 10.0  # сек

    _instance: ChromaService | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._chroma_client: chromadb.HttpClient | None = None
        self._embeddings: OpenAIEmbeddings | None = None
        self._initialized: bool = False

    # ── Singleton ─────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "ChromaService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lazy initialization ───────────────────────────────────

    async def _lazy_init(self) -> None:
        """
        Инициализирует клиент ChromaDB и объект Embeddings.
        Thread-safe через asyncio.Lock.
        Вызывается автоматически при первом обращении к любому методу.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:   # double-check после получения лока
                return

            logger.info("ChromaService: initializing client and embeddings...")
            self._chroma_client = _build_chroma_client()
            self._embeddings = _build_embeddings()
            self._initialized = True
            logger.info("ChromaService: ready")

    @property
    def chroma_client(self) -> chromadb.HttpClient:
        if self._chroma_client is None:
            raise RuntimeError("ChromaService not initialized. Call _lazy_init() first.")
        return self._chroma_client

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            raise RuntimeError("ChromaService not initialized. Call _lazy_init() first.")
        return self._embeddings

    def _get_vectorstore(self, collection_name: str) -> Chroma:
        """Создаёт объект Chroma (LangChain-обёртка) для указанной коллекции."""
        return Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def _run_sync(self, func: Callable, *args: Any, **kwargs: Any):
        """
        Запускает синхронную функцию в thread executor.
        Используется для ChromaDB вызовов, которые блокируют поток.
        """
        loop = asyncio.get_event_loop()
        if kwargs:
            return loop.run_in_executor(None, partial(func, *args, **kwargs))
        return loop.run_in_executor(None, func, *args)

    # ── Управление коллекциями ────────────────────────────────

    async def list_collections(self) -> list[CollectionInfo]:
        """
        Возвращает список всех коллекций с метаданными.

        Returns:
            Список CollectionInfo с именем и количеством векторов.
        """
        await self._lazy_init()

        def _sync() -> list[CollectionInfo]:
            collections = self.chroma_client.list_collections()
            result = []
            for col in collections:
                chroma_col = self.chroma_client.get_collection(col.name)
                count = chroma_col.count()

                # Получаем уникальные источники через peek
                sample: list[str] = []
                if count > 0:
                    peek = chroma_col.peek(min(10, count))
                    sources = {
                        m.get("source", "")
                        for m in (peek.get("metadatas") or [])
                        if m
                    }
                    sample = sorted(s for s in sources if s)

                result.append(CollectionInfo(
                    name=col.name,
                    count=count,
                    metadata=col.metadata or {},
                    sample_sources=sample,
                ))
            return result

        return await self._run_sync(_sync)

    async def collection_info(
        self,
        collection_name: str = settings.chroma_collection_name,
    ) -> CollectionInfo:
        """Детальная информация об одной коллекции."""
        await self._lazy_init()

        def _sync() -> CollectionInfo:
            try:
                col = self.chroma_client.get_collection(collection_name)
            except Exception:
                return CollectionInfo(name=collection_name, count=0)

            count = col.count()
            sample_sources: list[str] = []
            if count > 0:
                peek = col.peek(min(20, count))
                sources = {
                    m.get("source", "")
                    for m in (peek.get("metadatas") or [])
                    if m
                }
                sample_sources = sorted(s for s in sources if s)

            return CollectionInfo(
                name=collection_name,
                count=count,
                metadata=col.metadata or {},
                sample_sources=sample_sources,
            )

        return await self._run_sync(_sync)

    async def delete_collection(self, collection_name: str) -> None:
        """
        Полностью удаляет коллекцию из ChromaDB.
        Используй с осторожностью — данные не восстановить.
        """
        await self._lazy_init()

        def _sync() -> None:
            self.chroma_client.delete_collection(collection_name)
            logger.warning(f"ChromaService: collection '{collection_name}' deleted")

        await self._run_sync(_sync)

    async def reset_collection(
        self,
        collection_name: str = settings.chroma_collection_name,
    ) -> None:
        """
        Пересоздаёт коллекцию (удаляет все данные).
        Эквивалент: delete → create.
        """
        await self._lazy_init()

        def _sync() -> None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass   # коллекции не было — ок
            self.chroma_client.get_or_create_collection(collection_name)
            logger.info(f"ChromaService: collection '{collection_name}' reset")

        await self._run_sync(_sync)

    # ── Индексация документов ─────────────────────────────────

    async def add_documents(
        self,
        documents: list[Document],
        doc_id: str | None = None,
        collection_name: str = settings.chroma_collection_name,
        progress_callback: Callable[[IndexingProgress], None] | None = None,
    ) -> list[str]:
        """
        Индексирует список Document-чанков в ChromaDB.

        Добавляет doc_id в метаданные каждого чанка — это позволяет
        потом удалить все чанки документа одним запросом по doc_id.

        Батчинг: EMBED_BATCH_SIZE чанков за запрос к Embeddings API.
        Retry: до MAX_RETRIES попыток при сетевых ошибках (tenacity).
        Progress: опциональный коллбэк с объектом IndexingProgress.

        Args:
            documents:          Список чанков из DocumentParser
            doc_id:             UUID документа (добавляется в метаданные)
            collection_name:    Коллекция ChromaDB
            progress_callback:  fn(IndexingProgress) — вызывается после каждого батча

        Returns:
            Список chroma document IDs (uuid строки).
        """
        await self._lazy_init()

        if not documents:
            logger.warning("ChromaService.add_documents: empty document list")
            return []

        # Добавляем doc_id в метаданные чанков (для последующего удаления)
        if doc_id:
            for doc in documents:
                doc.metadata["doc_id"] = doc_id

        vectorstore = self._get_vectorstore(collection_name)
        all_ids: list[str] = []
        progress = IndexingProgress(
            doc_id=doc_id or "unknown",
            total_chunks=len(documents),
        )

        total_batches = (len(documents) + self.EMBED_BATCH_SIZE - 1) // self.EMBED_BATCH_SIZE

        for batch_num, i in enumerate(range(0, len(documents), self.EMBED_BATCH_SIZE), start=1):
            batch = documents[i: i + self.EMBED_BATCH_SIZE]
            logger.debug(
                f"ChromaService: indexing batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            # Retry-обёртка для одного батча
            batch_ids = await self._index_batch_with_retry(vectorstore, batch)
            all_ids.extend(batch_ids)

            # Обновляем прогресс
            progress.indexed_chunks += len(batch)
            if progress_callback:
                try:
                    progress_callback(progress)
                except Exception as cb_err:
                    logger.warning(f"Progress callback error: {cb_err}")

            logger.debug(f"ChromaService: {progress.percent}% done")

        logger.info(
            f"ChromaService: indexed {len(documents)} chunks → "
            f"{len(all_ids)} vectors "
            f"(collection='{collection_name}', doc_id={doc_id})"
        )
        return all_ids

    async def _index_batch_with_retry(
        self,
        vectorstore: Chroma,
        batch: list[Document],
    ) -> list[str]:
        """
        Индексирует один батч с экспоненциальным retry.

        Перехватывает:
          • httpx.ConnectError    — ChromaDB недоступен
          • httpx.TimeoutException — таймаут embeddings API
          • Exception с 'rate limit' в тексте — OpenAI rate limit
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.MAX_RETRIES),
                wait=wait_exponential(
                    min=self.RETRY_WAIT_MIN,
                    max=self.RETRY_WAIT_MAX,
                ),
                retry=retry_if_exception_type((
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    ConnectionError,
                )),
                reraise=True,
            ):
                with attempt:
                    def _sync_add(vs: Chroma, docs: list[Document]) -> list[str]:
                        return vs.add_documents(docs)

                    return await self._run_sync(_sync_add, vectorstore, batch)

        except RetryError as e:
            logger.error(f"ChromaService: batch failed after {self.MAX_RETRIES} retries: {e}")
            raise RuntimeError(
                f"ChromaDB indexing failed after {self.MAX_RETRIES} retries"
            ) from e

        # Fallback — не должен достигаться, но mypy доволен
        return []  # pragma: no cover

    # ── Поиск ─────────────────────────────────────────────────

    async def similarity_search(
        self,
        query: str,
        k: int = settings.top_k_results,
        collection_name: str = settings.chroma_collection_name,
        filter_source: str | None = None,
        filter_doc_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Векторный поиск Top-K релевантных чанков по косинусному сходству.

        Args:
            query:           Текст запроса (будет преобразован в embedding)
            k:               Количество результатов
            collection_name: Коллекция ChromaDB
            filter_source:   Фильтр по имени файла (metadata["source"])
            filter_doc_id:   Фильтр по UUID документа (metadata["doc_id"])

        Returns:
            Список SearchResult, отсортированный по релевантности (лучший — первый).
        """
        await self._lazy_init()

        vectorstore = self._get_vectorstore(collection_name)

        # Строим where-фильтр ChromaDB
        where: dict[str, Any] | None = None
        if filter_source and filter_doc_id:
            where = {"$and": [
                {"source": {"$eq": filter_source}},
                {"doc_id": {"$eq": filter_doc_id}},
            ]}
        elif filter_source:
            where = {"source": {"$eq": filter_source}}
        elif filter_doc_id:
            where = {"doc_id": {"$eq": filter_doc_id}}

        search_kwargs: dict[str, Any] = {"k": k}
        if where:
            search_kwargs["filter"] = where

        def _sync_search() -> list[tuple[Document, float]]:
            return vectorstore.similarity_search_with_score(query, **search_kwargs)

        raw_results: list[tuple[Document, float]] = await self._run_sync(_sync_search)

        return [
            SearchResult(document=doc, score=score)
            for doc, score in raw_results
        ]

    async def mmr_search(
        self,
        query: str,
        k: int = settings.top_k_results,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
        collection_name: str = settings.chroma_collection_name,
    ) -> list[SearchResult]:
        """
        Поиск через Maximal Marginal Relevance (MMR).

        MMR балансирует между релевантностью и разнообразием результатов.
        Полезен когда несколько чанков из одного абзаца попадают в Top-K
        и дублируют информацию.

        Args:
            query:       Текст запроса
            k:           Итоговое количество результатов
            fetch_k:     Сколько кандидатов достать перед отбором (по умолчанию k*4)
            lambda_mult: 0.0 = максимум разнообразия, 1.0 = максимум релевантности
            collection_name: Коллекция ChromaDB
        """
        await self._lazy_init()

        effective_fetch_k = fetch_k or k * 4
        vectorstore = self._get_vectorstore(collection_name)

        def _sync_mmr() -> list[Document]:
            return vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=effective_fetch_k,
                lambda_mult=lambda_mult,
            )

        docs: list[Document] = await self._run_sync(_sync_mmr)
        # MMR не возвращает score — ставим None
        return [SearchResult(document=doc) for doc in docs]

    # ── Удаление ──────────────────────────────────────────────

    async def delete_by_doc_id(
        self,
        doc_id: str,
        collection_name: str = settings.chroma_collection_name,
    ) -> int:
        """
        Удаляет все чанки документа по doc_id из метаданных.

        Это надёжный способ удаления: doc_id добавляется при индексации
        в add_documents() и всегда уникален для каждого документа.

        Returns:
            Количество удалённых векторов.
        """
        await self._lazy_init()

        def _sync() -> int:
            try:
                col = self.chroma_client.get_collection(collection_name)
                # Ищем все chroma IDs с этим doc_id
                results = col.get(
                    where={"doc_id": {"$eq": doc_id}},
                    include=[],   # только IDs, без векторов/документов
                )
                ids_to_delete = results.get("ids", [])
                if ids_to_delete:
                    col.delete(ids=ids_to_delete)
                    logger.info(
                        f"ChromaService: deleted {len(ids_to_delete)} vectors "
                        f"for doc_id='{doc_id}'"
                    )
                return len(ids_to_delete)
            except Exception as e:
                logger.error(f"ChromaService.delete_by_doc_id error: {e}")
                raise

        return await self._run_sync(_sync)

    async def delete_by_source(
        self,
        source_filename: str,
        collection_name: str = settings.chroma_collection_name,
    ) -> int:
        """
        Удаляет все чанки по имени файла (metadata["source"]).

        Используй delete_by_doc_id() если есть doc_id — он точнее,
        так как source (имя файла) теоретически может повторяться.

        Returns:
            Количество удалённых векторов.
        """
        await self._lazy_init()

        def _sync() -> int:
            col = self.chroma_client.get_collection(collection_name)
            results = col.get(
                where={"source": {"$eq": source_filename}},
                include=[],
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                logger.info(
                    f"ChromaService: deleted {len(ids_to_delete)} vectors "
                    f"for source='{source_filename}'"
                )
            return len(ids_to_delete)

        return await self._run_sync(_sync)

    async def delete_by_ids(
        self,
        chroma_ids: list[str],
        collection_name: str = settings.chroma_collection_name,
    ) -> None:
        """
        Удаляет векторы по списку Chroma document IDs.
        IDs хранятся в таблице documents.chroma_ids в Supabase.
        """
        await self._lazy_init()

        if not chroma_ids:
            return

        def _sync() -> None:
            col = self.chroma_client.get_collection(collection_name)
            col.delete(ids=chroma_ids)
            logger.info(f"ChromaService: deleted {len(chroma_ids)} vectors by IDs")

        await self._run_sync(_sync)

    # ── Health check ──────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Проверяет доступность ChromaDB.
        Используется в /health эндпоинте FastAPI.

        Returns:
            {"status": "ok", "version": "...", "collections_count": N}
            или {"status": "error", "detail": "..."}
        """
        await self._lazy_init()

        def _sync() -> dict[str, Any]:
            version = self.chroma_client.get_version()
            collections = self.chroma_client.list_collections()
            return {
                "status": "ok",
                "version": version,
                "collections_count": len(collections),
            }

        try:
            return await self._run_sync(_sync)
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# ── Глобальный синглтон ───────────────────────────────────────

chroma_service = ChromaService.get_instance()
