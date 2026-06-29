"""
src/rag/embeddings.py
Работа с ChromaDB и OpenAI Embeddings.

Обеспечивает: создание коллекций, добавление чанков, поиск по вектору.
Прокси для OpenAI передаётся через httpx в SDK.
"""

import httpx
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from loguru import logger

from src.core.config import settings


def _build_openai_embeddings() -> OpenAIEmbeddings:
    """
    Создаёт OpenAIEmbeddings с поддержкой прокси.
    Прокси передаётся через httpx.AsyncClient → openai SDK.
    """
    http_client_kwargs: dict = {}
    if settings.https_proxy:
        http_client_kwargs["proxy"] = settings.https_proxy

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        http_async_client=httpx.AsyncClient(**http_client_kwargs) if http_client_kwargs else None,
    )


def _build_chroma_client() -> chromadb.HttpClient:
    """Создаёт HTTP-клиент для ChromaDB."""
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )


class VectorStore:
    """
    Обёртка над ChromaDB + LangChain Chroma.

    Поддерживает батчинг при индексации больших документов
    (важно для 500-страничных PDF — не уходим в rate limit OpenAI).
    """

    BATCH_SIZE = 100  # кол-во чанков за один запрос к Embeddings API

    def __init__(self) -> None:
        self.embeddings = _build_openai_embeddings()
        self.client = _build_chroma_client()

    def get_vectorstore(self, collection_name: str = settings.chroma_collection_name) -> Chroma:
        """Возвращает Chroma vectorstore для данной коллекции."""
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    async def add_documents(
        self,
        documents: list[Document],
        collection_name: str = settings.chroma_collection_name,
    ) -> list[str]:
        """
        Добавляет чанки в ChromaDB батчами по BATCH_SIZE.
        Возвращает список chroma document IDs.
        """
        vectorstore = self.get_vectorstore(collection_name)
        all_ids: list[str] = []

        for i in range(0, len(documents), self.BATCH_SIZE):
            batch = documents[i: i + self.BATCH_SIZE]
            logger.debug(f"Indexing batch {i // self.BATCH_SIZE + 1}: {len(batch)} chunks")
            ids = vectorstore.add_documents(batch)
            all_ids.extend(ids)

        logger.info(f"Indexed {len(documents)} chunks → {len(all_ids)} vectors in ChromaDB")
        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = settings.top_k_results,
        collection_name: str = settings.chroma_collection_name,
    ) -> list[Document]:
        """Векторный поиск Top-K релевантных чанков."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.similarity_search(query, k=k)

    def delete_by_source(
        self,
        source_filename: str,
        collection_name: str = settings.chroma_collection_name,
    ) -> None:
        """Удаляет все чанки документа по имени файла."""
        vectorstore = self.get_vectorstore(collection_name)
        vectorstore.delete(where={"source": source_filename})
        logger.info(f"Deleted all chunks for source: {source_filename}")

    def list_collections(self) -> list[str]:
        """Возвращает список коллекций ChromaDB."""
        return [col.name for col in self.client.list_collections()]


# Глобальный инстанс
vector_store = VectorStore()
