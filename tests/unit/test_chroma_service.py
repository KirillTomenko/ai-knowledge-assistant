"""
tests/unit/test_chroma_service.py

Unit-тесты для ChromaService.
ChromaDB SDK и OpenAI Embeddings полностью замоканы.

Проверяем:
  • add_documents(): батчинг, retry, progress_callback, doc_id в metadata
  • similarity_search(): where-фильтры, возврат SearchResult
  • delete_by_doc_id(): правильный вызов col.get() + col.delete()
  • health_check(): возврат статуса
  • list_collections(): CollectionInfo с count
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio

import pytest
from langchain_core.documents import Document

from src.services.chroma_service import ChromaService, SearchResult, IndexingProgress, CollectionInfo


# ── Фикстура изолированного сервиса ──────────────────────────

@pytest.fixture
def chroma_svc(mock_chroma_http_client):
    """
    Свежий (не singleton) экземпляр ChromaService с замоканным клиентом.
    Используем новый инстанс чтобы не загрязнять глобальный singleton.
    """
    svc = ChromaService()
    svc._chroma_client = mock_chroma_http_client
    svc._embeddings = MagicMock()
    svc._initialized = True
    return svc


# ══════════════════════════════════════════════════════════════
# add_documents()
# ══════════════════════════════════════════════════════════════

class TestChromaServiceAddDocuments:

    @pytest.mark.asyncio
    async def test_add_documents_returns_ids(self, chroma_svc):
        """add_documents возвращает список строк-идентификаторов."""
        docs = [
            Document(page_content=f"Чанк {i}", metadata={"source": "test.pdf", "page": i})
            for i in range(3)
        ]

        # Мокаем vectorstore.add_documents через патч Chroma
        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents = MagicMock(return_value=["id-0", "id-1", "id-2"])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            result = await chroma_svc.add_documents(docs, doc_id="test-doc-id")

        assert result == ["id-0", "id-1", "id-2"]

    @pytest.mark.asyncio
    async def test_add_documents_injects_doc_id_into_metadata(self, chroma_svc):
        """doc_id должен быть добавлен в metadata каждого чанка."""
        docs = [
            Document(page_content="Текст", metadata={"source": "test.pdf"})
        ]
        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents = MagicMock(return_value=["id-0"])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            await chroma_svc.add_documents(docs, doc_id="my-doc-id")

        # После вызова doc_id должен быть в metadata
        assert docs[0].metadata["doc_id"] == "my-doc-id"

    @pytest.mark.asyncio
    async def test_add_documents_calls_progress_callback(self, chroma_svc):
        """progress_callback вызывается после каждого батча."""
        docs = [
            Document(page_content=f"Чанк {i}", metadata={"source": "f.pdf"})
            for i in range(5)
        ]
        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents = MagicMock(return_value=["id"] * 5)

        progress_calls: list[IndexingProgress] = []

        def capture_progress(p: IndexingProgress):
            progress_calls.append(p)

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            await chroma_svc.add_documents(
                docs, doc_id="x", progress_callback=capture_progress
            )

        assert len(progress_calls) >= 1
        assert progress_calls[-1].indexed_chunks == 5
        assert progress_calls[-1].is_done is True

    @pytest.mark.asyncio
    async def test_add_documents_empty_list_returns_empty(self, chroma_svc):
        result = await chroma_svc.add_documents([], doc_id="x")
        assert result == []

    @pytest.mark.asyncio
    async def test_add_documents_batches_correctly(self, chroma_svc):
        """250 чанков → 3 батча по 100/100/50 при BATCH_SIZE=100."""
        chroma_svc.EMBED_BATCH_SIZE = 100
        docs = [
            Document(page_content=f"Чанк {i}", metadata={"source": "big.pdf"})
            for i in range(250)
        ]

        batch_sizes: list[int] = []
        def fake_add(docs_batch):
            batch_sizes.append(len(docs_batch))
            return [f"id-{j}" for j in range(len(docs_batch))]

        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents = MagicMock(side_effect=fake_add)

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            result = await chroma_svc.add_documents(docs)

        assert batch_sizes == [100, 100, 50]
        assert len(result) == 250


# ══════════════════════════════════════════════════════════════
# similarity_search()
# ══════════════════════════════════════════════════════════════

class TestChromaServiceSimilaritySearch:

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self, chroma_svc):
        docs_with_scores = [
            (Document(page_content="Текст 1", metadata={"source": "doc.pdf", "page": 1}), 0.15),
            (Document(page_content="Текст 2", metadata={"source": "doc.pdf", "page": 2}), 0.30),
        ]
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = MagicMock(return_value=docs_with_scores)

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            results = await chroma_svc.similarity_search("Вопрос про работу")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score == 0.15
        assert results[0].source == "doc.pdf"
        assert results[0].page == 1

    @pytest.mark.asyncio
    async def test_search_passes_filter_by_source(self, chroma_svc):
        """filter_source добавляет where-фильтр в запрос."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = MagicMock(return_value=[])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            await chroma_svc.similarity_search(
                "Вопрос", filter_source="handbook.pdf"
            )

        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "filter" in call_kwargs
        assert call_kwargs["filter"] == {"source": {"$eq": "handbook.pdf"}}

    @pytest.mark.asyncio
    async def test_search_passes_filter_by_doc_id(self, chroma_svc):
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = MagicMock(return_value=[])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            await chroma_svc.similarity_search(
                "Вопрос", filter_doc_id="doc-uuid-123"
            )

        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert call_kwargs["filter"] == {"doc_id": {"$eq": "doc-uuid-123"}}

    @pytest.mark.asyncio
    async def test_search_no_filter_when_none(self, chroma_svc):
        """Без фильтров — no 'filter' key в kwargs."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = MagicMock(return_value=[])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            await chroma_svc.similarity_search("Вопрос")

        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "filter" not in call_kwargs

    @pytest.mark.asyncio
    async def test_search_empty_results(self, chroma_svc):
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score = MagicMock(return_value=[])

        with patch("src.services.chroma_service.Chroma", return_value=mock_vectorstore):
            results = await chroma_svc.similarity_search("Несуществующий вопрос")

        assert results == []


# ══════════════════════════════════════════════════════════════
# delete_by_doc_id()
# ══════════════════════════════════════════════════════════════

class TestChromaServiceDelete:

    @pytest.mark.asyncio
    async def test_delete_by_doc_id_calls_collection_delete(
        self, chroma_svc, mock_chroma_collection
    ):
        mock_chroma_collection.get = MagicMock(return_value={
            "ids": ["chroma-id-1", "chroma-id-2"],
        })
        chroma_svc._chroma_client.get_collection = MagicMock(
            return_value=mock_chroma_collection
        )

        count = await chroma_svc.delete_by_doc_id("doc-uuid-123")

        assert count == 2
        mock_chroma_collection.delete.assert_called_once_with(
            ids=["chroma-id-1", "chroma-id-2"]
        )

    @pytest.mark.asyncio
    async def test_delete_by_doc_id_returns_zero_if_not_found(
        self, chroma_svc, mock_chroma_collection
    ):
        mock_chroma_collection.get = MagicMock(return_value={"ids": []})
        chroma_svc._chroma_client.get_collection = MagicMock(
            return_value=mock_chroma_collection
        )

        count = await chroma_svc.delete_by_doc_id("nonexistent-doc")

        assert count == 0
        mock_chroma_collection.delete.assert_not_called()


# ══════════════════════════════════════════════════════════════
# health_check()
# ══════════════════════════════════════════════════════════════

class TestChromaServiceHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_ok(self, chroma_svc, mock_chroma_http_client):
        result = await chroma_svc.health_check()

        assert result["status"] == "ok"
        assert result["version"] == "0.5.20"
        assert "collections_count" in result

    @pytest.mark.asyncio
    async def test_health_check_error(self, chroma_svc):
        chroma_svc._chroma_client.get_version = MagicMock(
            side_effect=ConnectionError("ChromaDB unreachable")
        )

        result = await chroma_svc.health_check()

        assert result["status"] == "error"
        assert "ChromaDB unreachable" in result["detail"]
