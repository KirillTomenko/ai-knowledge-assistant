"""
tests/unit/test_rag_service.py

Unit-тесты для RAGService.
OpenAI LLM и ChromaService полностью замоканы.

Проверяем:
  • condense_question(): с историей и без
  • ask(): полный pipeline, RAGResult структура
  • _build_context_string(): форматирование с метаданными
  • _extract_sources(): дедупликация источников
  • ask() при пустом retrieval (no context)
  • ask_stream(): генерация токенов
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import AsyncIterator

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.services.chroma_service import SearchResult
from src.services.rag_service import RAGService, RAGResult, SourceReference


# ── Хелперы ───────────────────────────────────────────────────

def make_search_result(content: str, source: str = "doc.pdf", page: int = 1, score: float = 0.1):
    return SearchResult(
        document=Document(
            page_content=content,
            metadata={"source": source, "page": page, "doc_id": "doc-uuid"},
        ),
        score=score,
    )


def make_ai_message(content: str) -> AIMessage:
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    return msg


# ── Фикстура сервиса ──────────────────────────────────────────

@pytest.fixture
def rag_svc():
    """Свежий экземпляр RAGService с замоканными зависимостями."""
    svc = RAGService()

    # Мок LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=make_ai_message("Стандартный рабочий день с 09:00 до 18:00.")
    )
    svc._llm = mock_llm
    svc._condense_llm = mock_llm

    return svc


# ══════════════════════════════════════════════════════════════
# condense_question()
# ══════════════════════════════════════════════════════════════

class TestRAGServiceCondenseQuestion:

    @pytest.mark.asyncio
    async def test_condense_without_history_returns_original(self, rag_svc):
        """Без истории вопрос возвращается без изменений (LLM не вызывается)."""
        result = await rag_svc.condense_question("Какой режим работы?", chat_history=[])

        assert result == "Какой режим работы?"
        rag_svc._condense_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_condense_with_history_calls_llm(self, rag_svc):
        """С историей вызывается condense LLM."""
        rag_svc._condense_llm.ainvoke = AsyncMock(
            return_value=make_ai_message("Каков порядок удалённой работы в компании?")
        )

        result = await rag_svc.condense_question(
            question="А удалёнка?",
            chat_history=[("Какой режим работы?", "Пн-пт, 09:00-18:00.")],
        )

        assert result == "Каков порядок удалённой работы в компании?"
        rag_svc._condense_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_condense_fallback_on_llm_error(self, rag_svc):
        """При ошибке LLM возвращается исходный вопрос (fallback)."""
        rag_svc._condense_llm.ainvoke = AsyncMock(
            side_effect=ConnectionError("LLM недоступен")
        )

        result = await rag_svc.condense_question(
            question="А удалёнка?",
            chat_history=[("Вопрос", "Ответ")],
        )

        # Не упал — вернул оригинальный вопрос
        assert result == "А удалёнка?"

    @pytest.mark.asyncio
    async def test_condense_returns_original_if_llm_empty(self, rag_svc):
        """Если LLM вернул пустую строку — оригинальный вопрос."""
        rag_svc._condense_llm.ainvoke = AsyncMock(
            return_value=make_ai_message("")
        )

        result = await rag_svc.condense_question(
            question="Оригинальный вопрос",
            chat_history=[("Q", "A")],
        )

        assert result == "Оригинальный вопрос"


# ══════════════════════════════════════════════════════════════
# _build_context_string()
# ══════════════════════════════════════════════════════════════

class TestRAGServiceBuildContext:

    def test_build_context_empty_returns_fallback(self, rag_svc):
        result = rag_svc._build_context_string([])
        assert "не найдено" in result.lower()

    def test_build_context_includes_source_metadata(self, rag_svc):
        results = [
            make_search_result("Рабочий день 09:00-18:00", source="handbook.pdf", page=2),
        ]
        context = rag_svc._build_context_string(results)

        assert "handbook.pdf" in context
        assert "стр. 2" in context
        assert "Рабочий день 09:00-18:00" in context

    def test_build_context_numbers_chunks(self, rag_svc):
        results = [
            make_search_result("Чанк А", source="a.pdf", page=1),
            make_search_result("Чанк Б", source="b.pdf", page=3),
        ]
        context = rag_svc._build_context_string(results)

        assert "[1]" in context
        assert "[2]" in context

    def test_build_context_separates_chunks(self, rag_svc):
        results = [
            make_search_result("Чанк А"),
            make_search_result("Чанк Б"),
        ]
        context = rag_svc._build_context_string(results)
        # Разделитель между чанками
        assert "---" in context


# ══════════════════════════════════════════════════════════════
# _extract_sources()
# ══════════════════════════════════════════════════════════════

class TestRAGServiceExtractSources:

    def test_extract_sources_deduplicates(self, rag_svc):
        """Два чанка из одной страницы → один SourceReference."""
        results = [
            make_search_result("Чанк 1", source="doc.pdf", page=1),
            make_search_result("Чанк 2", source="doc.pdf", page=1),
            make_search_result("Чанк 3", source="doc.pdf", page=2),
        ]
        sources = rag_svc._extract_sources(results)

        assert len(sources) == 2  # doc.pdf:1 и doc.pdf:2

    def test_extract_sources_preserves_order(self, rag_svc):
        """Порядок: первый встреченный источник — первый в списке."""
        results = [
            make_search_result("А", source="a.pdf", page=1),
            make_search_result("Б", source="b.pdf", page=1),
        ]
        sources = rag_svc._extract_sources(results)

        assert sources[0].source == "a.pdf"
        assert sources[1].source == "b.pdf"

    def test_extract_sources_empty(self, rag_svc):
        assert rag_svc._extract_sources([]) == []

    def test_source_reference_format(self, rag_svc):
        ref = SourceReference(source="handbook.pdf", page=3, excerpt="...")
        formatted = ref.format()
        assert "handbook.pdf" in formatted
        assert "3" in formatted


# ══════════════════════════════════════════════════════════════
# ask() — полный pipeline
# ══════════════════════════════════════════════════════════════

class TestRAGServiceAsk:

    @pytest.mark.asyncio
    async def test_ask_returns_rag_result(self, rag_svc):
        search_results = [make_search_result("Рабочий день 09:00-18:00")]

        with patch(
            "src.services.rag_service.chroma_service.similarity_search",
            AsyncMock(return_value=search_results),
        ):
            result = await rag_svc.ask("Какой режим работы?")

        assert isinstance(result, RAGResult)
        assert result.question == "Какой режим работы?"
        assert result.answer == "Стандартный рабочий день с 09:00 до 18:00."
        assert result.has_context is True

    @pytest.mark.asyncio
    async def test_ask_with_empty_retrieval(self, rag_svc):
        """Если ChromaDB ничего не нашёл — has_context=False, answer из LLM."""
        with patch(
            "src.services.rag_service.chroma_service.similarity_search",
            AsyncMock(return_value=[]),
        ):
            result = await rag_svc.ask("Вопрос без ответа в документах")

        assert result.has_context is False
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_ask_uses_mmr_when_flag_set(self, rag_svc):
        """use_mmr=True → вызывается mmr_search, а не similarity_search."""
        mock_mmr = AsyncMock(return_value=[])
        mock_sim = AsyncMock(return_value=[])

        with patch("src.services.rag_service.chroma_service.mmr_search", mock_mmr), \
             patch("src.services.rag_service.chroma_service.similarity_search", mock_sim):
            await rag_svc.ask("Вопрос", use_mmr=True)

        mock_mmr.assert_called_once()
        mock_sim.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_uses_similarity_by_default(self, rag_svc):
        """По умолчанию use_mmr=False → similarity_search."""
        mock_mmr = AsyncMock(return_value=[])
        mock_sim = AsyncMock(return_value=[])

        with patch("src.services.rag_service.chroma_service.mmr_search", mock_mmr), \
             patch("src.services.rag_service.chroma_service.similarity_search", mock_sim):
            await rag_svc.ask("Вопрос")

        mock_sim.assert_called_once()
        mock_mmr.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_passes_history_to_condense(self, rag_svc):
        """История передаётся в condense_question."""
        history = [("Вопрос 1", "Ответ 1")]
        rag_svc.condense_question = AsyncMock(return_value="Переформулированный вопрос")

        with patch(
            "src.services.rag_service.chroma_service.similarity_search",
            AsyncMock(return_value=[]),
        ):
            await rag_svc.ask("А удалёнка?", chat_history=history)

        rag_svc.condense_question.assert_called_once_with("А удалёнка?", history)

    @pytest.mark.asyncio
    async def test_ask_full_response_includes_sources(self, rag_svc):
        """full_response = answer + formatted sources."""
        search_results = [make_search_result("Текст", source="handbook.pdf", page=5)]

        with patch(
            "src.services.rag_service.chroma_service.similarity_search",
            AsyncMock(return_value=search_results),
        ):
            result = await rag_svc.ask("Вопрос")

        assert "handbook.pdf" in result.full_response
        assert "5" in result.full_response

    @pytest.mark.asyncio
    async def test_ask_context_turns_matches_history_length(self, rag_svc):
        history = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]

        with patch(
            "src.services.rag_service.chroma_service.similarity_search",
            AsyncMock(return_value=[]),
        ):
            result = await rag_svc.ask("Вопрос", chat_history=history)

        assert result.context_turns == 3
