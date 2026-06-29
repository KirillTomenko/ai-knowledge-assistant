"""
src/services/rag_service.py

Сервис RAG-pipeline: поиск в ChromaDB → построение промпта → GPT-4o → ответ.

Отвечает за:
  • Построение LLM-клиента (GPT-4o через ProxyAPI, с retry)
  • Переформулировку вопроса с учётом истории диалога (condense step)
  • Retrieval из ChromaDB (similarity / MMR)
  • Сборку промпта с контекстом и историей
  • Streaming-ответа (опционально)
  • Typed результаты через dataclasses

Архитектурное решение — отказ от ConversationalRetrievalChain:
  ┌──────────────────────────────────────────────────────────────┐
  │ LangChain ConversationalRetrievalChain удобна, но чёрный     │
  │ ящик: сложно контролировать промпт, нет streaming из коробки,│
  │ сложная отладка. В этом сервисе реализован явный pipeline:   │
  │                                                              │
  │  1. condense_question()  — переформулировка с историей       │
  │  2. chroma_service.similarity_search() / mmr_search()        │
  │  3. _build_rag_prompt()  — сборка промпта с контекстом       │
  │  4. llm.ainvoke()        — запрос к GPT-4o                   │
  │                                                              │
  │ Это даёт полный контроль над каждым шагом и позволяет        │
  │ добавить streaming, кастомные промпты, A/B тестирование.     │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.services.chroma_service import SearchResult, chroma_service


# ═══════════════════════════════════════════════════════════════
# Typed результаты
# ═══════════════════════════════════════════════════════════════

@dataclass
class SourceReference:
    """Ссылка на источник в ответе RAG."""
    source: str          # имя файла
    page: int | str      # номер страницы
    excerpt: str         # первые 200 символов чанка (для отладки)

    def format(self) -> str:
        """Форматирует ссылку для вывода пользователю."""
        return f"📎 {self.source}, стр. {self.page}"


@dataclass
class RAGResult:
    """
    Полный результат RAG-запроса.

    Содержит ответ, источники и диагностические метаданные.
    """
    question: str                        # исходный вопрос
    condensed_question: str              # переформулированный вопрос (после condense step)
    answer: str                          # ответ GPT-4o
    sources: list[SourceReference]       # дедуплицированные источники
    search_results: list[SearchResult]   # сырые результаты поиска
    retrieval_method: str                # "similarity" | "mmr"
    context_turns: int                   # сколько пар истории было использовано
    has_context: bool                    # нашёл ли контекст в документах

    @property
    def formatted_sources(self) -> str:
        """Строка с источниками для вывода в Telegram."""
        if not self.sources:
            return ""
        lines = [ref.format() for ref in self.sources]
        return "\n\n*Источники:*\n" + "\n".join(lines)

    @property
    def full_response(self) -> str:
        """Полный ответ для пользователя (текст + источники)."""
        return self.answer + self.formatted_sources


# ═══════════════════════════════════════════════════════════════
# Промпты
# ═══════════════════════════════════════════════════════════════

# Системный промпт — инструктирует GPT не галлюцинировать
SYSTEM_PROMPT = """Ты корпоративный ассистент по работе с документами.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из документов.
2. Если в контексте нет ответа — скажи: "В загруженных документах нет информации по этому вопросу."
3. Не придумывай факты и не дополняй ответ из общих знаний.
4. Отвечай на том языке, на котором задан вопрос.
5. Будь лаконичен: сначала прямой ответ, потом детали если нужны.
6. Не упоминай, что работаешь на основе "контекста" или "фрагментов" — просто отвечай."""

# Промпт для переформулировки вопроса с учётом истории
CONDENSE_SYSTEM_PROMPT = """Твоя задача — переформулировать вопрос пользователя.

Если вопрос является уточнением или продолжением предыдущей беседы (использует местоимения "это", "он", "она", "там", "тогда" и т.д.), 
перепиши его как самостоятельный вопрос, добавив весь необходимый контекст из истории.

Если вопрос уже самодостаточен — верни его без изменений.
Отвечай ТОЛЬКО переформулированным вопросом, без пояснений."""

# Шаблон RAG-промпта с контекстом
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="""Контекст из документов:

{context}

---

Вопрос: {question}"""),
])


# ═══════════════════════════════════════════════════════════════
# RAGService
# ═══════════════════════════════════════════════════════════════

class RAGService:
    """
    Сервис RAG-pipeline.

    Singleton: создаётся один раз через RAGService.get_instance().
    LLM-клиент инициализируется лениво при первом вызове ask().

    Пример использования:
        service = RAGService.get_instance()
        result = await service.ask(
            question="Какой режим работы?",
            chat_history=[("Привет", "Привет! Чем помочь?")],
        )
        print(result.full_response)
    """

    # Retry-настройки для LLM вызовов
    MAX_RETRIES: int = 3
    RETRY_WAIT_MIN: float = 1.0
    RETRY_WAIT_MAX: float = 15.0

    # Количество символов из чанка для SourceReference.excerpt
    EXCERPT_LEN: int = 200

    _instance: RAGService | None = None

    def __init__(self) -> None:
        self._llm: ChatOpenAI | None = None
        self._condense_llm: ChatOpenAI | None = None

    @classmethod
    def get_instance(cls) -> "RAGService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── LLM фабрика ───────────────────────────────────────────

    def _build_llm(
        self,
        temperature: float = 0.1,
        streaming: bool = False,
    ) -> ChatOpenAI:
        """
        Создаёт ChatOpenAI с поддержкой прокси.

        Прокси пробрасывается через httpx.AsyncClient — единственный
        рабочий способ для langchain-openai + ProxyAPI + SOCKS5.
        """
        async_client_kwargs: dict[str, Any] = {}
        if settings.https_proxy:
            async_client_kwargs["proxy"] = settings.https_proxy

        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
            temperature=temperature,
            streaming=streaming,
            http_async_client=httpx.AsyncClient(**async_client_kwargs)
            if async_client_kwargs
            else None,
        )

    @property
    def llm(self) -> ChatOpenAI:
        """Основной LLM для генерации ответов (температура 0.1)."""
        if self._llm is None:
            self._llm = self._build_llm(temperature=0.1)
            logger.info(f"RAGService: LLM initialized (model={settings.llm_model})")
        return self._llm

    @property
    def condense_llm(self) -> ChatOpenAI:
        """
        LLM для переформулировки вопроса (температура 0.0).
        Отдельный инстанс — детерминированные переформулировки, не нужна случайность.
        """
        if self._condense_llm is None:
            self._condense_llm = self._build_llm(temperature=0.0)
        return self._condense_llm

    # ── Condense Step ─────────────────────────────────────────

    async def condense_question(
        self,
        question: str,
        chat_history: list[tuple[str, str]],
    ) -> str:
        """
        Переформулирует вопрос с учётом истории диалога.

        Пример:
            История: "Расскажи про режим работы" → "Режим работы — пн-пт, 9-18"
            Вопрос:  "А как насчёт удалёнки?"
            Результат: "Каков порядок удалённой работы в компании?"

        Если истории нет или вопрос самодостаточен — возвращает исходный вопрос.

        Args:
            question:     Текущий вопрос пользователя
            chat_history: Список (human_msg, ai_msg), от старых к новым

        Returns:
            Переформулированный (или исходный) вопрос.
        """
        if not chat_history:
            return question

        # Собираем историю как текст для промпта
        history_text = "\n".join(
            f"Пользователь: {human}\nАссистент: {ai}"
            for human, ai in chat_history[-settings.dialog_history_limit:]
        )

        prompt = [
            SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"История диалога:\n{history_text}\n\n"
                f"Текущий вопрос: {question}\n\n"
                "Переформулированный вопрос:"
            )),
        ]

        try:
            response = await self._call_llm_with_retry(self.condense_llm, prompt)
            condensed = response.content.strip()
            if condensed and condensed != question:
                logger.debug(f"RAGService: condensed '{question}' → '{condensed}'")
            return condensed or question
        except Exception as e:
            # Condense — не критичный шаг, при ошибке используем исходный вопрос
            logger.warning(f"RAGService: condense failed, using original: {e}")
            return question

    # ── Промпт-сборка ─────────────────────────────────────────

    def _build_context_string(self, search_results: list[SearchResult]) -> str:
        """
        Собирает строку контекста из результатов поиска.

        Каждый чанк форматируется с метаданными:
            [Источник: filename.pdf, стр. 3]
            ...текст чанка...

        Это позволяет модели ссылаться на конкретный источник.
        """
        if not search_results:
            return "Релевантных документов не найдено."

        parts: list[str] = []
        for i, result in enumerate(search_results, start=1):
            header = f"[{i}] Источник: {result.source}, стр. {result.page}"
            parts.append(f"{header}\n{result.content}")

        return "\n\n---\n\n".join(parts)

    def _build_history_messages(
        self,
        chat_history: list[tuple[str, str]],
    ) -> list[HumanMessage | AIMessage]:
        """
        Преобразует историю диалога в список LangChain-сообщений.
        Ограничиваем по dialog_history_limit из конфига.
        """
        limited = chat_history[-settings.dialog_history_limit:]
        messages: list[HumanMessage | AIMessage] = []
        for human_msg, ai_msg in limited:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def _extract_sources(self, search_results: list[SearchResult]) -> list[SourceReference]:
        """
        Дедуплицирует и форматирует источники из результатов поиска.
        Порядок сохраняется (первый встреченный уникальный source:page).
        """
        seen: set[str] = set()
        sources: list[SourceReference] = []

        for result in search_results:
            key = f"{result.source}:{result.page}"
            if key not in seen:
                seen.add(key)
                sources.append(SourceReference(
                    source=result.source,
                    page=result.page,
                    excerpt=result.content[:self.EXCERPT_LEN],
                ))

        return sources

    # ── LLM вызовы с retry ───────────────────────────────────

    async def _call_llm_with_retry(
        self,
        llm: ChatOpenAI,
        messages: list,
    ) -> AIMessage:
        """
        Вызывает LLM с экспоненциальным retry при сетевых ошибках.

        Перехватывает:
          • httpx.ConnectError    — нет сети / прокси недоступен
          • httpx.TimeoutException — таймаут
          • httpx.RemoteProtocolError — обрыв соединения
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
                    httpx.RemoteProtocolError,
                    ConnectionError,
                )),
                reraise=True,
            ):
                with attempt:
                    return await llm.ainvoke(messages)  # type: ignore[return-value]

        except RetryError as e:
            raise RuntimeError(
                f"LLM call failed after {self.MAX_RETRIES} retries"
            ) from e

        raise RuntimeError("Unreachable")  # pragma: no cover

    # ── Основной метод ────────────────────────────────────────

    async def ask(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
        collection_name: str = settings.chroma_collection_name,
        use_mmr: bool = False,
        k: int = settings.top_k_results,
    ) -> RAGResult:
        """
        Основной RAG-запрос.

        Pipeline:
            1. condense_question()         — переформулировка с историей
            2. chroma_service.*_search()   — поиск Top-K чанков
            3. _build_context_string()     — сборка контекста
            4. RAG_PROMPT_TEMPLATE         — формирование промпта
            5. llm.ainvoke()               — запрос к GPT-4o
            6. _extract_sources()          — форматирование источников

        Args:
            question:         Вопрос пользователя
            chat_history:     История диалога [(human, ai), ...]
            collection_name:  Коллекция ChromaDB для поиска
            use_mmr:          True → использовать MMR для разнообразия результатов
            k:                Количество чанков для поиска

        Returns:
            RAGResult с полным ответом, источниками и метаданными.

        Raises:
            RuntimeError: если LLM недоступен после всех retry
        """
        history = chat_history or []
        logger.info(
            f"RAGService.ask: '{question[:80]}...' "
            f"(history_turns={len(history)}, collection='{collection_name}', mmr={use_mmr})"
        )

        # ── Шаг 1: Переформулировка вопроса ──────────────────
        condensed_question = await self.condense_question(question, history)

        # ── Шаг 2: Поиск в ChromaDB ───────────────────────────
        if use_mmr:
            search_results = await chroma_service.mmr_search(
                query=condensed_question,
                k=k,
                collection_name=collection_name,
            )
            retrieval_method = "mmr"
        else:
            search_results = await chroma_service.similarity_search(
                query=condensed_question,
                k=k,
                collection_name=collection_name,
            )
            retrieval_method = "similarity"

        has_context = len(search_results) > 0
        logger.debug(
            f"RAGService: retrieved {len(search_results)} chunks "
            f"via {retrieval_method}"
        )

        # ── Шаг 3: Сборка контекста ───────────────────────────
        context = self._build_context_string(search_results)

        # ── Шаг 4: Формирование промпта ───────────────────────
        history_messages = self._build_history_messages(history)

        prompt_messages = RAG_PROMPT_TEMPLATE.format_messages(
            context=context,
            question=condensed_question,
            history=history_messages,
        )

        # ── Шаг 5: Вызов LLM ──────────────────────────────────
        response = await self._call_llm_with_retry(self.llm, prompt_messages)
        answer = response.content.strip()

        # ── Шаг 6: Источники ──────────────────────────────────
        sources = self._extract_sources(search_results)

        result = RAGResult(
            question=question,
            condensed_question=condensed_question,
            answer=answer,
            sources=sources,
            search_results=search_results,
            retrieval_method=retrieval_method,
            context_turns=len(history),
            has_context=has_context,
        )

        logger.info(
            f"RAGService.ask: done "
            f"(answer_len={len(answer)}, sources={len(sources)})"
        )
        return result

    # ── Streaming ─────────────────────────────────────────────

    async def ask_stream(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
        collection_name: str = settings.chroma_collection_name,
        use_mmr: bool = False,
        k: int = settings.top_k_results,
    ) -> AsyncIterator[str]:
        """
        Streaming-версия ask().

        Генерирует токены по мере их получения от GPT-4o.
        Полезно для Telegram, чтобы пользователь видел ответ в реальном времени
        (с aiogram можно обновлять сообщение через bot.edit_message_text).

        Yields:
            Токены (строки) по мере генерации.
            Последний элемент — пустая строка "" как сигнал завершения.

        Usage:
            async for token in rag_service.ask_stream("Вопрос"):
                if token:
                    full_text += token
                    await message.edit_text(full_text)
        """
        history = chat_history or []

        # Шаги 1-4 идентичны ask()
        condensed_question = await self.condense_question(question, history)

        if use_mmr:
            search_results = await chroma_service.mmr_search(
                query=condensed_question, k=k, collection_name=collection_name
            )
        else:
            search_results = await chroma_service.similarity_search(
                query=condensed_question, k=k, collection_name=collection_name
            )

        context = self._build_context_string(search_results)
        history_messages = self._build_history_messages(history)

        prompt_messages = RAG_PROMPT_TEMPLATE.format_messages(
            context=context,
            question=condensed_question,
            history=history_messages,
        )

        # Streaming LLM (отдельный инстанс с streaming=True)
        streaming_llm = self._build_llm(temperature=0.1, streaming=True)

        async for chunk in streaming_llm.astream(prompt_messages):
            if chunk.content:
                yield chunk.content

    # ── Health check ──────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Проверяет доступность LLM (короткий тестовый запрос).
        Используется в /health эндпоинте FastAPI.
        """
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content="ping")]),
                timeout=10.0,
            )
            return {"status": "ok", "model": settings.llm_model}
        except asyncio.TimeoutError:
            return {"status": "error", "detail": "LLM timeout (10s)"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# ── Глобальный синглтон ───────────────────────────────────────

rag_service = RAGService.get_instance()
