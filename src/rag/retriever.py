"""
src/rag/retriever.py
Основной RAG-pipeline: поиск в ChromaDB → промпт → GPT-4o → ответ.

Использует ConversationalRetrievalChain для поддержки истории диалога.
"""

import httpx
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from src.core.config import settings
from src.rag.embeddings import vector_store


# Системный промпт — инструктируем модель не галлюцинировать
SYSTEM_PROMPT = """Ты корпоративный ассистент. Отвечай ТОЛЬКО на основе предоставленного контекста из документов.
Если ответа в контексте нет — прямо скажи: "В загруженных документах нет информации по этому вопросу."
Всегда указывай источник (имя файла и страницу) в конце ответа в формате: [Источник: filename.pdf, стр. X].
Отвечай на том языке, на котором задан вопрос."""


def _build_llm() -> ChatOpenAI:
    """Создаёт LLM с поддержкой прокси."""
    http_client_kwargs: dict = {}
    if settings.https_proxy:
        http_client_kwargs["proxy"] = settings.https_proxy

    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        temperature=0.1,   # низкая температура для фактических ответов
        http_async_client=httpx.AsyncClient(**http_client_kwargs) if http_client_kwargs else None,
    )


class RAGRetriever:
    """
    RAG Query Pipeline.

    Поток:
    1. История диалога (последние N сообщений из Supabase)
    2. ConversationalRetrievalChain переформулирует вопрос с учётом истории
    3. ChromaDB: Top-K similarity search
    4. GPT-4o: генерация ответа на основе retrieved контекста
    """

    def __init__(self) -> None:
        self.llm = _build_llm()

    async def ask(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
        collection_name: str = settings.chroma_collection_name,
    ) -> dict:
        """
        Задаёт вопрос RAG-системе.

        Args:
            question: Вопрос пользователя
            chat_history: Список (user_message, assistant_message) для контекста
            collection_name: Коллекция ChromaDB

        Returns:
            {"answer": str, "source_documents": list[Document]}
        """
        vectorstore = vector_store.get_vectorstore(collection_name)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.top_k_results},
        )

        # Формируем историю для LangChain
        lc_history = []
        if chat_history:
            for human_msg, ai_msg in chat_history[-settings.dialog_history_limit:]:
                lc_history.append(HumanMessage(content=human_msg))
                lc_history.append(AIMessage(content=ai_msg))

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
        )

        logger.debug(f"RAG query: '{question}' with {len(lc_history)//2} history turns")

        result = await chain.ainvoke({
            "question": question,
            "chat_history": lc_history,
        })

        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", []),
        }


# Глобальный инстанс
rag_retriever = RAGRetriever()
