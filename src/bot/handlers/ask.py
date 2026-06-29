"""
src/bot/handlers/ask.py

RAG-хендлер: принимает вопрос пользователя, запускает RAG-pipeline,
сохраняет диалог в Supabase.

Использует обновлённый HistoryRepository:
  • get_as_langchain_pairs()  — история в формате LangChain
  • save_pair()               — сохранить пару user/assistant
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger

from src.db.repositories import history_repo
from src.rag.retriever import rag_retriever

router = Router()


@router.message(Command("ask"))
async def handle_ask_command(message: Message) -> None:
    """Обработка команды /ask <вопрос>"""
    question = message.text.removeprefix("/ask").strip()
    if not question:
        await message.answer(
            "Напиши вопрос после команды:\n"
            "`/ask Какой режим рабочего дня?`"
        )
        return
    await _process_rag_query(message, question)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_plain_message(message: Message) -> None:
    """Любое текстовое сообщение обрабатывается как RAG-запрос."""
    await _process_rag_query(message, message.text or "")


async def _process_rag_query(message: Message, question: str) -> None:
    user = message.from_user
    if user is None:
        return

    await message.bot.send_chat_action(message.chat.id, "typing")  # type: ignore

    try:
        # 1. Загружаем историю в формате LangChain [(human, ai), ...]
        chat_history = await history_repo.get_as_langchain_pairs(user.id)

        # 2. RAG-запрос
        result = await rag_retriever.ask(question, chat_history=chat_history)
        answer: str = result["answer"]
        source_docs = result["source_documents"]

        # 3. Форматируем ответ с источниками
        sources_text = _format_sources(source_docs)
        full_answer = answer + sources_text

        await message.answer(full_answer)

        # 4. Сохраняем пару в историю
        metadata = {
            "source_docs": [
                {
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                }
                for d in source_docs
            ]
        }
        await history_repo.save_pair(
            user_id=user.id,
            user_message=question,
            assistant_message=answer,
            username=user.username,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"RAG error for user_id={user.id}: {e}")
        await message.answer("⚠️ Ошибка при обработке запроса. Попробуй позже.")


def _format_sources(source_docs: list) -> str:
    """Форматирует список источников для вывода в Telegram."""
    if not source_docs:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for doc in source_docs:
        src = doc.metadata.get("source", "неизвестный документ")
        page = doc.metadata.get("page", "?")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"• {src}, стр. {page}")
    return "\n\n📎 *Источники:*\n" + "\n".join(lines) if lines else ""
