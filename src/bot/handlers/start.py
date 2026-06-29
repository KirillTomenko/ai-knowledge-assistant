"""src/bot/handlers/start.py — /start, /help, /clear, /stats"""

from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from src.db.repositories import history_repo, notes_repo

router = Router()

WELCOME_TEXT = """👋 Привет! Я *AI Knowledge Assistant*.

Задай вопрос по корпоративным документам — найду ответ и укажу источник.

*Работа с базой знаний:*
• Просто напиши вопрос — или используй `/ask <вопрос>`

*Заметки:*
• `/note <текст>` — сохранить заметку
• `/notes` — посмотреть все заметки
• `/notes <запрос>` — поиск по заметкам
• `/note_delete <id>` — удалить заметку
• `/clear_notes confirm` — удалить все заметки

*Прочее:*
• `/stats` — твоя статистика
• `/clear` — очистить историю диалога
• `/help` — показать это сообщение

_Пример: «Какой режим рабочего дня в компании?»_"""


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    name = message.from_user.first_name if message.from_user else "Привет"
    await message.answer(f"👋 {name}!\n\n" + WELCOME_TEXT.lstrip("👋 Привет! "))


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(WELCOME_TEXT)


@router.message(Command("clear"))
async def handle_clear_history(message: Message) -> None:
    """Очистить историю диалога пользователя."""
    user = message.from_user
    if user is None:
        return
    count = await history_repo.delete_by_user(user.id)
    await message.answer(f"🗑️ История очищена. Удалено сообщений: {count}.")


@router.message(Command("stats"))
async def handle_stats(message: Message) -> None:
    """Показать статистику пользователя."""
    user = message.from_user
    if user is None:
        return
    msg_count = await history_repo.count_by_user(user.id)
    notes_count = await notes_repo.count_by_user(user.id)
    last_msg = await history_repo.get_last_message(user.id)

    last_active = (
        last_msg.created_at.strftime("%d.%m.%Y %H:%M")
        if last_msg else "нет данных"
    )

    await message.answer(
        f"📊 *Твоя статистика:*\n\n"
        f"• Сообщений в истории: *{msg_count}*\n"
        f"• Заметок: *{notes_count}*\n"
        f"• Последняя активность: _{last_active}_"
    )
