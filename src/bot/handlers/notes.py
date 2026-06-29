"""
src/bot/handlers/notes.py

Хендлеры для работы с персональными заметками.

Команды:
  /note <текст>       — сохранить заметку
  /notes              — посмотреть все заметки (с пагинацией)
  /notes <запрос>     — поиск по заметкам
  /note_delete <N>    — удалить заметку #N из списка
  /clear_notes        — удалить все заметки (с подтверждением)
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger

from src.db.models import NoteUpdate
from src.db.repositories import notes_repo

router = Router()

NOTES_PAGE_SIZE = 5  # заметок на одну страницу


@router.message(Command("note"))
async def handle_save_note(message: Message) -> None:
    """Сохранить заметку: /note <текст>"""
    user = message.from_user
    if user is None:
        return

    content = message.text.removeprefix("/note").strip()
    if not content:
        await message.answer(
            "Напиши текст заметки после команды:\n"
            "`/note Важно: удалёнка только по согласованию с руководителем`"
        )
        return

    note = await notes_repo.create(user_id=user.id, content=content)
    await message.answer(f"✅ Заметка сохранена!\n\n_{note.content}_")


@router.message(Command("notes"))
async def handle_list_notes(message: Message) -> None:
    """
    Показать заметки: /notes          — все заметки
                      /notes <запрос> — поиск по тексту
    """
    user = message.from_user
    if user is None:
        return

    query = message.text.removeprefix("/notes").strip()

    if query:
        # Режим поиска
        notes = await notes_repo.search(user_id=user.id, query=query, limit=10)
        if not notes:
            await message.answer(f"🔍 По запросу «{query}» заметок не найдено.")
            return
        header = f"🔍 *Результаты поиска «{query}»:*\n\n"
    else:
        # Режим просмотра
        total = await notes_repo.count_by_user(user.id)
        if total == 0:
            await message.answer(
                "У тебя пока нет заметок.\n"
                "Создай первую: `/note текст заметки`"
            )
            return
        notes = await notes_repo.get_by_user(user_id=user.id, limit=NOTES_PAGE_SIZE)
        header = f"📝 *Твои заметки* (всего {total}):\n\n"

    lines = [header]
    for i, note in enumerate(notes, start=1):
        date_str = note.created_at.strftime("%d.%m.%Y")
        title_part = f"*{note.title}*\n" if note.title else ""
        source_part = f"\n   _из: {note.source_document}_" if note.source_document else ""
        lines.append(
            f"*{i}.* {title_part}{note.content}{source_part}\n"
            f"   _{date_str}_\n"
        )

    if not query and await notes_repo.count_by_user(user.id) > NOTES_PAGE_SIZE:
        lines.append(f"\n_Показаны первые {NOTES_PAGE_SIZE}. Для поиска: /notes <текст>_")

    await message.answer("\n".join(lines))


@router.message(Command("note_delete"))
async def handle_delete_note(message: Message) -> None:
    """
    Удалить заметку: /note_delete <UUID>

    UUID заметки можно узнать — в будущем через inline-кнопки.
    Пока принимаем напрямую.
    """
    user = message.from_user
    if user is None:
        return

    note_id = message.text.removeprefix("/note_delete").strip()
    if not note_id:
        await message.answer("Укажи ID заметки: `/note_delete <uuid>`")
        return

    deleted = await notes_repo.delete(note_id=note_id, user_id=user.id)
    if deleted:
        await message.answer("🗑️ Заметка удалена.")
    else:
        await message.answer("❌ Заметка не найдена или не принадлежит тебе.")


@router.message(Command("clear_notes"))
async def handle_clear_notes(message: Message) -> None:
    """Удалить все заметки: /clear_notes"""
    user = message.from_user
    if user is None:
        return

    count = await notes_repo.count_by_user(user.id)
    if count == 0:
        await message.answer("У тебя нет заметок для удаления.")
        return

    # Простое подтверждение через повторную команду с флагом
    if "confirm" not in (message.text or "").lower():
        await message.answer(
            f"⚠️ Будет удалено *{count}* заметок.\n"
            "Это действие необратимо.\n\n"
            "Для подтверждения отправь: `/clear_notes confirm`"
        )
        return

    deleted = await notes_repo.delete_all_by_user(user.id)
    await message.answer(f"🗑️ Удалено заметок: {deleted}.")
