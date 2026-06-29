"""src/bot/middlewares/logging.py — логирование всех входящих сообщений."""

from typing import Any, Awaitable, Callable
from aiogram import BaseMiddleware
from aiogram.types import Message
from loguru import logger


class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        user = event.from_user
        logger.info(
            f"Message from {user.id} (@{user.username}): {event.text[:80] if event.text else '[non-text]'}"
        )
        return await handler(event, data)
