"""
src/bot/main.py
Точка входа Telegram-бота на aiogram 3.x.

Ключевая особенность: прокси через AiohttpSession + ProxyConnector
для работы в России (Karing SOCKS5 на порту 3067).
"""

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger

from src.bot.handlers import ask, notes, start
from src.bot.middlewares.logging import LoggingMiddleware
from src.core.config import settings


def _build_bot() -> Bot:
    """
    Создаёт экземпляр бота с прокси (если настроен).

    Для SOCKS5 прокси (Karing) используем aiohttp-socks:
        pip install aiohttp-socks
    """
    bot_kwargs: dict = {
        "token": settings.telegram_bot_token,
        "default": DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    }

    if settings.http_proxy:
        from aiohttp import TCPConnector
        from aiohttp_socks import ProxyConnector
        from aiogram.client.session.aiohttp import AiohttpSession

        connector = ProxyConnector.from_url(settings.http_proxy)
        session = AiohttpSession(connector=connector)
        bot_kwargs["session"] = session
        logger.info(f"Bot: using proxy {settings.http_proxy}")

    return Bot(**bot_kwargs)


async def main() -> None:
    bot = _build_bot()
    dp = Dispatcher()

    # Middleware
    dp.message.middleware(LoggingMiddleware())

    # Регистрируем хендлеры
    dp.include_router(start.router)
    dp.include_router(ask.router)
    dp.include_router(notes.router)

    logger.info("Starting bot polling...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
