"""
src/api/auth.py

Аутентификация для Admin Panel.

Реализован HTTP Basic Auth через FastAPI Security:
  • Простой, без токенов — подходит для внутреннего инструмента
  • Credentials из .env: ADMIN_USERNAME + ADMIN_PASSWORD
  • Пароль сравнивается через secrets.compare_digest — защита от timing attacks
  • Dependency: просто добавь Depends(require_admin) к любому роутеру

Если позже нужен JWT — заменить только этот модуль,
все роутеры останутся без изменений (Depends-контракт).
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.core.config import settings

security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    FastAPI dependency — проверяет Basic Auth.

    Использование в роутере:
        @router.post("/upload")
        async def upload(
            _: str = Depends(require_admin),
            file: UploadFile = File(...),
        ): ...

    Returns:
        username (str) если аутентификация прошла успешно.

    Raises:
        HTTPException 401 при неверных credentials.
    """
    correct_username = settings.admin_username.encode("utf-8")
    correct_password = settings.admin_password.encode("utf-8")

    given_username = credentials.username.encode("utf-8")
    given_password = credentials.password.encode("utf-8")

    # Оба сравнения выполняются всегда — защита от timing attack
    username_ok = secrets.compare_digest(given_username, correct_username)
    password_ok = secrets.compare_digest(given_password, correct_password)

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учётные данные",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
