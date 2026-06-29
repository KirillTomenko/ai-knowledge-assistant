"""
src/api/main.py

FastAPI приложение — Admin Panel.

Что добавлено по сравнению с начальной версией:
  • /health — детальная проверка всех сервисов (ChromaDB + LLM + Supabase)
  • lifespan — прогрев ChromaService и RAGService при старте
  • /api/v1/ask — RAG эндпоинт для тестирования через Swagger
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.api.auth import require_admin
from src.api.routers import collections, documents
from src.api.schemas.documents import HealthResponse
from src.core.config import settings
from src.services import chroma_service, rag_service


# ── Lifespan: прогрев при старте ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Knowledge Assistant API...")

    # Инициализируем ChromaService (проверяет соединение)
    try:
        await chroma_service._lazy_init()
        logger.info("ChromaService: OK")
    except Exception as e:
        logger.warning(f"ChromaService init failed (non-fatal): {e}")

    yield

    logger.info("Shutting down API...")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="AI Knowledge Assistant — Admin API",
    description="""
RAG-система для корпоративного поиска по документам.

**Аутентификация:** HTTP Basic Auth (admin / password из .env)

**Основной флоу:**
1. `POST /api/v1/documents/upload` — загрузить PDF/DOCX/TXT
2. `GET /api/v1/documents/{id}/status` — дождаться status=done
3. `POST /api/v1/ask` — задать вопрос (или через Telegram бота)
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Роутеры ──────────────────────────────────────────────────

app.include_router(
    documents.router,
    prefix="/api/v1/documents",
    tags=["📄 Documents"],
    dependencies=[Depends(require_admin)],   # весь роутер защищён
)
app.include_router(
    collections.router,
    prefix="/api/v1/collections",
    tags=["🗄 Collections"],
    dependencies=[Depends(require_admin)],
)


# ── Health Check ─────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["🔍 System"],
    summary="Детальный health check всех сервисов",
)
async def health_check() -> HealthResponse:
    """
    Проверяет доступность всех зависимостей:
    - **chromadb**: соединение с векторным хранилищем
    - **llm**: короткий ping к GPT-4o через ProxyAPI
    - **overall**: ok если все сервисы ok, иначе degraded
    """
    import asyncio

    # Запускаем проверки параллельно
    chroma_task = asyncio.create_task(chroma_service.health_check())
    llm_task = asyncio.create_task(rag_service.health_check())

    chroma_status, llm_status = await asyncio.gather(
        chroma_task, llm_task, return_exceptions=True
    )

    # Если gather вернул исключение — оборачиваем в error-dict
    if isinstance(chroma_status, Exception):
        chroma_status = {"status": "error", "detail": str(chroma_status)}
    if isinstance(llm_status, Exception):
        llm_status = {"status": "error", "detail": str(llm_status)}

    services = {
        "chromadb": chroma_status,
        "llm": llm_status,
    }

    all_ok = all(s.get("status") == "ok" for s in services.values())
    overall = "ok" if all_ok else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        services=services,
    )


# ── RAG Ask эндпоинт (для тестирования через Swagger) ────────

class AskRequest(BaseModel):
    question: str
    collection_name: str = settings.chroma_collection_name
    use_mmr: bool = False


class AskResponse(BaseModel):
    question: str
    condensed_question: str
    answer: str
    sources: list[dict]
    retrieval_method: str
    has_context: bool


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    tags=["🤖 RAG"],
    summary="Задать вопрос RAG-системе (тест)",
    dependencies=[Depends(require_admin)],
)
async def ask_question(body: AskRequest) -> AskResponse:
    """
    Прямой RAG-запрос без Telegram. Удобно для тестирования через Swagger UI.
    """
    try:
        result = await rag_service.ask(
            question=body.question,
            collection_name=body.collection_name,
            use_mmr=body.use_mmr,
        )
    except Exception as e:
        logger.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        question=result.question,
        condensed_question=result.condensed_question,
        answer=result.answer,
        sources=[
            {"source": s.source, "page": s.page, "excerpt": s.excerpt}
            for s in result.sources
        ],
        retrieval_method=result.retrieval_method,
        has_context=result.has_context,
    )
