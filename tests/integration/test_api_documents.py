"""
tests/integration/test_api_documents.py

Integration-тесты для FastAPI эндпоинтов документов.
Используем httpx.AsyncClient с ASGITransport — реальные HTTP-запросы
через ASGI интерфейс без поднятия сервера.

Supabase и ChromaDB замоканы через patch.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from src.db.models import DocumentRow


# ── Хелперы ───────────────────────────────────────────────────

def make_doc_row(**kwargs) -> DocumentRow:
    base = {
        "id": "doc-uuid-1234",
        "filename": "test.pdf",
        "file_type": "pdf",
        "file_size": 10240,
        "status": "pending",
        "chunk_count": 0,
        "chroma_ids": None,
        "error_msg": None,
        "created_at": datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
    }
    base.update(kwargs)
    return DocumentRow(**base)


ADMIN_AUTH = ("admin", "testpassword")   # из mock_env фикстуры


@pytest.fixture
async def client():
    from src.api.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        auth=ADMIN_AUTH,
    ) as c:
        yield c


# ══════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        chroma_ok = {"status": "ok", "version": "0.5.20", "collections_count": 1}
        llm_ok = {"status": "ok", "model": "gpt-4o"}

        with patch("src.api.main.chroma_service.health_check", AsyncMock(return_value=chroma_ok)), \
             patch("src.api.main.rag_service.health_check", AsyncMock(return_value=llm_ok)):
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "chromadb" in data["services"]
        assert "llm" in data["services"]

    @pytest.mark.asyncio
    async def test_health_degraded_when_chroma_down(self, client):
        chroma_err = {"status": "error", "detail": "Connection refused"}
        llm_ok = {"status": "ok", "model": "gpt-4o"}

        with patch("src.api.main.chroma_service.health_check", AsyncMock(return_value=chroma_err)), \
             patch("src.api.main.rag_service.health_check", AsyncMock(return_value=llm_ok)):
            resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self):
        """Health check публичный — без авторизации."""
        from src.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            with patch("src.api.main.chroma_service.health_check", AsyncMock(return_value={"status": "ok"})), \
                 patch("src.api.main.rag_service.health_check", AsyncMock(return_value={"status": "ok"})):
                resp = await c.get("/health")

        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════

class TestAuth:

    @pytest.mark.asyncio
    async def test_upload_without_auth_returns_401(self):
        from src.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            resp = await c.post("/api/v1/documents/upload")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_upload_with_wrong_password_returns_401(self):
        from src.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            auth=("admin", "wrongpassword"),
        ) as c:
            resp = await c.post("/api/v1/documents/upload")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_list_documents_with_correct_auth_passes(self, client):
        with patch(
            "src.api.routers.documents.documents_repo.list_all",
            AsyncMock(return_value=[]),
        ):
            resp = await client.get("/api/v1/documents/")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# POST /api/v1/documents/upload
# ══════════════════════════════════════════════════════════════

class TestDocumentsUpload:

    @pytest.mark.asyncio
    async def test_upload_pdf_returns_202(self, client):
        mock_doc = make_doc_row()

        with patch("src.api.routers.documents.documents_repo.get_by_filename",
                   AsyncMock(return_value=None)), \
             patch("src.api.routers.documents.documents_repo.create",
                   AsyncMock(return_value=mock_doc)), \
             patch("src.api.routers.documents._run_indexing", AsyncMock()):

            pdf_content = b"%PDF-1.4 fake content"
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("handbook.pdf", io.BytesIO(pdf_content), "application/pdf")},
            )

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert data["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_upload_unsupported_format_returns_400(self, client):
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert resp.status_code == 400
        assert "Неподдерживаемый тип файла" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_duplicate_returns_409(self, client):
        existing = make_doc_row(status="done")

        with patch("src.api.routers.documents.documents_repo.get_by_filename",
                   AsyncMock(return_value=existing)):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
            )

        assert resp.status_code == 409
        assert "уже проиндексирован" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_txt_file(self, client):
        mock_doc = make_doc_row(filename="notes.txt", file_type="txt")

        with patch("src.api.routers.documents.documents_repo.get_by_filename",
                   AsyncMock(return_value=None)), \
             patch("src.api.routers.documents.documents_repo.create",
                   AsyncMock(return_value=mock_doc)), \
             patch("src.api.routers.documents._run_indexing", AsyncMock()):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
            )

        assert resp.status_code == 202


# ══════════════════════════════════════════════════════════════
# GET /api/v1/documents/{id}/status
# ══════════════════════════════════════════════════════════════

class TestDocumentsStatus:

    @pytest.mark.asyncio
    async def test_status_done(self, client):
        doc = make_doc_row(status="done", chunk_count=42)

        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=doc)):
            resp = await client.get("/api/v1/documents/doc-uuid-1234/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["chunk_count"] == 42

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, client):
        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=None)):
            resp = await client.get("/api/v1/documents/missing-id/status")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_status_error_includes_error_msg(self, client):
        doc = make_doc_row(status="error", error_msg="Parsing failed")

        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=doc)):
            resp = await client.get("/api/v1/documents/doc-uuid-1234/status")

        assert resp.status_code == 200
        assert resp.json()["error_msg"] == "Parsing failed"


# ══════════════════════════════════════════════════════════════
# DELETE /api/v1/documents/{id}
# ══════════════════════════════════════════════════════════════

class TestDocumentsDelete:

    @pytest.mark.asyncio
    async def test_delete_returns_204(self, client):
        doc = make_doc_row(status="done")

        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=doc)), \
             patch("src.api.routers.documents.chroma_service.delete_by_doc_id",
                   AsyncMock(return_value=42)), \
             patch("src.api.routers.documents.documents_repo.delete",
                   AsyncMock(return_value=True)):
            resp = await client.delete("/api/v1/documents/doc-uuid-1234")

        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_processing_returns_409(self, client):
        doc = make_doc_row(status="processing")

        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=doc)):
            resp = await client.delete("/api/v1/documents/doc-uuid-1234")

        assert resp.status_code == 409
        assert "в процессе индексации" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_404(self, client):
        with patch("src.api.routers.documents.documents_repo.get_by_id",
                   AsyncMock(return_value=None)):
            resp = await client.delete("/api/v1/documents/missing")

        assert resp.status_code == 404
