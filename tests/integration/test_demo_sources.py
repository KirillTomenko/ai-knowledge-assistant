"""Offline regression: real upload/parser/source formatting, external stores mocked.
These tests do not measure semantic retrieval or live LLM quality.
"""
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.rag.parser import document_parser
from src.services.chroma_service import SearchResult
from src.services.rag_service import RAGService
from tests.integration.test_api_documents import make_doc_row

DOCUMENTS = Path(__file__).resolve().parents[2] / "demo" / "documents"

@pytest.mark.asyncio
async def test_uploaded_document_keeps_findable_source(tmp_path):
    filename = "sever_remote_work.txt"
    captured = []

    async def capture(documents, **kwargs):
        captured.extend(documents)
        return ["vector-1"]

    with patch("src.api.routers.documents.UPLOAD_DIR", tmp_path), \
         patch("src.api.routers.documents.documents_repo.get_by_filename", AsyncMock(return_value=None)), \
         patch("src.api.routers.documents.documents_repo.create", AsyncMock(return_value=make_doc_row(filename=filename, file_type="txt"))), \
         patch("src.api.routers.documents.documents_repo.update_status", AsyncMock()) as update, \
         patch("src.api.routers.documents.chroma_service.add_documents", AsyncMock(side_effect=capture)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", auth=("admin", "testpassword")) as client:
            response = await client.post("/api/v1/documents/upload", files={"file": (filename, (DOCUMENTS / filename).read_bytes(), "text/plain")})

    assert response.status_code == 202
    assert update.await_args.kwargs["status"] == "done"
    assert captured
    assert all(chunk.metadata["source"] == filename for chunk in captured)
    source = RAGService()._extract_sources([SearchResult(document=chunk) for chunk in captured])[0]
    original = (DOCUMENTS / source.source).read_text(encoding="utf-8")
    assert source.page == 1  # TXT uses a synthetic page, not physical pagination.
    assert source.excerpt in original
    assert "16:00" in original
    assert any("16:00" in chunk.page_content for chunk in captured)

@pytest.mark.asyncio
async def test_demo_documents_parse_and_expected_passage_survives():
    import json
    cases = json.loads((DOCUMENTS.parent / "questions.json").read_text(encoding="utf-8"))
    for path in DOCUMENTS.glob("*.txt"):
        chunks = await document_parser.parse_file(path)
        assert chunks
        assert all(chunk.metadata["source"] == path.name for chunk in chunks)
        for case in cases:
            if case["expected_source"] == path.name:
                assert any(case["expected_excerpt"] in chunk.page_content for chunk in chunks)
