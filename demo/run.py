"""Record this demo against the real API; never substitute mock model answers."""
import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.core.config import settings
from src.rag.parser import document_parser

ROOT = Path(__file__).resolve().parent


def error_summary(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__  # Do not copy credentials or remote bodies into reports.


async def run(args):
    cases = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if args.live else "preflight",
        "status": "not_run",
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "collection": settings.chroma_collection_name,
        "documents": [],
        "cases": [{"id": c["id"], "question": c["question"], "status": "not_run",
                   "actual_answer": None, "actual_sources": [], "source_checks": []}
                  for c in cases],
        "review": "Source checks only locate excerpts. A human must verify answer support and refusal.",
    }
    for path in sorted((ROOT / "documents").glob("*.txt")):
        chunks = await document_parser.parse_file(path)
        report["documents"].append({"filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "chunks": len(chunks),
            "expected_passages_preserved": all(
                any(c["expected_excerpt"] in chunk.page_content for chunk in chunks)
                for c in cases if c["expected_source"] == path.name)})
    required = {"OPENAI_API_KEY": settings.openai_api_key,
                "SUPABASE_URL": settings.supabase_url, "SUPABASE_KEY": settings.supabase_key,
                "ADMIN_PASSWORD": settings.admin_password}
    report["missing_settings"] = [key for key, value in required.items()
        if not value or "your_" in value or "your-project" in value or "change_me" in value]
    async with httpx.AsyncClient(base_url=args.api, timeout=10, trust_env=False) as client:
        try:
            response = await client.get("/openapi.json")
            response.raise_for_status()
            report["api_openapi"] = response.status_code
        except Exception as exc:
            report["api_openapi"] = error_summary(exc)
    if report["missing_settings"]:
        report["status"] = "blocked_missing_settings"
        for case in report["cases"]:
            case["status"] = "not_run_missing_settings"
    elif not args.live:
        report["status"] = "ready_for_live_run"
    else:
        try:
            async with httpx.AsyncClient(base_url=args.api, timeout=120, trust_env=False,
                    auth=(settings.admin_username, settings.admin_password)) as client:
                # Use a fresh demo collection and a test Supabase project. Never delete existing data.
                for doc in report["documents"]:
                    path = ROOT / "documents" / doc["filename"]
                    response = await client.post("/api/v1/documents/upload",
                        files={"file": (path.name, path.read_bytes(), "text/plain")})
                    response.raise_for_status()  # On 409, stop; do not silently reuse an old index.
                    doc["id"] = response.json()["id"]
                    deadline = time.monotonic() + 180
                    while True:
                        response = await client.get(f"/api/v1/documents/{doc['id']}/status")
                        response.raise_for_status()
                        doc["indexing_status"] = response.json()["status"]
                        if doc["indexing_status"] == "done":
                            break
                        if doc["indexing_status"] == "error" or time.monotonic() > deadline:
                            raise RuntimeError("Indexing failed or timed out")
                        await asyncio.sleep(1)
                for case in report["cases"]:
                    try:
                        response = await client.post("/api/v1/ask", json={
                            "question": case["question"], "collection_name": settings.chroma_collection_name})
                        case["http_status"] = response.status_code
                        response.raise_for_status()
                        result = response.json()
                        case.update(status="needs_human_review", actual_answer=result["answer"],
                            actual_sources=result["sources"], has_context=result["has_context"])
                        for source in result["sources"]:
                            known = next((d for d in report["documents"] if d["filename"] == source["source"]), None)
                            text = (ROOT / "documents" / known["filename"]).read_text(encoding="utf-8") if known else ""
                            case["source_checks"].append({"source": source["source"],
                                "file_found": known is not None,
                                "excerpt_found": bool(source["excerpt"]) and source["excerpt"] in text})
                    except Exception as exc:
                        case.update(status="request_failed", error=error_summary(exc))
                report["status"] = "needs_human_review" if all(c["status"] == "needs_human_review" for c in report["cases"]) else "request_failed"
        except Exception as exc:
            report.update(status="upload_or_indexing_failed", error=error_summary(exc))
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}; status: {report['status']}")
    return 0 if report["status"] in {"ready_for_live_run", "needs_human_review"} else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--live", action="store_true", help="Upload demo documents and use paid embedding/LLM APIs")
    parser.add_argument("--output", default="demo/results.json")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
