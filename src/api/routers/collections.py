"""src/api/routers/collections.py — управление коллекциями ChromaDB."""

from fastapi import APIRouter
from src.services.chroma_service import chroma_service

router = APIRouter()


@router.get("/")
async def list_collections() -> dict:
    """Список коллекций ChromaDB."""
    collections = [item.name for item in await chroma_service.list_collections()]
    return {"collections": collections, "count": len(collections)}
