"""src/api/routers/collections.py — управление коллекциями ChromaDB."""

from fastapi import APIRouter
from src.rag.embeddings import vector_store

router = APIRouter()


@router.get("/")
async def list_collections() -> dict:
    """Список коллекций ChromaDB."""
    collections = vector_store.list_collections()
    return {"collections": collections, "count": len(collections)}
