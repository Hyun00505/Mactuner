from fastapi import APIRouter

router = APIRouter(tags=["rag"])


@router.get("/health")
def rag_health() -> dict:
    """Simple health endpoint for the RAG pipeline routes."""
    return {"status": "ok"}
