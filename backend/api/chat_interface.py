from fastapi import APIRouter

router = APIRouter(tags=["chat"])


@router.get("/health")
def chat_health() -> dict:
    """Simple health endpoint for the chat interface routes."""
    return {"status": "ok"}
