from fastapi import APIRouter

router = APIRouter(tags=["training"])


@router.get("/health")
def training_health() -> dict:
    """Simple health endpoint for the training routes."""
    return {"status": "ok"}
