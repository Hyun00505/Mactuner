from fastapi import APIRouter

router = APIRouter(tags=["model"])


@router.get("/health")
def model_health() -> dict:
    """Simple health endpoint for the model loader routes."""
    return {"status": "ok"}
