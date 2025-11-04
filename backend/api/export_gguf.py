from fastapi import APIRouter

router = APIRouter(tags=["export"])


@router.get("/health")
def export_health() -> dict:
    """Simple health endpoint for the export routes."""
    return {"status": "ok"}
