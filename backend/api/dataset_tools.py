from fastapi import APIRouter

router = APIRouter(tags=["dataset"])


@router.get("/health")
def dataset_health() -> dict:
    """Simple health endpoint for the dataset tools routes."""
    return {"status": "ok"}
