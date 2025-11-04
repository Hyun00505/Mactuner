"""FastAPI routes for model loading operations."""

from __future__ import annotations

from typing import Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from model_loader import load_model
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "model_loader module could not be imported. "
        "Ensure the project root is on PYTHONPATH."
    ) from exc

router = APIRouter(tags=["model"])


class ModelLoadRequest(BaseModel):
    """Request payload for loading a model."""
    model_source: Literal["hub", "local"] = Field(
        ..., description="Whether to pull the model from the Hugging Face hub or local path"
    )
    model_id_or_path: str = Field(..., description="Hugging Face model ID or local path")
    access_token: Optional[str] = Field(None, description="Optional Hugging Face access token")


class ModelLoadResponse(BaseModel):
    """Successful load response payload."""
    status: str
    metadata: Dict[str, object]


_MODEL_CACHE: Dict[str, object] = {}


@router.get("/health")
def model_health() -> Dict[str, str]:
    """Simple health endpoint for the model loader routes."""
    return {"status": "ok"}


@router.post("/load", response_model=ModelLoadResponse)
def load_model_endpoint(payload: ModelLoadRequest) -> ModelLoadResponse:
    """Load a model/tokenizer pair and return metadata for the client."""
    try:
        model, tokenizer, metadata = load_model(
            payload.model_source,
            payload.model_id_or_path,
            payload.access_token,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _MODEL_CACHE.update({
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "model_source": payload.model_source,
        "model_id_or_path": payload.model_id_or_path,
    })

    return ModelLoadResponse(status="success", metadata=metadata)


def get_cached_model() -> Dict[str, object]:
    """Expose cached model artifacts to other routers."""
    if not _MODEL_CACHE:
        raise RuntimeError("No model is currently loaded. Call /model/load first.")
    return _MODEL_CACHE


__all__ = ["router", "get_cached_model"]