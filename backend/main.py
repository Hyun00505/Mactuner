"""FastAPI entrypoint for the MacTuner backend service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import (
    chat_interface,
    dataset_tools,
    export_gguf,
    model_loader,
    rag_pipeline,
    training,
)

app = FastAPI(title="MacTuner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_loader.router, prefix="/model")
app.include_router(dataset_tools.router, prefix="/dataset")
app.include_router(training.router, prefix="/train")
app.include_router(chat_interface.router, prefix="/chat")
app.include_router(rag_pipeline.router, prefix="/rag")
app.include_router(export_gguf.router, prefix="/export")


@app.get("/health")
def health_check() -> dict:
    """Root health endpoint for service monitoring."""
    return {"status": "ok"}
