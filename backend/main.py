"""MacTuner FastAPI 메인 애플리케이션"""
import os

# HuggingFace tokenizers 병렬 처리 경고 억제
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api import chat_interface, dataset_tools, export_gguf, model_loader, rag_pipeline, training

# FastAPI 앱 생성
app = FastAPI(
    title="MacTuner API",
    description="MAC 환경 최적화 LLM 파인튜닝 & 배포 플랫폼",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS 미들웨어 설정
origins = (
    os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    .split(",")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# API 라우터 등록
# ========================================
app.include_router(model_loader.router, prefix="/model", tags=["Model"])
app.include_router(dataset_tools.router, prefix="/dataset", tags=["Dataset"])
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(chat_interface.router, prefix="/chat", tags=["Chat"])
app.include_router(rag_pipeline.router, prefix="/rag", tags=["RAG"])
app.include_router(export_gguf.router, prefix="/gguf", tags=["GGUF"])

# ========================================
# 추가 라우터 (향후)
# ========================================
# from backend.api import training, chat_interface, rag_pipeline, export_gguf
# app.include_router(training.router, prefix="/train", tags=["Training"])
# app.include_router(chat_interface.router, prefix="/chat", tags=["Chat"])
# app.include_router(rag_pipeline.router, prefix="/rag", tags=["RAG"])
# app.include_router(export_gguf.router, prefix="/export", tags=["Export"])

# ========================================
# 헬스 체크 및 기본 엔드포인트
# ========================================


@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🍎 MacTuner API v0.1.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
        },
        "paths": {
            "data_dir": str(settings.DATA_DIR),
            "output_dir": str(settings.OUTPUT_DIR),
            "log_dir": str(settings.LOG_DIR),
        },
    }


@app.get("/config", tags=["Config"])
async def get_config():
    """현재 설정 정보 반환"""
    return {
        "api": {
            "host": settings.API_HOST,
            "port": settings.API_PORT,
            "debug": settings.API_DEBUG,
        },
        "model": {
            "cache_dir": settings.MODEL_CACHE_DIR,
        },
        "training": {
            "default_batch_size": settings.DEFAULT_BATCH_SIZE,
            "default_learning_rate": settings.DEFAULT_LEARNING_RATE,
            "default_epochs": settings.DEFAULT_EPOCHS,
        },
        "lora": {
            "rank": settings.LORA_RANK,
            "alpha": settings.LORA_ALPHA,
            "dropout": settings.LORA_DROPOUT,
        },
        "rag": {
            "chunk_size": settings.RAG_CHUNK_SIZE,
            "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
            "embedding_model": settings.RAG_EMBEDDING_MODEL,
            "top_k": settings.RAG_TOP_K,
        },
        "optimization": {
            "use_mac_mps": settings.USE_MAC_MPS,
            "gradient_checkpointing": settings.GRADIENT_CHECKPOINTING,
            "mixed_precision": settings.MIXED_PRECISION,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
