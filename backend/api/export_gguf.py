"""GGUF 변환 및 배포 API"""
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.model_loader import get_cached_model
from backend.services.quantization_service import QuantizationService

router = APIRouter(tags=["gguf"])
quantization_service = QuantizationService()

# ========================================
# 요청/응답 모델
# ========================================


class ConvertToGGUFRequest(BaseModel):
    """GGUF 변환 요청"""

    model_path: str = Field(..., description="모델 경로 (로컬)")
    output_dir: str = Field(..., description="출력 디렉토리")
    quantization_method: str = Field("Q4_K", description="양자화 방식")
    use_gpu: bool = Field(True, description="GPU 사용 여부")


class RecommendQuantizationRequest(BaseModel):
    """양자화 방식 추천 요청"""

    model_size_gb: float = Field(..., description="모델 크기 (GB)")
    target_size_gb: Optional[float] = Field(None, description="목표 크기 (GB)")


class ValidateGGUFRequest(BaseModel):
    """GGUF 파일 검증 요청"""

    gguf_path: str = Field(..., description="GGUF 파일 경로")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def gguf_health() -> Dict[str, str]:
    """GGUF 변환 서비스 헬스 체크"""
    return {"status": "ok", "service": "gguf_export"}


# ========================================
# 양자화 방식 조회
# ========================================


@router.get("/methods")
async def get_quantization_methods() -> Dict:
    """지원되는 양자화 방식 조회"""
    try:
        result = quantization_service.get_available_methods()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/methods/recommended")
async def get_recommended_methods(
    model_size_gb: float = Query(..., description="모델 크기 (GB)"),
    target_size_gb: Optional[float] = Query(None, description="목표 크기"),
) -> Dict:
    """권장 양자화 방식 조회"""
    try:
        result = quantization_service.get_recommended_method(
            model_size_gb=model_size_gb,
            target_size_gb=target_size_gb,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# GGUF 변환
# ========================================


@router.post("/convert")
async def convert_to_gguf(request: ConvertToGGUFRequest) -> Dict:
    """모델을 GGUF 형식으로 변환"""
    try:
        result = quantization_service.convert_to_gguf(
            model_path=request.model_path,
            output_dir=request.output_dir,
            quantization_method=request.quantization_method,
            use_gpu=request.use_gpu,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"GGUF 변환 실패: {str(e)}")


# ========================================
# GGUF 검증
# ========================================


@router.post("/validate")
async def validate_gguf(request: ValidateGGUFRequest) -> Dict:
    """GGUF 파일 유효성 검증"""
    try:
        result = quantization_service.validate_gguf(request.gguf_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/validate/{file_path:path}")
async def validate_gguf_by_path(file_path: str) -> Dict:
    """경로로 GGUF 파일 검증"""
    try:
        result = quantization_service.validate_gguf(file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 변환 이력
# ========================================


@router.get("/history")
async def get_conversion_history() -> Dict:
    """변환 이력 조회"""
    try:
        result = quantization_service.get_conversion_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history/clear")
async def clear_conversion_history() -> Dict:
    """변환 이력 초기화"""
    try:
        result = quantization_service.clear_conversion_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 압축 통계
# ========================================


@router.get("/statistics")
async def get_compression_statistics() -> Dict:
    """압축 통계 조회"""
    try:
        result = quantization_service.get_compression_statistics()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
