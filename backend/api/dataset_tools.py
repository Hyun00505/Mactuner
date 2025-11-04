"""데이터셋 도구 API"""
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from pydantic import BaseModel, Field

from backend.services.dataset_service import DatasetService

router = APIRouter(tags=["dataset"])
dataset_service = DatasetService()

# ========================================
# 요청/응답 모델
# ========================================


class DataUploadResponse(BaseModel):
    """데이터 업로드 응답"""

    status: str
    file_info: Dict


class DataInfoResponse(BaseModel):
    """데이터 정보 응답"""

    shape: Dict
    size_mb: float
    dtypes: Dict
    columns: List[str]


class CleaningRequest(BaseModel):
    """데이터 정제 요청"""

    operation: str = Field(..., description="정제 작업 (missing_values, duplicates, normalize_text, filter_by_length)")
    kwargs: Dict = Field(default_factory=dict, description="작업별 파라미터")


class EDASummary(BaseModel):
    """EDA 요약"""

    total_rows: int
    total_columns: int
    memory_mb: float


class SplitRequest(BaseModel):
    """데이터 분할 요청"""

    test_size: float = Field(0.2, ge=0.01, le=0.99)
    random_state: int = Field(42)


class SaveRequest(BaseModel):
    """데이터셋 저장 요청"""

    filepath: str = Field(..., description="저장 경로")
    format: str = Field("csv", description="저장 형식 (csv, json, jsonl)")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def dataset_health() -> Dict[str, str]:
    """데이터셋 도구 헬스 체크"""
    return {"status": "ok", "service": "dataset_tools"}


# ========================================
# 데이터 업로드
# ========================================


@router.post("/upload", response_model=DataUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DataUploadResponse:
    """데이터셋 파일 업로드"""
    try:
        if not file.filename:
            raise ValueError("파일명이 없습니다.")

        # 파일 읽기
        content = await file.read()

        # 데이터셋 로드
        df, file_info = dataset_service.load_dataset(content, file.filename)

        return DataUploadResponse(status="success", file_info=file_info)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 업로드 실패: {str(e)}")


# ========================================
# 데이터 조회
# ========================================


@router.get("/info", response_model=DataInfoResponse)
async def get_data_info() -> DataInfoResponse:
    """로드된 데이터 정보 조회"""
    try:
        info = dataset_service.get_data_info()
        return DataInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/preview")
async def get_preview(n_rows: int = Query(5)) -> Dict:
    """데이터 미리보기"""
    try:
        preview = dataset_service.get_preview(n_rows)
        return {"status": "success", "data": preview}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 데이터 정제
# ========================================


@router.post("/clean")
async def clean_data(request: CleaningRequest) -> Dict:
    """데이터 정제"""
    try:
        df = dataset_service.clean_data(
            cleaning_type=request.operation,
            **request.kwargs
        )
        return {"status": "success", "message": f"정제 완료: {len(df)} 행"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 토큰 분석
# ========================================


@router.post("/analyze-tokens")
async def analyze_tokens(
    column: str = Query(..., description="분석할 텍스트 컬럼"),
    model_name: str = Query("gpt2", description="토크나이저 모델 이름"),
) -> Dict:
    """토큰 길이 분석"""
    try:
        result = dataset_service.analyze_token_length(column, model_name)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# EDA (탐색적 데이터 분석)
# ========================================


@router.get("/eda/statistics")
async def get_statistics() -> Dict:
    """기본 통계"""
    try:
        stats = dataset_service.get_statistics()
        return {"status": "success", "data": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eda/missing-values")
async def get_missing_values() -> Dict:
    """결측치 분석"""
    try:
        missing = dataset_service.get_missing_values()
        return {"status": "success", "data": missing}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eda/value-distribution")
async def get_value_distribution(
    column: str = Query(..., description="분석할 컬럼"),
    top_n: int = Query(10, ge=1, le=100),
) -> Dict:
    """값 분포 분석"""
    try:
        dist = dataset_service.get_value_distribution(column, top_n)
        return {"status": "success", "data": dist}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eda/correlation")
async def get_correlation() -> Dict:
    """상관관계 분석"""
    try:
        corr = dataset_service.get_correlation()
        return {"status": "success", "data": corr}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eda/summary")
async def get_eda_summary() -> Dict:
    """EDA 종합 요약"""
    try:
        stats = dataset_service.get_statistics()
        missing = dataset_service.get_missing_values()

        return {
            "status": "success",
            "summary": {
                "total_rows": stats["total_rows"],
                "total_columns": stats["total_columns"],
                "memory_mb": stats["memory_mb"],
                "missing_values": missing["total_missing_values"],
                "columns": dataset_service.data.columns.tolist(),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 데이터 분할
# ========================================


@router.post("/split")
async def split_dataset(request: SplitRequest) -> Dict:
    """Train/Test 데이터 분할"""
    try:
        train_df, test_df = dataset_service.train_test_split(
            test_size=request.test_size, random_state=request.random_state
        )

        return {
            "status": "success",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_ratio": len(train_df) / (len(train_df) + len(test_df)),
            "test_ratio": len(test_df) / (len(train_df) + len(test_df)),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 데이터 저장
# ========================================


@router.post("/save")
async def save_dataset(request: SaveRequest) -> Dict:
    """데이터셋 저장"""
    try:
        result = dataset_service.save_dataset(request.filepath, request.format)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 데이터 초기화
# ========================================


@router.post("/reset")
async def reset_data() -> Dict[str, str]:
    """데이터를 원본으로 복원"""
    try:
        result = dataset_service.reset_data()
        return {**result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
