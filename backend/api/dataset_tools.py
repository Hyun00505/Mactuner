"""데이터셋 도구 API"""
from typing import Any, Dict, List, Optional

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
    file_info: Dict[str, Any]  # 모든 타입 허용


class DataInfoResponse(BaseModel):
    """데이터 정보 응답"""

    shape: Dict
    size_mb: float
    dtypes: Dict[str, str]  # dtype을 문자열로
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
    format: str = Field("csv", description="저장 형식 (csv, json, jsonl, excel)")
    encoding: Optional[str] = Field(None, description="파일 인코딩 (utf-8, euc-kr, cp949 등)")


class HFDownloadRequest(BaseModel):
    """HuggingFace 데이터셋 다운로드 요청"""

    dataset_id: str = Field(..., description="HuggingFace 데이터셋 ID (예: 'datasets/username/dataset_name')")
    hf_token: Optional[str] = Field(None, description="HuggingFace API 토큰 (선택사항)")
    split: str = Field("train", description="데이터셋 split (train, validation, test 등)")
    max_samples: Optional[int] = Field(None, description="다운로드할 최대 샘플 수")


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
async def upload_dataset(file: UploadFile = File(...), data_format: str = Query(None)) -> DataUploadResponse:
    """데이터셋 파일 업로드 - 자동 형식 감지"""
    try:
        if not file.filename:
            raise ValueError("파일명이 없습니다.")

        # 파일 읽기
        content = await file.read()

        # 파일명에서 확장자 추출
        original_filename = file.filename.lower()
        
        # 확장자 기반 형식 결정
        if original_filename.endswith(".csv"):
            filename = f"upload.csv"
        elif original_filename.endswith(".json"):
            filename = f"upload.json"
        elif original_filename.endswith(".jsonl"):
            filename = f"upload.jsonl"
        elif original_filename.endswith(".xlsx") or original_filename.endswith(".xls"):
            filename = f"upload.xlsx"
        elif data_format and data_format in ("csv", "json", "jsonl", "excel"):
            # data_format이 명시적으로 주어진 경우
            ext = "xlsx" if data_format == "excel" else data_format
            filename = f"upload.{ext}"
        else:
            # 기본값
            filename = file.filename

        # 데이터셋 로드
        df, file_info = dataset_service.load_dataset(content, filename)
        
        # 히스토리에 추가 (로컬 파일) - 인코딩 정보 포함
        dataset_service.add_to_history(
            source="file", 
            filename=file.filename or filename,
            encoding=dataset_service.file_encoding
        )

        return DataUploadResponse(status="success", file_info=file_info)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 업로드 실패: {str(e)}")


# ========================================
# 데이터 조회
# ========================================


@router.get("/info")
async def get_data_info() -> Dict:
    """로드된 데이터 정보 조회"""
    try:
        info = dataset_service.get_data_info()
        return {"status": "success", "data": info}
    except ValueError as e:
        # 데이터가 로드되지 않은 경우
        return {"status": "no_data", "data": None, "message": str(e)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/preview")
async def get_preview(n_rows: int = Query(5)) -> Dict:
    """데이터 미리보기 - 데이터가 없으면 빈 응답 반환"""
    try:
        preview = dataset_service.get_preview(n_rows)
        return {"status": "success", "data": preview}
    except ValueError as e:
        # 데이터가 로드되지 않은 경우
        return {"status": "no_data", "data": None, "message": str(e)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/full-data")
async def get_full_data(limit: Optional[int] = Query(None), offset: int = Query(0)) -> Dict:
    """완전한 데이터 반환 (페이징 지원)"""
    try:
        # 파라미터 안전 처리
        safe_limit = None if limit is None else max(1, int(limit))
        safe_offset = max(0, int(offset))
        
        print(f"📥 [GET /full-data] limit={safe_limit}, offset={safe_offset}")
        data = dataset_service.get_full_data(limit=safe_limit, offset=safe_offset)
        print(f"✅ [GET /full-data] 반환 데이터: {len(data.get('rows', []))} 행")
        return {"status": "success", "data": data}
    except ValueError as e:
        print(f"❌ [GET /full-data] ValueError: {str(e)}")
        # 데이터가 없으면 빈 결과 반환 (에러 대신)
        return {"status": "no_data", "data": {"rows": [], "total_rows": 0, "columns": [], "dtypes": {}}, "message": str(e)}
    except Exception as e:
        print(f"❌ [GET /full-data] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        # 에러 대신 빈 결과 반환
        return {"status": "no_data", "data": {"rows": [], "total_rows": 0, "columns": [], "dtypes": {}}, "message": str(e)}


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


@router.get("/cached-datasets")
async def get_cached_datasets() -> Dict:
    """HuggingFace 캐시에 있는 다운로드된 데이터셋 목록"""
    try:
        import os
        from pathlib import Path
        
        # HuggingFace 캐시 디렉토리
        hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
        
        if not hf_cache.exists():
            return {"status": "success", "data": []}
        
        cached_datasets = []
        
        # 캐시 디렉토리의 모든 데이터셋 폴더 탐색
        for dataset_dir in hf_cache.iterdir():
            if dataset_dir.is_dir():
                dataset_id = dataset_dir.name
                # datasets--username--dataset-name 형식을 username/dataset-name으로 변환
                if dataset_id.startswith("datasets--"):
                    parts = dataset_id.replace("datasets--", "").split("--")
                    if len(parts) >= 2:
                        dataset_id = "/".join(parts)
                
                # 디렉토리 크기 계산
                size_bytes = sum(
                    f.stat().st_size
                    for f in dataset_dir.rglob("*")
                    if f.is_file()
                )
                size_mb = size_bytes / (1024 ** 2)
                
                cached_datasets.append({
                    "dataset_id": dataset_id,
                    "cache_dir": str(dataset_dir),
                    "size_mb": round(size_mb, 2),
                })
        
        return {"status": "success", "data": cached_datasets}
    except Exception as e:
        return {"status": "error", "data": [], "message": str(e)}


@router.get("/history")
async def get_dataset_history() -> Dict:
    """데이터셋 로드 히스토리 조회"""
    try:
        history = dataset_service.get_history()
        return {"status": "success", "data": history}
    except Exception as e:
        return {"status": "error", "data": [], "message": str(e)}


@router.post("/history/reload/{index}")
async def reload_from_history(index: int) -> Dict:
    """히스토리에서 데이터셋 다시 로드"""
    try:
        history = dataset_service.get_history()
        if index < 0 or index >= len(history):
            raise ValueError("유효하지 않은 인덱스")
        
        item = history[index]
        
        # 소스에 따라 다르게 처리
        if item["source"] == "hf":
            # HuggingFace 데이터셋 다시 로드
            result = dataset_service.download_hf_dataset(
                dataset_id=item.get("hf_dataset_id"),
                hf_token=item.get("hf_token"),
                split=item.get("hf_split", "train"),
                max_samples=item.get("hf_max_samples"),
            )
            return result
        else:
            return {"status": "info", "message": "파일 기반 데이터셋은 재업로드가 필요합니다"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history/clear")
async def clear_history(delete_data: bool = False) -> Dict:
    """히스토리 삭제"""
    try:
        result = dataset_service.clear_history(delete_data=delete_data)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/history/{index}")
async def delete_history_item(index: int, delete_data: bool = False) -> Dict:
    """특정 히스토리 항목 삭제"""
    try:
        result = dataset_service.delete_history_item(index, delete_data=delete_data)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eda/statistics")
async def get_statistics() -> Dict:
    """기본 통계 - 데이터가 없으면 빈 응답 반환"""
    try:
        stats = dataset_service.get_statistics()
        return {"status": "success", "data": stats}
    except ValueError as e:
        # 데이터가 로드되지 않은 경우
        return {"status": "no_data", "data": None, "message": str(e)}
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
        result = dataset_service.save_dataset(request.filepath, request.format, request.encoding)
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


# ========================================
# HuggingFace 데이터셋 다운로드
# ========================================


@router.post("/download-hf")
async def download_hf_dataset(request: HFDownloadRequest) -> Dict:
    """HuggingFace 데이터셋 다운로드"""
    try:
        result = dataset_service.download_hf_dataset(
            dataset_id=request.dataset_id,
            hf_token=request.hf_token,
            split=request.split,
            max_samples=request.max_samples,
        )
        # 히스토리에 추가 - 인코딩 정보 포함
        dataset_service.add_to_history(
            source="hf",
            hf_dataset_id=request.dataset_id,
            hf_token=request.hf_token,
            hf_split=request.split,
            hf_max_samples=request.max_samples,
            encoding=dataset_service.file_encoding,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 로컬 데이터셋 목록
# ========================================


@router.get("/local-datasets")
async def get_local_datasets() -> Dict:
    """로컬에 있는 데이터셋 목록 조회 (히스토리 기반)"""
    try:
        from pathlib import Path
        import json
        
        datasets = []
        history_file = Path(__file__).parent.parent.parent / "data" / "dataset_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                
                # 히스토리 형식 처리 (배열 또는 객체)
                history_list = []
                if isinstance(history_data, list):
                    # 배열 형식
                    history_list = history_data
                elif isinstance(history_data, dict) and "history" in history_data:
                    # 객체 with "history" 키 형식
                    history_list = history_data["history"]
                
                # 히스토리에서 데이터셋 정보 추출
                for item in history_list:
                    # 파일 업로드
                    if item.get("source") in ["file", "upload"]:
                        datasets.append({
                            "dataset_id": item.get("filename") or item.get("file_name", "Unknown"),
                            "source": "upload",
                            "format": item.get("format", "unknown"),
                            "timestamp": item.get("timestamp", ""),
                            "size_mb": item.get("size_mb", 0),
                            "rows": item.get("rows", 0),
                            "columns": item.get("columns", 0),
                        })
                    # HuggingFace 데이터셋
                    elif item.get("source") == "hf":
                        datasets.append({
                            "dataset_id": item.get("hf_dataset_id", "Unknown"),
                            "source": "huggingface",
                            "split": item.get("hf_split", "train"),
                            "timestamp": item.get("timestamp", ""),
                            "rows": item.get("rows", 0),
                            "columns": item.get("columns", 0),
                            "format": item.get("format", "HuggingFace Dataset"),
                        })
                
                print(f"✅ 로컬 데이터셋 조회 완료: {len(datasets)}개 발견")
            except Exception as e:
                print(f"⚠️ 히스토리 파일 읽기 오류: {str(e)}")
        else:
            print(f"⚠️ 히스토리 파일 없음: {history_file}")
        
        return {"status": "success", "datasets": datasets}
    
    except Exception as e:
        print(f"❌ 로컬 데이터셋 조회 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "success", "datasets": []}  # 에러도 빈 배열 반환


@router.post("/load-by-id")
async def load_dataset_by_id(request: Dict) -> Dict:
    """데이터셋 ID로 데이터셋 로드"""
    try:
        dataset_id = request.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id가 필요합니다")
        
        from pathlib import Path
        import json
        
        history_file = Path(__file__).parent.parent.parent / "data" / "dataset_history.json"
        
        if not history_file.exists():
            raise ValueError("히스토리 파일이 없습니다")
        
        with open(history_file, "r") as f:
            history_data = json.load(f)
        
        # 히스토리 형식 처리
        history_list = []
        if isinstance(history_data, list):
            history_list = history_data
        elif isinstance(history_data, dict) and "history" in history_data:
            history_list = history_data["history"]
        
        # 데이터셋 ID로 히스토리에서 찾기
        dataset_item = None
        for item in history_list:
            # 파일 업로드
            if item.get("source") in ["file", "upload"]:
                if item.get("filename") == dataset_id or item.get("file_name") == dataset_id:
                    dataset_item = item
                    break
            # HuggingFace 데이터셋
            elif item.get("source") == "hf":
                if item.get("hf_dataset_id") == dataset_id:
                    dataset_item = item
                    break
        
        if not dataset_item:
            raise ValueError(f"데이터셋 ID '{dataset_id}'를 찾을 수 없습니다")
        
        # 소스에 따라 다르게 처리
        if dataset_item.get("source") == "hf":
            # HuggingFace 데이터셋 다시 로드
            result = dataset_service.download_hf_dataset(
                dataset_id=dataset_item.get("hf_dataset_id"),
                hf_token=dataset_item.get("hf_token"),
                split=dataset_item.get("hf_split", "train"),
                max_samples=dataset_item.get("hf_max_samples"),
            )
            return {"status": "success", "message": f"데이터셋 '{dataset_id}' 로드 완료", "data": result}
        else:
            # 파일 기반 데이터셋은 파일 경로에서 로드
            # 파일 경로가 없으면 에러 (파일은 메모리에만 있으므로 재업로드 필요)
            file_path = dataset_item.get("file_path") or dataset_item.get("filename") or dataset_item.get("file_name")
            if not file_path:
                raise ValueError("파일 경로를 찾을 수 없습니다. 파일 기반 데이터셋은 재업로드가 필요합니다.")
            
            # 파일 경로가 상대 경로인 경우 절대 경로로 변환
            if not Path(file_path).is_absolute():
                # data 폴더에서 찾기
                data_dir = Path(__file__).parent.parent.parent / "data"
                file_path = data_dir / file_path
            
            if not Path(file_path).exists():
                raise ValueError(f"파일을 찾을 수 없습니다: {file_path}. 파일 기반 데이터셋은 재업로드가 필요합니다.")
            
            # 파일 읽기
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            # 데이터셋 로드
            result = dataset_service.load_dataset(file_content, Path(file_path).name)
            return {"status": "success", "message": f"데이터셋 '{dataset_id}' 로드 완료", "data": result[1]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
