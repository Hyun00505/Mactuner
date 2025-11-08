"""모델 로더 API"""
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import logging
import os
import io
import sys
import threading
from contextlib import redirect_stdout, redirect_stderr

from pydantic import BaseModel, Field

from backend.services.model_service import ModelService
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model"])
model_service = ModelService()

# HuggingFace 라이브러리 로그 캡처용
class LogCapture(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.last_progress = 0
    
    def write(self, message):
        if message and message.strip():
            # "Loading checkpoint shards" 메시지 추출
            if "Loading checkpoint shards" in message:
                # 진행도 추출 (예: "0%", "25%", "50%", "75%", "100%")
                import re
                # 정규표현식으로 백분율 추출
                match = re.search(r'(\d+)%', message)
                if match:
                    try:
                        progress_pct = int(match.group(1))
                        # 중복 진행도 피하기
                        if progress_pct != self.last_progress and progress_pct % 25 == 0:
                            self.last_progress = progress_pct
                            # 진행도를 25-85% 범위로 매핑
                            mapped_progress = 25 + (progress_pct / 100) * 60
                            
                            # 샤드 정보 추출 (예: "2/4")
                            shard_match = re.search(r'(\d+)/(\d+)', message)
                            shard_info = f" {shard_match.group(1)}/{shard_match.group(2)}" if shard_match else ""
                            
                            self.callback({
                                "status": "loading_checkpoint",
                                "message": f"체크포인트 로드 중{shard_info}... {progress_pct}%",
                                "progress": mapped_progress
                            })
                    except Exception as e:
                        logger.debug(f"로그 파싱 오류: {e}")
        return super().write(message)

# 글로벌 모델 캐시
_MODEL_CACHE: Dict = {}


# ========================================
# 요청/응답 모델
# ========================================


class ModelDownloadRequest(BaseModel):
    """모델 다운로드 요청"""

    model_id: str = Field(..., description="Hugging Face 모델 ID (예: gpt2, meta-llama/Llama-2-7b)")
    access_token: Optional[str] = Field(None, description="Hugging Face 액세스 토큰 (선택사항)")


class ModelDownloadResponse(BaseModel):
    """모델 다운로드 응답"""

    status: str = Field(..., description="상태 ('success' 또는 'error')")
    metadata: Dict = Field(..., description="모델 메타데이터")


class ModelInfo(BaseModel):
    """모델 정보"""

    model_id: str
    source: str
    model_type: str
    num_parameters: int
    estimated_memory_gb: float
    device: str
    dtype: str


class ModelListResponse(BaseModel):
    """모델 목록 응답"""

    models: list = Field(..., description="모델 목록")
    count: int = Field(..., description="모델 개수")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def model_health() -> Dict[str, str]:
    """모델 로더 헬스 체크"""
    return {"status": "ok", "service": "model_loader"}


# ========================================
# 모델 다운로드
# ========================================


@router.post("/download", response_model=ModelDownloadResponse)
async def download_model(payload: ModelDownloadRequest) -> ModelDownloadResponse:
    """Hugging Face에서 모델 다운로드"""
    try:
        model, tokenizer, metadata = model_service.load_from_hub(
            payload.model_id, payload.access_token
        )

        # 모델 캐시에 저장 (인메모리)
        _MODEL_CACHE.update(
            {
                "model": model,
                "tokenizer": tokenizer,
                "metadata": metadata,
                "model_id": payload.model_id,
                "source": "hub",
            }
        )

        return ModelDownloadResponse(status="success", metadata=metadata)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"모델 다운로드 실패: {str(e)}",
        )


@router.post("/download-stream")
async def download_model_stream(model_id: str, access_token: str = None):
    """모델 다운로드 (스트리밍 진행상황)"""
    
    def generate():
        try:
            yield json.dumps({"status": "starting", "message": f"모델 다운로드 시작: {model_id}"}).encode() + b'\n'
            
            # 토큰 설정
            hf_token = access_token or os.getenv("HUGGINGFACE_TOKEN", "")
            
            yield json.dumps({"status": "downloading_tokenizer", "progress": 10}).encode() + b'\n'
            
            # 토크나이저 다운로드
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    token=hf_token if hf_token else None,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    trust_remote_code=True,
                )
                yield json.dumps({"status": "tokenizer_done", "progress": 20}).encode() + b'\n'
            except Exception as e:
                yield json.dumps({"status": "error", "message": f"토크나이저 로드 실패: {str(e)}"}).encode() + b'\n'
                return
            
            yield json.dumps({"status": "downloading_model", "progress": 25}).encode() + b'\n'
            
            # 모델 다운로드
            from transformers import AutoModelForCausalLM
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=hf_token if hf_token else None,
                    device_map="auto",
                    torch_dtype="auto",
                    cache_dir=settings.MODEL_CACHE_DIR,
                    trust_remote_code=True,
                )
                yield json.dumps({"status": "model_loaded", "progress": 90}).encode() + b'\n'
            except Exception as e:
                yield json.dumps({"status": "error", "message": f"모델 로드 실패: {str(e)}"}).encode() + b'\n'
                return
            
            # 메타데이터 추출
            num_params = sum(p.numel() for p in model.parameters())
            metadata = {
                "model_id": model_id,
                "model_type": getattr(model.config, "model_type", "unknown"),
                "num_params": num_params,
                "hidden_size": getattr(model.config, "hidden_size", 0),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", 0),
            }
            
            # 모델 캐시에 저장
            _MODEL_CACHE.update({
                "model": model,
                "tokenizer": tokenizer,
                "metadata": metadata,
                "model_id": model_id,
                "source": "hub",
            })
            
            yield json.dumps({
                "status": "completed",
                "progress": 100,
                "message": f"✅ {model_id} 다운로드 완료!",
                "metadata": metadata
            }).encode() + b'\n'
            
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            yield json.dumps({
                "status": "error",
                "message": f"❌ 오류: {str(e)}"
            }).encode() + b'\n'
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ========================================
# 로컬 모델 업로드
# ========================================


@router.post("/upload")
async def upload_model(
    request_body: dict,
) -> ModelDownloadResponse:
    """로컬에서 모델 업로드"""
    try:
        model_path = request_body.get("model_path")
        if not model_path:
            raise ValueError("model_path가 필요합니다")
            
        model, tokenizer, metadata = model_service.load_local(model_path)

        # 모델 캐시에 저장
        _MODEL_CACHE.update(
            {
                "model": model,
                "tokenizer": tokenizer,
                "metadata": metadata,
                "model_id": model_path,
                "source": "local",
            }
        )

        return ModelDownloadResponse(status="success", metadata=metadata)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"모델 업로드 실패: {str(e)}",
        )


@router.post("/upload-stream")
async def upload_model_stream(request_body: dict):
    """로컬에서 모델 업로드 (스트리밍 진행상황)"""
    
    def generate():
        try:
            model_path = request_body.get("model_path")
            if not model_path:
                raise ValueError("model_path가 필요합니다")
            
            yield json.dumps({"status": "starting", "message": f"모델 로드 시작: {model_path}", "progress": 0}).encode() + b'\n'
            
            # GGUF 모델 감지
            is_gguf = model_path.lower().endswith('.gguf')
            
            if is_gguf:
                yield json.dumps({"status": "detecting", "message": "GGUF 모델 감지됨! llama.cpp 사용...", "progress": 10}).encode() + b'\n'
                
                # llama.cpp 사용
                from backend.services.llama_cpp_service import LlamaCppService
                
                llama_service = LlamaCppService()
                
                if not llama_service.is_available():
                    error_msg = "❌ llama-cpp-python이 설치되어 있지 않습니다.\n"
                    error_msg += "GGUF 모델을 사용하려면 다음 명령을 실행해주세요:\n"
                    error_msg += "python -m pip install llama-cpp-python"
                    
                    yield json.dumps({"status": "error", "message": error_msg}).encode() + b'\n'
                    raise Exception(error_msg)
                else:
                    yield json.dumps({"status": "loading_gguf", "message": "llama.cpp로 GGUF 모델 로드 중...", "progress": 20}).encode() + b'\n'
                    
                    load_result = llama_service.load_model(model_path)
                    
                    if load_result["status"] != "success":
                        raise Exception(load_result.get("message", "GGUF 로드 실패"))
                    
                    yield json.dumps({"status": "gguf_loaded", "message": "GGUF 모델 로드 완료!", "progress": 90}).encode() + b'\n'
                    
                    # GGUF 모델을 캐시에 저장
                    _MODEL_CACHE.update({
                        "model": llama_service,
                        "tokenizer": None,  # GGUF는 토크나이저 불필요
                        "metadata": {
                            "model_id": model_path,
                            "model_type": "GGUF",
                            "framework": "llama.cpp",
                            "device": "Metal (Mac GPU) + CPU",
                            "path": model_path,
                        },
                        "model_id": model_path,
                        "source": "local",
                        "is_gguf": True,
                    })
                    
                    yield json.dumps({
                        "status": "completed",
                        "progress": 100,
                        "message": "✅ GGUF 모델 로드 완료!",
                        "is_gguf": True,
                        "model_size": os.path.getsize(model_path) / (1024**3),  # GB 단위로 변환
                        "metadata": {
                            "model_type": "GGUF",
                            "framework": "llama.cpp",
                            "device": "Metal (Mac GPU) + CPU",
                        }
                    }).encode() + b'\n'
                    return
            
            # Transformers 사용 (일반 모델)
            if not is_gguf:
                yield json.dumps({"status": "preparing", "message": "모델 파일 준비 중...", "progress": 15}).encode() + b'\n'
                
                model_service_instance = model_service
                
                # 진행 정보를 실시간으로 스트리밍하기 위한 큐
                progress_queue = []
                
                def stream_progress(progress_data):
                    """진행 정보를 큐에 추가"""
                    progress_queue.append(progress_data)
                
                # 로그 캡처용 콜백
                def log_callback(log_data):
                    """로그에서 진행 정보 추출"""
                    yield_data = {
                        "status": log_data.get("status"),
                        "message": log_data.get("message"),
                        "progress": log_data.get("progress")
                    }
                    progress_queue.append(yield_data)
                
                # 표준 출력/에러를 캡처하는 스트림
                log_capture = LogCapture(log_callback)
                
                # 모델과 토크나이저 로드 (로그 캡처와 함께)
                model = None
                tokenizer = None
                metadata = None
                
                try:
                    with redirect_stdout(log_capture), redirect_stderr(log_capture):
                        model, tokenizer, metadata = model_service_instance.load_local(model_path, stream_progress)
                except Exception as e:
                    logger.error(f"Model load failed during streaming: {str(e)}")
                    
                    # 수집된 진행 정보를 모두 스트리밍
                    for update in progress_queue:
                        yield json.dumps({
                            "status": update.get("status"),
                            "message": update.get("message"),
                            "progress": update.get("progress")
                        }).encode() + b'\n'
                    
                    # 오류 메시지 전송
                    yield json.dumps({
                        "status": "error",
                        "message": f"❌ 모델 로드 실패: {str(e)}"
                    }).encode() + b'\n'
                    return
                
                # 수집된 진행 정보를 모두 스트리밍
                for update in progress_queue:
                    yield json.dumps({
                        "status": update.get("status"),
                        "message": update.get("message"),
                        "progress": update.get("progress")
                    }).encode() + b'\n'
                
                # 모델 로드 성공 확인
                if model is None or tokenizer is None or metadata is None:
                    yield json.dumps({
                        "status": "error",
                        "message": "❌ 모델 로드 실패: 변수가 정의되지 않았습니다"
                    }).encode() + b'\n'
                    return
                
                # 모델 캐시에 저장
                _MODEL_CACHE.update({
                    "model": model,
                    "tokenizer": tokenizer,
                    "metadata": metadata,
                    "model_id": model_path,
                    "source": "local",
                    "is_gguf": False,
                })
                
                yield json.dumps({
                    "status": "completed",
                    "progress": 100,
                    "message": f"✅ 모델 로드 완료!",
                    "is_gguf": False,
                    "model_size": os.path.getsize(model_path) / (1024**3) if os.path.isfile(model_path) else sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(model_path)
                        for filename in filenames
                    ) / (1024**3),
                    "metadata": metadata
                }).encode() + b'\n'
            
        except Exception as e:
            logger.error(f"Model upload failed: {str(e)}")
            yield json.dumps({
                "status": "error",
                "message": f"❌ 오류: {str(e)}"
            }).encode() + b'\n'
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ========================================
# 모델 정보 조회
# ========================================


@router.get("/info/{model_id}")
async def get_model_info(
    model_id: str,
    access_token: Optional[str] = None,
) -> Dict:
    """모델 정보 조회"""
    try:
        info = model_service.get_model_info(model_id, access_token)
        return {"status": "success", "data": info}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"모델 정보 조회 실패: {str(e)}",
        )


# ========================================
# 현재 로드된 모델 정보
# ========================================


@router.get("/current")
async def get_current_model() -> Dict:
    """현재 로드된 모델 정보"""
    if not _MODEL_CACHE:
        raise HTTPException(
            status_code=404,
            detail="로드된 모델이 없습니다. /download 또는 /upload를 사용하세요.",
        )

    return {
        "status": "success",
        "model_id": _MODEL_CACHE.get("model_id"),
        "source": _MODEL_CACHE.get("source"),
        "metadata": _MODEL_CACHE.get("metadata"),
    }


# ========================================
# 로컬 모델 목록
# ========================================


@router.get("/local-models", response_model=ModelListResponse)
async def list_local_models() -> ModelListResponse:
    """로컬에 있는 모델 목록 조회"""
    try:
        models = model_service.list_local_models()
        return ModelListResponse(models=models, count=len(models))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 목록 조회 실패: {str(e)}",
        )


# ========================================
# 모델 언로드
# ========================================


@router.post("/unload")
async def unload_model() -> Dict[str, str]:
    """로드된 모델 언로드"""
    if not _MODEL_CACHE:
        raise HTTPException(
            status_code=404,
            detail="언로드할 모델이 없습니다.",
        )

    _MODEL_CACHE.clear()
    return {"status": "success", "message": "모델이 언로드되었습니다."}


# ========================================
# 모델 관리 (삭제, 폴더 열기)
# ========================================


@router.post("/delete/{model_id}")
async def delete_model(model_id: str) -> Dict[str, str]:
    """모델 삭제"""
    try:
        import shutil
        from pathlib import Path
        
        # model_id를 경로로 변환
        # models--namespace--model-name 형식
        parts = model_id.split("/")
        if len(parts) == 2:
            model_dir_name = f"models--{parts[0]}--{parts[1]}"
        else:
            model_dir_name = model_id
        
        model_dir = Path(settings.MODEL_CACHE_DIR) / model_dir_name
        
        if not model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"모델을 찾을 수 없습니다: {model_id}",
            )
        
        # 모델 삭제
        shutil.rmtree(model_dir)
        
        return {
            "status": "success",
            "message": f"✅ {model_id} 모델이 삭제되었습니다."
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"모델 삭제 실패: {str(e)}",
        )


@router.post("/open-folder")
async def open_model_folder(request_body: dict) -> Dict[str, str]:
    """모델 폴더를 Finder에서 열기"""
    try:
        import subprocess
        from pathlib import Path
        
        model_path = request_body.get("path")
        if not model_path:
            raise ValueError("path가 필요합니다")
        
        model_path = Path(model_path).expanduser()
        
        if not model_path.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {model_path}")
        
        # macOS에서 Finder 열기
        subprocess.Popen(["open", "-R", str(model_path)])
        
        return {
            "status": "success",
            "message": f"✅ Finder에서 폴더를 열었습니다"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"폴더 열기 실패: {str(e)}",
        )


# ========================================
# 로컬 models 폴더 스캔
# ========================================


@router.get("/folder-models")
async def scan_local_models_folder() -> Dict:
    """로컬 models 폴더에서 모델 자동 감지"""
    try:
        from pathlib import Path
        
        # models 폴더 경로
        models_folder = Path(__file__).parent.parent.parent / "models"
        
        if not models_folder.exists():
            return {
                "status": "success",
                "models": [],
                "message": "models 폴더가 존재하지 않습니다"
            }
        
        models = []
        
        # models 폴더의 모든 항목 스캔
        for item in models_folder.iterdir():
            if item.is_dir():
                # 폴더: HuggingFace 모델로 간주
                config_path = item / "config.json"
                if config_path.exists():
                    # 폴더 크기 계산
                    folder_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    size_gb = folder_size / (1024 ** 3)
                    
                    models.append({
                        "model_id": item.name,
                        "path": str(item.absolute()),
                        "size_gb": round(size_gb, 2),
                        "source": "local",
                        "model_type": "HuggingFace"
                    })
            
            elif item.is_file() and item.suffix.lower() == ".gguf":
                # GGUF 파일
                file_size = item.stat().st_size
                size_gb = file_size / (1024 ** 3)
                
                models.append({
                    "model_id": item.stem,  # 파일명 (확장자 제외)
                    "path": str(item.absolute()),
                    "size_gb": round(size_gb, 2),
                    "source": "local",
                    "model_type": "GGUF"
                })
        
        return {
            "status": "success",
            "models": models,
            "message": f"✅ {len(models)}개의 모델을 찾았습니다"
        }
    
    except Exception as e:
        logger.error(f"모델 폴더 스캔 실패: {str(e)}")
        return {
            "status": "error",
            "models": [],
            "message": f"❌ 모델 스캔 실패: {str(e)}"
        }


# ========================================
# 헬퍼 함수
# ========================================


def get_cached_model():
    """캐시된 모델 반환 (다른 라우터에서 사용)"""
    if not _MODEL_CACHE:
        raise RuntimeError(
            "로드된 모델이 없습니다. /model/download 또는 /model/upload를 사용하세요."
        )
    return _MODEL_CACHE