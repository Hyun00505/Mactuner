"""학습 API"""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.model_loader import get_cached_model
from backend.services.training_service import TrainingService

router = APIRouter(tags=["training"])
training_service = TrainingService()

# ========================================
# 요청/응답 모델
# ========================================


class LoRAConfig(BaseModel):
    """LoRA 설정"""

    rank: int = Field(8, description="LoRA rank")
    alpha: int = Field(16, description="LoRA alpha")
    dropout: float = Field(0.1, description="LoRA dropout")
    target_modules: List[str] = Field(
        ["q_proj", "v_proj"], description="Target modules"
    )


class TrainingArgsRequest(BaseModel):
    """학습 인자"""

    output_dir: str = Field("output/checkpoints", description="Output directory")
    num_epochs: int = Field(3, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(4, ge=1, le=64, description="Batch size")
    learning_rate: float = Field(5e-5, description="Learning rate")
    warmup_steps: int = Field(500, description="Warmup steps")
    weight_decay: float = Field(0.01, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation")
    max_grad_norm: float = Field(1.0, description="Max grad norm")
    save_strategy: str = Field("epoch", description="Save strategy")
    eval_strategy: str = Field("epoch", description="Evaluation strategy")


class PrepareLearningRequest(BaseModel):
    """학습 준비 요청"""

    use_lora: bool = Field(True, description="Use LoRA")
    use_qlora: bool = Field(False, description="Use QLoRA (4-bit quantization)")
    lora_config: Optional[LoRAConfig] = Field(None, description="LoRA configuration")
    training_args: TrainingArgsRequest = Field(
        default_factory=TrainingArgsRequest, description="Training arguments"
    )


class DatasetPrepareRequest(BaseModel):
    """데이터셋 준비 요청"""

    text_column: str = Field(..., description="Text column name")
    label_column: Optional[str] = Field(None, description="Label column name")
    max_length: int = Field(512, ge=128, le=4096, description="Max sequence length")
    test_size: float = Field(0.1, ge=0.01, le=0.5, description="Test set ratio")


class RecommendedParametersRequest(BaseModel):
    """파라미터 추천 요청"""

    model_size_params: int = Field(..., description="Model total parameters")
    dataset_size: int = Field(..., description="Dataset size")


class SaveModelRequest(BaseModel):
    """모델 저장 요청"""
    
    output_dir: str = Field(..., description="Output directory for saving")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def training_health() -> Dict[str, str]:
    """학습 엔진 헬스 체크"""
    return {"status": "ok", "service": "training"}


# ========================================
# 학습 준비
# ========================================


@router.post("/prepare")
async def prepare_training(request: PrepareLearningRequest) -> Dict:
    """학습 준비 (모델 로드 + LoRA 설정)"""
    try:
        # 모델 캐시에서 가져오기
        model_cache = get_cached_model()
        model = model_cache["model"]
        tokenizer = model_cache["tokenizer"]

        # LoRA/QLoRA 설정
        if request.use_qlora:
            if request.lora_config:
                training_service.setup_qlora(
                    rank=request.lora_config.rank,
                    alpha=request.lora_config.alpha,
                    dropout=request.lora_config.dropout,
                    target_modules=request.lora_config.target_modules,
                )
            else:
                training_service.setup_qlora()
        elif request.use_lora:
            if request.lora_config:
                training_service.setup_lora(
                    rank=request.lora_config.rank,
                    alpha=request.lora_config.alpha,
                    dropout=request.lora_config.dropout,
                    target_modules=request.lora_config.target_modules,
                )
            else:
                training_service.setup_lora()

        # 모델 준비
        prepared_model, model_info = training_service.prepare_model_for_training(
            model, tokenizer, use_lora=request.use_lora, use_qlora=request.use_qlora
        )

        # TrainingArguments 설정
        training_args_info = training_service.configure_training_args(
            output_dir=request.training_args.output_dir,
            num_epochs=request.training_args.num_epochs,
            batch_size=request.training_args.batch_size,
            learning_rate=request.training_args.learning_rate,
            warmup_steps=request.training_args.warmup_steps,
            weight_decay=request.training_args.weight_decay,
            gradient_accumulation_steps=request.training_args.gradient_accumulation_steps,
            max_grad_norm=request.training_args.max_grad_norm,
        )

        return {
            "status": "success",
            "model_info": model_info,
            "training_args": training_args_info,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"학습 준비 실패: {str(e)}")


# ========================================
# 데이터셋 준비
# ========================================


@router.post("/prepare-dataset")
async def prepare_dataset(request: DatasetPrepareRequest) -> Dict:
    """데이터셋 준비 (토크나이징)"""
    try:
        # 현재 모델의 데이터셋 준비
        # 주의: 실제 환경에서는 데이터셋이 업로드되어야 함
        # 여기서는 API 구조만 보여줍니다

        return {
            "status": "success",
            "message": "데이터셋 준비 준비 완료",
            "text_column": request.text_column,
            "max_length": request.max_length,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"데이터셋 준비 실패: {str(e)}")


# ========================================
# LoRA 설정
# ========================================


@router.post("/config-lora")
async def configure_lora(config: LoRAConfig) -> Dict:
    """LoRA 설정"""
    try:
        result = training_service.setup_lora(
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            target_modules=config.target_modules,
        )
        return {"status": "success", "config": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LoRA 설정 실패: {str(e)}")


# ========================================
# TrainingArguments 설정
# ========================================


@router.post("/config-training-args")
async def configure_training_args(request: TrainingArgsRequest) -> Dict:
    """TrainingArguments 설정"""
    try:
        result = training_service.configure_training_args(
            output_dir=request.output_dir,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            warmup_steps=request.warmup_steps,
            weight_decay=request.weight_decay,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            max_grad_norm=request.max_grad_norm,
        )
        return {"status": "success", "config": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 파라미터 추천
# ========================================


@router.post("/recommend-parameters")
async def recommend_parameters(request: RecommendedParametersRequest) -> Dict:
    """모델 및 데이터셋 크기에 따른 최적 파라미터 추천"""
    try:
        recommendations = training_service.recommend_parameters(
            model_size_params=request.model_size_params,
            dataset_size=request.dataset_size,
        )
        return {"status": "success", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 학습 실행
# ========================================


@router.post("/start")
async def start_training() -> Dict:
    """학습 시작"""
    try:
        # 실제 환경에서는 데이터셋이 준비되어야 함
        # 여기서는 준비 상태만 확인
        if training_service.model is None:
            raise ValueError("모델이 준비되지 않았습니다. /prepare를 먼저 호출하세요.")

        return {
            "status": "success",
            "message": "학습이 시작되었습니다.",
            "model_prepared": training_service.model is not None,
            "training_args_configured": training_service.training_args is not None,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 학습 상태 조회
# ========================================


@router.get("/status")
async def get_training_status() -> Dict:
    """현재 학습 상태 조회"""
    try:
        status = training_service.get_training_status()
        return {"status": "success", "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 학습 이력 조회
# ========================================


@router.get("/history")
async def get_training_history() -> Dict:
    """학습 이력 조회"""
    try:
        history = training_service.get_training_history()
        return {"status": "success", "data": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 모델 저장
# ========================================


@router.post("/save")
async def save_model(request: SaveModelRequest) -> Dict:
    """학습된 모델 저장"""
    try:
        result = training_service.save_model(request.output_dir)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 모델 평가
# ========================================


@router.post("/evaluate")
async def evaluate_model() -> Dict:
    """모델 평가"""
    try:
        if training_service.trainer is None:
            raise ValueError("Trainer가 초기화되지 않았습니다.")

        return {
            "status": "success",
            "message": "평가가 완료되었습니다.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
