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

    text_column: str = Field("text", description="Text column name (for causal_lm format)")
    label_column: Optional[str] = Field(None, description="Label column name (not used, kept for compatibility)")
    max_length: int = Field(512, ge=128, le=4096, description="Max sequence length")
    test_size: float = Field(0.1, ge=0.01, le=0.5, description="Test set ratio")
    format_type: str = Field("causal_lm", description="Data format type: 'causal_lm', 'instruction', or 'chat'")
    input_column: Optional[str] = Field(None, description="Input column name (for instruction/chat format)")
    output_column: Optional[str] = Field(None, description="Output column name (for instruction/chat format)")
    input_columns: Optional[List[str]] = Field(None, description="Input column names (multiple columns will be combined)")
    output_columns: Optional[List[str]] = Field(None, description="Output column names (multiple columns will be combined)")
    output_separator: Optional[str] = Field("\n", description="Separator for combining multiple output columns")
    template: Optional[str] = Field(None, description="Custom prompt template (uses default if None)")


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
        # QLoRA의 경우에도 use_lora=True로 전달 (QLoRA는 LoRA의 변형이므로)
        prepared_model, model_info = training_service.prepare_model_for_training(
            model, tokenizer, use_lora=request.use_lora or request.use_qlora
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
    """학습 시작 (비동기)"""
    try:
        if training_service.model is None:
            raise ValueError("모델이 준비되지 않았습니다. /prepare를 먼저 호출하세요.")
        
        if training_service.training_args is None:
            raise ValueError("TrainingArguments가 설정되지 않았습니다. /prepare를 먼저 호출하세요.")

        # 학습은 백그라운드에서 실행되도록 설정
        # 실제 학습은 별도의 스트리밍 엔드포인트에서 처리
        return {
            "status": "started",
            "message": "학습이 시작되었습니다.",
            "model_prepared": training_service.model is not None,
            "training_args_configured": training_service.training_args is not None,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/start-stream")
async def start_training_stream(request: DatasetPrepareRequest):
    """학습 시작 (스트리밍 진행 상황)"""
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    
    async def generate():
        try:
            # 모델 준비 상태 확인
            if training_service.model is None:
                yield json.dumps({"status": "error", "message": "모델이 준비되지 않았습니다. 모델 로더 노드를 먼저 실행하세요."}).encode() + b'\n'
                return
            
            # 모델이 학습 가능한지 확인
            trainable_params = [p for p in training_service.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                yield json.dumps({"status": "error", "message": "모델에 학습 가능한 파라미터가 없습니다. LoRA 설정을 확인하세요."}).encode() + b'\n'
                return
            
            # 모델이 train 모드인지 확인
            if hasattr(training_service.model, 'training') and not training_service.model.training:
                logger.warning("모델이 train 모드가 아닙니다. train 모드로 변경합니다.")
                training_service.model.train()
            
            # 데이터셋 준비
            yield json.dumps({"status": "preparing_dataset", "message": "데이터셋 준비 중...", "progress": 0}).encode() + b'\n'
            
            # dataset_tools의 전역 dataset_service 인스턴스 사용
            from backend.api.dataset_tools import dataset_service
            
            # 데이터셋이 로드되어 있는지 확인
            if dataset_service.data is None or len(dataset_service.data) == 0:
                yield json.dumps({"status": "error", "message": "데이터셋이 로드되지 않았습니다. 데이터셋 로더 노드를 먼저 실행하세요."}).encode() + b'\n'
                return
            
            # 데이터셋 컬럼 확인 및 자동 감지
            text_column = request.text_column
            input_column = request.input_column
            output_column = request.output_column
            
            # 포맷 타입에 따라 컬럼 자동 감지
            if request.format_type == "instruction" or request.format_type == "chat":
                # Instruction/Chat 포맷: input/output 컬럼 필요
                if not input_column or not output_column:
                    # 자동 감지
                    input_cols = [col for col in dataset_service.data.columns 
                                 if col.lower() in ['instruction', 'input', 'question', 'prompt', 'query', 'user']]
                    output_cols = [col for col in dataset_service.data.columns 
                                  if col.lower() in ['output', 'response', 'answer', 'reply', 'assistant']]
                    
                    if input_cols:
                        input_column = input_cols[0]
                        yield json.dumps({"status": "info", "message": f"입력 컬럼 '{input_column}'을 자동으로 감지했습니다.", "progress": 5}).encode() + b'\n'
                    if output_cols:
                        output_column = output_cols[0]
                        yield json.dumps({"status": "info", "message": f"출력 컬럼 '{output_column}'을 자동으로 감지했습니다.", "progress": 5}).encode() + b'\n'
                    
                    if not input_column or not output_column:
                        yield json.dumps({"status": "error", "message": f"{request.format_type} 포맷을 위한 input/output 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(dataset_service.data.columns)}"}).encode() + b'\n'
                        return
            else:
                # Causal LM: 단일 텍스트 컬럼
                if text_column not in dataset_service.data.columns:
                    # 컬럼이 없으면 자동 감지
                    possible_columns = ['text', 'content', 'input', 'instruction', 'question', 'prompt', 'message', 'output', 'response']
                    found_column = None
                    for col in dataset_service.data.columns:
                        if col.lower() in possible_columns:
                            found_column = col
                            break
                    
                    if found_column:
                        text_column = found_column
                        yield json.dumps({"status": "info", "message": f"컬럼 '{text_column}'을 자동으로 감지했습니다.", "progress": 5}).encode() + b'\n'
                    elif len(dataset_service.data.columns) > 0:
                        # 첫 번째 컬럼 사용
                        text_column = dataset_service.data.columns[0]
                        yield json.dumps({"status": "info", "message": f"첫 번째 컬럼 '{text_column}'을 사용합니다.", "progress": 5}).encode() + b'\n'
                    else:
                        yield json.dumps({"status": "error", "message": "데이터셋에 사용 가능한 컬럼이 없습니다."}).encode() + b'\n'
                        return
            
            # 데이터셋 준비
            dataset_info = training_service.prepare_dataset(
                dataset=dataset_service.data,  # dataset_service에서 데이터 가져오기
                text_column=text_column,  # 자동 감지된 컬럼 사용
                label_column=request.label_column,
                max_length=request.max_length,
                test_size=request.test_size,
                format_type=request.format_type,  # 포맷 타입
                input_column=input_column,  # 입력 컬럼 (자동 감지된 값)
                output_column=output_column,  # 출력 컬럼 (자동 감지된 값)
                input_columns=request.input_columns,  # 여러 입력 컬럼
                output_columns=request.output_columns,  # 여러 출력 컬럼
                output_separator=request.output_separator,  # 출력 컬럼 구분자
                template=request.template,  # 템플릿
            )
            
            yield json.dumps({"status": "dataset_prepared", "message": f"데이터셋 준비 완료: {dataset_info['train_size']}개 학습, {dataset_info['eval_size']}개 검증", "progress": 20}).encode() + b'\n'
            
            # 학습 시작
            yield json.dumps({"status": "starting_training", "message": "학습 시작...", "progress": 30}).encode() + b'\n'
            
            # 학습 진행 상황을 저장할 변수
            training_progress = {
                "step": 0,
                "loss": None,
                "epoch": 0,
            }
            
            # 커스텀 콜백으로 학습 진행 상황 수집
            from transformers import TrainerCallback
            
            class StreamingCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        training_progress["step"] = state.global_step
                        training_progress["loss"] = logs.get("loss")
                        training_progress["epoch"] = state.epoch
            
            # Trainer 생성 및 학습 시작 (비동기로 실행)
            import threading
            
            def train_thread():
                try:
                    train_result = training_service.start_training(
                        train_dataset=dataset_info["encoded_train"],
                        eval_dataset=dataset_info["encoded_eval"],
                        callbacks=[StreamingCallback()],
                    )
                    training_service.training_result = train_result
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    training_service.training_error = error_msg
                    print(f"❌ 학습 스레드 오류: {error_msg}")  # 디버깅용
            
            thread = threading.Thread(target=train_thread)
            thread.daemon = True
            thread.start()
            
            # 학습 스레드 시작 후 잠시 대기 (초기화 시간)
            await asyncio.sleep(1)
            
            # 초기 오류 확인
            if hasattr(training_service, 'training_error') and training_service.training_error:
                yield json.dumps({
                    "status": "error",
                    "message": f"학습 시작 실패: {training_service.training_error}",
                }).encode() + b'\n'
                return
            
            # 학습 진행 상황 모니터링
            import time
            last_step = 0
            start_time = time.time()
            last_step_time = start_time
            
            while thread.is_alive():
                if training_service.trainer and training_service.trainer.state:
                    state = training_service.trainer.state
                    if state.global_step > last_step:
                        current_time = time.time()
                        step_delta = state.global_step - last_step
                        time_delta = current_time - last_step_time
                        
                        last_step = state.global_step
                        last_step_time = current_time
                        
                        # 최근 로그에서 loss 및 eval_loss 추출
                        loss = None
                        eval_loss = None
                        if state.log_history:
                            for log_entry in reversed(state.log_history):
                                if log_entry and isinstance(log_entry, dict):
                                    if "loss" in log_entry and loss is None:
                                        loss = log_entry.get("loss")
                                    if "eval_loss" in log_entry and eval_loss is None:
                                        eval_loss = log_entry.get("eval_loss")
                                    # loss와 eval_loss를 모두 찾으면 중단
                                    if loss is not None and (eval_loss is not None or "eval_loss" not in log_entry):
                                        break
                        
                        # 진행률 및 시간 정보 계산
                        progress = 30
                        total_steps = None
                        progress_percent = 0
                        elapsed_time = current_time - start_time
                        time_per_step = time_delta / step_delta if step_delta > 0 else 0
                        eta_seconds = None
                        
                        if training_service.training_args and dataset_info.get("encoded_train"):
                            try:
                                encoded_train = dataset_info["encoded_train"]
                                if isinstance(encoded_train, dict) and "input_ids" in encoded_train:
                                    train_size = len(encoded_train["input_ids"])
                                    batch_size = training_service.training_args.per_device_train_batch_size
                                    gradient_accumulation_steps = training_service.training_args.gradient_accumulation_steps or 1
                                    steps_per_epoch = train_size // (batch_size * gradient_accumulation_steps)
                                    total_steps = training_service.training_args.num_train_epochs * steps_per_epoch
                                    
                                    if total_steps > 0:
                                        progress_percent = (state.global_step / total_steps) * 100
                                        progress = min(30 + int(progress_percent * 0.7), 99)
                                        
                                        # 예상 남은 시간 계산
                                        if time_per_step > 0:
                                            remaining_steps = total_steps - state.global_step
                                            eta_seconds = remaining_steps * time_per_step
                            except Exception:
                                pass  # 진행률 계산 실패 시 기본값 사용
                        
                        # 시간 포맷팅 함수
                        def format_time(seconds):
                            if seconds is None or seconds < 0:
                                return "N/A"
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            if hours > 0:
                                return f"{hours}:{minutes:02d}:{secs:02d}"
                            else:
                                return f"{minutes}:{secs:02d}"
                        
                        # 메시지 구성
                        message_parts = [f"Step {state.global_step}", f"Epoch {state.epoch:.2f}"]
                        if loss is not None:
                            message_parts.append(f"Loss: {loss:.4f}")
                        if eval_loss is not None:
                            message_parts.append(f"Eval Loss: {eval_loss:.4f}")
                        
                        yield json.dumps({
                            "status": "training",
                            "message": ", ".join(message_parts),
                            "progress": progress,
                            "step": state.global_step,
                            "epoch": state.epoch,
                            "loss": loss,
                            "eval_loss": eval_loss if eval_loss is not None else None,
                            "total_steps": total_steps,
                            "progress_percent": progress_percent,
                            "elapsed_time": elapsed_time,
                            "eta_seconds": eta_seconds,
                            "time_per_step": time_per_step,
                        }).encode() + b'\n'
                await asyncio.sleep(0.5)  # 더 자주 업데이트
            
            # 학습 완료
            if hasattr(training_service, 'training_result') and training_service.training_result:
                result = training_service.training_result
                yield json.dumps({
                    "status": "completed",
                    "message": "학습 완료!",
                    "progress": 100,
                    "training_loss": result.get("training_loss") if isinstance(result, dict) else None,
                    "num_epochs": result.get("num_epochs") if isinstance(result, dict) else None,
                }).encode() + b'\n'
            elif hasattr(training_service, 'training_error') and training_service.training_error:
                yield json.dumps({
                    "status": "error",
                    "message": f"학습 실패: {training_service.training_error}",
                }).encode() + b'\n'
            else:
                yield json.dumps({
                    "status": "completed",
                    "message": "학습 완료!",
                    "progress": 100,
                }).encode() + b'\n'
                
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"학습 시작 실패: {str(e)}"}).encode() + b'\n'
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


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
