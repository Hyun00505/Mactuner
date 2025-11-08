"""학습 엔진 서비스"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from backend.config import settings
from backend.utils.mac_optimization import MACOptimizer
from backend.services.device_manager import get_device_manager

logger = logging.getLogger(__name__)


class TrainingService:
    """모델 학습 서비스"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_args = None
        self.lora_config = None
        self.training_result = None
        self.training_error = None
        # 선택된 디바이스 또는 기본값 사용
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_current_device()
        self.training_history = {
            "loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

    # ========================================
    # LoRA 설정
    # ========================================

    def setup_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """LoRA 설정"""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        return {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": target_modules,
            "status": "configured",
        }

    # ========================================
    # 모델 준비
    # ========================================

    def prepare_model_for_training(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        use_lora: bool = True,
        gradient_checkpointing: bool = True,
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """학습을 위한 모델 준비"""
        try:
            # 모델을 train 모드로 먼저 설정
            model.train()
            
            # 모든 파라미터의 requires_grad를 False로 설정 (LoRA 적용 전)
            for param in model.parameters():
                param.requires_grad = False
            
            if use_lora:
                # LoRA 준비
                if self.lora_config is None:
                    self.setup_lora()
                
                # LoRA 모델 적용
                model = get_peft_model(model, self.lora_config)
                
                # LoRA 적용 후 train 모드 유지
                model.train()
                
                # 디바이스로 이동 (LoRA 적용 후)
                if self.device_manager:
                    device = self.device_manager.get_current_device()
                    if device:
                        model = model.to(device)
                        logger.info(f"LoRA 모델을 {device}로 이동했습니다.")
                
                # LoRA 모델이 제대로 설정되었는지 확인
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if len(trainable_params) == 0:
                    raise ValueError("LoRA 모델에 학습 가능한 파라미터가 없습니다. LoRA 설정을 확인하세요.")
                
                logger.info(f"LoRA 적용 완료: {len(trainable_params)}개 학습 가능 파라미터 그룹")
                logger.info(f"학습 가능 파라미터 수: {sum(p.numel() for p in trainable_params):,}개")
                
                # LoRA 파라미터 이름 확인
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.debug(f"학습 가능 파라미터: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}, device: {param.device}")
            else:
                # LoRA를 사용하지 않는 경우 모든 파라미터를 학습 가능하도록 설정
                for param in model.parameters():
                    param.requires_grad = True
                model.train()

            # Gradient checkpointing 비활성화 (LoRA 학습 시 문제 발생 가능)
            # LoRA를 사용할 때는 gradient checkpointing이 필수는 아니며,
            # 입력 텐서의 requires_grad 문제로 인해 비활성화
            if gradient_checkpointing:
                logger.warning("⚠️ Gradient checkpointing은 LoRA 학습 시 문제를 일으킬 수 있습니다. 비활성화합니다.")
                # Gradient checkpointing 비활성화
                if hasattr(model, 'gradient_checkpointing_disable'):
                    model.gradient_checkpointing_disable()
                if hasattr(model, 'config'):
                    model.config.use_cache = True  # use_cache를 True로 설정
            else:
                # Gradient checkpointing이 비활성화되어 있으면 use_cache를 True로 설정
                if hasattr(model, 'config'):
                    model.config.use_cache = True

            self.model = model
            self.tokenizer = tokenizer

            # 모델 통계
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_percent = (num_trainable / num_params) * 100 if num_params > 0 else 0
            
            logger.info(f"모델 준비 완료: 총 {num_params:,}개 파라미터, 학습 가능 {num_trainable:,}개 ({trainable_percent:.2f}%)")
            
            # 최종 확인: 모델이 train 모드인지 확인
            if hasattr(model, 'training'):
                if not model.training:
                    logger.warning("모델이 train 모드가 아닙니다. train 모드로 변경합니다.")
                    model.train()
            
            return model, {
                "total_parameters": num_params,
                "trainable_parameters": num_trainable,
                "trainable_percent": trainable_percent,
                "status": "prepared",
            }

        except Exception as e:
            logger.error(f"모델 준비 실패: {str(e)}")
            raise RuntimeError(f"모델 준비 실패: {str(e)}")

    # ========================================
    # TrainingArguments 설정
    # ========================================

    def configure_training_args(
        self,
        output_dir: str = "output/checkpoints",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_strategy: str = "epoch",
        eval_strategy: str = "epoch",
        logging_steps: int = 100,
        save_total_limit: int = 3,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        optim: str = "adamw_torch",  # 기본값을 범용 호환 optimizer로 변경 (Mac/Windows/Linux 모두 지원)
        fp16: bool = False,
        bf16: bool = False,  # 기본값을 False로 변경 (Mac/Windows 호환성)
    ) -> Dict[str, Any]:
        """TrainingArguments 구성"""
        try:
            import platform
            import torch
            
            # CUDA 사용 가능 여부 확인
            has_cuda = torch.cuda.is_available()
            is_mac = platform.system() == "Darwin"
            
            # bitsandbytes는 CUDA 환경에서만 작동
            # Mac이나 CUDA가 없는 환경에서는 일반 optimizer 사용
            if not has_cuda or is_mac:
                # CUDA가 없거나 Mac인 경우 bitsandbytes 기반 optimizer 사용 불가
                if optim in ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_bnb_8bit", "adamw_bnb_4bit"]:
                    logger.info(f"CUDA 사용 불가 또는 Mac 환경 감지. Optimizer를 '{optim}'에서 'adamw_torch'로 변경합니다.")
                    optim = "adamw_torch"  # 범용 호환 optimizer
                # Mac MPS는 bf16을 지원하지만, 안정성을 위해 fp16 또는 float32 사용
                if is_mac and bf16:
                    logger.info("Mac 환경에서 bf16을 fp16으로 변경합니다.")
                    bf16 = False
                    fp16 = True  # Mac에서는 fp16 사용
            else:
                # CUDA가 있는 경우 bitsandbytes 설치 여부 확인
                try:
                    import bitsandbytes
                    logger.info("CUDA 환경에서 bitsandbytes를 사용합니다.")
                except ImportError:
                    # bitsandbytes가 설치되지 않은 경우 일반 optimizer 사용
                    if optim in ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_bnb_8bit", "adamw_bnb_4bit"]:
                        logger.warning(f"bitsandbytes가 설치되지 않았습니다. Optimizer를 '{optim}'에서 'adamw_torch'로 변경합니다.")
                        optim = "adamw_torch"
            
            output_dir = Path(output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            self.training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                save_strategy=save_strategy,
                eval_strategy=eval_strategy,  # evaluation_strategy 대신 eval_strategy 사용
                logging_steps=logging_steps,
                save_total_limit=save_total_limit,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                optim=optim,
                fp16=fp16,
                bf16=bf16,
                gradient_checkpointing=False,  # LoRA 학습 시 비활성화 (입력 텐서 requires_grad 문제)
                logging_dir=str(output_dir / "logs"),
                report_to=["tensorboard"],
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )

            return {
                "output_dir": str(output_dir),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "status": "configured",
            }

        except Exception as e:
            raise RuntimeError(f"TrainingArguments 설정 실패: {str(e)}")

    # ========================================
    # 데이터셋 처리
    # ========================================

    def prepare_dataset(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        max_length: int = 512,
        test_size: float = 0.1,
        format_type: str = "causal_lm",  # "causal_lm", "instruction", "chat"
        input_column: Optional[str] = None,
        output_column: Optional[str] = None,
        input_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None,
        output_separator: Optional[str] = "\n",
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """학습 데이터셋 준비
        
        Args:
            dataset: 학습 데이터셋
            text_column: 텍스트 컬럼 (causal_lm용)
            label_column: 라벨 컬럼 (사용 안 함, 호환성 유지)
            max_length: 최대 토큰 길이
            test_size: 테스트 세트 비율
            format_type: 데이터 포맷 타입 ("causal_lm", "instruction", "chat")
            input_column: 입력 컬럼 (instruction/chat용)
            output_column: 출력 컬럼 (instruction/chat용)
            template: 프롬프트 템플릿 (None이면 기본 템플릿 사용)
        """
        try:
            if self.tokenizer is None:
                raise ValueError("토크나이저가 로드되지 않았습니다.")

            # 포맷에 따라 텍스트 준비
            if format_type == "instruction":
                # Instruction Tuning: input + output 결합
                
                # 여러 컬럼 지원: input_columns와 output_columns가 있으면 사용
                if input_columns and len(input_columns) > 0:
                    # 여러 입력 컬럼 합치기
                    def combine_input(row):
                        values = [str(row[col]) for col in input_columns if col in dataset.columns]
                        return " ".join(values)
                    input_texts = [combine_input(row) for _, row in dataset.iterrows()]
                elif input_column:
                    input_texts = dataset[input_column].tolist()
                else:
                    # 자동 감지
                    input_cols = [col for col in dataset.columns if col.lower() in ['instruction', 'input', 'question', 'prompt', 'query']]
                    if input_cols:
                        input_texts = dataset[input_cols[0]].tolist()
                    else:
                        raise ValueError(f"Instruction 포맷을 위한 input 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(dataset.columns)}")
                
                if output_columns and len(output_columns) > 0:
                    # 여러 출력 컬럼 합치기
                    separator = output_separator.replace("\\n", "\n").replace("\\t", "\t") if output_separator else "\n"
                    def combine_output(row):
                        values = [str(row[col]) for col in output_columns if col in dataset.columns]
                        return separator.join(values)
                    output_texts = [combine_output(row) for _, row in dataset.iterrows()]
                elif output_column:
                    output_texts = dataset[output_column].tolist()
                else:
                    # 자동 감지
                    output_cols = [col for col in dataset.columns if col.lower() in ['output', 'response', 'answer', 'reply']]
                    if output_cols:
                        output_texts = dataset[output_cols[0]].tolist()
                    else:
                        raise ValueError(f"Instruction 포맷을 위한 output 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(dataset.columns)}")
                
                # 템플릿 적용
                if template:
                    texts = [
                        template.format(input=input_texts[i], output=output_texts[i])
                        for i in range(len(input_texts))
                    ]
                else:
                    # 기본 Instruction 템플릿
                    texts = [
                        f"### Instruction:\n{input_texts[i]}\n### Response:\n{output_texts[i]}"
                        for i in range(len(input_texts))
                    ]
                    
            elif format_type == "chat":
                # Chat Format: system + user + assistant
                if input_column is None or output_column is None:
                    # 자동 감지
                    user_cols = [col for col in dataset.columns if col.lower() in ['user', 'input', 'question', 'prompt']]
                    assistant_cols = [col for col in dataset.columns if col.lower() in ['assistant', 'output', 'response', 'answer']]
                    system_cols = [col for col in dataset.columns if col.lower() == 'system']
                    
                    if user_cols and assistant_cols:
                        input_column = user_cols[0]
                        output_column = assistant_cols[0]
                    else:
                        raise ValueError(f"Chat 포맷을 위한 user/assistant 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(dataset.columns)}")
                
                system_text = dataset[system_cols[0]].iloc[0] if system_cols and len(system_cols) > 0 else "You are a helpful AI assistant."
                
                if template:
                    texts = [
                        template.format(
                            system=system_text,
                            user=row[input_column],
                            assistant=row[output_column]
                        )
                        for _, row in dataset.iterrows()
                    ]
                else:
                    # 기본 Chat 템플릿
                    texts = [
                        f"<system>{system_text}</system>\n<user>{row[input_column]}</user>\n<assistant>{row[output_column]}</assistant>"
                        for _, row in dataset.iterrows()
                    ]
            else:
                # Causal LM: 단일 컬럼 사용
                texts = dataset[text_column].tolist()

            # 토크나이징
            encodings = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # DataCollatorForLanguageModeling이 자동으로 labels를 생성하므로,
            # 여기서는 labels를 생성하지 않음

            # 데이터셋 분할
            total_size = len(texts)
            test_idx = int(total_size * (1 - test_size))

            # 텐서를 리스트로 변환 (Dataset.from_dict를 위해)
            train_encodings = {
                k: v[:test_idx].tolist() if hasattr(v, 'tolist') else v[:test_idx] 
                for k, v in encodings.items()
            }
            eval_encodings = {
                k: v[test_idx:].tolist() if hasattr(v, 'tolist') else v[test_idx:] 
                for k, v in encodings.items()
            }

            # DataCollatorForLanguageModeling이 자동으로 labels를 생성하므로,
            # 여기서는 labels를 추가하지 않음
            # labels는 DataCollator에서 자동으로 생성됨

            return {
                "train_size": test_idx,
                "eval_size": total_size - test_idx,
                "total_size": total_size,
                "max_length": max_length,
                "encoded_train": train_encodings,
                "encoded_eval": eval_encodings,
                "status": "prepared",
            }

        except Exception as e:
            raise RuntimeError(f"데이터셋 준비 실패: {str(e)}")

    # ========================================
    # 학습 실행
    # ========================================

    def start_training(
        self,
        train_dataset,
        eval_dataset=None,
        callbacks=None,
    ) -> Dict[str, Any]:
        """학습 실행"""
        try:
            if self.model is None or self.training_args is None:
                raise ValueError("모델 또는 TrainingArguments가 설정되지 않았습니다.")

            # 모델이 학습 가능한지 확인
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError("모델에 학습 가능한 파라미터가 없습니다. LoRA가 제대로 적용되었는지 확인하세요.")
            
            logger.info(f"학습 시작: 학습 가능한 파라미터 {sum(p.numel() for p in trainable_params):,}개")

            # 모델을 train 모드로 설정 (확실하게)
            self.model.train()
            
            # 모든 학습 가능한 파라미터가 requires_grad=True인지 확인 및 강제 설정
            trainable_found = False
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_found = True
                    # requires_grad가 True인 파라미터를 명시적으로 확인
                    if not param.requires_grad:
                        logger.warning(f"파라미터 {name}의 requires_grad가 False입니다. True로 설정합니다.")
                        param.requires_grad = True
                    logger.debug(f"학습 가능 파라미터: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")
            
            if not trainable_found:
                raise ValueError("모델에 학습 가능한 파라미터가 없습니다. LoRA가 제대로 적용되었는지 확인하세요.")
            
            # 모델이 실제로 train 모드인지 확인
            if hasattr(self.model, 'training') and not self.model.training:
                logger.warning("모델이 train 모드가 아닙니다. train 모드로 강제 변경합니다.")
                self.model.train()
            
            # 디바이스 확인 및 이동
            if self.device_manager:
                device = self.device_manager.get_current_device()
                if device:
                    # 모델이 올바른 디바이스에 있는지 확인
                    first_param = next(self.model.parameters())
                    if first_param.device != device:
                        logger.info(f"모델을 {device}로 이동합니다.")
                        self.model = self.model.to(device)

            # 딕셔너리를 Dataset 객체로 변환
            from datasets import Dataset
            
            # train_dataset이 딕셔너리인 경우 Dataset으로 변환
            if isinstance(train_dataset, dict):
                # DataCollator가 자동으로 labels를 생성하므로, 
                # 이미 있는 labels는 제거하거나 유지할 수 있음
                # 하지만 DataCollator가 자동으로 생성하는 것이 더 안전함
                train_dataset = Dataset.from_dict(train_dataset)
            
            # eval_dataset이 딕셔너리인 경우 Dataset으로 변환
            if eval_dataset is not None and isinstance(eval_dataset, dict):
                eval_dataset = Dataset.from_dict(eval_dataset)

            # DataCollator 설정
            # DataCollatorForLanguageModeling은 자동으로 labels를 생성함
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM이므로 MLM은 False
            )

            # 모델이 실제로 학습 가능한지 최종 확인
            trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
            if trainable_count == 0:
                raise ValueError(
                    "모델에 학습 가능한 파라미터가 없습니다. "
                    "LoRA가 제대로 적용되었는지 확인하세요. "
                    "모델이 prepare_model_for_training()을 통해 준비되었는지 확인하세요."
                )
            
            logger.info(f"✅ 학습 준비 완료: {trainable_count}개 학습 가능 파라미터 그룹, "
                       f"총 {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}개 학습 가능 파라미터")
            
            # 모델이 train 모드인지 최종 확인
            if hasattr(self.model, 'training') and not self.model.training:
                logger.warning("⚠️ 모델이 train 모드가 아닙니다. train 모드로 강제 변경합니다.")
                self.model.train()
            
            # 테스트 forward pass로 모델이 제대로 작동하는지 확인
            try:
                # 샘플 데이터로 테스트
                sample_batch = data_collator([{"input_ids": [1, 2, 3, 4, 5]}])
                if sample_batch:
                    # 디바이스로 이동
                    if self.device_manager:
                        device = self.device_manager.get_current_device()
                        if device:
                            sample_batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in sample_batch.items()}
                    
                    # Forward pass 테스트 (gradient enabled)
                    self.model.train()  # 확실하게 train 모드
                    torch.set_grad_enabled(True)
                    
                    outputs = self.model(**sample_batch)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                        logger.info(f"✅ Forward pass 테스트 성공. Loss: {loss.item() if hasattr(loss, 'item') else loss}")
                        if hasattr(loss, 'requires_grad'):
                            logger.info(f"Loss requires_grad: {loss.requires_grad}")
                            if not loss.requires_grad:
                                logger.error("❌ Loss가 requires_grad=False입니다. 모델 파라미터를 확인하세요.")
                                # 학습 가능한 파라미터 재확인
                                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                                logger.error(f"학습 가능한 파라미터 수: {len(trainable_params)}")
                                if len(trainable_params) > 0:
                                    for name, param in list(self.model.named_parameters())[:5]:
                                        if param.requires_grad:
                                            logger.error(f"  - {name}: requires_grad={param.requires_grad}, device={param.device}")
                        if hasattr(loss, 'grad_fn'):
                            logger.info(f"Loss grad_fn: {loss.grad_fn}")
                        else:
                            logger.warning("⚠️ Loss에 grad_fn이 없습니다.")
                    else:
                        logger.warning("⚠️ Forward pass에서 loss를 반환하지 않았습니다.")
            except Exception as e:
                logger.warning(f"⚠️ Forward pass 테스트 실패 (계속 진행): {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())

            # Trainer 생성 직전 최종 검증
            logger.info("=" * 60)
            logger.info("Trainer 생성 전 최종 검증")
            logger.info("=" * 60)
            
            # 1. 모델이 train 모드인지 확인
            if hasattr(self.model, 'training'):
                if not self.model.training:
                    logger.error("❌ 모델이 train 모드가 아닙니다!")
                    self.model.train()
                    logger.info("✅ 모델을 train 모드로 변경했습니다.")
                else:
                    logger.info("✅ 모델이 train 모드입니다.")
            
            # 2. 학습 가능한 파라미터 확인
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError("❌ 학습 가능한 파라미터가 없습니다!")
            logger.info(f"✅ 학습 가능한 파라미터: {len(trainable_params)}개 그룹, 총 {sum(p.numel() for p in trainable_params):,}개")
            
            # 3. 첫 번째 학습 가능한 파라미터 확인
            first_trainable = next((p for p in self.model.parameters() if p.requires_grad), None)
            if first_trainable is not None:
                logger.info(f"✅ 첫 번째 학습 가능 파라미터: requires_grad={first_trainable.requires_grad}, device={first_trainable.device}")
            else:
                raise ValueError("❌ 학습 가능한 파라미터를 찾을 수 없습니다!")
            
            # 4. 디바이스 확인
            if self.device_manager:
                device = self.device_manager.get_current_device()
                if device:
                    logger.info(f"✅ 목표 디바이스: {device}")
                    logger.info(f"✅ 모델 디바이스: {first_trainable.device}")
                    if first_trainable.device != device:
                        logger.warning(f"⚠️ 모델이 목표 디바이스와 다릅니다. {device}로 이동합니다.")
                        self.model = self.model.to(device)
            
            logger.info("=" * 60)

            # 커스텀 콜백: 학습 시작 전 모델 상태 확인
            from transformers import TrainerCallback
            
            class ModelValidationCallback(TrainerCallback):
                def on_train_begin(self, args, state, control, model=None, **kwargs):
                    if model is not None:
                        logger.info("=" * 60)
                        logger.info("학습 시작 시점 모델 상태 확인")
                        logger.info("=" * 60)
                        
                        # 모델이 train 모드인지 확인
                        if hasattr(model, 'training'):
                            if not model.training:
                                logger.error("❌ 학습 시작 시점에 모델이 train 모드가 아닙니다!")
                                model.train()
                            else:
                                logger.info("✅ 모델이 train 모드입니다.")
                        
                        # 학습 가능한 파라미터 확인
                        trainable_params = [p for p in model.parameters() if p.requires_grad]
                        logger.info(f"✅ 학습 가능한 파라미터: {len(trainable_params)}개 그룹")
                        
                        # 첫 번째 학습 가능한 파라미터 확인
                        first_trainable = next((p for p in model.parameters() if p.requires_grad), None)
                        if first_trainable is not None:
                            logger.info(f"✅ 첫 번째 학습 가능 파라미터: requires_grad={first_trainable.requires_grad}, device={first_trainable.device}")
                        
                        logger.info("=" * 60)

            # 기존 콜백에 검증 콜백 추가
            all_callbacks = [ModelValidationCallback()]
            if callbacks:
                all_callbacks.extend(callbacks)

            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=all_callbacks,
                data_collator=data_collator,
            )
            
            # Trainer 생성 후 모델 상태 재확인
            logger.info("Trainer 생성 후 모델 상태 확인")
            if hasattr(self.model, 'training') and not self.model.training:
                logger.warning("⚠️ Trainer 생성 후 모델이 train 모드가 아닙니다. train 모드로 변경합니다.")
                self.model.train()
            
            # 학습 시작 전 최종 확인
            trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
            if trainable_count == 0:
                raise ValueError("❌ 학습 시작 직전: 학습 가능한 파라미터가 없습니다!")
            logger.info(f"✅ 학습 시작 직전: {trainable_count}개 학습 가능 파라미터 그룹 확인됨")

            # 학습 시작
            train_result = self.trainer.train()

            return {
                "status": "completed",
                "training_loss": train_result.training_loss,
                "final_learning_rate": train_result.training_loss,
                "num_epochs": train_result.epoch,
                "checkpoint_dir": self.training_args.output_dir,
            }

        except Exception as e:
            raise RuntimeError(f"학습 실패: {str(e)}")

    # ========================================
    # 모델 저장
    # ========================================

    def save_model(self, output_dir: str) -> Dict[str, str]:
        """학습된 모델 저장"""
        try:
            output_dir = Path(output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.model is None:
                raise ValueError("저장할 모델이 없습니다.")

            # 모델 저장
            self.model.save_pretrained(str(output_dir))

            # 토크나이저 저장
            if self.tokenizer:
                self.tokenizer.save_pretrained(str(output_dir))

            # 설정 저장
            config = {
                "model_type": "finetuned",
                "lora_config": self.lora_config.to_dict() if self.lora_config else None,
                "training_args": self.training_args.to_dict() if self.training_args else None,
            }

            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

            return {
                "status": "success",
                "output_dir": str(output_dir),
                "files_saved": ["model", "tokenizer", "config"],
            }

        except Exception as e:
            raise RuntimeError(f"모델 저장 실패: {str(e)}")

    # ========================================
    # 모델 평가
    # ========================================

    def evaluate(self, eval_dataset) -> Dict[str, Any]:
        """모델 평가"""
        try:
            if self.trainer is None:
                raise ValueError("Trainer가 초기화되지 않았습니다.")

            metrics = self.trainer.evaluate(eval_dataset)

            return {
                "status": "completed",
                "eval_loss": metrics.get("eval_loss"),
                "eval_perplexity": metrics.get("eval_perplexity"),
                "metrics": metrics,
            }

        except Exception as e:
            raise RuntimeError(f"평가 실패: {str(e)}")

    # ========================================
    # 최적 파라미터 추천
    # ========================================

    def recommend_parameters(
        self,
        model_size_params: int,
        dataset_size: int,
    ) -> Dict[str, Any]:
        """모델 및 데이터셋 크기에 따른 최적 파라미터 추천"""
        recommendations = {}

        # 배치 크기 추천
        memory_stats = MACOptimizer.get_memory_stats()
        available_memory_gb = memory_stats["available_gb"]

        if model_size_params > 7e9:  # 7B+
            recommendations["batch_size"] = max(1, int(available_memory_gb / 8))
            recommendations["lora_rank"] = 8
        elif model_size_params > 1e9:  # 1B-7B
            recommendations["batch_size"] = max(2, int(available_memory_gb / 4))
            recommendations["lora_rank"] = 16
        else:  # <1B
            recommendations["batch_size"] = max(4, int(available_memory_gb / 2))
            recommendations["lora_rank"] = 32

        # 학습률 추천
        if dataset_size < 1000:
            recommendations["learning_rate"] = 1e-4
            recommendations["num_epochs"] = 10
        elif dataset_size < 10000:
            recommendations["learning_rate"] = 5e-5
            recommendations["num_epochs"] = 5
        else:
            recommendations["learning_rate"] = 2e-5
            recommendations["num_epochs"] = 3

        # Warmup steps 추천
        recommendations["warmup_steps"] = max(100, dataset_size // recommendations["batch_size"] // 10)

        return {
            "model_size_params": model_size_params,
            "dataset_size": dataset_size,
            "available_memory_gb": available_memory_gb,
            "recommendations": recommendations,
        }

    # ========================================
    # 학습 모니터링
    # ========================================

    def get_training_status(self) -> Dict[str, Any]:
        """현재 학습 상태 조회"""
        if self.trainer is None:
            return {"status": "not_started"}

        return {
            "status": "training",
            "current_epoch": self.trainer.state.epoch,
            "total_epochs": self.training_args.num_train_epochs,
            "global_step": self.trainer.state.global_step,
            "logging_steps": self.training_args.logging_steps,
        }

    def get_training_history(self) -> Dict[str, List]:
        """학습 이력 조회"""
        if self.trainer is None:
            return {}

        history = self.trainer.state.log_history
        return {
            "history": history,
            "total_steps": len(history),
        }
