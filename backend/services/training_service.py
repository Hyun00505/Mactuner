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

logger = logging.getLogger(__name__)


class TrainingService:
    """모델 학습 서비스"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_args = None
        self.lora_config = None
        self.device = MACOptimizer.get_device()
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
            if use_lora:
                # LoRA 준비
                if self.lora_config is None:
                    self.setup_lora()
                model = get_peft_model(model, self.lora_config)

            # Gradient checkpointing 활성화
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()

            self.model = model
            self.tokenizer = tokenizer

            # 모델 통계
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_percent = (num_trainable / num_params) * 100

            return model, {
                "total_parameters": num_params,
                "trainable_parameters": num_trainable,
                "trainable_percent": round(trainable_percent, 2),
                "device": str(self.device),
                "use_lora": use_lora,
            }

        except Exception as e:
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
        optim: str = "paged_adamw_32bit",
        fp16: bool = False,
        bf16: bool = True,
    ) -> Dict[str, Any]:
        """TrainingArguments 구성"""
        try:
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
                evaluation_strategy=eval_strategy,
                logging_steps=logging_steps,
                save_total_limit=save_total_limit,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                optim=optim,
                fp16=fp16,
                bf16=bf16,
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
    ) -> Dict[str, Any]:
        """학습 데이터셋 준비"""
        try:
            if self.tokenizer is None:
                raise ValueError("토크나이저가 로드되지 않았습니다.")

            # 텍스트 추출
            texts = dataset[text_column].tolist()

            # 토크나이징
            encodings = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # 라벨 설정
            if label_column and label_column in dataset.columns:
                labels = dataset[label_column].tolist()
            else:
                # 언어 모델링 태스크: 입력 자체가 라벨
                labels = encodings["input_ids"].clone()

            # 데이터셋 분할
            total_size = len(texts)
            test_idx = int(total_size * (1 - test_size))

            train_encodings = {
                k: v[:test_idx] for k, v in encodings.items()
            }
            eval_encodings = {
                k: v[test_idx:] for k, v in encodings.items()
            }

            train_labels = labels[:test_idx] if isinstance(labels, list) else labels[:test_idx]
            eval_labels = labels[test_idx:] if isinstance(labels, list) else labels[test_idx:]

            return {
                "train_size": test_idx,
                "eval_size": total_size - test_idx,
                "total_size": total_size,
                "max_length": max_length,
                "encoded_train": train_encodings,
                "encoded_eval": eval_encodings,
                "train_labels": train_labels,
                "eval_labels": eval_labels,
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

            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=callbacks or [],
            )

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
