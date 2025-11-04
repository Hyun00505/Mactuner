"""LoRA-based fine-tuning utilities optimized for macOS MPS environments."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """Container for artifacts generated during training."""

    trainer: Trainer
    peft_model: PeftModel
    training_output: Any


def _ensure_output_dir(path: str) -> None:
    if not path:
        raise ValueError("❌ output_dir must be provided for saving adapters.")
    os.makedirs(path, exist_ok=True)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("✅ Using Apple Metal (MPS) backend for training.")
        return torch.device("mps")
    logger.warning("❌ MPS backend unavailable. Falling back to CPU.")
    return torch.device("cpu")


def _build_lora_config(config: Dict[str, Any]) -> LoraConfig:
    default_config: Dict[str, Any] = {
        "r": 8,
        "lora_alpha": config.get("alpha", 16),
        "lora_dropout": config.get("dropout", 0.05),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    if "target_modules" in config:
        default_config["target_modules"] = config["target_modules"]
    if "modules_to_save" in config:
        default_config["modules_to_save"] = config["modules_to_save"]

    # Respect overrides for canonical LoRA argument names
    for key in ("r", "lora_alpha", "lora_dropout"):
        if key in config:
            default_config[key] = config[key]

    return LoraConfig(**default_config)


def _prepare_datasets(
    dataset: Union[Dataset, DatasetDict],
) -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("❌ DatasetDict must contain a 'train' split.")
        train_dataset = dataset["train"]
        eval_dataset = (
            dataset.get("validation")
            or dataset.get("eval")
            or dataset.get("test")
        )
        return train_dataset, eval_dataset

    if isinstance(dataset, Dataset):
        return dataset, None

    raise TypeError(
        "❌ dataset must be a datasets.Dataset or datasets.DatasetDict instance."
    )


def _configure_training_arguments(
    params: Dict[str, Any], output_dir: str
) -> TrainingArguments:
    params = dict(params)  # shallow copy to avoid side-effects

    save_merged = bool(params.pop("save_merged_model", False))

    defaults: Dict[str, Any] = {
        "output_dir": output_dir,
        "remove_unused_columns": False,
        "report_to": [],
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_total_limit": 2,
    }

    for key, value in defaults.items():
        params.setdefault(key, value)

    training_args = TrainingArguments(**params)
    setattr(training_args, "_save_merged_model", save_merged)
    return training_args


def train_model(
    base_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: Union[Dataset, DatasetDict],
    output_dir: str,
    lora_config: Dict[str, Any],
    training_args: Dict[str, Any],
) -> TrainingArtifacts:
    """LoRA 기반 학습을 실행하고 체크포인트를 저장합니다."""

    logging.basicConfig(level=logging.INFO)

    _ensure_output_dir(output_dir)

    device = _select_device()

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    lora_cfg = _build_lora_config(lora_config)
    logger.info("Applying LoRA with config: %s", lora_cfg)

    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()
    peft_model.to(device)

    train_dataset, eval_dataset = _prepare_datasets(dataset)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    hf_training_args = _configure_training_arguments(training_args, output_dir)

    if hf_training_args.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=peft_model,
        args=hf_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("🚀 Starting LoRA fine-tuning")
    train_output = trainer.train()

    logger.info("💾 Saving LoRA adapter to %s", output_dir)
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if getattr(hf_training_args, "_save_merged_model", False):
        merged_output_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_output_dir, exist_ok=True)
        logger.info("🔁 Saving merged model to %s", merged_output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

    return TrainingArtifacts(
        trainer=trainer,
        peft_model=peft_model,
        training_output=train_output,
    )

