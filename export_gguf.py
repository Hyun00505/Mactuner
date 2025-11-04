"""Utility helpers for exporting fine-tuned Hugging Face models to GGUF format.

This module wraps the llama.cpp conversion and quantization scripts so that
fine-tuned models produced elsewhere in this repository can be converted into
runtime-friendly GGUF artifacts.  It keeps concerns such as LoRA adapter merge,
script validation, and metadata reporting in one place for CLI or API callers.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


LOGGER = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Metadata returned after a conversion or quantization step."""

    output_path: Path
    file_size_bytes: int
    extra: Dict[str, str]

    def describe(self) -> str:
        size_mb = self.file_size_bytes / (1024 ** 2)
        details = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"{self.output_path} ({size_mb:.2f} MB{', ' + details if details else ''})"


def _ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"❌ {kind} 경로를 찾을 수 없습니다: {path}")


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = torch.float16,
) -> Path:
    """Merge a PEFT LoRA adapter into its base model for export."""

    base_dir = Path(base_model_path)
    adapter_dir = Path(adapter_path)
    output_dir = Path(output_path)

    _ensure_exists(base_dir, "기본 모델")
    _ensure_exists(adapter_dir, "LoRA 어댑터")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading base model from %s", base_dir)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=dtype,
    )
    model.to(device)
    LOGGER.info("Applying LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    peft_model = peft_model.merge_and_unload()

    LOGGER.info("Saving merged model to %s", output_dir)
    peft_model.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def convert_to_gguf(
    model_path: str,
    output_path: str,
    outtype: str = "f16",
    converter_script: str = "convert-hf-to-gguf.py",
    extra_args: Optional[list[str]] = None,
) -> ExportResult:
    """Convert a Hugging Face model directory into a GGUF file."""

    model_dir = Path(model_path)
    output_file = Path(output_path)
    script_path = Path(converter_script)

    _ensure_exists(model_dir, "모델")
    if script_path.suffix == "":
        search = shutil.which(converter_script)
        if search:
            script_path = Path(search)
    _ensure_exists(script_path, "converter 스크립트")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script_path), "--model", str(model_dir), "--outfile", str(output_file), "--outtype", outtype]
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running GGUF conversion: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    size_bytes = output_file.stat().st_size
    return ExportResult(output_file, size_bytes, {"outtype": outtype})


def quantize_gguf(
    input_gguf: str,
    output_gguf: str,
    quant_type: str,
    quant_bin: str = "./quantize",
    extra_args: Optional[list[str]] = None,
) -> ExportResult:
    """Quantize a GGUF file using llama.cpp's quantize binary."""

    input_path = Path(input_gguf)
    output_path = Path(output_gguf)

    _ensure_exists(input_path, "GGUF 입력")

    quant_path = Path(quant_bin)
    if not quant_path.exists():
        search = shutil.which(quant_bin)
        if search:
            quant_path = Path(search)
    _ensure_exists(quant_path, "quantize 바이너리")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(quant_path), str(input_path), str(output_path), quant_type]
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running quantization: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    size_bytes = output_path.stat().st_size
    return ExportResult(output_path, size_bytes, {"quant_type": quant_type})


__all__ = [
    "ExportResult",
    "merge_lora_adapter",
    "convert_to_gguf",
    "quantize_gguf",
]
