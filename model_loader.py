"""Model loader module for loading LLMs from Hugging Face or local directories.

This module exposes a single public function, :func:`load_model`, which can be
used by CLI tools or other Python modules to obtain a model, tokenizer, and
basic metadata about the loading process.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


__all__ = ["load_model"]


def _select_device() -> torch.device:
    """Selects the best available device, preferring Apple MPS if available."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ MPS 디바이스 감지. MPS를 사용합니다.")
        return torch.device("mps")
    print("❌ MPS 사용 불가. CPU로 대체합니다.")
    return torch.device("cpu")


def _format_param_count(num_params: int) -> str:
    """Formats the parameter count into a human-readable string."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def _load_components(
    model_id_or_path: str,
    access_token: Optional[str],
) -> Tuple[torch.nn.Module, PreTrainedTokenizer]:
    """Loads the model and tokenizer using transformers Auto classes."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        use_auth_token=access_token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        use_auth_token=access_token,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_model(
    model_source: str,
    model_id_or_path: str,
    access_token: Optional[str] = None,
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, Dict[str, object]]:
    """Loads an LLM model and tokenizer from Hugging Face hub or local path.

    Args:
        model_source: "hub" or "local" indicating where to load the model from.
        model_id_or_path: The Hugging Face model ID or local path to load from.
        access_token: Optional Hugging Face access token, required for private models.

    Returns:
        tuple: (model, tokenizer, metadata)

    Raises:
        ValueError: If ``model_source`` is not "hub" or "local".
        RuntimeError: If loading fails for any reason.
    """
    if model_source not in {"hub", "local"}:
        raise ValueError("model_source must be either 'hub' or 'local'")

    if model_source == "hub" and not access_token:
        print("ℹ️ 허깅페이스 허브 모델 로드 – 토큰이 필요한 경우 제공해주세요.")

    device = _select_device()

    token = access_token if model_source == "hub" else None

    print(f"⏳ 모델 로드 시작 – source: {model_source}, target: {model_id_or_path}")
    start_time = time.time()
    try:
        model, tokenizer = _load_components(model_id_or_path, token)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("❌ 모델 로드 실패 – 경로 또는 토큰 확인 필요.") from exc

    model.to(device)
    load_time_sec = time.time() - start_time
    num_params = sum(p.numel() for p in model.parameters())

    metadata: Dict[str, object] = {
        "device": device.type,
        "num_params": _format_param_count(num_params),
        "load_time_sec": round(load_time_sec, 2),
    }

    print("✅ 모델 로드 완료")
    print(
        f"📊 메타데이터 – device: {metadata['device']}, "
        f"num_params: {metadata['num_params']}, load_time_sec: {metadata['load_time_sec']}"
    )

    model.eval()
    return model, tokenizer, metadata
