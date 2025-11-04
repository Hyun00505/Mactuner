"""Conversational interface utilities for instruction/chat tuned models.

This module exposes a high-level :func:`chat` function that keeps lightweight
conversation state, prepares prompts for instruction or chat style models, and
invokes a provided model/tokenizer pair to generate responses.  The utilities
are intended for CLI or API based experiences where models are injected from
`model_loader` and datasets from `trainer` fine-tuning outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizer


__all__ = ["chat", "ChatConfig"]

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class ChatConfig:
    """Configuration parameters for chat generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    prompt_style: str = "instruction"  # or "chat"
    retain_history: bool = True
    max_history_turns: int = 5


def _trim_history(history: Sequence[Tuple[str, str]], limit: int) -> List[Tuple[str, str]]:
    if limit <= 0:
        return []
    return list(history[-limit:])


def _build_instruction_prompt(
    message: str,
    history: Sequence[Tuple[str, str]],
    system_prompt: Optional[str],
) -> str:
    sections: List[str] = []
    if system_prompt:
        sections.append(f"### System:\n{system_prompt.strip()}")
    for user, assistant in history:
        sections.append(f"### Instruction:\n{user.strip()}\n### Response:\n{assistant.strip()}")
    sections.append(f"### Instruction:\n{message.strip()}\n### Response:\n")
    return "\n\n".join(sections).strip()


def _build_chat_prompt(
    message: str,
    history: Sequence[Tuple[str, str]],
    system_prompt: Optional[str],
) -> str:
    system_block = system_prompt.strip() if system_prompt else _DEFAULT_SYSTEM_PROMPT
    lines: List[str] = ["[INST] <<SYS>>", system_block, "<</SYS>>", ""]
    for user, assistant in history:
        lines.append(f"User: {user.strip()}")
        lines.append(f"Assistant: {assistant.strip()}")
        lines.append("")
    lines.append(f"User: {message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


_PROMPT_BUILDERS: Dict[str, Callable[[str, Sequence[Tuple[str, str]], Optional[str]], str]] = {
    "instruction": _build_instruction_prompt,
    "chat": _build_chat_prompt,
}


def _prepare_generation_kwargs(
    tokenizer: PreTrainedTokenizer,
    config: ChatConfig,
) -> Dict[str, object]:
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    return {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "do_sample": config.do_sample,
        "eos_token_id": eos,
        "pad_token_id": pad,
    }


def chat(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    message: str,
    history: Optional[Sequence[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    prompt_style: str = "instruction",
    retain_history: bool = True,
    max_history_turns: int = 5,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Generate a model response for a single user message.

    Args:
        model: Loaded causal language model (optionally LoRA-adapted).
        tokenizer: Tokenizer aligned with the model.
        message: Current user message.
        history: Optional existing history of (user, assistant) pairs.
        system_prompt: Optional instruction to prepend to the prompt.
        max_new_tokens: Generation limit for new tokens.
        temperature, top_p, top_k, do_sample: Sampling controls.
        prompt_style: ``"instruction"`` or ``"chat"`` prompt template.
        retain_history: Whether to append the new exchange to history.
        max_history_turns: Maximum history turns to keep for the prompt.

    Returns:
        response text and updated history list.
    """

    history = list(history or [])
    if not retain_history:
        history = []
    trimmed_history = _trim_history(history, max_history_turns)
    active_history = trimmed_history if retain_history else []

    builder = _PROMPT_BUILDERS.get(prompt_style.lower())
    if builder is None:
        raise ValueError(
            "Unsupported prompt_style. Use 'instruction' or 'chat'."
        )

    prompt = builder(message, trimmed_history, system_prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_config = ChatConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        prompt_style=prompt_style,
        retain_history=retain_history,
        max_history_turns=max_history_turns,
    )
    generation_kwargs = _prepare_generation_kwargs(tokenizer, gen_config)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_sequence = output_ids[0]
    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = generated_sequence[prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    updated_history = active_history
    if retain_history:
        updated_history = active_history + [(message, response)]

    return response, updated_history
