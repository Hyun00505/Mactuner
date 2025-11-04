"""Utility functions for loading datasets and running lightweight EDA.

This module focuses on local data files that can be consumed by
`datasets.Dataset`.  It exposes helpers for loading, cleaning, normalising and
analysing text datasets that follow common instruction-tuning schemas (e.g.
`instruction`/`input`/`output` or `prompt`/`response`).
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv"}


class DatasetLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded."""


def _detect_loader(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetLoadError(
            "❌ 지원되지 않는 파일 형식입니다 (지원: .json, .jsonl, .csv)"
        )
    if ext in {".json", ".jsonl"}:
        return "json"
    return "csv"


def load_dataset(path: str) -> Dataset:
    """Load a dataset from JSON/JSONL/CSV using Hugging Face datasets.

    Args:
        path: Local filesystem path to the dataset file.

    Returns:
        A :class:`datasets.Dataset` instance containing the data.

    Raises:
        DatasetLoadError: If the file cannot be loaded or is unsupported.
    """

    if not os.path.exists(path):
        raise DatasetLoadError("❌ 데이터 파일을 찾을 수 없습니다.")

    loader_name = _detect_loader(path)
    data_files = {"train": path}

    try:
        ds_dict = load_dataset(loader_name, data_files=data_files)
    except Exception as err:  # pragma: no cover - datasets raises many types
        raise DatasetLoadError(
            "❌ 데이터셋을 로드할 수 없습니다. 경로와 형식을 확인하세요."
        ) from err

    return ds_dict["train"]


def clean_dataset(
    ds: Dataset,
    remove_empty: bool = True,
    dedup: bool = True,
    text_fields: Optional[Iterable[str]] = None,
) -> Dataset:
    """Apply lightweight cleaning to the dataset.

    Args:
        ds: Input dataset.
        remove_empty: Remove rows where tracked text fields are empty.
        dedup: Drop duplicate rows.
        text_fields: Iterable of fields to validate. When ``None`` the
            intersection of available standard fields is used.
    """

    cleaned = ds

    if text_fields is None:
        candidates = (
            "instruction",
            "input",
            "output",
            "prompt",
            "response",
        )
        text_fields = [field for field in candidates if field in cleaned.column_names]

    tracked_fields = tuple(text_fields)

    if remove_empty and tracked_fields:
        def _non_empty(example: Dict[str, object]) -> bool:
            for field in tracked_fields:
                value = example.get(field)
                if value is None:
                    return False
                if isinstance(value, str) and not value.strip():
                    return False
            return True

        cleaned = cleaned.filter(_non_empty)

    if dedup:
        seen = set()

        def _is_new(example: Dict[str, object]) -> bool:
            if tracked_fields:
                key = tuple(
                    (field, _to_hashable(example.get(field))) for field in tracked_fields
                )
            else:
                key = tuple(
                    sorted((k, _to_hashable(v)) for k, v in example.items())
                )
            if key in seen:
                return False
            seen.add(key)
            return True

        cleaned = cleaned.filter(_is_new)

    return cleaned


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(example: Dict[str, object]) -> Dict[str, object]:
    """Normalise text fields in a dataset example.

    The transformation lowercases strings, strips punctuation/symbols and
    collapses repeated whitespace.
    """

    normalised = dict(example)
    for key, value in example.items():
        if isinstance(value, str):
            text = value.lower()
            text = _PUNCT_RE.sub("", text)
            text = _WHITESPACE_RE.sub(" ", text)
            normalised[key] = text.strip()
    return normalised


def _describe_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"avg": 0.0, "min": 0, "max": 0, "median": 0.0}
    return {
        "avg": float(mean(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "median": float(median(lengths)),
    }


def _histogram(lengths: List[int], bins: int = 10) -> Dict[str, int]:
    if not lengths:
        return {}
    max_len = max(lengths)
    bin_size = max(1, math.ceil(max_len / bins))
    counter: Counter[str] = Counter()
    for length in lengths:
        bucket_index = length // bin_size
        if bucket_index >= bins:
            bucket_index = bins - 1
        start = bucket_index * bin_size
        end = start + bin_size
        label = f"{start}-{end}"
        counter[label] += 1
    return dict(counter)


def _to_hashable(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in value.items()))
    return value


def analyze_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    input_field: str,
    output_field: str,
) -> Dict[str, object]:
    """Compute descriptive statistics for the dataset.

    Raises:
        ValueError: If the specified fields are missing.
    """

    missing_fields = {}
    num_examples = len(ds)

    for field in (input_field, output_field):
        if field not in ds.column_names:
            raise ValueError(
                "❌ 필수 필드(instruction/output)가 일부 샘플에 없습니다."
            )

    input_lengths: List[int] = []
    output_lengths: List[int] = []
    missing_counts = {input_field: 0, output_field: 0}

    for example in ds:
        input_text = example.get(input_field)
        output_text = example.get(output_field)

        if input_text in (None, ""):
            missing_counts[input_field] += 1
            input_text = ""
        if output_text in (None, ""):
            missing_counts[output_field] += 1
            output_text = ""

        input_lengths.append(
            len(tokenizer.encode(str(input_text), add_special_tokens=False))
        )
        output_lengths.append(
            len(tokenizer.encode(str(output_text), add_special_tokens=False))
        )

    missing_fields = {
        field: (missing_counts[field] / num_examples if num_examples else 0.0)
        for field in (input_field, output_field)
    }

    stats = {
        "num_examples": num_examples,
        "input_length": _describe_lengths(input_lengths),
        "output_length": _describe_lengths(output_lengths),
        "input_histogram": _histogram(input_lengths),
        "output_histogram": _histogram(output_lengths),
        "missing_fields": missing_fields,
    }

    return stats


def preview_samples(ds: Dataset, n: int = 3) -> List[Dict[str, object]]:
    """Return up to ``n`` preview samples from the dataset."""

    n = max(0, n)
    if n == 0 or len(ds) == 0:
        return []

    subset = ds.shuffle(seed=int(time.time())).select(range(min(n, len(ds))))
    previews: List[Dict[str, object]] = []

    for example in subset:
        snippet = {}
        for field_pair in (
            ("instruction", "input", "output"),
            ("prompt", "response"),
        ):
            if len(field_pair) == 3:
                instr, inp, out = field_pair
                if instr in example:
                    snippet[instr] = example.get(instr)
                if inp in example:
                    snippet[inp] = example.get(inp)
                if out in example:
                    snippet[out] = example.get(out)
            else:
                prompt, response = field_pair
                if prompt in example:
                    snippet[prompt] = example.get(prompt)
                if response in example:
                    snippet[response] = example.get(response)
        previews.append(snippet)

    return previews


__all__ = [
    "DatasetLoadError",
    "analyze_dataset",
    "clean_dataset",
    "load_dataset",
    "normalize_text",
    "preview_samples",
]
