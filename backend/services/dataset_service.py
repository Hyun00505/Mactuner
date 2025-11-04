"""데이터셋 처리 서비스"""
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from backend.config import settings


class DatasetService:
    """데이터셋 처리 및 분석 서비스"""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.file_info: Dict[str, Any] = {}

    def load_dataset(
        self, file_content: bytes, filename: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """파일에서 데이터셋 로드"""
        try:
            if filename.endswith(".csv"):
                self.data = pd.read_csv(io.BytesIO(file_content))
            elif filename.endswith(".json"):
                self.data = pd.read_json(io.BytesIO(file_content))
            elif filename.endswith(".jsonl"):
                self.data = pd.read_json(io.BytesIO(file_content), lines=True)
            else:
                raise ValueError(f"지원되지 않는 파일 형식: {filename}")

            # 원본 데이터 백업
            self.original_data = self.data.copy()

            # 파일 정보 저장
            self.file_info = {
                "filename": filename,
                "size_bytes": len(file_content),
                "size_mb": len(file_content) / (1024**2),
                "rows": len(self.data),
                "columns": self.data.columns.tolist(),
                "dtypes": self.data.dtypes.to_dict(),
            }

            return self.data, self.file_info

        except Exception as e:
            raise RuntimeError(f"데이터셋 로드 실패: {str(e)}")

    def get_data_info(self) -> Dict[str, Any]:
        """데이터셋 기본 정보"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        return {
            "shape": {"rows": len(self.data), "columns": len(self.data.columns)},
            "size_mb": self.data.memory_usage(deep=True).sum() / (1024**2),
            "dtypes": self.data.dtypes.to_dict(),
            "columns": self.data.columns.tolist(),
        }

    def get_preview(self, n_rows: int = 5) -> Dict[str, Any]:
        """데이터 미리보기"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        return {
            "head": self.data.head(n_rows).to_dict(orient="records"),
            "tail": self.data.tail(n_rows).to_dict(orient="records"),
            "total_rows": len(self.data),
            "preview_rows": n_rows * 2,
        }

    # ========================================
    # 데이터 정제
    # ========================================

    def handle_missing_values(
        self, strategy: str = "drop", value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """결측치 처리"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        missing_before = self.data.isnull().sum().sum()

        if strategy == "drop":
            self.data = self.data.dropna()
        elif strategy == "fill":
            self.data = self.data.fillna(value or 0)
        elif strategy == "forward_fill":
            self.data = self.data.fillna(method="ffill")
        else:
            raise ValueError(f"지원되지 않는 전략: {strategy}")

        missing_after = self.data.isnull().sum().sum()

        return {
            "strategy": strategy,
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "removed_rows": max(0, len(self.original_data) - len(self.data)),
        }

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, int]:
        """중복 행 제거"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        rows_before = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset)
        rows_after = len(self.data)

        return {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": rows_before - rows_after,
        }

    def normalize_text(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """텍스트 정규화"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        if columns is None:
            # 문자열 컬럼 자동 감지
            columns = self.data.select_dtypes(include=["object"]).columns.tolist()

        changes = {}
        for col in columns:
            if col not in self.data.columns:
                continue

            # 소문자 변환
            self.data[col] = self.data[col].str.lower()
            # 특수문자 제거
            self.data[col] = self.data[col].str.replace(r"[^a-z0-9\s]", "", regex=True)
            # 연속 공백 제거
            self.data[col] = self.data[col].str.replace(r"\s+", " ", regex=True)
            # 공백 제거
            self.data[col] = self.data[col].str.strip()

            changes[col] = "정규화 완료"

        return {"normalized_columns": changes}

    def filter_by_text_length(
        self, column: str, min_length: int = 0, max_length: int = 10000
    ) -> Dict[str, Any]:
        """텍스트 길이로 필터링"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        if column not in self.data.columns:
            raise ValueError(f"컬럼을 찾을 수 없음: {column}")

        rows_before = len(self.data)

        # 텍스트 길이 계산
        self.data = self.data[
            (self.data[column].str.len() >= min_length)
            & (self.data[column].str.len() <= max_length)
        ]

        rows_after = len(self.data)

        return {
            "column": column,
            "min_length": min_length,
            "max_length": max_length,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
        }

    def analyze_token_length(
        self, text_column: str, model_name: str = "gpt2"
    ) -> Dict[str, Any]:
        """토큰 길이 분석"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"토크나이저 로드 실패: {str(e)}")

        token_lengths = self.data[text_column].apply(
            lambda x: len(tokenizer.encode(str(x))) if pd.notna(x) else 0
        )

        return {
            "model": model_name,
            "column": text_column,
            "min_tokens": int(token_lengths.min()),
            "max_tokens": int(token_lengths.max()),
            "mean_tokens": float(token_lengths.mean()),
            "median_tokens": float(token_lengths.median()),
            "std_tokens": float(token_lengths.std()),
        }

    # ========================================
    # 탐색적 데이터 분석 (EDA)
    # ========================================

    def get_statistics(self) -> Dict[str, Any]:
        """기본 통계"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        stats = {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "memory_mb": float(self.data.memory_usage(deep=True).sum() / (1024**2)),
            "dtypes": self.data.dtypes.value_counts().to_dict(),
            "describe": self.data.describe().to_dict(),
        }

        return stats

    def get_missing_values(self) -> Dict[str, Any]:
        """결측치 분석"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data) * 100).round(2)

        return {
            "columns_with_missing": missing[missing > 0].to_dict(),
            "missing_percentage": missing_percent[missing_percent > 0].to_dict(),
            "total_missing_values": int(missing.sum()),
            "total_cells": int(len(self.data) * len(self.data.columns)),
        }

    def get_value_distribution(self, column: str, top_n: int = 10) -> Dict[str, Any]:
        """값 분포 분석"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        if column not in self.data.columns:
            raise ValueError(f"컬럼을 찾을 수 없음: {column}")

        value_counts = self.data[column].value_counts().head(top_n)

        return {
            "column": column,
            "unique_values": int(self.data[column].nunique()),
            "top_values": value_counts.to_dict(),
            "total_count": len(self.data),
        }

    def get_correlation(self) -> Dict[str, Any]:
        """수치형 컬럼 상관관계"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        numeric_data = self.data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) == 0:
            return {"message": "수치형 컬럼이 없습니다."}

        correlation = numeric_data.corr()

        return {
            "correlation": correlation.to_dict(),
            "numeric_columns": numeric_data.columns.tolist(),
        }

    # ========================================
    # 데이터 분할
    # ========================================

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 분할"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        if not 0 < test_size < 1:
            raise ValueError("test_size는 0과 1 사이여야 합니다.")

        indices = np.arange(len(self.data))
        np.random.seed(random_state)
        np.random.shuffle(indices)

        split_point = int(len(self.data) * (1 - test_size))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        train_data = self.data.iloc[train_indices].reset_index(drop=True)
        test_data = self.data.iloc[test_indices].reset_index(drop=True)

        return train_data, test_data

    def save_dataset(self, filepath: str, format: str = "csv") -> Dict[str, str]:
        """데이터셋 저장"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        filepath = Path(filepath).expanduser()

        try:
            if format == "csv":
                self.data.to_csv(filepath, index=False)
            elif format == "json":
                self.data.to_json(filepath, orient="records")
            elif format == "jsonl":
                self.data.to_json(filepath, orient="records", lines=True)
            else:
                raise ValueError(f"지원되지 않는 형식: {format}")

            return {
                "status": "success",
                "filepath": str(filepath),
                "rows": len(self.data),
                "format": format,
            }

        except Exception as e:
            raise RuntimeError(f"데이터셋 저장 실패: {str(e)}")

    def reset_data(self) -> Dict[str, str]:
        """데이터 초기화 (원본으로 복원)"""
        if self.original_data is None:
            raise ValueError("원본 데이터가 없습니다.")

        self.data = self.original_data.copy()
        return {"status": "success", "message": "데이터가 원본으로 복원되었습니다."}
