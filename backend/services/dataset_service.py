"""데이터셋 처리 서비스"""
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from backend.config import settings
import json


class DatasetService:
    """데이터셋 처리 및 분석 서비스"""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.file_info: Dict[str, Any] = {}
        self.file_encoding: str = "utf-8"  # 감지된 인코딩
        self.detected_format: str = "unknown"  # 감지된 포맷
        self.dataset_history: List[Dict[str, Any]] = []  # 데이터셋 로드 히스토리
        self.history_file = Path(settings.DATA_DIR) / "dataset_history.json"  # 히스토리 저장 파일
        self._load_history_from_file()  # 시작할 때 히스토리 로드

    def detect_encoding(self, content: bytes) -> str:
        """파일 인코딩 자동 감지"""
        try:
            detected = chardet.detect(content)
            encoding = detected.get("encoding", "utf-8")
            confidence = detected.get("confidence", 0)
            
            # 신뢰도가 낮으면 utf-8 사용
            if confidence < 0.7:
                return "utf-8"
            
            return encoding if encoding else "utf-8"
        except Exception:
            return "utf-8"

    def load_dataset(
        self, file_content: bytes, filename: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """파일에서 데이터셋 로드"""
        try:
            # 인코딩 감지
            self.file_encoding = self.detect_encoding(file_content)
            
            if filename.endswith(".csv"):
                self.detected_format = "CSV"
                self.data = pd.read_csv(io.BytesIO(file_content), encoding=self.file_encoding)
            elif filename.endswith(".json"):
                self.detected_format = "JSON"
                self.data = pd.read_json(io.BytesIO(file_content))
            elif filename.endswith(".jsonl"):
                self.detected_format = "JSONL"
                self.data = pd.read_json(io.BytesIO(file_content), lines=True)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                self.detected_format = "Excel"
                # Excel 파일 처리
                self.data = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError(f"지원되지 않는 파일 형식: {filename}")

            # 원본 데이터 백업
            self.original_data = self.data.copy()

            # dtype을 문자열로 변환하여 직렬화 가능하게 함
            dtypes_dict = {}
            for col, dtype in self.data.dtypes.items():
                dtypes_dict[col] = str(dtype)

            # 파일 정보 저장
            self.file_info = {
                "filename": filename,
                "size_bytes": len(file_content),
                "size_mb": float(len(file_content) / (1024**2)),
                "rows": len(self.data),
                "columns": self.data.columns.tolist(),
                "dtypes": dtypes_dict,
                "encoding": self.file_encoding,  # 감지된 인코딩
                "format": self.detected_format,  # 감지된 포맷
            }

            return self.data, self.file_info

        except Exception as e:
            raise RuntimeError(f"데이터셋 로드 실패: {str(e)}")

    def get_data_info(self) -> Dict[str, Any]:
        """데이터셋 기본 정보"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        # dtype을 문자열로 변환하여 직렬화 가능하게 함
        dtypes_dict = {}
        for col, dtype in self.data.dtypes.items():
            dtypes_dict[col] = str(dtype)

        return {
            "shape": {"rows": len(self.data), "columns": len(self.data.columns)},
            "size_mb": float(self.data.memory_usage(deep=True).sum() / (1024**2)),
            "dtypes": dtypes_dict,
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

    def clean_data(self, cleaning_type: str, **kwargs) -> Dict[str, Any]:
        """통합 데이터 정제"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")
        
        if cleaning_type == "missing_values":
            return self.handle_missing_values(**kwargs)
        elif cleaning_type == "duplicates":
            return self.remove_duplicates(**kwargs)
        elif cleaning_type == "normalize_text":
            return self.normalize_text(**kwargs)
        elif cleaning_type == "filter_by_length":
            return self.filter_by_text_length(**kwargs)
        else:
            raise ValueError(f"지원되지 않는 정제 작업: {cleaning_type}")

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
            "status": "success",
            "operation": "missing_values",
            "strategy": strategy,
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "removed_rows": max(0, len(self.original_data) - len(self.data)),
            "remaining_rows": len(self.data),
        }

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, int]:
        """중복 행 제거"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        rows_before = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset)
        rows_after = len(self.data)

        return {
            "status": "success",
            "operation": "duplicates",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": rows_before - rows_after,
            "remaining_rows": rows_after,
        }

    def normalize_text(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """텍스트 정규화 - LLM 파인튜닝용"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        if columns is None:
            # 문자열 컬럼 자동 감지
            columns = self.data.select_dtypes(include=["object"]).columns.tolist()

        changes = {}
        for col in columns:
            if col not in self.data.columns:
                continue

            # 원본 저장
            original_col = self.data[col].copy()

            # 소문자 변환
            self.data[col] = self.data[col].str.lower()
            # 특수문자 제거 (기본 영어/숫자만 유지)
            self.data[col] = self.data[col].str.replace(r"[^a-z0-9\s]", "", regex=True)
            # 연속 공백 제거
            self.data[col] = self.data[col].str.replace(r"\s+", " ", regex=True)
            # 공백 제거
            self.data[col] = self.data[col].str.strip()

            changes[col] = "정규화 완료"

        return {
            "status": "success",
            "operation": "normalize_text",
            "normalized_columns": changes,
            "total_normalized": len(changes),
            "remaining_rows": len(self.data),
        }

    def filter_by_text_length(
        self, column: str = None, min_length: int = 10, max_length: int = 10000
    ) -> Dict[str, Any]:
        """텍스트 길이로 필터링 - LLM 파인튜닝용"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        # 컬럼 자동 선택 (컬럼이 없으면 첫 번째 문자열 컬럼 사용)
        if column is None:
            text_columns = self.data.select_dtypes(include=["object"]).columns.tolist()
            if not text_columns:
                raise ValueError("텍스트 컬럼을 찾을 수 없습니다.")
            column = text_columns[0]

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
            "status": "success",
            "operation": "filter_by_length",
            "column": column,
            "min_length": min_length,
            "max_length": max_length,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "remaining_rows": rows_after,
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

        # dtype을 문자열로 변환
        dtypes_str = {str(k): str(v) for k, v in self.data.dtypes.value_counts().to_dict().items()}
        describe_dict = self.data.describe().to_dict()
        
        # describe 결과의 NaN을 처리
        for col in describe_dict:
            for key in describe_dict[col]:
                if isinstance(describe_dict[col][key], float):
                    if np.isnan(describe_dict[col][key]):
                        describe_dict[col][key] = None

        stats = {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "memory_mb": float(self.data.memory_usage(deep=True).sum() / (1024**2)),
            "dtypes": dtypes_str,
            "describe": describe_dict,
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
        
        # NaN을 None으로 변환
        corr_dict = correlation.to_dict()
        for col in corr_dict:
            for key in corr_dict[col]:
                if isinstance(corr_dict[col][key], float):
                    if np.isnan(corr_dict[col][key]):
                        corr_dict[col][key] = None

        return {
            "correlation": corr_dict,
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

    def save_dataset(self, filepath: str, format: str = "csv", encoding: str = None) -> Dict[str, str]:
        """데이터셋 저장"""
        if self.data is None:
            raise ValueError("로드된 데이터가 없습니다.")

        filepath = Path(filepath).expanduser()
        
        # 인코딩 설정 (기본값: 원래 파일의 인코딩 또는 utf-8)
        save_encoding = encoding or self.file_encoding or "utf-8"

        try:
            if format == "csv":
                self.data.to_csv(filepath, index=False, encoding=save_encoding)
            elif format == "json":
                self.data.to_json(filepath, orient="records", force_ascii=False)
            elif format == "jsonl":
                self.data.to_json(filepath, orient="records", lines=True, force_ascii=False)
            elif format == "excel":
                self.data.to_excel(filepath, index=False, sheet_name="Data")
            else:
                raise ValueError(f"지원되지 않는 형식: {format}")

            return {
                "status": "success",
                "filepath": str(filepath),
                "rows": len(self.data),
                "columns": len(self.data.columns),
                "format": format,
                "encoding": save_encoding,
                "size_mb": float(filepath.stat().st_size / (1024**2)),
            }

        except Exception as e:
            raise RuntimeError(f"데이터셋 저장 실패: {str(e)}")

    def reset_data(self) -> Dict[str, str]:
        """데이터 초기화 (원본으로 복원)"""
        if self.original_data is None:
            raise ValueError("원본 데이터가 없습니다.")

        self.data = self.original_data.copy()
        return {"status": "success", "message": "데이터가 원본으로 복원되었습니다."}
    
    def _load_demo_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """테스트용 더미 데이터셋 로드 (인터넷 불필요)"""
        import random
        
        # 테스트 데이터 생성
        num_rows = max_samples or 1000
        
        if dataset_name.lower() == "test":
            # 기본 테스트 데이터
            data = {
                "id": list(range(1, num_rows + 1)),
                "text": [f"샘플 텍스트 {i}" for i in range(1, num_rows + 1)],
                "label": [random.choice(["긍정", "부정", "중립"]) for _ in range(num_rows)],
                "score": [round(random.uniform(0, 1), 2) for _ in range(num_rows)],
            }
        elif dataset_name.lower() == "demo":
            # 더미 한국어 데이터
            data = {
                "한글_제목": [f"제목_{i}" for i in range(1, num_rows + 1)],
                "내용": [f"이것은 테스트 데이터셋입니다. 행 번호: {i}" for i in range(1, num_rows + 1)],
                "카테고리": [random.choice(["뉴스", "블로그", "SNS", "리뷰"]) for _ in range(num_rows)],
                "좋아요": [random.randint(0, 1000) for _ in range(num_rows)],
                "댓글_수": [random.randint(0, 500) for _ in range(num_rows)],
            }
        else:
            # 일반 더미 데이터
            data = {
                "index": list(range(1, num_rows + 1)),
                "value_a": [random.uniform(0, 100) for _ in range(num_rows)],
                "value_b": [random.uniform(0, 100) for _ in range(num_rows)],
                "category": [random.choice(["A", "B", "C"]) for _ in range(num_rows)],
            }
        
        # DataFrame 생성
        self.data = pd.DataFrame(data)
        self.original_data = self.data.copy()
        
        # dtype을 문자열로 변환
        dtypes_dict = {}
        for col, dtype in self.data.dtypes.items():
            dtypes_dict[col] = str(dtype)
        
        # 파일 정보 저장
        self.file_encoding = "utf-8"
        self.detected_format = "Demo Dataset"
        
        self.file_info = {
            "filename": f"{dataset_name}_demo.parquet",
            "rows": len(self.data),
            "columns": self.data.columns.tolist(),
            "dtypes": dtypes_dict,
            "encoding": self.file_encoding,
            "format": self.detected_format,
            "dataset_id": f"demo/{dataset_name}",
            "split": "train",
        }
        
        return {
            "status": "success",
            "message": f"✅ 테스트 데이터셋 로드 완료! (더미 데이터)",
            "dataset_info": self.file_info,
            "file_info": self.file_info,
        }

    def download_hf_dataset(
        self,
        dataset_id: str,
        hf_token: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """HuggingFace 데이터셋 다운로드 및 로드 (로컬 캐시 자동 활용)"""
        try:
            import logging
            from pathlib import Path
            logger = logging.getLogger(__name__)
            
            # dataset_id 정규화
            dataset_id = dataset_id.strip()
            
            # 테스트용 더미 데이터셋
            if dataset_id.lower() in ["test", "demo", "example"]:
                return self._load_demo_dataset(dataset_id, max_samples)
            
            logger.info(f"🔄 HuggingFace 데이터셋 로드 시작: {dataset_id}")
            
            # HuggingFace에서 데이터셋 로드 (자동 캐싱)
            # cache_dir을 명시적으로 설정하여 로컬 캐시 활용
            hf_cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
            
            kwargs = {
                "cache_dir": str(hf_cache_dir),  # 로컬 캐시 디렉토리 명시
                "trust_remote_code": True,  # 원격 코드 신뢰
            }
            if hf_token:
                kwargs["token"] = hf_token
            
            try:
                logger.info(f"📥 로컬 캐시 확인: {hf_cache_dir}")
                dataset = load_dataset(dataset_id, split=split, **kwargs)
                logger.info(f"✅ 데이터셋 로드 완료 (캐시 활용: {hf_cache_dir})")
            except Exception as first_error:
                # 첫 번째 시도 실패 시, 여러 형식을 시도
                alternate_ids = []
                
                # username/name 형식이면 datasets/ prefix 추가 시도
                if "/" in dataset_id and not dataset_id.startswith("datasets/"):
                    alternate_ids.append(f"datasets/{dataset_id}")
                
                # 다른 split 시도 (첫 번째 시도가 split 문제일 수 있음)
                alternate_splits = ["validation", "test", None]
                
                dataset_loaded = False
                last_error = first_error
                
                # 다른 형식 및 split 시도
                for alt_id in alternate_ids:
                    for alt_split in alternate_splits:
                        try:
                            logger.info(f"Trying: {alt_id} with split: {alt_split}")
                            if alt_split:
                                dataset = load_dataset(alt_id, split=alt_split, **kwargs)
                            else:
                                # split 없이 전체 데이터셋 로드
                                dataset_full = load_dataset(alt_id, **kwargs)
                                # 첫 번째 split 사용
                                if isinstance(dataset_full, dict):
                                    dataset = dataset_full[list(dataset_full.keys())[0]]
                                else:
                                    dataset = dataset_full
                            
                            dataset_id = alt_id
                            dataset_loaded = True
                            logger.info(f"✅ Successfully loaded: {alt_id}")
                            break
                        except Exception as e:
                            last_error = e
                            continue
                    
                    if dataset_loaded:
                        break
                
                if not dataset_loaded:
                    logger.error(f"All attempts failed. Last error: {last_error}")
                    raise first_error
            
            # 샘플 수 제한
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # DataFrame으로 변환
            self.data = dataset.to_pandas()
            self.original_data = self.data.copy()
            
            # dtype을 문자열로 변환
            dtypes_dict = {}
            for col, dtype in self.data.dtypes.items():
                dtypes_dict[col] = str(dtype)
            
            # 파일 정보 저장
            self.file_encoding = "utf-8"
            self.detected_format = "HuggingFace Dataset"
            
            self.file_info = {
                "filename": f"{dataset_id.split('/')[-1]}.parquet",
                "rows": len(self.data),
                "columns": self.data.columns.tolist(),
                "dtypes": dtypes_dict,
                "encoding": self.file_encoding,
                "format": self.detected_format,
                "dataset_id": dataset_id,
                "split": split,
            }
            
            return {
                "status": "success",
                "message": f"✅ HuggingFace 데이터셋 로드 완료!",
                "dataset_info": self.file_info,
                "file_info": self.file_info,
            }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Dataset loading failed: {error_msg}")
            
            # 더 친절한 에러 메시지 제공
            if "doesn't exist" in error_msg or "cannot be accessed" in error_msg:
                detail = (
                    f"❌ 데이터셋을 찾을 수 없습니다.\n"
                    f"• 데이터셋 ID: {dataset_id}\n"
                    f"• Split: {split}\n\n"
                    f"💡 다음을 확인하세요:\n"
                    f"1. 인터넷 연결 확인 (HuggingFace Hub 접근 필요)\n"
                    f"2. 정확한 ID 입력 (예: username/dataset-name)\n"
                    f"3. Split 이름 확인 (train, validation, test 등)\n"
                    f"4. Private 데이터셋이면 토큰 입력\n"
                    f"5. HuggingFace 웹사이트에서 데이터셋 존재 확인"
                )
                raise RuntimeError(detail)
            elif "Couldn't reach" in error_msg or "Failed to resolve" in error_msg or "Network" in error_msg:
                detail = (
                    f"❌ 네트워크 연결 오류\n\n"
                    f"HuggingFace Hub에 연결할 수 없습니다.\n"
                    f"• 데이터셋 ID: {dataset_id}\n\n"
                    f"💡 해결 방법:\n"
                    f"1. 인터넷 연결 확인\n"
                    f"2. 방화벽/VPN 설정 확인\n"
                    f"3. HuggingFace 서버 상태 확인"
                )
                raise RuntimeError(detail)
            else:
                raise RuntimeError(f"HuggingFace 데이터셋 다운로드 실패: {error_msg}")
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(val) for key, val in obj.items()}
        else:
            return str(obj)
    
    def add_to_history(self, source: str = "file", **kwargs) -> None:
        """데이터셋 히스토리에 항목 추가"""
        import datetime
        
        if not self.file_info:
            return
        
        # 중복 제거 (같은 파일 정보가 이미 있으면 기존 항목 제거)
        self.dataset_history = [
            h for h in self.dataset_history
            if h.get("filename") != self.file_info.get("filename")
        ]
        
        # file_info의 모든 값을 직렬화 가능한 타입으로 변환
        safe_file_info = {
            key: self._convert_to_serializable(value)
            for key, value in self.file_info.items()
        }
        
        # kwargs의 모든 값을 직렬화 가능한 타입으로 변환 (None 제외)
        safe_kwargs = {
            key: self._convert_to_serializable(value)
            for key, value in kwargs.items()
            if value is not None
        }
        
        history_item = {
            "id": len(self.dataset_history) + 1,
            "source": source,  # "file", "hf", "demo"
            "filename": str(safe_file_info.get("filename", "unknown")),
            "rows": int(safe_file_info.get("rows", 0)),
            "columns": int(len(safe_file_info.get("columns", []))),
            "size_mb": float(safe_file_info.get("size_mb", 0)),
            "format": str(safe_file_info.get("format", "unknown")),
            "timestamp": datetime.datetime.now().isoformat(),
            **safe_kwargs,  # dataset_id, hf_token 등 추가 정보 (모두 안전함)
        }
        
        # 최신 항목을 리스트 앞에 추가
        self.dataset_history.insert(0, history_item)
        
        # 최대 10개 항목만 유지
        self.dataset_history = self.dataset_history[:10]
        
        # 파일에 저장
        self._save_history_to_file()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """데이터셋 로드 히스토리 조회"""
        return self.dataset_history
    
    def _load_history_from_file(self) -> None:
        """파일에서 히스토리 로드"""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.dataset_history = json.load(f)
        except Exception as e:
            print(f"히스토리 로드 실패: {e}")
            self.dataset_history = []
    
    def _save_history_to_file(self) -> None:
        """히스토리를 파일에 저장"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.dataset_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
    
    def clear_history(self, delete_data: bool = False) -> Dict[str, Any]:
        """히스토리 삭제"""
        result = {
            "status": "success",
            "message": "히스토리가 삭제되었습니다.",
            "deleted_count": len(self.dataset_history),
        }
        
        if delete_data:
            result["message"] = "히스토리와 데이터가 모두 삭제되었습니다."
            self.data = None
            self.original_data = None
            self.file_info = {}
        
        self.dataset_history = []
        self._save_history_to_file()
        
        return result
    
    def delete_history_item(self, index: int, delete_data: bool = False) -> Dict[str, Any]:
        """특정 히스토리 항목 삭제"""
        if index < 0 or index >= len(self.dataset_history):
            raise ValueError("유효하지 않은 인덱스")
        
        deleted_item = self.dataset_history.pop(index)
        self._save_history_to_file()
        
        message = f"{deleted_item.get('filename', 'Unknown')} 히스토리가 삭제되었습니다."
        
        if delete_data:
            message = f"{deleted_item.get('filename', 'Unknown')} 파일이 삭제되었습니다."
            # 현재 로드된 데이터가 이 항목과 같으면 삭제
            if self.file_info and self.file_info.get("filename") == deleted_item.get("filename"):
                self.data = None
                self.original_data = None
                self.file_info = {}
        
        return {
            "status": "success",
            "message": message,
            "deleted_item": deleted_item,
        }
