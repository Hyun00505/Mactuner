"""양자화 및 GGUF 변환 서비스"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.config import settings
from backend.utils.mac_optimization import MACOptimizer

logger = logging.getLogger(__name__)


class QuantizationService:
    """모델 양자화 및 GGUF 변환 서비스"""

    # 지원되는 양자화 방식
    QUANTIZATION_METHODS = {
        "Q2_K": {"description": "초저용량 (극단적 압축)", "quality": 1, "size": 1},
        "Q3_K": {"description": "매우 낮음 (최소 크기)", "quality": 2, "size": 1},
        "Q4_0": {"description": "낮음 (빠른 실행)", "quality": 3, "size": 2},
        "Q4_K": {"description": "중간 낮음 (권장)", "quality": 4, "size": 2},
        "Q5_0": {"description": "중간", "quality": 5, "size": 3},
        "Q5_K": {"description": "중간 높음", "quality": 6, "size": 3},
        "Q6_K": {"description": "높음 (좋은 품질)", "quality": 7, "size": 4},
        "Q8_0": {"description": "매우 높음 (최고 품질)", "quality": 8, "size": 5},
        "F16": {"description": "Full 16-bit (원본 수준)", "quality": 9, "size": 6},
        "F32": {"description": "Full 32-bit (최고 정확도)", "quality": 10, "size": 7},
    }

    def __init__(self):
        self.device = MACOptimizer.get_device()
        self.conversion_history: List[Dict[str, Any]] = []

    # ========================================
    # 양자화 방식 조회
    # ========================================

    def get_available_methods(self) -> Dict[str, Any]:
        """사용 가능한 양자화 방식 조회"""
        return {
            "status": "success",
            "available_methods": self.QUANTIZATION_METHODS,
            "total_methods": len(self.QUANTIZATION_METHODS),
            "recommended_method": "Q4_K",  # 권장 방식
        }

    def get_recommended_method(
        self,
        model_size_gb: float,
        target_size_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """모델 크기에 따른 권장 양자화 방식"""
        recommendations = []

        # 메모리 상태 확인
        memory_stats = MACOptimizer.get_memory_stats()
        available_memory_gb = memory_stats["available_gb"]

        # 모델 크기에 따른 권장
        if model_size_gb > 50:  # 50GB+
            primary = "Q2_K"  # 초저용량 필수
            secondary = ["Q3_K", "Q4_0"]
        elif model_size_gb > 30:  # 30-50GB
            primary = "Q3_K"
            secondary = ["Q4_0", "Q4_K"]
        elif model_size_gb > 15:  # 15-30GB
            primary = "Q4_0"
            secondary = ["Q4_K", "Q5_K"]
        elif model_size_gb > 8:  # 8-15GB
            primary = "Q4_K"
            secondary = ["Q5_K", "Q6_K"]
        elif model_size_gb > 4:  # 4-8GB
            primary = "Q5_K"
            secondary = ["Q6_K", "Q8_0"]
        else:  # <4GB
            primary = "Q6_K"
            secondary = ["Q8_0", "F16"]

        recommendations.append(
            {
                "rank": 1,
                "method": primary,
                "reason": "메모리 효율성 최적",
                "info": self.QUANTIZATION_METHODS[primary],
            }
        )

        for i, method in enumerate(secondary, start=2):
            recommendations.append(
                {
                    "rank": i,
                    "method": method,
                    "reason": "대안",
                    "info": self.QUANTIZATION_METHODS[method],
                }
            )

        return {
            "status": "success",
            "model_size_gb": model_size_gb,
            "available_memory_gb": available_memory_gb,
            "primary_recommendation": primary,
            "all_recommendations": recommendations,
        }

    # ========================================
    # GGUF 변환
    # ========================================

    def convert_to_gguf(
        self,
        model_path: str,
        output_dir: str,
        quantization_method: str = "Q4_K",
        use_gpu: bool = True,
    ) -> Dict[str, Any]:
        """모델을 GGUF 형식으로 변환"""
        try:
            model_path = Path(model_path).expanduser()
            output_dir = Path(output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            if quantization_method not in self.QUANTIZATION_METHODS:
                raise ValueError(
                    f"지원되지 않는 양자화 방식: {quantization_method}. "
                    f"사용 가능한 방식: {list(self.QUANTIZATION_METHODS.keys())}"
                )

            # 1. 모델 메타데이터 확인
            model_info = self._get_model_info(model_path)

            # 2. 변환 명령어 구성
            output_file = output_dir / f"model-{quantization_method}.gguf"

            # llama-cpp-python의 convert 스크립트 사용
            conversion_result = self._run_conversion(
                model_path=str(model_path),
                output_file=str(output_file),
                quantization_method=quantization_method,
                use_gpu=use_gpu,
            )

            # 3. 결과 검증
            if not output_file.exists():
                raise RuntimeError("GGUF 변환 실패: 출력 파일이 생성되지 않음")

            # 4. 파일 정보 수집
            file_size_gb = output_file.stat().st_size / (1024**3)
            compression_ratio = model_info["original_size_gb"] / file_size_gb

            result = {
                "status": "success",
                "model_path": str(model_path),
                "output_file": str(output_file),
                "quantization_method": quantization_method,
                "method_info": self.QUANTIZATION_METHODS[quantization_method],
                "original_size_gb": round(model_info["original_size_gb"], 2),
                "compressed_size_gb": round(file_size_gb, 2),
                "compression_ratio": round(compression_ratio, 2),
                "device": str(self.device),
            }

            # 히스토리 저장
            self.conversion_history.append(result)

            return result

        except Exception as e:
            raise RuntimeError(f"GGUF 변환 실패: {str(e)}")

    def _get_model_info(self, model_path: Path) -> Dict[str, float]:
        """모델 파일 정보 조회"""
        total_size = 0

        if model_path.is_dir():
            for file in model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
        else:
            total_size = model_path.stat().st_size

        return {
            "original_size_gb": total_size / (1024**3),
        }

    def _run_conversion(
        self,
        model_path: str,
        output_file: str,
        quantization_method: str,
        use_gpu: bool,
    ) -> Dict[str, Any]:
        """llama.cpp 변환 스크립트 실행"""
        try:
            # llama.cpp의 convert 스크립트 사용
            # 설치: pip install llama-cpp-python

            cmd = [
                "python",
                "-m",
                "llama_cpp",
                "convert",
                "--model",
                model_path,
                "--output",
                output_file,
                "--quantize",
                quantization_method,
            ]

            # GPU 사용 옵션
            if use_gpu and self.device.type == "mps":
                cmd.extend(["--device", "metal"])
            elif use_gpu and self.device.type == "cuda":
                cmd.extend(["--device", "cuda"])

            logger.info(f"GGUF 변환 시작: {quantization_method}")
            logger.info(f"명령어: {' '.join(cmd)}")

            # 대체 방법: llama.cpp Python 바인딩 직접 사용
            return self._convert_with_python_binding(
                model_path, output_file, quantization_method
            )

        except Exception as e:
            logger.warning(f"명령어 실행 실패, Python 바인딩으로 시도: {str(e)}")
            return self._convert_with_python_binding(
                model_path, output_file, quantization_method
            )

    def _convert_with_python_binding(
        self,
        model_path: str,
        output_file: str,
        quantization_method: str,
    ) -> Dict[str, Any]:
        """Python 바인딩을 사용한 변환"""
        try:
            import llama_cpp

            logger.info(f"llama-cpp-python을 사용한 변환: {quantization_method}")

            # 모델 로드
            model = llama_cpp.Llama(
                model_path=model_path,
                verbose=False,
                quantization=quantization_method,
            )

            # 메타데이터 저장
            metadata = {
                "original_model": model_path,
                "quantization_method": quantization_method,
                "format": "GGUF",
            }

            return {
                "status": "success",
                "method": "python_binding",
                "metadata": metadata,
            }

        except ImportError:
            raise RuntimeError(
                "llama-cpp-python이 설치되지 않았습니다. "
                "설치: pip install llama-cpp-python"
            )

    # ========================================
    # 변환 검증
    # ========================================

    def validate_gguf(self, gguf_path: str) -> Dict[str, Any]:
        """GGUF 파일 유효성 검증"""
        try:
            gguf_path = Path(gguf_path).expanduser()

            if not gguf_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {gguf_path}")

            if not gguf_path.suffix == ".gguf":
                raise ValueError(f"GGUF 파일이 아님: {gguf_path}")

            # 파일 크기 확인
            file_size_gb = gguf_path.stat().st_size / (1024**3)

            # GGUF 파일 헤더 검증
            with open(gguf_path, "rb") as f:
                magic = f.read(4)
                is_valid_gguf = magic == b"GGUF"

            return {
                "status": "success",
                "file_path": str(gguf_path),
                "file_size_gb": round(file_size_gb, 2),
                "is_valid_gguf": is_valid_gguf,
                "file_exists": gguf_path.exists(),
            }

        except Exception as e:
            raise RuntimeError(f"GGUF 검증 실패: {str(e)}")

    # ========================================
    # 변환 이력
    # ========================================

    def get_conversion_history(self) -> Dict[str, Any]:
        """변환 이력 조회"""
        return {
            "status": "success",
            "history": self.conversion_history,
            "total_conversions": len(self.conversion_history),
        }

    def clear_conversion_history(self) -> Dict[str, str]:
        """변환 이력 초기화"""
        self.conversion_history = []
        return {
            "status": "success",
            "message": "Conversion history cleared",
        }

    # ========================================
    # 압축 통계
    # ========================================

    def get_compression_statistics(self) -> Dict[str, Any]:
        """압축 통계"""
        if not self.conversion_history:
            return {
                "status": "success",
                "total_conversions": 0,
                "message": "No conversion history",
            }

        total_original = sum(h["original_size_gb"] for h in self.conversion_history)
        total_compressed = sum(h["compressed_size_gb"] for h in self.conversion_history)
        avg_ratio = total_original / total_compressed if total_compressed > 0 else 0

        method_stats = {}
        for history in self.conversion_history:
            method = history["quantization_method"]
            if method not in method_stats:
                method_stats[method] = {
                    "count": 0,
                    "total_original_gb": 0,
                    "total_compressed_gb": 0,
                }
            method_stats[method]["count"] += 1
            method_stats[method]["total_original_gb"] += history["original_size_gb"]
            method_stats[method]["total_compressed_gb"] += history["compressed_size_gb"]

        return {
            "status": "success",
            "total_conversions": len(self.conversion_history),
            "total_original_gb": round(total_original, 2),
            "total_compressed_gb": round(total_compressed, 2),
            "average_compression_ratio": round(avg_ratio, 2),
            "by_method": method_stats,
        }
