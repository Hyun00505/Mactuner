"""MAC (Apple Silicon) 최적화 관련 유틸리티"""
import psutil
import torch


class MACOptimizer:
    """MAC (Apple Silicon) 최적화 관련 유틸리티"""

    @staticmethod
    def get_device():
        """MAC MPS 또는 CPU 선택"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def get_optimal_batch_size(model_params: int) -> int:
        """모델 크기에 따른 최적 배치 크기"""
        # 가용 메모리 확인 (GB)
        memory_gb = psutil.virtual_memory().available / (1024**3)

        if model_params > 7e9:  # 7B+
            return max(1, int(memory_gb / 8))
        elif model_params > 1e9:  # 1B-7B
            return max(4, int(memory_gb / 4))
        else:  # <1B
            return max(8, int(memory_gb / 2))

    @staticmethod
    def get_memory_stats() -> dict:
        """메모리 사용 통계"""
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent": vm.percent,
            "device": str(MACOptimizer.get_device()),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        }

    @staticmethod
    def estimate_model_memory(model_params: int, dtype_bytes: int = 2) -> float:
        """모델 메모리 사용량 추정 (GB)"""
        # 모델 파라미터 + 옵티마이저 상태 + 활성화
        return (model_params * dtype_bytes * 3) / (1024**3)
