"""
디바이스 관리 모듈
GPU/CPU 감지 및 선택, 최적화된 디바이스 관리를 제공합니다.
"""

import torch
import logging
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """지원되는 디바이스 타입"""
    MPS = "mps"          # Mac Metal Performance Shaders
    CUDA = "cuda"        # NVIDIA GPU
    CPU = "cpu"          # CPU


@dataclass
class DeviceInfo:
    """디바이스 정보"""
    type: DeviceType
    name: str
    is_available: bool
    memory_total: Optional[float] = None      # GB
    memory_allocated: Optional[float] = None  # GB
    memory_reserved: Optional[float] = None   # GB
    compute_capability: Optional[str] = None  # CUDA만 해당
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "type": self.type.value,
            "name": self.name,
            "is_available": self.is_available,
            "memory_total": self.memory_total,
            "memory_allocated": self.memory_allocated,
            "memory_reserved": self.memory_reserved,
            "compute_capability": self.compute_capability,
        }


class DeviceManager:
    """디바이스 관리 클래스 (싱글톤 패턴)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """초기화 (한 번만 실행)"""
        if not DeviceManager._initialized:
            self._current_device: Optional[torch.device] = None
            self._selected_device_type: Optional[DeviceType] = None
            self._available_devices: List[DeviceInfo] = []
            self._detect_devices()
            DeviceManager._initialized = True
    
    def _detect_devices(self) -> None:
        """사용 가능한 모든 디바이스 감지"""
        logger.info("🔍 시스템 디바이스 감지 중...")
        
        # 1. MPS (Mac Metal Performance Shaders) 감지
        if torch.backends.mps.is_available():
            self._available_devices.append(
                DeviceInfo(
                    type=DeviceType.MPS,
                    name="Apple Metal Performance Shaders",
                    is_available=True,
                )
            )
            logger.info("✅ MPS (Mac GPU) 감지됨")
        
        # 2. CUDA (NVIDIA GPU) 감지
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024 ** 3)  # GB로 변환
                    
                    self._available_devices.append(
                        DeviceInfo(
                            type=DeviceType.CUDA,
                            name=f"{props.name} (ID: {i})",
                            is_available=True,
                            memory_total=total_memory,
                            compute_capability=f"{props.major}.{props.minor}",
                        )
                    )
                    logger.info(f"✅ CUDA GPU 감지됨: {props.name} ({total_memory:.2f}GB)")
                except Exception as e:
                    logger.warning(f"⚠️  CUDA 정보 조회 실패: {e}")
        
        # 3. CPU (항상 사용 가능)
        self._available_devices.append(
            DeviceInfo(
                type=DeviceType.CPU,
                name="CPU (Intel/AMD)",
                is_available=True,
            )
        )
        logger.info("✅ CPU 감지됨")
        
        logger.info(f"📊 총 {len(self._available_devices)}개 디바이스 감지됨")
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """사용 가능한 모든 디바이스 반환"""
        return self._available_devices
    
    def select_device(self, device_type: str) -> bool:
        """디바이스 선택
        
        Args:
            device_type: 선택할 디바이스 타입 ("mps", "cuda", "cpu")
        
        Returns:
            성공 여부
        """
        try:
            device_type = device_type.lower()
            
            # 유효한 타입 확인
            if device_type not in [d.value for d in DeviceType]:
                logger.error(f"❌ 지원하지 않는 디바이스 타입: {device_type}")
                return False
            
            # 사용 가능 여부 확인
            device_enum = DeviceType(device_type)
            available_device = next(
                (d for d in self._available_devices 
                 if d.type == device_enum and d.is_available),
                None
            )
            
            if not available_device:
                logger.error(f"❌ 사용 불가능한 디바이스: {device_type}")
                return False
            
            # 디바이스 설정
            if device_type == "mps":
                self._current_device = torch.device("mps")
            elif device_type == "cuda":
                self._current_device = torch.device("cuda")
            else:  # cpu
                self._current_device = torch.device("cpu")
            
            self._selected_device_type = device_enum
            logger.info(f"✅ 디바이스 선택 완료: {device_type}")
            logger.info(f"🎯 현재 디바이스: {self._current_device}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ 디바이스 선택 중 오류: {e}")
            return False
    
    def get_current_device(self) -> torch.device:
        """현재 선택된 디바이스 반환"""
        if self._current_device is None:
            # 기본값: 최적의 디바이스 선택
            self.auto_select_device()
        return self._current_device
    
    def get_current_device_type(self) -> Optional[DeviceType]:
        """현재 선택된 디바이스 타입 반환"""
        return self._selected_device_type
    
    def auto_select_device(self) -> bool:
        """최적의 디바이스 자동 선택
        우선순위: MPS > CUDA > CPU
        """
        logger.info("🤖 최적 디바이스 자동 선택 중...")
        
        # MPS 우선
        if any(d.type == DeviceType.MPS and d.is_available for d in self._available_devices):
            return self.select_device("mps")
        
        # CUDA 다음
        if any(d.type == DeviceType.CUDA and d.is_available for d in self._available_devices):
            return self.select_device("cuda")
        
        # CPU 최후
        return self.select_device("cpu")
    
    def move_tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """텐서를 현재 선택된 디바이스로 이동"""
        device = self.get_current_device()
        return tensor.to(device)
    
    def move_model_to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """모델을 현재 선택된 디바이스로 이동"""
        device = self.get_current_device()
        return model.to(device)
    
    def get_device_memory_info(self) -> Dict:
        """현재 디바이스의 메모리 정보 반환"""
        device = self.get_current_device()
        
        if device.type == "cuda":
            return {
                "device": str(device),
                "allocated": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "reserved": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "total": torch.cuda.get_device_properties(device.index).total_memory / (1024 ** 3),
            }
        elif device.type == "mps":
            # MPS는 메모리 정보를 제공하지 않음
            return {
                "device": str(device),
                "allocated": None,
                "reserved": None,
                "total": None,
            }
        else:  # cpu
            return {
                "device": str(device),
                "allocated": None,
                "reserved": None,
                "total": None,
            }
    
    def clear_cache(self) -> None:
        """디바이스 캐시 메모리 정리"""
        device = self.get_current_device()
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("✅ CUDA 캐시 정리 완료")
        elif device.type == "mps":
            # MPS는 수동 메모리 관리 지원하지 않음
            logger.info("ℹ️  MPS는 자동 메모리 관리를 사용합니다")
        else:  # cpu
            logger.info("ℹ️  CPU는 캐시 정리가 필요하지 않습니다")
    
    def get_optimal_dtype(self) -> torch.dtype:
        """현재 디바이스에 최적화된 데이터 타입 반환"""
        device = self.get_current_device()
        
        if device.type == "cuda":
            # CUDA에서는 float16 또는 bfloat16 권장
            if torch.cuda.is_available() and torch.cuda.get_device_capability(device.index)[0] >= 8:
                return torch.bfloat16  # Ampere 이상
            return torch.float16
        elif device.type == "mps":
            # MPS는 float32 또는 float16 지원
            return torch.float32
        else:  # cpu
            return torch.float32
    
    def get_status_info(self) -> Dict:
        """현재 디바이스 상태 정보 반환"""
        current_device = self.get_current_device()
        memory_info = self.get_device_memory_info()
        
        return {
            "current_device": str(current_device),
            "device_type": current_device.type,
            "available_devices": [d.to_dict() for d in self._available_devices],
            "selected_type": self._selected_device_type.value if self._selected_device_type else None,
            "memory": memory_info,
            "optimal_dtype": str(self.get_optimal_dtype()),
        }


# 전역 인스턴스 (싱글톤)
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """디바이스 매니저 인스턴스 반환"""
    return device_manager


