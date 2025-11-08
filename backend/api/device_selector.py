"""
디바이스 선택 API
GPU/CPU 감지, 선택, 상태 조회 엔드포인트를 제공합니다.
"""

import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException

from backend.services.device_manager import get_device_manager, DeviceInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/devices/available", response_model=Dict, tags=["Device"])
async def get_available_devices() -> Dict:
    """
    사용 가능한 모든 디바이스 조회
    
    Returns:
        사용 가능한 디바이스 목록과 정보
    
    Example:
        ```json
        {
            "devices": [
                {
                    "type": "mps",
                    "name": "Apple Metal Performance Shaders",
                    "is_available": true
                },
                {
                    "type": "cuda",
                    "name": "NVIDIA RTX 3090 (ID: 0)",
                    "is_available": true,
                    "memory_total": 24.0,
                    "compute_capability": "8.6"
                },
                {
                    "type": "cpu",
                    "name": "CPU (Intel/AMD)",
                    "is_available": true
                }
            ],
            "count": 3
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        devices = device_manager.get_available_devices()
        
        return {
            "devices": [d.to_dict() for d in devices],
            "count": len(devices),
        }
    except Exception as e:
        logger.error(f"❌ 디바이스 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/select/{device_type}", response_model=Dict)
async def select_device(device_type: str) -> Dict:
    """
    디바이스 선택
    
    Args:
        device_type: 선택할 디바이스 타입 ("mps", "cuda", "cpu")
    
    Returns:
        선택 결과 및 현재 상태
    
    Example:
        Request: POST /api/device/devices/select/cuda
        Response:
        ```json
        {
            "success": true,
            "selected_device": "cuda",
            "current_device": "cuda:0",
            "message": "CUDA 디바이스가 선택되었습니다"
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        success = device_manager.select_device(device_type)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"디바이스 선택 실패: {device_type}"
            )
        
        return {
            "success": True,
            "selected_device": device_type,
            "current_device": str(device_manager.get_current_device()),
            "message": f"{device_type.upper()} 디바이스가 선택되었습니다",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 디바이스 선택 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/current", response_model=Dict)
async def get_current_device() -> Dict:
    """
    현재 선택된 디바이스 정보 조회
    
    Returns:
        현재 디바이스 타입, 이름, 메모리 정보
    
    Example:
        ```json
        {
            "current_device": "cuda:0",
            "device_type": "cuda",
            "selected_type": "cuda",
            "memory": {
                "allocated": 2.5,
                "reserved": 5.0,
                "total": 24.0
            },
            "optimal_dtype": "torch.float16"
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        status_info = device_manager.get_status_info()
        
        return status_info
    except Exception as e:
        logger.error(f"❌ 현재 디바이스 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/auto-select", response_model=Dict)
async def auto_select_device() -> Dict:
    """
    최적의 디바이스 자동 선택
    우선순위: MPS > CUDA > CPU
    
    Returns:
        선택된 디바이스 정보
    
    Example:
        ```json
        {
            "success": true,
            "selected_device": "mps",
            "message": "최적 디바이스 자동 선택 완료",
            "current_device": "mps"
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        success = device_manager.auto_select_device()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="자동 디바이스 선택 실패"
            )
        
        current_device = device_manager.get_current_device()
        device_type = device_manager.get_current_device_type()
        
        return {
            "success": True,
            "selected_device": device_type.value if device_type else "unknown",
            "current_device": str(current_device),
            "message": "최적 디바이스 자동 선택 완료",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 자동 디바이스 선택 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/memory", response_model=Dict)
async def get_memory_info() -> Dict:
    """
    현재 디바이스의 메모리 정보 조회
    
    Returns:
        메모리 할당, 예약, 총 크기 정보 (단위: GB)
    
    Example:
        ```json
        {
            "device": "cuda:0",
            "allocated": 2.5,
            "reserved": 5.0,
            "total": 24.0,
            "available": 19.0
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        memory_info = device_manager.get_device_memory_info()
        
        # available 계산
        if memory_info.get("total") and memory_info.get("allocated"):
            memory_info["available"] = memory_info["total"] - memory_info["allocated"]
        
        return memory_info
    except Exception as e:
        logger.error(f"❌ 메모리 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/clear-cache", response_model=Dict)
async def clear_device_cache() -> Dict:
    """
    현재 디바이스의 캐시 메모리 정리
    
    Returns:
        정리 결과
    
    Example:
        ```json
        {
            "success": true,
            "device": "cuda:0",
            "message": "캐시 메모리 정리 완료"
        }
        ```
    """
    try:
        device_manager = get_device_manager()
        device_manager.clear_cache()
        current_device = str(device_manager.get_current_device())
        
        return {
            "success": True,
            "device": current_device,
            "message": "캐시 메모리 정리 완료",
        }
    except Exception as e:
        logger.error(f"❌ 캐시 정리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


