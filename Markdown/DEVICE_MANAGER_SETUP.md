# 🖥️ 디바이스 관리 시스템 (Device Manager Setup)

## 개요

MacTuner에 **GPU/CPU 자동 감지 및 선택 시스템**을 구현했습니다. 사용자가 시작 시 사용할 디바이스(MPS, CUDA, CPU)를 선택하면, 모든 학습, 추론, RAG 기능이 선택된 디바이스를 사용합니다.

## 주요 기능

### ✅ 1. 자동 디바이스 감지
- **MPS** (Apple Metal Performance Shaders) - Mac GPU
- **CUDA** (NVIDIA GPU) - 커스텀 GPU
- **CPU** - 폴백 옵션

### ✅ 2. 유연한 디바이스 선택
- 대시보드에서 직관적인 UI로 디바이스 선택
- 메모리 정보 실시간 조회
- 캐시 메모리 관리 기능

### ✅ 3. 모듈화된 구조
- 기존 코드와 완전히 독립적
- 모든 서비스가 `DeviceManager`를 사용
- 전역 상태 관리로 일관성 보장

---

## 백엔드 구조

### 📁 `backend/services/device_manager.py` (핵심 모듈)

```python
class DeviceManager:
    """디바이스 관리 (싱글톤 패턴)"""
    
    def get_available_devices() -> List[DeviceInfo]
    def select_device(device_type: str) -> bool
    def get_current_device() -> torch.device
    def auto_select_device() -> bool
    def get_device_memory_info() -> Dict
    def clear_cache() -> None
```

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `detect_devices()` | 시스템 디바이스 자동 감지 |
| `select_device()` | 특정 디바이스 선택 |
| `get_current_device()` | 현재 선택된 디바이스 반환 |
| `auto_select_device()` | 최적 디바이스 자동 선택 (우선순위: MPS > CUDA > CPU) |
| `move_model_to_device()` | 모델을 선택된 디바이스로 이동 |
| `move_tensor_to_device()` | 텐서를 선택된 디바이스로 이동 |
| `get_optimal_dtype()` | 디바이스 최적 데이터 타입 (float16, bfloat16, etc.) |

### 📁 `backend/api/device_selector.py` (API 엔드포인트)

**사용 가능한 엔드포인트:**

```http
GET  /device/devices/available      # 모든 디바이스 조회
POST /device/devices/select/{type}  # 디바이스 선택 (mps/cuda/cpu)
GET  /device/devices/current        # 현재 디바이스 정보
POST /device/devices/auto-select    # 최적 디바이스 자동 선택
GET  /device/devices/memory         # 메모리 정보 조회
POST /device/devices/clear-cache    # 캐시 정리
```

### 통합된 서비스들

✅ **training_service.py**
```python
self.device_manager = get_device_manager()
self.device = self.device_manager.get_current_device()
```

✅ **model_service.py**
✅ **chat_service.py**
✅ **quantization_service.py**

---

## 프론트엔드 구조

### 📁 `frontend/src/stores/deviceStore.ts` (Zustand 상태 관리)

```typescript
interface DeviceStore {
  // 상태
  availableDevices: Device[]
  selectedDevice: DeviceType | null
  currentDevice: string | null
  memoryInfo: MemoryInfo | null
  isLoading: boolean
  error: string | null
  
  // 액션
  fetchAvailableDevices()
  selectDevice(deviceType)
  autoSelectDevice()
  fetchMemoryInfo()
  clearCache()
}
```

**사용 예제:**
```typescript
const store = useDeviceStore();

// 디바이스 선택
await store.selectDevice('cuda');

// 현재 상태 조회
console.log(store.currentDevice); // "cuda:0"
console.log(store.memoryInfo);    // { allocated: 2.5, total: 24.0, ... }
```

### 📁 `frontend/src/components/DeviceSelector.tsx` (UI 컴포넌트)

**기능:**
- 🎯 디바이스 카드형 선택 UI
- 📊 실시간 메모리 사용량 표시
- 🧹 캐시 정리 버튼
- 💾 가용 메모리 표시
- ⚠️ 에러 처리

**사용 예제:**
```tsx
import DeviceSelector from '../components/DeviceSelector';

<DeviceSelector 
  onDeviceSelected={(device) => console.log(`Selected: ${device}`)}
/>
```

### 📁 Dashboard에 통합

`frontend/src/pages/Dashboard.tsx`에 다음이 추가됨:

```tsx
<section className="mb-12">
  <DeviceSelector />
</section>
```

---

## 사용 흐름

### 1️⃣ 백엔드 시작 (자동 감지)

```bash
python app.py
```

백엔드 로그:
```
🔍 시스템 디바이스 감지 중...
✅ MPS (Mac GPU) 감지됨
✅ CUDA GPU 감지됨: NVIDIA RTX 3090 (24.0GB)
✅ CPU 감지됨
📊 총 3개 디바이스 감지됨
🤖 최적 디바이스 자동 선택 중...
✅ 디바이스 선택 완료: mps
🎯 현재 디바이스: mps
```

### 2️⃣ 프론트엔드 시작 (사용자 선택)

1. Dashboard 열기
2. **🎯 컴퓨팅 디바이스 선택** 섹션 표시
3. 사용자가 GPU 선택
4. 선택된 디바이스로 모든 기능 실행

### 3️⃣ API 호출 흐름

```
프론트엔드 (DeviceSelector)
    ↓
POST /device/devices/select/cuda
    ↓
백엔드 (device_selector.py)
    ↓
DeviceManager.select_device('cuda')
    ↓
모든 서비스 (training_service, model_service, etc.)
    ↓
선택된 GPU 사용
```

---

## API 응답 예제

### 1. 사용 가능한 디바이스 조회

**요청:**
```http
GET /device/devices/available
```

**응답:**
```json
{
  "devices": [
    {
      "type": "mps",
      "name": "Apple Metal Performance Shaders",
      "is_available": true,
      "memory_total": null,
      "memory_allocated": null,
      "memory_reserved": null,
      "compute_capability": null
    },
    {
      "type": "cuda",
      "name": "NVIDIA RTX 3090 (ID: 0)",
      "is_available": true,
      "memory_total": 24.0,
      "memory_allocated": null,
      "memory_reserved": null,
      "compute_capability": "8.6"
    },
    {
      "type": "cpu",
      "name": "CPU (Intel/AMD)",
      "is_available": true,
      "memory_total": null,
      "memory_allocated": null,
      "memory_reserved": null,
      "compute_capability": null
    }
  ],
  "count": 3
}
```

### 2. 디바이스 선택

**요청:**
```http
POST /device/devices/select/cuda
```

**응답:**
```json
{
  "success": true,
  "selected_device": "cuda",
  "current_device": "cuda:0",
  "message": "CUDA 디바이스가 선택되었습니다"
}
```

### 3. 메모리 정보 조회

**요청:**
```http
GET /device/devices/memory
```

**응답:**
```json
{
  "device": "cuda:0",
  "allocated": 2.5,
  "reserved": 5.0,
  "total": 24.0,
  "available": 21.5
}
```

---

## 개발자 가이드

### 새로운 서비스에 DeviceManager 통합

```python
from backend.services.device_manager import get_device_manager

class MyNewService:
    def __init__(self):
        # 디바이스 매니저 초기화
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_current_device()
    
    def process_model(self, model):
        # 모델을 현재 선택된 디바이스로 이동
        model = self.device_manager.move_model_to_device(model)
        
        # 텐서를 디바이스로 이동
        tensor = self.device_manager.move_tensor_to_device(tensor)
        
        # 메모리 정보 조회
        memory_info = self.device_manager.get_device_memory_info()
        
        # 캐시 정리
        self.device_manager.clear_cache()
```

### 프론트엔드에서 디바이스 정보 사용

```typescript
import { useDeviceStore } from '../stores/deviceStore';

function MyComponent() {
  const { selectedDevice, currentDevice, memoryInfo } = useDeviceStore();
  
  return (
    <div>
      <p>선택: {selectedDevice}</p>
      <p>현재: {currentDevice}</p>
      <p>메모리: {memoryInfo?.allocated} GB / {memoryInfo?.total} GB</p>
    </div>
  );
}
```

---

## 트러블슈팅

### ❓ CUDA 감지 안 됨

**확인 사항:**
```bash
# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"

# GPU 확인 (Linux/Mac)
nvidia-smi

# macOS Metal 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"
```

**해결:**
- NVIDIA 드라이버 업데이트
- PyTorch CUDA 버전 일치 확인
- `requirements.txt` 재설치

### ❓ MPS가 느림

**최적화:**
```python
# 적절한 데이터 타입 선택
dtype = device_manager.get_optimal_dtype()  # float32 권장
model = model.to(dtype)
```

### ❓ 메모리 부족

**해결:**
1. 캐시 정리: `POST /device/devices/clear-cache`
2. 배치 크기 감소
3. 그래디언트 체크포인팅 활성화

---

## 성능 비교

| 디바이스 | 속도 | 메모리 | 호환성 | 추천 |
|---------|------|--------|--------|------|
| **MPS** | 🚀🚀🚀 | 자동 | M1/M2/M3 | ✅ Mac 최고 |
| **CUDA** | 🚀🚀🚀 | 자동 | NVIDIA | ✅ 고성능 PC |
| **CPU** | 🐌 | 좋음 | 모든 시스템 | ✅ 호환성 |

---

## 다음 단계

1. ✅ LoRA/QLoRA 지원 (기존 기능 유지)
2. ✅ 자동 파라미터 추천
3. ✅ 실시간 학습 모니터링
4. ✅ 자동 체크포인트 관리

모든 기능이 선택된 디바이스에 대응합니다!

---

## 문제 해결 및 피드백

문제 발생 시:
1. `/device/health` API 호출로 진단
2. 백엔드 로그 확인
3. 브라우저 콘솔 확인

---

**작성 날짜:** 2025-11-08  
**버전:** 1.0.0


