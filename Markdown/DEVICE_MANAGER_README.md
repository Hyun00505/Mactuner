# 🎯 MacTuner Device Manager System

## 🎉 완성된 기능

MacTuner에 **멀티-디바이스 지원 시스템**이 완료되었습니다! 🚀

### ✨ 주요 기능

✅ **자동 GPU/CPU 감지**
- Apple Silicon (MPS) 자동 감지
- NVIDIA CUDA 자동 감지  
- CPU 폴백 옵션

✅ **동적 디바이스 선택**
- Dashboard에서 실시간 선택
- 런타임 중 변경 가능
- 직관적인 UI

✅ **메모리 관리**
- 실시간 메모리 모니터링
- 한 클릭 캐시 정리
- 사용 가능 메모리 표시

✅ **완전한 모듈화**
- 기존 코드와 독립적
- 모든 서비스가 자동으로 적용
- 싱글톤 패턴으로 일관성 보장

---

## 🚀 5분 시작 가이드

### Step 1: 시스템 요구사항 확인

```bash
# Python 버전 확인
python --version  # 3.10 이상

# PyTorch 설치 확인
python -c "import torch; print(torch.__version__)"
```

### Step 2: 앱 시작

```bash
cd Mactuner
python app.py
```

### Step 3: 브라우저 열기

```
http://localhost:3000
```

### Step 4: Dashboard에서 디바이스 선택

1. Dashboard 페이지 열기
2. **🎯 컴퓨팅 디바이스 선택** 섹션 보기
3. 원하는 GPU 선택 (MPS, CUDA, 또는 CPU)
4. ✅ 완료!

---

## 📁 생성된 파일 목록

### 백엔드 (Backend)

```
backend/
├── services/
│   ├── device_manager.py          ← 🆕 핵심 모듈 (싱글톤)
│   ├── training_service.py         ← ✏️ 수정 (device_manager 통합)
│   ├── model_service.py            ← ✏️ 수정 (device_manager 통합)
│   ├── chat_service.py             ← ✏️ 수정 (device_manager 통합)
│   └── quantization_service.py     ← ✏️ 수정 (device_manager 통합)
│
├── api/
│   ├── device_selector.py          ← 🆕 API 엔드포인트
│   └── __init__.py                 ← ✏️ 수정 (device_selector 추가)
│
└── main.py                         ← ✏️ 수정 (device API 라우터 등록)
```

### 프론트엔드 (Frontend)

```
frontend/src/
├── stores/
│   └── deviceStore.ts              ← 🆕 전역 상태 (Zustand)
│
├── components/
│   └── DeviceSelector.tsx          ← 🆕 UI 컴포넌트
│
├── pages/
│   └── Dashboard.tsx               ← ✏️ 수정 (DeviceSelector 통합)
│
└── utils/
    └── api.ts                      ← ✏️ 수정 (deviceAPI 추가)
```

### 문서 (Documentation)

```
Markdown/
├── DEVICE_MANAGER_SETUP.md         ← 📄 상세 설정 가이드
├── DEVICE_MANAGER_QUICK_START.md   ← 📄 5분 빠른 시작
├── DEVICE_MANAGER_SUMMARY.md       ← 📄 전체 요약
└── (이 파일)
```

---

## 🔧 API 엔드포인트

### 6개의 REST API 추가됨

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/device/devices/available` | 모든 디바이스 조회 |
| POST | `/device/devices/select/{type}` | 디바이스 선택 |
| GET | `/device/devices/current` | 현재 디바이스 조회 |
| POST | `/device/devices/auto-select` | 최적 디바이스 선택 |
| GET | `/device/devices/memory` | 메모리 정보 조회 |
| POST | `/device/devices/clear-cache` | 캐시 정리 |

### API 테스트

```bash
# 1. 디바이스 조회
curl http://localhost:8001/device/devices/available

# 2. CUDA 선택
curl -X POST http://localhost:8001/device/devices/select/cuda

# 3. 현재 상태
curl http://localhost:8001/device/devices/current
```

---

## 💻 코드 구조

### 백엔드 핵심

```python
from backend.services.device_manager import get_device_manager

# 1. 싱글톤 패턴으로 관리
device_manager = get_device_manager()

# 2. 현재 디바이스 가져오기
device = device_manager.get_current_device()

# 3. 모델/텐서 이동
model = device_manager.move_model_to_device(model)

# 4. 메모리 정보 조회
memory = device_manager.get_device_memory_info()

# 5. 캐시 정리
device_manager.clear_cache()
```

### 프론트엔드 상태

```typescript
import { useDeviceStore } from '../stores/deviceStore';

const {
  availableDevices,    // 사용 가능 디바이스 목록
  selectedDevice,      // 선택된 디바이스
  currentDevice,       // 현재 디바이스 (문자열)
  memoryInfo,          // 메모리 정보
  selectDevice,        // 디바이스 선택 함수
  clearCache,          // 캐시 정리 함수
} = useDeviceStore();
```

---

## 🎨 UI 미리보기

### Dashboard - 디바이스 선택 섹션

```
🎯 컴퓨팅 디바이스 선택
학습 및 추론에 사용할 GPU/CPU를 선택하세요

┌──────────────┬──────────────┬──────────────┐
│     🍎       │     ⚡       │     💾       │
│  Apple MPS   │   CUDA GPU   │     CPU      │
│   M1 Pro     │   RTX 3090   │  Intel Core  │
│  ✓ Available │  ✓ Available │ ✓ Available  │
└──────────────┴──────────────┴──────────────┘
       ↓ (Click)    ↓ (Click)    ↓ (Click)

✅ 선택된 디바이스: CUDA (cuda:0)

📊 메모리 정보 ▼
사용 중: 2.5 GB / 24.0 GB
████░░░░░░░░░░░░░░░░░░░░ 10.4%

[🧹 캐시 메모리 정리]
```

---

## 📊 성능 비교

| 항목 | MPS | CUDA | CPU |
|------|-----|------|-----|
| 속도 | ⚡⚡⚡ | ⚡⚡⚡ | 🐢 |
| 메모리 | 자동 | 수동 | 시스템 |
| 호환성 | M1/M2/M3 | NVIDIA | 모든 PC |
| 추천 | Mac | 고성능 PC | 테스트 |

---

## 🔄 워크플로우

```
1️⃣ 앱 시작
   └─ 자동 디바이스 감지

2️⃣ Dashboard 열기
   └─ 디바이스 선택 UI 표시

3️⃣ GPU 선택
   └─ 선택된 디바이스로 모든 기능 실행

4️⃣ 모델 다운로드
   └─ 선택된 GPU 사용

5️⃣ 학습 시작
   └─ 선택된 GPU 사용

6️⃣ 추론/Chat
   └─ 선택된 GPU 사용
```

---

## ✅ 통합된 서비스들

### 자동으로 디바이스 지원

- ✅ **Training Service** - 학습 (LoRA/QLoRA)
- ✅ **Model Service** - 모델 다운로드/로드
- ✅ **Chat Service** - 추론
- ✅ **Quantization Service** - GGUF 변환

### 사용자 선택 디바이스 적용

1. Dashboard에서 선택 → 2. 모든 서비스가 그 디바이스 사용

---

## 📚 문서 참고

### 빠른 시작
```
📄 Markdown/DEVICE_MANAGER_QUICK_START.md
   - 5분 안에 시작
   - 기본 사용법
   - 트러블슈팅
```

### 상세 설정
```
📄 Markdown/DEVICE_MANAGER_SETUP.md
   - 전체 아키텍처
   - API 상세 문서
   - 개발자 가이드
```

### 전체 요약
```
📄 Markdown/DEVICE_MANAGER_SUMMARY.md
   - 구현 완료 내역
   - 코드 구조
   - 성능 비교
```

---

## 🚨 문제 해결

### ❓ 디바이스가 표시되지 않음

```bash
# 1. 백엔드 로그 확인
# "🔍 시스템 디바이스 감지 중..." 메시지 확인

# 2. API 테스트
curl http://localhost:8001/device/devices/available

# 3. 재시작
# Ctrl+C → python app.py
```

### ❓ CUDA가 느림

```bash
# GPU 사용 확인
nvidia-smi

# 드라이버 업데이트 필요한지 확인
nvidia-smi --query-gpu=driver_version --format=csv
```

### ❓ 메모리 부족

1. UI에서 "🧹 캐시 메모리 정리" 클릭
2. 배치 크기 감소
3. CPU로 변경

---

## 🎯 다음 단계

### 이미 구현됨 ✅
- ✅ Multi-Device 지원 (MPS, CUDA, CPU)
- ✅ 동적 디바이스 선택
- ✅ 실시간 메모리 관리
- ✅ 완전한 모듈화

### 향후 계획 (Phase 2)
- 다중 GPU 지원 (여러 GPU 동시 사용)
- 분산 학습 (DDP)
- 성능 프로파일링 UI

---

## 📞 지원

### 기술 지원

1. 📖 상단 문서 읽기
2. 🔍 백엔드 로그 확인
3. 🌐 API 문서: `http://localhost:8001/docs`

### 버그 리포트

- GitHub Issues에 버그 보고
- 상세한 로그/스크린샷 포함

---

## 🎉 완성!

모든 기능이 구현되었고 테스트 준비가 완료되었습니다! 🚀

**Happy Training!** 💪

---

## 📝 체크리스트

구현된 항목:

- ✅ MPS 감지 및 지원
- ✅ CUDA 감지 및 지원
- ✅ CPU 폴백
- ✅ 동적 디바이스 선택
- ✅ 실시간 메모리 모니터링
- ✅ 캐시 관리
- ✅ API 엔드포인트 (6개)
- ✅ Zustand Store
- ✅ DeviceSelector UI
- ✅ Dashboard 통합
- ✅ 모든 서비스 통합
- ✅ 문서 작성
- ✅ 타입 안전성 (TypeScript)
- ✅ 에러 처리
- ✅ 로깅

---

## 📅 업데이트

**마지막 업데이트:** 2025-11-08  
**상태:** ✅ Production Ready

---

**작성:** MacTuner Team  
**버전:** 1.0.0


