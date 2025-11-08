# 🚀 디바이스 관리 시스템 - 빠른 시작 가이드

## 5분 안에 시작하기

### 1️⃣ 시스템 요구사항 확인

```bash
# Python 버전 확인 (3.10 이상)
python --version

# PyTorch 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 디바이스 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

### 2️⃣ 애플리케이션 시작

```bash
# 프로젝트 디렉토리
cd Mactuner

# 모든 서비스 시작 (백엔드 + 프론트엔드)
python app.py
```

**예상 출력:**
```
🔍 시스템 디바이스 감지 중...
✅ MPS (Mac GPU) 감지됨
✅ CPU 감지됨
📊 총 2개 디바이스 감지됨

🍎 MacTuner - 통합 서비스
============================================================

🔧 백엔드 시작 중... (포트 8001)
✅ 백엔드 시작됨
🎨 프론트엔드 시작 중... (포트 3000)
✅ 프론트엔드 시작됨

============================================================
✅ 모든 서비스가 시작되었습니다!
============================================================

📱 접속 정보:
  🌐 프론트엔드:  http://localhost:3000
  🔌 백엔드 API:  http://localhost:8001
  📚 API 문서:    http://localhost:8001/docs

🎯 다음 단계:
  1. 브라우저에서 http://localhost:3000 열기
  2. Dashboard에서 기능 테스트
  3. Ctrl+C를 누르면 모든 서비스가 종료됩니다

============================================================
```

### 3️⃣ 브라우저에서 사용

1. **http://localhost:3000** 방문
2. **Dashboard** 페이지 확인
3. **🎯 컴퓨팅 디바이스 선택** 섹션 보기

### 4️⃣ 디바이스 선택

**UI 화면:**

```
🎯 컴퓨팅 디바이스 선택
학습 및 추론에 사용할 GPU/CPU를 선택하세요

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     🍎       │  │     ⚡       │  │     💾       │
│     mps      │  │    cuda      │  │     cpu      │
│  Apple Metal │  │  NVIDIA RTX  │  │  Intel/AMD   │
│              │  │   3090 24GB  │  │              │
│  ✓ 사용가능  │  │  ✓ 사용가능  │  │  ✓ 사용가능  │
└──────────────┘  └──────────────┘  └──────────────┘
     (선택)           (선택)           (선택)

선택된 디바이스: ✅ mps
```

### 5️⃣ 메모리 정보 확인

**"📊 메모리 정보" 버튼 클릭:**

```
📊 메모리 정보 ▼

사용 중:  0.00 GB / 12.00 GB
█░░░░░░░░░░░░░░░░░░░░░░░░ 0.0%

디바이스:        mps
예약됨:          -
사용 가능:       12.00 GB (초록색)

🧹 캐시 메모리 정리 [클릭]
```

---

## 🔧 API 테스트 (선택사항)

### cURL로 테스트

```bash
# 1. 사용 가능한 디바이스 조회
curl http://localhost:8001/device/devices/available

# 2. CUDA 디바이스 선택
curl -X POST http://localhost:8001/device/devices/select/cuda

# 3. 현재 디바이스 확인
curl http://localhost:8001/device/devices/current

# 4. 메모리 정보 조회
curl http://localhost:8001/device/devices/memory

# 5. 캐시 정리
curl -X POST http://localhost:8001/device/devices/clear-cache
```

### Python으로 테스트

```python
import requests

BASE_URL = "http://localhost:8001"

# 1. 디바이스 조회
response = requests.get(f"{BASE_URL}/device/devices/available")
print(response.json())

# 2. 디바이스 선택
response = requests.post(f"{BASE_URL}/device/devices/select/cuda")
print(response.json())

# 3. 현재 상태
response = requests.get(f"{BASE_URL}/device/devices/current")
print(response.json())
```

---

## 💡 주요 기능

### 1️⃣ 자동 감지

앱 시작 시 자동으로:
- ✅ MPS (Mac GPU) 감지
- ✅ CUDA (NVIDIA) 감지  
- ✅ CPU 감지
- ✅ 최적 디바이스 선택

### 2️⃣ 사용자 선택

대시보드에서:
- 🎯 원하는 디바이스 선택
- 📊 실시간 메모리 확인
- 🧹 캐시 정리

### 3️⃣ 자동 적용

선택된 디바이스는:
- 📥 모델 다운로드
- 🎓 학습 (LoRA/QLoRA)
- 💬 Chat 추론
- 🔍 RAG 검색
- 📦 GGUF 양자화

모두에 자동으로 적용됩니다!

---

## ⚡ 성능 팁

### MPS (Mac GPU) 최적화

```
✅ 장점:
  • 가장 빠른 성능
  • Mac 에너지 효율성
  • 메모리 자동 관리

⚙️ 설정:
  • dtype: float32
  • 최대 배치 크기 활용
```

### CUDA 최적화

```
✅ 장점:
  • 가장 안정적
  • 대규모 모델 학습
  • 커뮤니티 지원

⚙️ 설정:
  • dtype: float16 또는 bfloat16
  • 그래디언트 체크포인팅
  • 메모리 최적화
```

### CPU 사용

```
✅ 장점:
  • 모든 시스템에서 작동
  • 호환성 최고

⚠️ 주의:
  • 느린 속도
  • 더 많은 시간 필요
  • 대규모 모델 불가
```

---

## 🆘 문제 해결

### ❓ 디바이스가 표시되지 않음

**해결 순서:**

1. **백엔드 로그 확인**
   ```bash
   # 백엔드 터미널에서 확인
   # 🔍 시스템 디바이스 감지 중... 메시지 확인
   ```

2. **API 건강 확인**
   ```bash
   curl http://localhost:8001/health
   ```

3. **백엔드 재시작**
   ```bash
   # Ctrl+C로 종료 후 다시 시작
   python app.py
   ```

### ❓ CUDA를 선택했는데 느림

**확인:**
- `nvidia-smi` 명령으로 GPU 사용 확인
- 백엔드 로그에서 CUDA 관련 에러 확인
- NVIDIA 드라이버 업데이트 필요한지 확인

### ❓ 메모리 부족

**해결:**

1. **캐시 정리**
   - UI에서 "🧹 캐시 메모리 정리" 클릭

2. **배치 크기 감소**
   - 학습 페이지에서 배치 크기 설정

3. **CPU로 변경**
   - 메모리가 충분한 CPU 사용

---

## 📚 다음 단계

디바이스 선택 후:

1. **📥 모델 다운로드**
   - ModelDownload 페이지에서 Hugging Face 모델 선택

2. **📊 데이터 처리**
   - DataProcessing 페이지에서 데이터셋 업로드

3. **🎓 모델 학습**
   - Training 페이지에서 LoRA/QLoRA 학습 시작

4. **💬 Chat**
   - 학습된 모델과 상호작용

---

## 🎯 실제 예제 워크플로우

```
[1] Dashboard 열기
         ↓
[2] 🎯 디바이스 선택 (예: CUDA)
         ↓
[3] ✅ "CUDA 선택되었습니다" 확인
         ↓
[4] ModelDownload로 이동
         ↓
[5] 모델 다운로드 (CUDA 사용)
         ↓
[6] DataProcessing로 이동
         ↓
[7] 데이터셋 업로드 및 정제
         ↓
[8] Training 페이지 이동
         ↓
[9] LoRA 구성 및 학습 시작 (CUDA 사용)
         ↓
[10] Chat에서 학습된 모델 테스트 (CUDA 사용)
```

---

## 📞 지원

**문제 발생 시:**

1. 📄 `Markdown/DEVICE_MANAGER_SETUP.md` 참고 (상세 설명)
2. 🔗 GitHub Issues 확인
3. 💬 백엔드 로그 확인

---

## ✨ 주요 개선사항

이전 버전 vs 신규 버전:

| 항목 | 이전 | 신규 |
|------|------|------|
| 디바이스 선택 | 고정 | 🎯 동적 선택 |
| CUDA 지원 | ❌ | ✅ CPU + CUDA |
| 메모리 관리 | 수동 | 📊 자동 + 관리 UI |
| 캐시 정리 | 없음 | 🧹 한 클릭 |
| 모듈성 | 강한 결합 | 🧩 독립적 |

---

**Happy Training! 🎉**

**마지막 업데이트:** 2025-11-08


