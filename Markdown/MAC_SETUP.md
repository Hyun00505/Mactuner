# 🍎 MacTuner MAC 호환성 설정 가이드

## ⚠️ MAC 환경 문제 해결

### 문제 1: bitsandbytes 불호환
```
❌ 오류: bitsandbytes==0.48.2 doesn't have wheels for macOS
✅ 해결: QLoRA 제거, LoRA만 사용
```

### 문제 2: 의존성 설치 실패
```
❌ 오류: Distribution can't be installed
✅ 해결: MAC 호환 버전 사용
```

---

## 🚀 MAC에서 올바른 설치 방법

### Step 1: 프로젝트 디렉토리로 이동
```bash
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner
```

### Step 2: .venv 디렉토리 제거 (있으면)
```bash
rm -rf .venv
```

### Step 3: 가상환경 생성 (Python 3.11 이상)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: pip 업그레이드
```bash
pip install --upgrade pip setuptools wheel
```

### Step 5: 의존성 설치
```bash
# requirements.txt에서 설치 (권장)
pip install -r backend/requirements.txt

# 또는 개별 설치
pip install fastapi uvicorn transformers peft torch sentence-transformers
pip install PyPDF2 python-dotenv pydantic-settings
```

### Step 6: 서버 실행
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: API 테스트
```bash
# 브라우저에서 열기
http://localhost:8000/docs
```

---

## ✨ MAC에서 지원되는 기능

### ✅ 완전 지원
```
✅ 모델 로더 (Hugging Face 다운로드)
✅ 데이터 처리 (업로드, 정제, EDA)
✅ LoRA 미세조정 (메모리 효율적)
✅ Chat 인터페이스
✅ RAG 파이프라인
✅ GGUF 변환
✅ Metal GPU 가속 (MPS)
```

### ⚠️ MAC에서 제한됨
```
❌ QLoRA (4-bit 양자화) - bitsandbytes 불호환
→ LoRA로 대체 (메모리 사용량 10배, 여전히 효율적)
```

### 🔧 MAC 최적화 자동 적용
```
✅ MPS (Metal Performance Shaders) 자동 감지
✅ 최적 배치 크기 자동 계산
✅ 메모리 효율적 파라미터 추천
```

---

## 📊 성능 비교

### MAC에서의 메모리 사용
```
Full Fine-tuning (FP32):    13GB 모델 = 26GB 메모리 필요
LoRA:                        13GB 모델 = 20GB 메모리 필요 (23% 절감)
QLoRA (미지원):             13GB 모델 = 4GB 메모리 필요 (불가능)

→ LoRA는 여전히 충분히 효율적! ✅
```

---

## 🆘 자주 나는 오류와 해결법

### 오류 1: ModuleNotFoundError: No module named 'transformers'
```bash
# 해결: 의존성 재설치
pip install --upgrade transformers

# 또는 가상환경 확인
which python  # 올바른 python이 선택되었나?
```

### 오류 2: torch ImportError: cannot import name 'MPS'
```bash
# 해결: PyTorch 재설치 (MAC용)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 오류 3: bitsandbytes not available
```bash
# 이미 해결됨 - 최신 requirements.txt 사용
# QLoRA 코드는 이미 제거됨
```

### 오류 4: CUDA device not available
```bash
# 정상 - MAC은 CUDA 미지원 (MPS 사용)
# 자동으로 MPS 또는 CPU로 폴백됨 ✅
```

---

## 🎯 권장 설치 순서

### 최초 설정 (처음 한번만)
```bash
# 1. 프로젝트 위치
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner

# 2. 가상환경 생성
python3.11 -m venv .venv  # Python 3.11 권장

# 3. 활성화
source .venv/bin/activate

# 4. pip 업그레이드
pip install --upgrade pip

# 5. 의존성 설치
pip install -r backend/requirements.txt
```

### 매번 실행할 때
```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. 서버 실행
python -m uvicorn backend.main:app --reload

# 3. 다른 터미널에서 테스트
# http://localhost:8000/docs
```

---

## 📈 메모리 모니터링

### 실시간 메모리 확인
```bash
# 새 터미널에서 실행
while true; do
  ps aux | grep python | grep uvicorn | awk '{print "Memory: " $6/1024 " MB"}'
  sleep 5
done
```

### Activity Monitor로 확인
```bash
# 1. Spotlight 검색 (cmd + space)
# 2. "Activity Monitor" 검색
# 3. Memory 탭에서 python 프로세스 확인
```

---

## 🔧 트러블슈팅

### 만약 설치가 여전히 실패한다면?

#### 1단계: 캐시 초기화
```bash
pip cache purge
rm -rf ~/.cache/pip
```

#### 2단계: 완전 재설치
```bash
# 가상환경 제거
rm -rf .venv

# 재생성
python3 -m venv .venv
source .venv/bin/activate

# 처음부터 설치
pip install -r backend/requirements.txt
```

#### 3단계: Python 버전 확인
```bash
python --version  # 3.11 이상 필요

# 만약 3.10 이하라면
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3 -m venv .venv
```

---

## 📱 시스템 정보 확인

### MAC 환경 정보
```bash
# 1. 시스템 정보
system_profiler SPSoftwareDataType

# 2. Python 정보
python -c "import sys; print(sys.version)"

# 3. PyTorch 정보
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# 4. 메모리 정보
vm_stat
```

---

## 🎓 학습: MAC 최적화 원리

### MPS (Metal Performance Shaders)
```
CPU                  MPS (GPU)
일반 계산            병렬 계산
순차 처리            동시 처리 (수천 개 병렬)
느림                 빠름 (10배)

PyTorch는 자동으로 MPS 선택! ✅
```

### LoRA vs QLoRA 메모리
```
LoRA:
- 원본 모델 가중치: FP32 (고정)
- LoRA 어댑터: FP32 (작음)
- 메모리 절감: ~15% (이미 충분)

QLoRA (MAC 미지원):
- 원본 모델 가중치: INT4 (초소형)
- LoRA 어댑터: FP32/BF16 (작음)
- 메모리 절감: ~90% (MAC에서 불가능)

→ LoRA만으로도 충분히 효율적! ✅
```

---

## 🎯 최종 체크리스트

- [ ] Python 3.11 이상 설치됨
- [ ] .venv 가상환경 생성됨
- [ ] requirements.txt 설치 완료
- [ ] `python -m uvicorn backend.main:app --reload` 실행됨
- [ ] http://localhost:8000/docs 접속 가능
- [ ] API 엔드포인트 응답 확인 (< 1초)

---

## 📞 추가 지원

### 문제 해결 순서
1. 이 가이드를 처음부터 따라하기
2. 오류 메시지 전체 읽기
3. 구글에서 오류 메시지 검색
4. Stack Overflow 확인

### 유용한 리소스
- [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [MAC M1/M2 개발 환경 설정](https://developer.apple.com/documentation/metal)

---

## ✨ 결론

**MAC에서 완벽하게 작동합니다!** 🍎

```
✅ 모든 기본 기능 지원
✅ LoRA로 메모리 효율화
✅ MPS로 GPU 가속
✅ 설정 간단 (이 가이드 따라하면 됨)
```

**이제 즉시 시작할 준비가 되었습니다!** 🚀

