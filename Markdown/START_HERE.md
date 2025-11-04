# 🚀 MacTuner 구현 시작하기

## ✅ 완료된 항목

✅ **문서 작성**
- `SPECIFICATION.md` - 전체 기능 명세 (900줄)
- `IMPLEMENTATION_GUIDE.md` - 구현 가이드 (900줄)
- `QUICK_START.md` - 빠른 시작 가이드 (330줄)
- `UV_SETUP.md` - uv 환경 설정 가이드
- `README.md` - 프로젝트 개요 (315줄)

✅ **프로젝트 설정**
- `pyproject.toml` - uv 기반 의존성 관리
- `.env.example` - 환경 변수 템플릿
- 디렉토리 구조 생성
- 초기 백엔드 파일 생성 (`config.py`, `main.py`)

---

## 🏗️ 지금 할 일

### 1단계: 환경 설정 (5분)

```bash
# 1. .env 파일 설정
cp .env.example .env

# Hugging Face 토큰 입력
# vi .env  (또는 원하는 에디터로 수정)
```

### 2단계: uv 가상환경 생성 (2분)

```bash
# 1. 가상환경 생성
uv venv .venv

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 의존성 설치 (처음엔 시간이 걸림)
uv sync --all-extras
```

### 3단계: 백엔드 실행 테스트 (1분)

```bash
# 백엔드 서버 시작
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4단계: API 확인 (1분)

```bash
# 새로운 터미널에서
curl http://localhost:8000/health

# 또는 브라우저에서
# http://localhost:8000/docs (Swagger UI)
```

---

## 📋 구현 로드맵

### Phase 1: 기본 기능 (우선순위)

#### 모델 로더 (Priority: HIGH)
```
파일: backend/api/model_loader.py
- [ ] Hugging Face 모델 다운로드 API
- [ ] 로컬 모델 업로드 API
- [ ] 모델 메타데이터 조회
- [ ] 테스트 작성
```

#### 데이터셋 도구 (Priority: HIGH)
```
파일: backend/api/dataset_tools.py
- [ ] CSV/JSON 업로드
- [ ] 데이터 정제 함수
- [ ] EDA 통계 계산
- [ ] 시각화 생성
- [ ] 테스트 작성
```

#### 학습 엔진 (Priority: HIGH)
```
파일: backend/services/training_service.py
- [ ] LoRA 설정 생성
- [ ] 학습 인자 설정
- [ ] 학습 루프 구현
- [ ] 체크포인트 저장
- [ ] 테스트 작성
```

#### Chat 인터페이스 (Priority: MEDIUM)
```
파일: backend/api/chat_interface.py
- [ ] 모델 로드
- [ ] 텍스트 생성 API
- [ ] 파라미터 조정
- [ ] 대화 히스토리 관리
- [ ] 테스트 작성
```

#### RAG 파이프라인 (Priority: MEDIUM)
```
파일: backend/services/rag_service.py
- [ ] PDF 파싱
- [ ] 텍스트 청킹
- [ ] 임베딩 생성
- [ ] 벡터 검색
- [ ] 컨텍스트 기반 답변
- [ ] 테스트 작성
```

#### GGUF 변환 (Priority: MEDIUM)
```
파일: backend/services/quantization_service.py
- [ ] 모델 양자화
- [ ] GGUF 변환
- [ ] 파일 최적화
- [ ] 변환 결과 검증
- [ ] 테스트 작성
```

---

## 🔍 현재 상태

### 파일 구조
```
MacTuner/
├── ✅ pyproject.toml         # uv 설정
├── ✅ .env.example           # 환경 변수
├── ✅ SPECIFICATION.md       # 전체 명세
├── ✅ IMPLEMENTATION_GUIDE.md # 구현 가이드
├── ✅ QUICK_START.md         # 빠른 시작
├── ✅ UV_SETUP.md            # uv 가이드
├── ✅ README.md              # 개요
├── ✅ backend/
│   ├── ✅ __init__.py
│   ├── ✅ config.py          # 설정 관리
│   ├── ✅ main.py            # FastAPI 앱
│   ├── api/                   # 라우터 (미구현)
│   ├── services/              # 비즈니스 로직 (미구현)
│   ├── utils/                 # 유틸리티 (미구현)
│   └── database/              # DB 모델 (미구현)
├── tests/                      # 테스트 (미구현)
├── data/                       # 데이터 폴더
├── output/                     # 출력 폴더
├── logs/                       # 로그 폴더
└── frontend/                   # 프론트엔드 (향후)
```

### 다음 구현 우선순위
1. ✅ 환경 설정
2. ⬜ **모델 로더 구현**
3. ⬜ 데이터셋 도구 구현
4. ⬜ 학습 엔진 구현
5. ⬜ Chat 인터페이스 구현
6. ⬜ RAG 파이프라인 구현
7. ⬜ GGUF 변환 구현
8. ⬜ 프론트엔드 구현

---

## 📚 참고 자료

### 추천 읽기 순서
1. `UV_SETUP.md` - 환경 설정 (현재 위치)
2. `QUICK_START.md` - 빠른 시작
3. `SPECIFICATION.md` - 각 기능 상세 명세
4. `IMPLEMENTATION_GUIDE.md` - 구현 예시 코드

### 유용한 명령어

```bash
# 개발 도구
uv run black backend/              # 코드 포맷
uv run ruff check backend/         # 린트 검사
uv run pytest tests/               # 테스트 실행
uv run pytest tests/ --cov=backend # 커버리지

# 패키지 관리
uv pip list                        # 설치된 패키지
uv pip tree                        # 의존성 트리
uv sync --all-extras               # 전체 의존성 설치

# API 문서
# http://localhost:8000/docs       # Swagger UI
# http://localhost:8000/redoc      # ReDoc
```

---

## 🎯 개발 팁

### 1. 작은 단위로 구현
각 API를 구현할 때마다:
- [ ] 함수 작성
- [ ] 타입 지정 (Type hints)
- [ ] 테스트 작성
- [ ] 커밋

### 2. MAC MPS 최적화 유의
```python
import torch

# 자동 감지
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

# 모든 모델/텐서를 device로 이동
model = model.to(device)
```

### 3. 환경 변수 사용
```python
from backend.config import settings

# 설정값 사용
batch_size = settings.DEFAULT_BATCH_SIZE
learning_rate = settings.DEFAULT_LEARNING_RATE
```

### 4. 에러 처리
```python
from fastapi import HTTPException

try:
    # 작업
except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
```

---

## 🚀 다음 단계

**지금 바로:**
```bash
# 1. .env 설정
cp .env.example .env

# 2. 가상환경 생성
uv venv .venv
source .venv/bin/activate
uv sync --all-extras

# 3. 백엔드 실행
python -m uvicorn backend.main:app --reload

# 4. API 확인
# http://localhost:8000/docs
```

**그 다음:**
1. `SPECIFICATION.md` → 기능 1 (모델 로더) 세부 내용 읽기
2. `IMPLEMENTATION_GUIDE.md` → 구현 코드 참고
3. `backend/api/model_loader.py` 구현 시작

---

## 💡 질문이 있으신가요?

- 📖 문서: `SPECIFICATION.md`, `IMPLEMENTATION_GUIDE.md`
- 🔧 설정: `UV_SETUP.md`, `.env.example`
- 🚀 시작: `QUICK_START.md`

**행운을 빕니다! 🍎**

