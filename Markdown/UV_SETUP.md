# MacTuner: uv 기반 환경 설정 가이드

## 📦 uv란?

`uv`는 Rust로 작성된 초고속 Python 패키지 관리자입니다. `pip`보다 훨씬 빠르고, `venv` + 의존성 관리를 통합으로 지원합니다.

- **빠른 설치**: pip 대비 10-100배 빠름
- **메모리 효율**: 저메모리 환경에서도 잘 작동
- **Python 관리**: 여러 Python 버전 관리 가능
- **락 파일**: `uv.lock` 파일로 재현 가능한 빌드 지원

---

## 🚀 Step 1: uv 설치 확인

```bash
uv --version
# uv 0.6.11 (또는 더 최신 버전)
```

이미 설치되어 있으므로 넘어갑니다.

---

## 🏗️ Step 2: 가상환경 생성 및 동기화

### 2.1 가상환경 생성

```bash
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner

# 가상환경 생성 (Python 3.11 사용)
uv venv .venv

# 가상환경 활성화
source .venv/bin/activate
```

### 2.2 의존성 설치

```bash
# 모든 의존성 설치 (프로덕션 + 개발 도구)
uv sync --all-extras

# 또는 프로덕션만 설치
uv sync
```

### 2.3 uv.lock 생성 확인

```bash
ls -la uv.lock
# uv.lock 파일이 생성되었으면 OK
```

---

## 📋 Step 3: 프로젝트 구조 생성

### 3.1 백엔드 디렉토리 구조

```bash
mkdir -p backend/{api,services,utils,database}
mkdir -p tests/backend
mkdir -p frontend/src/{components,pages,services,store,hooks,types,styles}
mkdir -p tests/frontend
mkdir -p data output logs docs

# __init__.py 파일 생성
touch backend/__init__.py
touch backend/api/__init__.py
touch backend/services/__init__.py
touch backend/utils/__init__.py
touch backend/database/__init__.py
touch tests/__init__.py
touch tests/backend/__init__.py
```

### 3.2 환경 파일 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 편집 (Hugging Face 토큰 등)
# vi .env  또는 원하는 에디터로 열기
```

---

## 🔧 Step 4: 초기 백엔드 파일 생성

### 4.1 config.py (설정 관리)

```python
# backend/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    """MacTuner 설정 클래스"""

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "True").lower() == "true"

    # Hugging Face
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    MODEL_CACHE_DIR: str = os.path.expanduser(os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface/hub"))

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "output")
    LOG_DIR: Path = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

    # Learning
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "4"))
    DEFAULT_LEARNING_RATE: float = float(os.getenv("DEFAULT_LEARNING_RATE", "5e-5"))
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "3"))

    # LoRA
    LORA_RANK: int = int(os.getenv("LORA_RANK", "8"))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "16"))

    # RAG
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # Performance
    USE_MAC_MPS: bool = os.getenv("USE_MAC_MPS", "True").lower() == "true"
    GRADIENT_CHECKPOINTING: bool = os.getenv("GRADIENT_CHECKPOINTING", "True").lower() == "true"

    def __init__(self):
        # 필요한 디렉토리 생성
        for dir_path in [self.DATA_DIR, self.OUTPUT_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
```

### 4.2 logger.py (로깅 설정)

```python
# backend/utils/logger.py
import logging
import os
from pathlib import Path

from backend.config import settings

def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    # 파일 핸들러
    log_file = settings.LOG_DIR / f"{name}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 포매터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = get_logger("MacTuner")
```

### 4.3 main.py (FastAPI 앱)

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.config import settings

# API 라우터 임포트 (향후)
# from backend.api import model_loader, training, chat_interface, rag_pipeline, export_gguf, dataset_tools

app = FastAPI(
    title="MacTuner API",
    description="MAC 환경 최적화 LLM 파인튜닝 & 배포 플랫폼",
    version="0.1.0"
)

# CORS 설정
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (향후)
# app.include_router(model_loader.router, prefix="/model")
# app.include_router(dataset_tools.router, prefix="/dataset")
# app.include_router(training.router, prefix="/train")
# app.include_router(chat_interface.router, prefix="/chat")
# app.include_router(rag_pipeline.router, prefix="/rag")
# app.include_router(export_gguf.router, prefix="/export")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MacTuner API v0.1.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
        log_level="info"
    )
```

---

## ▶️ Step 5: 백엔드 실행

### 5.1 개발 서버 시작

```bash
# 터미널 1: 백엔드
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 또는
cd backend
python main.py
```

### 5.2 API 테스트

```bash
# 터미널 2: API 테스트
curl http://localhost:8000/health
# 응답: {"status":"ok","version":"0.1.0","environment":"development"}

# Swagger UI 접속
# http://localhost:8000/docs
```

---

## 📦 Step 6: 자주 사용하는 uv 명령어

### 의존성 관리

```bash
# 모든 의존성 설치 (개발 도구 포함)
uv sync --all-extras

# 프로덕션 의존성만 설치
uv sync --no-dev

# 새로운 패키지 추가
uv pip install numpy pandas

# 패키지 제거
uv pip uninstall numpy

# 의존성 목록 표시
uv pip list

# 의존성 트리 표시
uv pip tree
```

### 개발 도구

```bash
# 코드 포맷 (black)
uv run black backend/

# 린트 검사 (ruff)
uv run ruff check backend/

# 타입 검사 (mypy)
uv run mypy backend/

# 테스트 실행 (pytest)
uv run pytest tests/

# 테스트 + 커버리지
uv run pytest tests/ --cov=backend
```

### 동기화 및 락 파일

```bash
# lock 파일 업데이트
uv lock

# lock 파일 기반으로 설치
uv sync
```

---

## 🐛 자주 발생하는 문제

### 문제 1: `torch` 설치 오류

**원인**: Mac에서 PyTorch는 MPS 지원 버전이 필요

```bash
# 해결책: torch 수동 설치
uv pip install torch torchvision torchaudio --no-index --find-links https://download.pytorch.org/whl/nightly/cpu_and_mps
```

### 문제 2: `bitsandbytes` 설치 실패

**원인**: Mac에서는 설치가 복잡할 수 있음

```bash
# 해결책: 선택사항으로 설정 (pyproject.toml에서 주석 처리)
```

### 문제 3: 가상환경 활성화 안 됨

```bash
# 확인
which python
# ~/.venv/bin/python이 나와야 함

# 다시 활성화
source .venv/bin/activate
```

---

## ✅ 체크리스트

- [ ] uv 설치 확인 (`uv --version`)
- [ ] 가상환경 생성 (`uv venv .venv`)
- [ ] 가상환경 활성화 (`source .venv/bin/activate`)
- [ ] 의존성 설치 (`uv sync --all-extras`)
- [ ] .env 파일 생성 및 설정
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] 초기 파일 생성 (config.py, main.py, logger.py)
- [ ] 백엔드 서버 실행 및 테스트
- [ ] Swagger UI 접속 확인 (`http://localhost:8000/docs`)

모든 항목을 확인했다면 개발을 시작할 준비가 되었습니다! 🎉
