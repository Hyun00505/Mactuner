# MacTuner 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 필수 요구사항
- macOS 12.0+ (Apple Silicon M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- Git

### 1단계: 저장소 클론 및 기본 설정

```bash
# 저장소 클론
git clone https://github.com/bunhine0452/Mactuner.git
cd Mactuner

# 백엔드 가상 환경 설정
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r backend/requirements.txt
```

### 2단계: 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# Hugging Face
HUGGINGFACE_TOKEN=your_token_here

# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Paths
MODEL_CACHE_DIR=~/.cache/huggingface/hub
DATA_DIR=./data
OUTPUT_DIR=./output
EOF
```

### 3단계: 백엔드 실행

```bash
# 터미널 1: 백엔드 서버
source venv/bin/activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 또는
cd backend && python main.py
```

### 4단계: 프론트엔드 실행 (선택사항)

```bash
# 터미널 2: 프론트엔드 (새로운 터미널 창)
cd frontend
npm install
npm run dev

# 브라우저에서 http://localhost:5173 접속
```

---

## 📚 주요 API 엔드포인트 테스트

### 모델 다운로드
```bash
curl -X POST http://localhost:8000/model/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'
```

### 모델 로컬 업로드
```bash
curl -X POST http://localhost:8000/model/upload \
  -F "file=@/path/to/model"
```

### 데이터셋 업로드
```bash
curl -X POST http://localhost:8000/dataset/upload \
  -F "file=@data.csv"
```

### 데이터셋 분석
```bash
curl -X POST http://localhost:8000/dataset/analyze \
  -F "file=@data.csv"
```

---

## 🏗️ 개발 구조 이해하기

### 백엔드 구조
```
backend/
├── main.py                 # FastAPI 앱 진입점
├── requirements.txt        # Python 패키지
├── api/                    # API 라우터
│   ├── model_loader.py     # 모델 로드 API
│   ├── dataset_tools.py    # 데이터셋 API
│   ├── training.py         # 학습 API
│   ├── chat_interface.py   # 채팅 API
│   ├── rag_pipeline.py     # RAG API
│   └── export_gguf.py      # GGUF 내보내기 API
├── services/               # 비즈니스 로직
│   ├── model_service.py
│   ├── training_service.py
│   ├── rag_service.py
│   └── quantization_service.py
└── utils/                  # 유틸리티
    ├── mac_optimization.py # MAC 최적화
    ├── logger.py           # 로깅
    └── validators.py       # 입력 검증
```

### 프론트엔드 구조
```
frontend/src/
├── components/        # 재사용 가능한 컴포넌트
├── pages/             # 전체 페이지
├── services/          # API 호출 로직
├── store/             # 상태 관리 (Zustand)
├── hooks/             # 커스텀 훅
├── types/             # TypeScript 타입
└── App.tsx            # 메인 앱
```

---

## 🔧 개발 팁

### 백엔드 개발

#### 새로운 API 추가
1. `backend/api/` 에서 새 파일 생성 (예: `new_feature.py`)
2. 라우터 정의:
```python
from fastapi import APIRouter

router = APIRouter(tags=["new_feature"])

@router.post("/action")
async def new_action(data: dict):
    return {"status": "success", "data": data}
```
3. `backend/main.py` 에 라우터 등록:
```python
from api import new_feature
app.include_router(new_feature.router, prefix="/feature")
```

#### 서비스 추가
1. `backend/services/` 에서 새 파일 생성
2. 서비스 클래스 정의:
```python
class MyService:
    def __init__(self):
        pass
    
    def do_something(self):
        pass
```
3. API에서 사용:
```python
from services.my_service import MyService

service = MyService()

@router.post("/action")
async def action():
    return service.do_something()
```

### 프론트엔드 개발

#### 새로운 페이지 추가
1. `src/pages/` 에서 새 파일 생성 (예: `MyPage.tsx`)
2. 컴포넌트 작성:
```typescript
import React from 'react';

export const MyPage: React.FC = () => {
  return (
    <div className="p-6">
      <h1>My Page</h1>
    </div>
  );
};
```
3. 라우팅 추가 (추후 React Router 설정)

#### API 호출 추가
1. `src/services/api.ts` 에 새로운 API 메서드 추가:
```typescript
export const myAPI = {
  fetchData: () => api.get('/my-endpoint'),
  postData: (data: any) => api.post('/my-endpoint', data),
};
```
2. 컴포넌트에서 사용:
```typescript
import { myAPI } from '../services/api';

const response = await myAPI.fetchData();
```

---

## 🐛 트러블슈팅

### 문제: MAC MPS 지원 안 됨
**해결책:**
```python
import torch
# 확인
print(torch.backends.mps.is_available())  # True여야 함

# 수동 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

### 문제: 메모리 부족
**해결책:**
1. 배치 크기 줄이기
2. LoRA 사용 (Full Fine-tuning 대신)
3. 그래디언트 체크포인팅 활성화

### 문제: 모델 다운로드 실패
**해결책:**
```bash
# Hugging Face 토큰 확인
huggingface-cli login

# 캐시 초기화
rm -rf ~/.cache/huggingface/hub/*
```

### 문제: 포트 이미 사용 중
**해결책:**
```bash
# 포트 사용 프로세스 찾기
lsof -i :8000

# 프로세스 종료
kill -9 <PID>
```

---

## 📖 다음 단계

1. **기능 구현 시작**: [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) 참조
2. **상세 명세**: [SPECIFICATION.md](./SPECIFICATION.md) 참조
3. **테스트 작성**: `tests/` 디렉토리 참조
4. **배포**: Docker 사용 권장

---

## 🛠️ 유용한 명령어

### 백엔드
```bash
# 의존성 설치
pip install -r backend/requirements.txt

# 개발 서버 실행
python -m uvicorn backend.main:app --reload

# 테스트 실행
pytest tests/backend/

# 코드 포맷
black backend/

# 린트 확인
flake8 backend/
```

### 프론트엔드
```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 테스트 실행
npm test

# 코드 포맷
npm run format
```

---

## 📞 지원 및 문의

- **이슈 보고**: GitHub Issues
- **토론**: GitHub Discussions
- **이메일**: support@mactuner.dev

---

## ✅ 체크리스트

시작하기 전에 확인하세요:

- [ ] macOS 12.0+ 설치 확인
- [ ] Python 3.11+ 설치 확인
- [ ] Node.js 18+ 설치 확인
- [ ] Git 설치 확인
- [ ] 저장소 클론 완료
- [ ] 가상 환경 생성 완료
- [ ] .env 파일 생성 완료
- [ ] 백엔드 실행 확인
- [ ] API 테스트 성공 확인

모든 항목을 확인했다면 개발을 시작할 준비가 되었습니다! 🎉
