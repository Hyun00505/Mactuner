# 🍎 MacTuner: MAC 환경 최적화 LLM 파인튜닝 & 배포 플랫폼

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Node.js](https://img.shields.io/badge/node.js-18%2B-green)
![macOS](https://img.shields.io/badge/macOS-12.0%2B-success)

MacTuner는 Apple Silicon MAC 사용자를 위해 특별히 최적화된 **LLM 파인튜닝, PEFT, RAG, GGUF 배포**를 통합하는 올인원 플랫폼입니다. 직관적인 UI와 강력한 백엔드를 통해 누구나 쉽게 자신의 데이터로 강력한 언어 모델을 만들고 배포할 수 있습니다.

## 🌟 주요 특징

### 1️⃣ **모델 관리**
- ✅ Hugging Face Hub에서 직접 모델 다운로드
- ✅ 로컬 파일에서 모델 업로드
- ✅ 모델 메타데이터 및 통계 제공
- ✅ 스마트 캐싱 및 버전 관리

### 2️⃣ **데이터 처리**
- ✅ CSV, JSON, JSONL, TXT 지원
- ✅ 자동 데이터 정제 (결측치, 중복 제거)
- ✅ 탐색적 데이터 분석 (EDA) 대시보드
- ✅ 토큰 길이 분석 및 시각화

### 3️⃣ **효율적인 학습**
- ✅ **LoRA** (Low-Rank Adaptation) 지원
- ✅ **QLoRA** (양자화 + LoRA) 지원
- ✅ MAC MPS 최적화
- ✅ 자동 파라미터 추천
- ✅ 실시간 학습 모니터링
- ✅ 자동 체크포인트 관리

### 4️⃣ **모델 검증**
- ✅ 학습된 모델과 인터랙티브 채팅
- ✅ 문맥 유지 및 대화 설정
- ✅ 생성 파라미터 조정 (temperature, top-k, top-p 등)
- ✅ 프롬프트 템플릿 관리

### 5️⃣ **RAG 기능**
- ✅ PDF, TXT, MARKDOWN 문서 지원
- ✅ 텍스트 청킹 및 오버랩 설정
- ✅ 임베딩 기반 검색
- ✅ 문서 기반 질의응답

### 6️⃣ **모델 배포**
- ✅ GGUF 형식 변환
- ✅ 다양한 양자화 옵션 (Q4, Q5, Q8, F16)
- ✅ 최적화된 파일 크기
- ✅ llama.cpp 호환성

## 📋 시스템 요구사항

| 요소 | 최소 | 권장 |
|------|------|------|
| **macOS** | 12.0 | 13.0+ |
| **CPU** | Apple Silicon M1 | M2/M3/M4 |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 50GB | 100GB+ |
| **Python** | 3.11 | 3.11+ |
| **Node.js** | 18 | 18+ |

## 🚀 빠른 시작

### 1단계: 저장소 클론
```bash
git clone https://github.com/bunhine0452/Mactuner.git
cd Mactuner
```

### 2단계: 환경 설정
```bash
# Python 가상 환경
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r backend/requirements.txt
```

### 3단계: 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 Hugging Face 토큰 등 설정
```

### 4단계: 백엔드 실행
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5단계: 프론트엔드 실행 (선택사항)
```bash
cd frontend
npm install
npm run dev
```

자세한 설정은 [QUICK_START.md](./QUICK_START.md)를 참조하세요.

## 📚 문서

| 문서 | 설명 |
|------|------|
| [SPECIFICATION.md](./SPECIFICATION.md) | 전체 기능 명세서 (각 기능별 상세 요구사항) |
| [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) | 개발자 구현 가이드 (코드 예시 포함) |
| [QUICK_START.md](./QUICK_START.md) | 빠른 시작 가이드 (5분 내 시작) |

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    MacTuner Platform                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   Frontend (React)   │      │  Backend (FastAPI)   │    │
│  │  - Dashboard         │◄────►│  - Model Service     │    │
│  │  - Model Manager     │      │  - Training Service  │    │
│  │  - Training Monitor  │      │  - RAG Service       │    │
│  │  - Chat Interface    │      │  - GGUF Converter    │    │
│  │  - RAG Setup         │      │  - Data Processor    │    │
│  │  - Export Control    │      │                      │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                           │                 │
│                                           ▼                 │
│                    ┌──────────────────────────────┐         │
│                    │   Storage & Cache            │         │
│                    │  - HF Models Cache           │         │
│                    │  - Training Checkpoints      │         │
│                    │  - RAG Vector Store          │         │
│                    │  - GGUF Models              │         │
│                    └──────────────────────────────┘         │
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   External Services  │      │   MAC Optimization   │    │
│  │  - Hugging Face HQ   │      │  - MPS Backend       │    │
│  │  - PyPI              │      │  - Memory Mgmt       │    │
│  │  - GitHub           │      │  - Performance       │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 💻 기술 스택

### 백엔드
- **Framework**: FastAPI
- **ML/DL**: PyTorch, Transformers, PEFT
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **RAG**: Sentence-transformers, FAISS, PyMuPDF
- **Quantization**: BitsAndBytes, llama-cpp-python

### 프론트엔드
- **Framework**: React 18+
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: Shadcn/ui, Radix UI
- **State**: Zustand
- **Data Fetching**: Axios, React Query
- **Charts**: Recharts

## 🎯 주요 기능 플로우

### 학습 흐름
```
모델 선택 → 데이터 준비 → EDA 분석 → 파라미터 설정 
    ↓
LoRA/QLoRA 학습 → 실시간 모니터링 → 체크포인트 저장
    ↓
최고 성능 모델 자동 저장
```

### RAG 흐름
```
문서 업로드 → 텍스트 추출 → 청킹 설정
    ↓
임베딩 생성 → 벡터 스토어 구축
    ↓
질문 입력 → 관련 문서 검색 → 답변 생성
```

### 배포 흐름
```
학습 완료 모델 선택 → 양자화 방식 선택 → GGUF 변환
    ↓
파일 크기 최적화 → 배포 준비 완료
```

## 📊 성능 최적화

### MAC MPS 활용
- PyTorch Metal Performance Shaders 자동 감지
- 최적의 배치 크기 자동 계산
- 메모리 사용 최적화

### 파인튜닝 효율성
- **LoRA**: 풀 파인튜닝 대비 메모리 90% 절감
- **QLoRA**: 추가로 50% 메모리 절감
- **그래디언트 체크포인팅**: 메모리 사용 30-50% 감소

### 모델 크기 감소
- **Q4_0**: 원본 대비 약 25% 크기
- **Q5_0**: 원본 대비 약 35% 크기
- **Q8_0**: 원본 대비 약 50% 크기

## 🔄 개발 상태

### Phase 1: 기본 기능 ✅ (계획)
- [ ] 모델 다운로드/업로드 API
- [ ] 기본 EDA 대시보드
- [ ] LoRA 파인튜닝
- [ ] 학습 모니터링 UI
- [ ] 기본 Chat 인터페이스

### Phase 2: 고급 기능 🔄 (계획)
- [ ] RAG 파이프라인
- [ ] GGUF 변환
- [ ] 파라미터 추천 엔진
- [ ] Advanced Chat 옵션

### Phase 3: 최적화 & 폴리싱 📋 (계획)
- [ ] MAC MPS 완전 최적화
- [ ] UI/UX 개선
- [ ] 성능 최적화
- [ ] 테스트 & 문서화

## 🔧 개발 가이드

### 새로운 기능 추가

#### 백엔드 API 추가
```python
# backend/api/new_feature.py
from fastapi import APIRouter

router = APIRouter(tags=["new_feature"])

@router.post("/action")
async def new_action(data: dict):
    return {"status": "success"}
```

#### 프론트엔드 페이지 추가
```typescript
// frontend/src/pages/NewPage.tsx
export const NewPage = () => {
  return <div>New Feature</div>;
};
```

더 자세한 가이드는 [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)를 참조하세요.

## 🐛 알려진 문제 및 해결책

| 문제 | 해결책 |
|------|--------|
| MAC MPS 미지원 | PyTorch 최신 버전 업데이트 |
| 메모리 부족 | 배치 크기 줄이기, LoRA 사용 |
| 모델 다운로드 실패 | Hugging Face 토큰 확인, 캐시 초기화 |
| 포트 충돌 | 다른 포트 사용 또는 프로세스 종료 |

## 📞 지원

- 📝 [GitHub Issues](https://github.com/bunhine0452/Mactuner/issues)
- 💬 [GitHub Discussions](https://github.com/bunhine0452/Mactuner/discussions)
- 📧 문의: support@mactuner.dev

## 📜 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 🙏 기여

MacTuner 개발에 기여하고 싶으신가요?

1. Fork 하기
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. Commit 하기 (`git commit -m 'Add some AmazingFeature'`)
4. Push 하기 (`git push origin feature/AmazingFeature`)
5. Pull Request 열기

자세한 기여 가이드는 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참조하세요.

## 🎓 학습 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Hugging Face 가이드](https://huggingface.co/docs)
- [PEFT 문서](https://huggingface.co/docs/peft)
- [RAG 최적 사례](https://python.langchain.com/docs/modules/data_connection/retrievers/)

## 🎉 사용 사례

MacTuner로 다음을 할 수 있습니다:
- 🏢 도메인 특화 챗봇 구축
- 📝 맞춤형 텍스트 생성 모델 개발
- 🔍 회사 문서 기반 QA 시스템
- 🌐 다국어 지원 애플리케이션
- 💼 프로덕션급 LLM 배포

## 🚀 로드맵

- Q1 2024: Phase 1 완료, 베타 릴리즈
- Q2 2024: Phase 2 완료, RAG/GGUF 지원
- Q3 2024: Phase 3 시작, 성능 최적화
- Q4 2024: v1.0 정식 릴리즈

## ⭐ 스타해주세요

MacTuner가 도움이 되었다면 ⭐를 눌러주세요!

---

**만든이**: Bun Hyun Bin  
**시작일**: 2024년 11월  
**상태**: 활발히 개발 중 🚀
