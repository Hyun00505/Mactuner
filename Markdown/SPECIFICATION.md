# MacTuner: MAC 환경 최적화 LLM 파인튜닝 및 배포 통합 플랫폼

## 프로젝트 개요

MacTuner는 Apple Silicon (M1/M2/M3/M4) MAC을 위해 설계된 LLM(대규모 언어 모델) 파인튜닝, PEFT, RAG, GGUF 배포까지를 통합으로 지원하는 플랫폼입니다. 사용자 친화적인 UI와 MAC 최적화 기술을 통해 누구나 쉽게 LLM을 학습하고 배포할 수 있습니다.

---

## 아키텍처 개요

```
MacTuner
├── Frontend (React + TypeScript + Tailwind CSS)
│   ├── Model Management UI
│   ├── Data Modeling Dashboard
│   ├── Training Monitor
│   ├── Chat Interface
│   ├── RAG Setup & Chat
│   └── GGUF Export Control
├── Backend (FastAPI + Python)
│   ├── Model Loader & Manager
│   ├── Data Processing Pipeline
│   ├── Training Engine (LoRA/PEFT)
│   ├── Chat Interface API
│   ├── RAG Pipeline
│   └── GGUF Converter
└── Database & Storage
    ├── Model Registry
    ├── Training Logs
    └── Model Checkpoints
```

---

## 기능 명세서

### 기능 1: 모델 다운로드 및 불러오기

#### 1.1 개요

사용자가 Hugging Face에서 원시 모델을 다운로드하거나 로컬 파일에서 모델을 업로드할 수 있는 기능.

#### 1.2 요구사항

- **Hugging Face 모델 다운로드**

  - Hugging Face 액세스 토큰 입력 UI
  - 모델 검색 기능 (검색어로 모델 목록 조회)
  - 모델 메타데이터 표시 (모델 크기, 파라미터 수, 아키텍처 등)
  - 진행률 표시 및 다운로드 상태 모니터링
  - 로컬 캐시 관리 (이미 다운로드된 모델 확인)

- **로컬 파일 업로드**
  - 로컬 폴더 선택 (파일 브라우저)
  - 지원되는 포맷: `.safetensors`, `.bin`, 폴더 구조
  - 모델 유효성 검사 (config.json, tokenizer 파일 확인)

#### 1.3 기술 구현

- 백엔드: FastAPI + `transformers` 라이브러리
- 모델 로딩: `AutoModelForCausalLM`, `AutoTokenizer`
- 디바이스 최적화: MAC MPS (Metal Performance Shaders) 활용
- 캐싱: `~/.cache/huggingface/hub/` 위치 활용

```python
# 구현 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_from_hub(model_id: str, token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map="auto",  # MAC MPS에 자동 배치
        torch_dtype="auto"
    )
    return model, tokenizer

def load_model_local(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto"
    )
    return model, tokenizer
```

---

### 기능 2: 데이터 모델링 (데이터 정제 및 EDA)

#### 2.1 개요

파인튜닝에 사용할 데이터를 정제하고 탐색적 데이터 분석(EDA)을 통해 데이터 특성을 파악하는 기능.

#### 2.2 요구사항

- **데이터 업로드 & 미리보기**

  - 지원 포맷: CSV, JSON, JSONL, TXT
  - 파일 크기, 행 수 표시
  - 샘플 데이터 미리보기 (처음 5-10개 행)

- **데이터 정제**

  - 결측치(NaN) 처리 (삭제/채우기 옵션)
  - 중복 행 제거
  - 텍스트 정규화 (소문자 변환, 특수문자 제거 등)
  - 토큰 길이 분석 및 필터링
  - 데이터 분할 (Train/Validation 비율 설정, 기본값: 8:2)

- **탐색적 데이터 분석 (EDA)**
  - 통계 요약: 행 수, 열 수, 메모리 사용량, 평균 텍스트 길이
  - 텍스트 길이 분포 시각화 (히스토그램)
  - 단어 빈도 분석 (WordCloud)
  - 토큰 분포 시각화
  - 결측치 비율 시각화

#### 2.3 기술 구현

- 백엔드: Pandas, Plotly, Matplotlib
- 토큰화: `transformers` 라이브러리의 tokenizer
- 시각화: Plotly (인터랙티브) 또는 Matplotlib

```python
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer

def analyze_data(df: pd.DataFrame, tokenizer) -> dict:
    stats = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum()
    }

    # 텍스트 길이 분석
    text_col = [col for col in df.columns if df[col].dtype == 'object'][0]
    df['text_length'] = df[text_col].str.len()
    df['token_count'] = df[text_col].apply(
        lambda x: len(tokenizer.encode(x))
    )

    return stats, df
```

---

### 기능 3: 학습 (파인튜닝 & PEFT)

#### 3.1 개요

MAC 최적화된 파인튜닝 또는 PEFT를 수행하고, 최적의 파라미터 추천 및 학습 모니터링 제공.

#### 3.2 요구사항

#### 3.2.1 파인튜닝 방식 선택

- **Full Fine-tuning**: 모든 파라미터 학습 (고사양 MAC용)
- **LoRA (Low-Rank Adaptation)**: 추가 파라미터만 학습 (권장)
- **QLoRA (Quantized LoRA)**: 양자화된 모델 + LoRA (메모리 효율적)

#### 3.2.2 파라미터 설정 UI

```yaml
기본 파라미터:
  - 학습률 (Learning Rate): 5e-5 (기본값), 범위: 1e-5 ~ 1e-3
  - 배치 크기 (Batch Size): 4 (기본값), 범위: 1 ~ 32
  - 에포크 (Epochs): 3 (기본값), 범위: 1 ~ 100
  - 워밍업 스텝 (Warmup Steps): 500, 범위: 0 ~ 5000
  - 가중치 감쇠 (Weight Decay): 0.01, 범위: 0 ~ 0.1

LoRA 특화 파라미터:
  - LoRA rank (r): 8 (기본값), 범위: 4 ~ 64
  - LoRA alpha: 16 (기본값), 범위: 1 ~ 256
  - LoRA dropout: 0.1 (기본값), 범위: 0 ~ 0.5
  - Target modules: ["q_proj", "v_proj"] (자동 감지)

MAC 최적화 파라미터:
  - 그래디언트 누적 스텝: 1 (기본값)
  - 혼합 정밀도 (fp16/bf16): 활성화 권장
  - 메모리 효율 설정: 그래디언트 체크포인팅 활성화
```

#### 3.2.3 최적 파라미터 추천 엔진

모델 크기와 데이터셋 크기에 따른 자동 추천:

```yaml
소형 모델 (<1B 파라미터):
  - 추천 배치 크기: 8-16
  - 추천 학습률: 5e-4 ~ 1e-3
  - LoRA rank: 16-32
  - 메소드: Full Fine-tuning 또는 LoRA

중형 모델 (1B-7B):
  - 추천 배치 크기: 4-8
  - 추천 학습률: 1e-4 ~ 5e-4
  - LoRA rank: 8-16
  - 메소드: LoRA (권장)

대형 모델 (>7B):
  - 추천 배치 크기: 1-4
  - 추천 학습률: 1e-5 ~ 1e-4
  - LoRA rank: 4-8
  - 메소드: QLoRA (권장)
```

#### 3.2.4 학습 모니터링 대시보드

- **실시간 지표**

  - 학습 손실 (Training Loss) 그래프
  - 검증 손실 (Validation Loss) 그래프
  - 학습 정확도 (Accuracy)
  - GPU/CPU 메모리 사용률
  - 배치당 소요 시간
  - 예상 남은 시간

- **체크포인트 관리**
  - 자동 체크포인트 저장 (N 스텝마다)
  - 최고 성능 모델 자동 저장
  - 체크포인트 목록 및 복구 옵션

#### 3.3 기술 구현

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
import torch

def get_peft_model_config(model_size_params: int) -> LoraConfig:
    """모델 크기에 따른 최적 LoRA 설정"""
    if model_size_params < 1e9:
        rank, alpha = 16, 32
    elif model_size_params < 7e9:
        rank, alpha = 8, 16
    else:
        rank, alpha = 4, 8

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def setup_training(model, train_dataset, eval_dataset, config: dict):
    """MAC 최적화된 트레이닝 설정"""
    peft_config = get_peft_model_config(model.num_parameters())
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'] * 2,
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,  # 메모리 효율
        fp16=torch.cuda.is_available() or torch.backends.mps.is_available(),  # MAC MPS 지원
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    return trainer
```

---

### 기능 4: 데이터 검증 (Chat Interface)

#### 4.1 개요

학습된 모델과 대화를 통해 모델의 성능을 검증하는 기능.

#### 4.2 요구사항

#### 4.2.1 모델 로딩

- 자동 감지: 최신 학습 완료 모델 자동 로드
- 수동 선택: 로컬 폴더 지정
- 원래 모델과 학습된 LoRA 어댑터 자동 병합 옵션

#### 4.2.2 대화 UI

- **메시지 히스토리**: 대화 내용 표시
- **컨텍스트 설정**:
  - 문맥 유지: 전체 대화 히스토리 사용 여부
  - 문맥 윈도우: 최근 N개 메시지만 사용
  - 히스토리 초기화 버튼

#### 4.2.3 생성 파라미터 조정

```yaml
기본 파라미터:
  - 최대 토큰 길이 (Max Tokens): 512 (범위: 1 ~ 2048)
  - 온도 (Temperature): 0.7 (범위: 0 ~ 2.0)
    * 0: 결정론적 (항상 같은 답변)
    * 1.0: 기본값 (균형)
    * >1.0: 창의적 (다양한 답변)
  - Top-K: 50 (범위: 0 ~ 100)
  - Top-P (Nucleus Sampling): 0.95 (범위: 0 ~ 1.0)
  - Repetition Penalty: 1.0 (범위: 1.0 ~ 2.0)

고급 파라미터:
  - 반복 패널티 감마 (Repeat Penalty Gamma): 1.0
  - No Repeat N-Gram Size: 0 (비활성화)
  - Length Penalty: 1.0
```

#### 4.2.4 프롬프트 관리

- 사전 정의된 프롬프트 템플릿 (예: 한국어 지시, 질문응답, 요약 등)
- 커스텀 프롬프트 입력
- 프롬프트 저장/로드 기능

#### 4.3 기술 구현

```python
from transformers import TextGenerationPipeline
import torch

def load_finetuned_model(base_model, lora_adapter_path=None):
    """학습된 모델 로드 및 병합"""
    if lora_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model = model.merge_and_unload()  # 병합

    return model

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    **kwargs
) -> str:
    """응답 생성"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

---

### 기능 5: RAG (Retrieval-Augmented Generation)

#### 5.1 개요

외부 문서(PDF 등)를 활용하여 모델의 답변을 보강하는 RAG 기능.

#### 5.2 요구사항

#### 5.2.1 문서 업로드 및 전처리

- **지원 포맷**: PDF, TXT, MARKDOWN
- **문서 정보 표시**:
  - 파일명, 파일 크기
  - 총 텍스트 길이 (문자 수, 페이지 수)
  - 예상 청크 개수

#### 5.2.2 청킹 설정

```yaml
청킹 파라미터:
  - 청크 크기 (Chunk Size): 512 토큰 (범위: 128 ~ 2048)
  - 오버랩 (Overlap): 50 토큰 (범위: 0 ~ 50% of chunk_size)
  - 오버랩 백분율 표시 (예: 512토큰 기준 50 = 9.8%)
```

#### 5.2.3 임베딩 설정

- 임베딩 모델 선택:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (다국어, 빠름)
  - `sentence-transformers/paraphrase-MiniLM-L6-v2` (영어 특화)
  - `sentence-transformers/all-MiniLM-L12-v2` (범용)
  - Custom 모델 선택 옵션

#### 5.2.4 벡터 스토어 관리

- FAISS 기반 벡터 데이터베이스 구축
- 벡터 저장/로드 기능
- 벡터 스토어 크기, 임베딩 수 표시

#### 5.2.5 RAG 검색 및 대화

- **검색 설정**:

  - 검색 결과 수 (Top-K): 5 (범위: 1 ~ 20)
  - 유사도 임계값: 0.3 (범위: 0 ~ 1.0)

- **대화 인터페이스**:
  - 사용자 질문 입력
  - 검색된 관련 문서 표시 (점수와 함께)
  - 모델 응답 생성 및 표시
  - 기능 4의 대화 파라미터 동일 적용

#### 5.3 기술 구현

```python
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """텍스트를 청크로 분할"""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

def build_vector_store(chunks: list[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """벡터 스토어 구축"""
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    return index, embeddings, chunks, embedding_model

def retrieve_relevant_chunks(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    embedding_model,
    top_k: int = 5
) -> list[tuple[str, float]]:
    """관련 청크 검색"""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)

    # 유사도 점수로 변환 (거리 -> 유사도)
    scores = 1 / (1 + distances[0])

    return [(chunks[i], scores[j]) for j, i in enumerate(indices[0])]

def generate_rag_response(
    query: str,
    model,
    tokenizer,
    relevant_chunks: list[tuple[str, float]],
    **generation_kwargs
) -> str:
    """RAG 기반 응답 생성"""
    context = "\n\n".join([chunk for chunk, _ in relevant_chunks])

    prompt = f"""다음 문맥을 기반으로 질문에 답변하세요:

문맥:
{context}

질문: {query}

답변:"""

    return generate_response(model, tokenizer, prompt, **generation_kwargs)
```

---

### 기능 6: GGUF 배포 (모델 양자화 및 변환)

#### 6.1 개요

학습된 모델을 GGUF 형식으로 변환하여 가벼운 배포가 가능하게 함.

#### 6.2 요구사항

#### 6.2.1 모델 선택 및 변환 준비

- 자동 감지 또는 수동 선택: 학습된 모델 선택
- LoRA 어댑터 병합 옵션
- 변환 전 모델 정보 표시 (크기, 파라미터 수 등)

#### 6.2.2 양자화 방식 선택

```yaml
양자화 옵션:
  - Q4_0 (4-bit, 최고 압축률):
      설명: "가장 작은 파일 크기, 약간의 품질 손실 가능"
      예상 크기: 원본의 약 25%
      추천: 제한된 스토리지/메모리 환경

  - Q4_1 (4-bit with better precision):
      설명: "Q4_0보다 약간 나은 정확도"
      예상 크기: 원본의 약 28%
      추천: Q4_0과 정확도 균형

  - Q5_0 (5-bit):
      설명: "Q4와 Q8 사이의 균형"
      예상 크기: 원본의 약 35%
      추천: 표준 권장안

  - Q5_1 (5-bit with better precision):
      설명: "Q5_0보다 약간 나은 정확도"
      예상 크기: 원본의 약 40%
      추천: Q5_0과 정확도 균형

  - Q8_0 (8-bit):
      설명: "거의 손실 없음, 충분한 압축"
      예상 크기: 원본의 약 50%
      추천: 고품질 필요 시

  - F16 (반정밀도):
      설명: "최소 압축, 최고 품질"
      예상 크기: 원본의 약 50%
      추천: 메모리 충분할 때만
```

#### 6.2.3 변환 프로세스

- 진행률 표시 바
- 예상 소요 시간
- 실시간 로그 표시
- 변환 완료 후 파일 크기 비교 (원본 vs 변환된 모델)

#### 6.2.4 GGUF 모델 테스트

- 변환된 GGUF 모델 간단한 테스트 실행 가능
- llama.cpp 또는 llama-cpp-python으로 로드
- 기본 대화 테스트

#### 6.3 기술 구현

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import json
from pathlib import Path

def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization_type: str = "q4_0",
    use_gpu: bool = True
) -> str:
    """모델을 GGUF 형식으로 변환"""

    # 1. 모델 로드 및 병합 (PEFT인 경우)
    try:
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    # 2. 모델 정규 형식으로 저장
    temp_dir = Path(output_path).parent / "temp_gguf"
    temp_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(temp_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(temp_dir))

    # 3. 변환 스크립트 실행 (llama.cpp 활용)
    convert_cmd = [
        "python",
        "llama/gguf/convert.py",
        str(temp_dir),
        "--outfile", output_path,
        "--outtype", quantization_type.upper()
    ]

    if use_gpu:
        convert_cmd.append("--use-f16")

    subprocess.run(convert_cmd, check=True)

    # 정리
    import shutil
    shutil.rmtree(temp_dir)

    return output_path

def estimate_model_size(quantization_type: str, original_size_gb: float) -> float:
    """양자화에 따른 예상 파일 크기"""
    ratios = {
        "q4_0": 0.25,
        "q4_1": 0.28,
        "q5_0": 0.35,
        "q5_1": 0.40,
        "q8_0": 0.50,
        "f16": 0.50
    }
    return original_size_gb * ratios.get(quantization_type.lower(), 1.0)
```

---

## 사용자 인터페이스 (UI/UX) 명세

### 디자인 철학

- **직관성**: 사용자가 각 기능을 쉽게 찾고 이해할 수 있어야 함
- **효율성**: 최소한의 클릭으로 원하는 작업 완료
- **현대성**: 최신 디자인 트렌드 적용 (다크/라이트 모드)
- **반응성**: 실시간 피드백 제공

### 기술 스택

- **Frontend**: React 18+ + TypeScript + Tailwind CSS + Shadcn/ui
- **State Management**: Zustand 또는 TanStack Query
- **Charts**: Recharts 또는 Chart.js
- **File Upload**: react-dropzone 또는 custom solution
- **WebSocket**: 실시간 학습 모니터링용

### 페이지 구조

```
1. Dashboard (메인 대시보드)
   - 빠른 시작 (Quick Start)
   - 최근 모델/프로젝트
   - 시스템 상태 (메모리, 디스크)
   - 주요 기능 접근 버튼

2. Model Management (모델 관리)
   - 모델 다운로드/업로드
   - 로컬 모델 목록
   - 모델 메타데이터 및 미리보기

3. Data Preparation (데이터 준비)
   - 데이터 업로드
   - EDA 대시보드 (통계, 시각화)
   - 데이터 정제 설정
   - Train/Validation 분할

4. Training (학습)
   - 학습 방식 선택 (Full/LoRA/QLoRA)
   - 파라미터 설정 UI
   - 자동 추천 파라미터 제시
   - 실시간 학습 모니터링 대시보드
   - 체크포인트 관리

5. Chat Interface (대화 인터페이스)
   - 모델 선택
   - 실시간 대화 UI
   - 생성 파라미터 사이드바
   - 대화 히스토리 관리

6. RAG Setup (RAG 설정)
   - 문서 업로드 및 미리보기
   - 청킹 파라미터 설정
   - 임베딩 모델 선택
   - 벡터 스토어 구축 진행률
   - RAG 대화 인터페이스

7. GGUF Export (GGUF 내보내기)
   - 모델 선택 및 정보
   - 양자화 방식 선택 및 설명
   - 예상 파일 크기 표시
   - 변환 진행률
   - 테스트 기능

### 디자인 상세사항

#### 색상 팔레트
```

Primary: #3B82F6 (파란색 - 주요 액션)
Secondary: #8B5CF6 (보라색 - 보조)
Success: #10B981 (초록색 - 완료/성공)
Warning: #F59E0B (주황색 - 주의)
Danger: #EF4444 (빨간색 - 오류)
Neutral: #6B7280 (회색 - 기본)

다크모드:
Background: #0F172A (거의 검은색)
Surface: #1E293B (깊은 파란-회색)
Text: #F1F5F9 (거의 흰색)

````

#### 구성 요소
- 네비게이션 사이드바 (축소/확장 가능)
- 상단 헤더 (로고, 사용자 정보, 설정)
- 메인 콘텐츠 영역
- 우측 패널 (설정, 모니터링 정보)
- 하단 상태 바 (시스템 리소스 사용량)

#### 상호작용 패턴
- 버튼: 명확한 라벨, 아이콘 포함
- 입력 필드: 자동 완성, 입력값 검증 피드백
- 진행률 바: 애니메이션 포함, 예상 시간 표시
- 모달: 확인 전 재확인 (특히 삭제 작업)
- 토스트 알림: 작업 완료/오류 메시지

---

## 기술 스택 및 라이브러리

### 백엔드
```python
# Core
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Model & Training
transformers>=4.36.0
torch>=2.0.0
peft>=0.7.0
datasets>=2.14.0
bitsandbytes>=0.41.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# RAG
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pymupdf>=1.23.0

# GGUF Conversion
llama-cpp-python>=0.2.0
[또는 llama.cpp를 시스템에 설치]

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
tqdm>=4.66.0
tensorboard>=2.14.0
matplotlib>=3.8.0
````

### 프론트엔드

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.28.0",
    "recharts": "^2.10.0",
    "react-dropzone": "^14.2.0",
    "tailwindcss": "^3.4.0",
    "shadcn-ui": "latest",
    "@radix-ui/react-*": "latest"
  }
}
```

### MAC 특화 최적화

- **Metal Performance Shaders (MPS)**: PyTorch MPS 백엔드 활용
- **메모리 효율**: 그래디언트 체크포인팅, 분할 배치
- **디스크 효율**: 저장소 관리, 모델 캐싱

---

## 배포 및 설치

### 요구사항

- macOS 12.0 이상
- Apple Silicon (M1/M2/M3/M4) 또는 Intel Mac (지원하지 않음)
- 8GB RAM 최소 (16GB 이상 권장)
- 50GB 이상 디스크 공간

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/bunhine0452/Mactuner.git
cd Mactuner

# 2. 가상 환경 설정
python -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt
pip install -r requirements-frontend.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# 5. 실행
python -m backend.main  # 백엔드
npm run dev             # 프론트엔드 (별도 터미널)
```

---

## 개발 일정 (우선순위)

### Phase 1 (기본 기능)

- [ ] 모델 다운로드/업로드 API
- [ ] 기본 EDA 대시보드
- [ ] LoRA 파인튜닝 엔진
- [ ] 학습 모니터링 UI
- [ ] 기본 Chat 인터페이스

### Phase 2 (고급 기능)

- [ ] RAG 파이프라인
- [ ] GGUF 변환 기능
- [ ] 파라미터 추천 엔진
- [ ] Advanced Chat 옵션

### Phase 3 (최적화 & 폴리싱)

- [ ] MAC MPS 최적화
- [ ] UI/UX 개선
- [ ] 성능 최적화
- [ ] 테스트 & 문서화

---

## 테스트 전략

### 단위 테스트

- 개별 API 엔드포인트 테스트
- 데이터 처리 함수 테스트
- 모델 로딩/저장 테스트

### 통합 테스트

- 전체 파인튜닝 파이프라인
- RAG 파이프라인
- GGUF 변환 프로세스

### 사용자 수용 테스트

- 실제 사용자 시나리오 테스트
- 성능 및 안정성 검증

---

## 문서화

### 사용자 문서

- 설치 가이드
- 튜토리얼 (기본부터 고급까지)
- FAQ
- 트러블슈팅

### 개발자 문서

- API 문서 (Swagger/OpenAPI)
- 아키텍처 설명
- 코드 컨벤션
- 기여 가이드

---

## 결론

MacTuner는 MAC 사용자들이 접근 가능한 LLM 파인튜닝 및 배포 플랫폼을 제공합니다. 직관적인 UI, 최적화된 기술 스택, 그리고 포괄적인 기능들을 통해 누구나 자신의 데이터로 강력한 LLM 모델을 만들고 배포할 수 있게 합니다.
