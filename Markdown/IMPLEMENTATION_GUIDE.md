# MacTuner 구현 가이드

## 📋 목차

1. [프로젝트 초기 설정](#프로젝트-초기-설정)
2. [백엔드 구현](#백엔드-구현)
3. [프론트엔드 구현](#프론트엔드-구현)
4. [통합 및 테스트](#통합-및-테스트)
5. [배포](#배포)

---

## 프로젝트 초기 설정

### 1.1 디렉토리 구조 생성

```bash
MacTuner/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── db.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── dataset_tools.py
│   │   ├── training.py
│   │   ├── chat_interface.py
│   │   ├── rag_pipeline.py
│   │   └── export_gguf.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py
│   │   ├── training_service.py
│   │   ├── rag_service.py
│   │   └── quantization_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── validators.py
│   │   └── mac_optimization.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── store/
│   │   ├── hooks/
│   │   ├── types/
│   │   ├── styles/
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── backend/
│   └── frontend/
├── docs/
├── SPECIFICATION.md
├── IMPLEMENTATION_GUIDE.md
└── README.md
```

### 1.2 백엔드 초기 설정

```bash
# Python 환경 설정
python -m venv venv
source venv/bin/activate

# requirements.txt 생성
pip install -r backend/requirements.txt

# 환경 변수 설정
cp .env.example .env
```

### 1.3 프론트엔드 초기 설정

```bash
# Node.js 프로젝트 초기화
cd frontend
npm create vite@latest . -- --template react-ts
npm install

# UI 라이브러리 설치
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install @radix-ui/react-* lucide-react recharts zustand @tanstack/react-query axios
```

---

## 백엔드 구현

### 2.1 MAC 최적화 유틸리티 (mac_optimization.py)

```python
# backend/utils/mac_optimization.py
import torch
import psutil
import os

class MACOptimizer:
    """MAC (Apple Silicon) 최적화 관련 유틸리티"""

    @staticmethod
    def get_device():
        """MAC MPS 또는 CPU 선택"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def get_optimal_batch_size(model_params: int) -> int:
        """모델 크기에 따른 최적 배치 크기"""
        # 가용 메모리 확인
        memory_gb = psutil.virtual_memory().available / (1024**3)

        if model_params > 7e9:  # 7B+
            return max(1, int(memory_gb / 8))
        elif model_params > 1e9:  # 1B-7B
            return max(4, int(memory_gb / 4))
        else:  # <1B
            return max(8, int(memory_gb / 2))

    @staticmethod
    def get_memory_stats():
        """메모리 사용 통계"""
        return {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "percent": psutil.virtual_memory().percent,
            "device": str(MACOptimizer.get_device())
        }
```

### 2.2 모델 로더 서비스 (model_service.py)

```python
# backend/services/model_service.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from utils.mac_optimization import MACOptimizer

class ModelService:
    def __init__(self):
        self.device = MACOptimizer.get_device()
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    def load_from_hub(self, model_id: str, token: str = None):
        """Hugging Face에서 모델 다운로드"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=token,
                cache_dir=self.cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=token,
                device_map="auto",
                torch_dtype=torch.float16 if self.device.type == "mps" else "auto",
                cache_dir=self.cache_dir
            )
            return model, tokenizer, self._extract_metadata(model)
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {str(e)}")

    def load_local(self, path: str):
        """로컬에서 모델 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"경로를 찾을 수 없음: {path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            return model, tokenizer, self._extract_metadata(model)
        except Exception as e:
            raise RuntimeError(f"로컬 모델 로드 실패: {str(e)}")

    def _extract_metadata(self, model):
        """모델 메타데이터 추출"""
        return {
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 0,
            "vocab_size": model.config.vocab_size,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(self.device)
        }
```

### 2.3 학습 서비스 (training_service.py)

```python
# backend/services/training_service.py
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
import torch
from utils.mac_optimization import MACOptimizer

class TrainingService:
    def __init__(self):
        self.device = MACOptimizer.get_device()

    def get_lora_config(self, model_size_params: int):
        """모델 크기에 맞는 LoRA 설정"""
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

    def prepare_model(self, model, config: dict):
        """모델 학습 준비"""
        if config.get("method") == "lora":
            peft_config = self.get_lora_config(model.num_parameters())
            model = get_peft_model(model, peft_config)

        return model

    def get_training_args(self, config: dict):
        """트레이닝 인자 설정"""
        return TrainingArguments(
            output_dir=config.get("output_dir", "./results"),
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            per_device_eval_batch_size=config.get("batch_size", 4) * 2,
            learning_rate=config.get("learning_rate", 5e-5),
            warmup_steps=config.get("warmup_steps", 500),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            gradient_checkpointing=True,
            fp16=self.device.type in ["cuda", "mps"],
            logging_dir="./logs",
        )
```

### 2.4 RAG 서비스 (rag_service.py)

```python
# backend/services/rag_service.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader

class RAGService:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.chunks = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50):
        """텍스트를 청크로 분할"""
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def build_vector_store(self, chunks: list[str], model_name: str):
        """벡터 스토어 구축"""
        self.embedding_model = SentenceTransformer(model_name)
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5):
        """관련 청크 검색"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        scores = 1 / (1 + distances[0])
        results = [(self.chunks[i], float(scores[j])) for j, i in enumerate(indices[0])]
        return results
```

### 2.5 API 라우터 구현

#### 2.5.1 모델 로더 API

```python
# backend/api/model_loader.py (수정)
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from services.model_service import ModelService

router = APIRouter(tags=["model"])
model_service = ModelService()
_MODEL_CACHE = {}

class ModelDownloadRequest(BaseModel):
    model_id: str
    access_token: str = None

class ModelDownloadResponse(BaseModel):
    status: str
    metadata: dict

@router.post("/download", response_model=ModelDownloadResponse)
async def download_model(payload: ModelDownloadRequest):
    """Hugging Face에서 모델 다운로드"""
    try:
        model, tokenizer, metadata = model_service.load_from_hub(
            payload.model_id,
            payload.access_token
        )
        _MODEL_CACHE.update({
            "model": model,
            "tokenizer": tokenizer,
            "metadata": metadata
        })
        return ModelDownloadResponse(status="success", metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """로컬 모델 파일 업로드"""
    try:
        # 파일 저장 및 검증
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        model, tokenizer, metadata = model_service.load_local(tmp_path)
        _MODEL_CACHE.update({
            "model": model,
            "tokenizer": tokenizer,
            "metadata": metadata
        })
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### 2.5.2 데이터셋 도구 API

```python
# backend/api/dataset_tools.py (구현)
from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io
import json

router = APIRouter(tags=["dataset"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """데이터셋 업로드"""
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise ValueError("지원되지 않는 파일 형식")

        stats = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "preview": df.head(5).to_dict()
        }

        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """데이터셋 분석"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        analysis = {
            "describe": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
        }

        return {"status": "success", "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### 2.5.3 학습 API

```python
# backend/api/training.py (구현)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.training_service import TrainingService
from transformers import Trainer
import asyncio

router = APIRouter(tags=["training"])
training_service = TrainingService()
_TRAINING_STATE = {}

class TrainingConfig(BaseModel):
    method: str  # "full", "lora", "qlora"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    output_dir: str = "./results"

@router.post("/start")
async def start_training(config: TrainingConfig):
    """학습 시작"""
    try:
        # 모델 및 데이터셋 준비 로직
        # (세부 구현은 실제 프로젝트에서 작성)
        return {"status": "started", "training_id": "train_001"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """학습 상태 조회"""
    return _TRAINING_STATE.get(training_id, {"status": "not_found"})
```

---

## 프론트엔드 구현

### 3.1 프로젝트 구조

```
frontend/src/
├── components/
│   ├── common/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Layout.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── ModelManagement.tsx
│   │   ├── DataPreparation.tsx
│   │   ├── Training.tsx
│   │   ├── Chat.tsx
│   │   ├── RAG.tsx
│   │   └── GGUFExport.tsx
│   └── forms/
│       ├── ModelDownloadForm.tsx
│       ├── TrainingConfigForm.tsx
│       └── RAGSetupForm.tsx
├── pages/
│   └── index.tsx
├── services/
│   ├── api.ts
│   └── websocket.ts
├── store/
│   └── appStore.ts
├── hooks/
│   └── useTrainingMonitor.ts
├── types/
│   └── index.ts
└── App.tsx
```

### 3.2 타입 정의

```typescript
// frontend/src/types/index.ts
export interface Model {
  id: string;
  name: string;
  source: "hub" | "local";
  size: number;
  parameters: number;
  metadata: Record<string, any>;
}

export interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: string[];
  size: number;
}

export interface TrainingConfig {
  method: "full" | "lora" | "qlora";
  epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  output_dir: string;
}

export interface TrainingState {
  status: "idle" | "running" | "completed" | "failed";
  epoch: number;
  step: number;
  loss: number;
  eval_loss?: number;
  progress: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}
```

### 3.3 상태 관리

```typescript
// frontend/src/store/appStore.ts
import { create } from "zustand";
import { Model, Dataset, TrainingState } from "../types";

interface AppState {
  // Models
  models: Model[];
  selectedModel: Model | null;
  setSelectedModel: (model: Model) => void;

  // Datasets
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  setSelectedDataset: (dataset: Dataset) => void;

  // Training
  trainingState: TrainingState;
  updateTrainingState: (state: Partial<TrainingState>) => void;

  // UI
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  models: [],
  selectedModel: null,
  setSelectedModel: (model) => set({ selectedModel: model }),

  datasets: [],
  selectedDataset: null,
  setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),

  trainingState: {
    status: "idle",
    epoch: 0,
    step: 0,
    loss: 0,
    progress: 0,
  },
  updateTrainingState: (state) =>
    set((prev) => ({
      trainingState: { ...prev.trainingState, ...state },
    })),

  sidebarOpen: true,
  toggleSidebar: () => set((prev) => ({ sidebarOpen: !prev.sidebarOpen })),
}));
```

### 3.4 API 서비스

```typescript
// frontend/src/services/api.ts
import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const modelAPI = {
  downloadFromHub: (modelId: string, token?: string) => api.post("/model/download", { model_id: modelId, access_token: token }),

  uploadLocal: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/model/upload", formData);
  },
};

export const datasetAPI = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/dataset/upload", formData);
  },

  analyze: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/dataset/analyze", formData);
  },
};

export const trainingAPI = {
  start: (config: any) => api.post("/train/start", config),

  getStatus: (trainingId: string) => api.get(`/train/status/${trainingId}`),
};

export const chatAPI = {
  chat: (message: string, modelId: string, config?: any) => api.post("/chat/generate", { message, model_id: modelId, ...config }),
};

export const ragAPI = {
  uploadDocument: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/rag/upload", formData);
  },

  buildIndex: (config: any) => api.post("/rag/build", config),

  search: (query: string, topK: number = 5) => api.post("/rag/search", { query, top_k: topK }),
};

export const ggufAPI = {
  convertModel: (modelPath: string, quantizationType: string) => api.post("/export/gguf", { model_path: modelPath, quantization_type: quantizationType }),
};
```

### 3.5 주요 컴포넌트

#### 3.5.1 모델 다운로드 폼

```typescript
// frontend/src/components/forms/ModelDownloadForm.tsx
import React, { useState } from "react";
import { modelAPI } from "../../services/api";

export const ModelDownloadForm: React.FC = () => {
  const [modelId, setModelId] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await modelAPI.downloadFromHub(modelId, token);
      console.log("모델 다운로드 성공:", response.data);
      // 성공 처리
    } catch (err: any) {
      setError(err.response?.data?.detail || "다운로드 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">모델 다운로드</h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">모델 ID</label>
          <input type="text" value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder="예: meta-llama/Llama-2-7b" className="w-full px-4 py-2 border rounded-lg" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Hugging Face Token</label>
          <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="토큰 입력" className="w-full px-4 py-2 border rounded-lg" />
        </div>

        {error && <div className="p-4 bg-red-50 text-red-700 rounded-lg">{error}</div>}

        <button onClick={handleDownload} disabled={!modelId || loading} className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
          {loading ? "다운로드 중..." : "다운로드"}
        </button>
      </div>
    </div>
  );
};
```

#### 3.5.2 학습 모니터 훅

```typescript
// frontend/src/hooks/useTrainingMonitor.ts
import { useEffect, useCallback } from "react";
import { useAppStore } from "../store/appStore";
import { trainingAPI } from "../services/api";

export const useTrainingMonitor = (trainingId: string | null) => {
  const updateTrainingState = useAppStore((state) => state.updateTrainingState);

  const pollTrainingStatus = useCallback(async () => {
    if (!trainingId) return;

    try {
      const response = await trainingAPI.getStatus(trainingId);
      updateTrainingState(response.data);
    } catch (error) {
      console.error("학습 상태 조회 실패:", error);
    }
  }, [trainingId, updateTrainingState]);

  useEffect(() => {
    const interval = setInterval(pollTrainingStatus, 2000);
    return () => clearInterval(interval);
  }, [pollTrainingStatus]);
};
```

---

## 통합 및 테스트

### 4.1 백엔드 테스트

```python
# tests/backend/test_model_loader.py
import pytest
from backend.services.model_service import ModelService

@pytest.fixture
def model_service():
    return ModelService()

def test_load_from_hub(model_service):
    """Hugging Face에서 모델 로드 테스트"""
    model, tokenizer, metadata = model_service.load_from_hub(
        "gpt2"
    )
    assert model is not None
    assert tokenizer is not None
    assert "num_parameters" in metadata
```

### 4.2 API 테스트

```python
# tests/backend/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_model_download_endpoint():
    """모델 다운로드 엔드포인트 테스트"""
    response = client.post(
        "/model/download",
        json={"model_id": "gpt2"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
```

### 4.3 프론트엔드 테스트

```typescript
// tests/frontend/ModelDownloadForm.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { ModelDownloadForm } from "../../src/components/forms/ModelDownloadForm";

test("renders download form", () => {
  render(<ModelDownloadForm />);
  expect(screen.getByText("모델 다운로드")).toBeInTheDocument();
});

test("handles model download", async () => {
  render(<ModelDownloadForm />);

  const input = screen.getByPlaceholderText(/모델 ID/i);
  fireEvent.change(input, { target: { value: "gpt2" } });

  const button = screen.getByText("다운로드");
  fireEvent.click(button);

  // 테스트 로직
});
```

---

## 배포

### 5.1 Docker 배포

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 백엔드 의존성
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node 설치
RUN apt-get update && apt-get install -y nodejs npm
RUN rm -rf /var/lib/apt/lists/*

# 프론트엔드 의존성
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm ci --omit=dev

# 프론트엔드 빌드
COPY frontend .
RUN npm run build

WORKDIR /app

# 코드 복사
COPY backend .

# 포트 노출
EXPOSE 8000

# 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 Docker Compose

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  mactuner:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
```

### 5.3 배포 체크리스트

- [ ] 환경 변수 설정 (.env)
- [ ] 데이터베이스 마이그레이션
- [ ] 정적 파일 최적화
- [ ] CORS 설정 확인
- [ ] 로깅 설정
- [ ] 모니터링 도구 연결
- [ ] 보안 헤더 설정
- [ ] 성능 테스트 완료

---

## 결론

이 구현 가이드는 MacTuner 개발의 기본 틀을 제시합니다. 각 단계를 순서대로 진행하면서 테스트를 통해 기능을 검증해야 합니다.
