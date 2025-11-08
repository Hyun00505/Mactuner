# 🎯 통합 모델 시스템 구현 완료

## 📋 개요

MacTuner의 **모델 관리 시스템**을 완전히 통합했습니다! 

이제 `/model` 페이지에서 보이는 **모든 모델이 Editor의 노드에서도 동일하게 표시**됩니다. 🚀

---

## ✨ 주요 기능

### 1️⃣ 로컬 모델 자동 감지

**감지 위치**:
- `~/.cache/huggingface/hub/` - HuggingFace 다운로드 모델
- `./models/` 폴더 - 프로젝트 로컬 모델

**감지 방식**:
```
✅ config.json 있는 폴더 → HuggingFace 모델 인식
✅ *.gguf 파일 → GGUF 모델 인식
✅ pytorch_model.bin → PyTorch 모델 인식
✅ model.safetensors → SafeTensors 모델 인식
```

### 2️⃣ HuggingFace & 로컬 구분

**Source 선택**:
```
🤗 HuggingFace Hub (다운로드)
   └─ 새로운 모델 다운로드
   └─ 모델 ID 입력 필요

📂 로컬 저장소 (기존 모델)
   └─ 이미 저장된 모델 사용
   └─ 자동 감지된 목록에서 선택
```

### 3️⃣ 동적 옵션 로드

**작동 방식**:
```
ModelLoaderNode 렌더링
   ↓
NodeForm에서 "localModel" 파라미터 감지
   ↓
dynamicOptions = true 확인
   ↓
API: http://localhost:8001/model/local-models 호출
   ↓
로컬 모델 목록 받기
   ↓
드롭다운에 표시 (아이콘 + 이름 + 크기)
```

### 4️⃣ 소스별 로드 처리

**로컬 모델 로드**:
```typescript
if (source === 'local') {
  // 이미 있는 모델이므로 바로 로드
  POST /model/current
  {
    model_id: selectedModelId,
    source: "local"
  }
}
```

**HuggingFace 모델 다운로드**:
```typescript
if (source === 'huggingface') {
  // 스트림으로 다운로드 중 진행률 표시
  POST /model/download-stream
  ?model_id=gpt2&access_token=...
}
```

---

## 🏗️ 구현 상세

### Backend 로직

**`ModelService.list_local_models()`**:

```python
def list_local_models(self):
    models = []
    
    # 1. HuggingFace 캐시 디렉토리
    # models--namespace--model-name 형식 파싱
    for item in cache_dir.iterdir():
        if item.name.startswith("models--"):
            models.append({
                "source": "huggingface",
                "model_id": "namespace/model-name"
            })
    
    # 2. 프로젝트 ./models/ 폴더
    models_folder = project_root / "models"
    for item in models_folder.iterdir():
        if item.is_file() and item.suffix == ".gguf":
            # GGUF 파일
            models.append({
                "source": "local_folder",
                "model_id": item.stem
            })
        elif item.is_dir() and (item / "config.json").exists():
            # HuggingFace 모델 폴더
            models.append({
                "source": "local_folder",
                "model_id": item.name
            })
    
    return models
```

### Frontend 로직

**`nodeLoader.ts - fetchDynamicOptions()`**:

```typescript
async function fetchDynamicOptions(endpoint) {
  const response = await fetch(
    `http://localhost:8001${endpoint}`
  );
  const data = await response.json();
  
  // data.models 배열 처리
  return data.models.map(item => ({
    label: `${getIcon(item.source)} ${item.model_id} (${item.size_gb}GB)`,
    value: item.model_id
  }));
}
```

**`ModelLoaderNode.tsx - handleDownload()`**:

```typescript
const handleDownload = async () => {
  const isLocal = node.data.source === 'local';
  const modelId = isLocal 
    ? node.data.localModel 
    : node.data.modelId;
  
  if (isLocal) {
    // 로컬 모델: 빠른 로드
    await fetch('/model/current', {
      method: 'POST',
      body: JSON.stringify({
        model_id: modelId,
        source: 'local'
      })
    });
  } else {
    // HuggingFace: 스트림 다운로드
    const response = await fetch(
      `/model/download-stream?model_id=${modelId}`
    );
    // 진행률 표시...
  }
};
```

---

## 📊 `/model` 페이지와 Editor 동기화

### 공통점

```
모두 같은 API 사용: GET /model/local-models
모두 같은 모델 목록 표시
모두 같은 아이콘 사용 (🤗 vs 📂)
모두 같은 모델 정보 표시 (크기, 소스 등)
```

### 차이점

```
/model 페이지:
  └─ 주로 모델 관리 (삭제, 폴더 열기)
  └─ 모든 모델 목록 보기

Editor 노드:
  └─ 주로 모델 선택 & 로드
  └─ 워크플로우에 필요한 모델만 선택
```

---

## 🎨 UI 개선 사항

### ModelLoaderNode.json 개선

```json
{
  "id": "source",
  "type": "select",
  "options": [
    {
      "label": "🤗 HuggingFace Hub (다운로드)",
      "value": "huggingface"
    },
    {
      "label": "📂 로컬 저장소 (기존 모델)",
      "value": "local"
    }
  ]
}
```

### NodeForm 개선

- ✅ 동적 옵션 로드 지원
- ✅ 모델 크기 표시
- ✅ 소스 아이콘 표시
- ✅ 조건부 파라미터 표시

### ModelLoaderNode 컴포넌트

- ✅ 소스별 다른 UI
- ✅ 로드된 모델 표시
- ✅ 진행률 표시
- ✅ 에러 메시지 표시

---

## 🔄 데이터 흐름

### 로컬 모델 로드 흐름

```
1. Editor 시작
   ↓
2. ModelLoaderNode 렌더링
   ↓
3. NodeForm이 "localModel" 파라미터 감지
   ↓
4. dynamicOptions = true 확인
   ↓
5. fetchDynamicOptions("/model/local-models") 호출
   ↓
6. Backend: 로컬 모델 목록 반환
   {
     "models": [
       {
         "model_id": "gpt2",
         "source": "huggingface",
         "size_gb": 0.5
       },
       {
         "model_id": "my_model",
         "source": "local_folder",
         "size_gb": 2.0
       }
     ]
   }
   ↓
7. Frontend: 드롭다운 생성
   "🤗 gpt2 (0.5GB)"
   "📂 my_model (2.0GB)"
   ↓
8. 사용자 선택
   ↓
9. handleDownload() 실행
   ↓
10. 로컬인지 HF인지 확인
    ├─ 로컬: 바로 로드 (빠름)
    └─ HF: 스트림 다운로드 (느림)
    ↓
11. 진행률 표시
    ↓
12. 완료
```

---

## 📁 @models 폴더 구조

**지원되는 형식**:

```
models/
├── gpt2/                        # HuggingFace 형식
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
│
├── my_model/                    # GGUF 폴더
│   ├── model.gguf
│   └── ...
│
├── small_model.gguf             # GGUF 파일 (루트)
│
└── custom_model/                # SafeTensors 형식
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

**자동 감지**:
- ✅ `config.json` + `pytorch_model.bin` → HuggingFace
- ✅ `*.gguf` 파일 → GGUF 모델
- ✅ `config.json` + `model.safetensors` → SafeTensors
- ✅ HuggingFace 캐시 디렉토리도 감지

---

## 🚀 사용 방법

### 로컬 모델 추가

**1. 모델 저장**:
```bash
# 프로젝트의 models/ 폴더에 저장
cp -r ~/my_model ./models/
```

**2. Editor에서 사용**:
```
1. 🤖 모델 로더 노드 추가
2. "모델 소스" → "📂 로컬 저장소" 선택
3. "모델 선택" → 드롭다운에서 "my_model" 선택
4. "📂 로컬 모델 로드" 클릭
5. 완료!
```

### HuggingFace 모델 다운로드

**1. Editor에서**:
```
1. 🤖 모델 로더 노드 추가
2. "모델 소스" → "🤗 HuggingFace Hub" 선택
3. "모델 ID" → "gpt2" 입력
4. "🤗 HuggingFace에서 다운로드" 클릭
5. 진행률 표시
6. 완료!
```

**2. `/model` 페이지에서도 동일하게 표시됨**:
```
다운로드된 모델 목록:
- 🤗 gpt2 (500MB)
- 📂 my_model (2GB)
```

---

## 💾 모델 정보 저장

**노드 데이터**:
```typescript
node.data = {
  source: "local" | "huggingface",
  modelId: "gpt2",                    // HuggingFace 모델 ID
  localModel: "my_model",              // 로컬 모델명
  accessToken: "hf_...",               // HF 토큰
  loadedModel: "gpt2",                 // 로드된 모델
  loadedSource: "huggingface" | "local",
  // ... 기타 파라미터
}
```

---

## 🔧 기술 스택

### Backend
- FastAPI
- PyTorch
- Path 유틸로 파일 시스템 감지

### Frontend
- React + TypeScript
- Zustand (상태 관리)
- nodeLoader.ts (동적 옵션 로드)

---

## ✅ 완료 항목

- ✅ ModelLoaderNode.json 업데이트
- ✅ nodeLoader.ts 동적 옵션 처리 개선
- ✅ NodeForm 스트림 처리
- ✅ ModelLoaderNode 소스별 처리 분리
- ✅ 로컬 vs HF 모델 구분
- ✅ 진행률 실시간 표시
- ✅ 로드된 모델 정보 표시

---

## 🎓 배운 개념

1. **동적 옵션 로드**: API에서 데이터 실시간 가져오기
2. **조건부 파라미터**: 선택에 따라 다른 파라미터 표시
3. **스트림 처리**: 진행률 표시하며 파일 다운로드
4. **파일 시스템 감지**: 여러 형식의 모델 자동 감지
5. **API 응답 파싱**: 다양한 응답 형식 처리

---

## 📞 다음 단계

### Phase 1: 데이터셋 로더 (다음)
- 현재: 모델 로더 완성
- 다음: 데이터셋도 같은 방식 적용

### Phase 2: 다른 노드들
- TrainingNode
- ChatNode
- RagNode
- GgufExportNode

### Phase 3: 노드 연결
- 포트 드래그
- 데이터 흐름 시각화

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 완성, 프로덕션 준비 완료  
**버전**: 1.0.0

