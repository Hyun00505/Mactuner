# 📋 JSON 기반 노드 정의 시스템 구현 완료

## 🎉 완성 내용

MacTuner의 Workflow Editor에 **JSON 기반 동적 노드 시스템**을 구현했습니다! 🚀

---

## 📁 생성된 파일 (총 5개)

### 1️⃣ 노드 정의 JSON 파일

```
frontend/src/@nodes/
├── ModelLoaderNode.json          📄 모델 로더 (완전)
└── DatasetLoaderNode.json        📄 데이터셋 로더 (완전)
```

### 2️⃣ 유틸리티

```
frontend/src/utils/
└── nodeLoader.ts                 ⚙️ 노드 로더 및 유효성 검사
```

### 3️⃣ 컴포넌트

```
frontend/src/components/Editor/
├── NodeForm.tsx                  🎨 동적 파라미터 폼
├── ModelLoaderNode.tsx           ✏️ 완전히 다시 구현
└── DatasetLoaderNode.tsx         ✏️ 완전히 다시 구현
```

---

## 🏗️ JSON 노드 정의 구조

### 기본 구조

```json
{
  "id": "model-loader",
  "name": "Model Loader",
  "icon": "🤖",
  "category": "input",
  "description": "설명",
  "color": "from-blue-500 to-blue-600",
  "inputs": [],
  "outputs": [
    {
      "id": "out-model",
      "name": "Model",
      "type": "model"
    }
  ],
  "parameters": [
    {
      "id": "source",
      "name": "Source",
      "type": "select",
      "label": "Data Source",
      "value": "huggingface",
      "options": [...]
    }
  ]
}
```

### 파라미터 타입

| 타입       | 설명        | 예제         |
| ---------- | ----------- | ------------ |
| `text`     | 텍스트 입력 | 모델 ID 입력 |
| `password` | 숨겨진 입력 | API 토큰     |
| `number`   | 숫자 입력   | 배치 크기    |
| `select`   | 드롭다운    | 포맷 선택    |
| `checkbox` | 체크박스    | 옵션 활성화  |
| `textarea` | 긴 텍스트   | 프롬프트     |
| `file`     | 파일 업로드 | 데이터셋     |

### 조건부 파라미터

```json
{
  "id": "modelId",
  "condition": {
    "parameter": "source",
    "operator": "equals",
    "value": "huggingface"
  }
}
```

### 동적 옵션

```json
{
  "id": "localModel",
  "type": "select",
  "dynamicOptions": true,
  "apiEndpoint": "/model/local-models"
}
```

---

## 💻 ModelLoaderNode.json 파라미터

### 1️⃣ Source (소스 선택)

```json
{
  "id": "source",
  "type": "select",
  "value": "huggingface",
  "options": [
    { "label": "HuggingFace Hub", "value": "huggingface" },
    { "label": "Local Storage", "value": "local" }
  ]
}
```

### 2️⃣ Model ID (HuggingFace)

```json
{
  "id": "modelId",
  "type": "text",
  "value": "gpt2",
  "condition": { "source equals huggingface" }
}
```

### 3️⃣ Local Model (로컬 모델 목록)

```json
{
  "id": "localModel",
  "type": "select",
  "dynamicOptions": true,
  "apiEndpoint": "/model/local-models",
  "condition": { "source equals local" }
}
```

### 4️⃣ Access Token

```json
{
  "id": "accessToken",
  "type": "password",
  "required": false,
  "condition": { "source equals huggingface" }
}
```

### 5️⃣ 기타 파라미터

- `torchDtype`: Float32, Float16, BFloat16 선택
- `deviceMap`: Auto, CPU, GPU 선택
- `trustRemoteCode`: 원격 코드 실행 허용 여부
- `cacheDir`: 캐시 디렉토리 경로

---

## 📊 DatasetLoaderNode.json 파라미터

### 1️⃣ Source (데이터 소스)

```json
{
  "options": [
    { "value": "upload", "label": "Upload File" },
    { "value": "huggingface", "label": "HuggingFace Dataset" },
    { "value": "local", "label": "Local File" }
  ]
}
```

### 2️⃣ Data Format

- CSV ✅
- JSON ✅
- Parquet ✅
- JSONL ✅

### 3️⃣ 조건부 파라미터

- **파일 업로드**: `source === "upload"`
- **HuggingFace 데이터셋**: `source === "huggingface"`
- **로컬 파일**: `source === "local"`

### 4️⃣ 텍스트 처리

- `textColumn`: 텍스트 열 이름
- `labelColumn`: 레이블 열 이름 (선택)
- `maxSamples`: 최대 샘플 수

### 5️⃣ 데이터 분할

- `testSize`: 테스트 분할 비율 (0.0-1.0)
- `randomSeed`: 재현성을 위한 시드
- `skipRows`: 건너뛸 행 수

---

## 🔧 nodeLoader.ts 유틸리티

### 주요 함수

```typescript
// 모든 노드 정의 로드
loadAllNodeDefinitions(): Promise<Map<string, NodeDefinition>>

// 특정 노드 로드
loadNodeDefinition(nodeType: string): Promise<NodeDefinition | null>

// 노드 정의 가져오기
getNodeDefinition(nodeType: string): NodeDefinition | null

// 모든 노드 정의
getAllNodeDefinitions(): NodeDefinition[]

// 조건 확인
checkParameterCondition(
  condition: ParameterCondition,
  parameterValues: Record<string, any>
): boolean

// 표시할 파라미터 필터링
getVisibleParameters(
  parameters: NodeParameter[],
  parameterValues: Record<string, any>
): NodeParameter[]

// 기본값 생성
createNodeDefaultValues(definition: NodeDefinition): Record<string, any>

// 유효성 검사
validateNodeParameters(
  parameters: NodeParameter[],
  values: Record<string, any>
): { valid: boolean; errors: Record<string, string> }

// API 요청 형식 변환
parametersToApiRequest(
  definition: NodeDefinition,
  values: Record<string, any>
): Record<string, any>

// 동적 옵션 가져오기
fetchDynamicOptions(apiEndpoint: string): Promise<ParameterOption[]>
```

---

## 🎨 NodeForm.tsx 컴포넌트

### 특징

- ✅ 모든 파라미터 타입 지원
- ✅ 조건부 파라미터 표시/숨김
- ✅ 동적 옵션 로드 (API)
- ✅ 실시간 유효성 검사
- ✅ 파일 업로드 지원
- ✅ 반응형 디자인

### 사용법

```tsx
<NodeForm parameters={definition.parameters} values={node.data} onChange={(id, value) => updateNode(id, value)} onFileSelect={(id, file) => handleFileUpload(id, file)} />
```

---

## 🎯 ModelLoaderNode.tsx 개선 사항

### ✨ 새로운 기능

1. **JSON 정의 기반**

   - 파라미터를 JSON에서 로드
   - 동적 UI 생성

2. **로컬 모델 지원**

   - HuggingFace Hub 모델
   - 로컬 저장소 모델
   - 자동 감지 및 목록

3. **조건부 파라미터**

   - Source에 따라 다른 파라미터 표시
   - 불필요한 필드 숨김

4. **실시간 진행률**
   - 모델 로드 진행 상황 표시
   - 진행 상황 %로 표시

---

## 📝 DatasetLoaderNode.tsx 개선 사항

### ✨ 새로운 기능

1. **JSON 정의 기반**

   - 모든 파라미터 JSON에서 정의
   - 동적 UI 생성

2. **다중 소스 지원**

   - 파일 업로드 (로컬)
   - HuggingFace Dataset
   - 로컬 파일

3. **파일 포맷 지원**

   - CSV ✅
   - JSON ✅
   - Parquet ✅
   - JSONL ✅

4. **데이터 정보**
   - 행/열 수 표시
   - 파일 크기
   - 데이터 미리보기

---

## 🔗 백엔드 API 지원

### 기존 API (이미 구현됨)

```
GET  /model/local-models          # 로컬 모델 목록
POST /model/download              # 모델 다운로드
GET  /dataset/info                # 데이터셋 정보
POST /dataset/upload              # 파일 업로드
```

### 응답 형식

**로컬 모델**:

```json
{
  "models": [
    {
      "id": "gpt2",
      "name": "gpt2",
      "size": "500MB"
    }
  ],
  "count": 1
}
```

**데이터셋 정보**:

```json
{
  "rows": 10000,
  "columns": 5,
  "size": "100MB"
}
```

---

## 🚀 사용 방법

### 1️⃣ 모델 로더 사용

```
1. 노드 팔레트에서 "🤖 모델 로더" 클릭
2. 노드 추가됨
3. "Source" 선택: HuggingFace 또는 Local
4. 모델 ID 또는 로컬 모델 선택
5. 필요한 파라미터 설정
6. "📥 모델 로드" 클릭
```

### 2️⃣ 데이터셋 로더 사용

```
1. 노드 팔레트에서 "📊 데이터셋" 클릭
2. 노드 추가됨
3. "Source" 선택: Upload, HuggingFace, Local
4. 포맷 선택: CSV, JSON, Parquet, JSONL
5. 파일 선택 또는 경로 입력
6. 텍스트/레이블 열 이름 입력
7. 선택사항: 테스트 분할, 최대 샘플 등
```

---

## 📊 파라미터 흐름

```
JSON 파일 (정의)
    ↓
nodeLoader.ts (로드)
    ↓
NodeForm.tsx (UI 생성)
    ↓
ModelLoaderNode.tsx (렌더링)
    ↓
Backend API (실행)
```

---

## 🔄 다음 단계

### Phase 1: 나머지 노드 JSON 정의 (1시간)

```
TrainingNode.json
- epochs, batch_size, learning_rate
- use_lora, lora_rank, lora_alpha
- gradient_checkpointing
```

### Phase 2: 노드 컴포넌트 구현 (2시간)

```
TrainingNode.tsx
ChatNode.tsx
RagNode.tsx
GgufExportNode.tsx
```

### Phase 3: 노드 연결 (2시간)

```
포트 드래그 UI
데이터 흐름 시각화
유효성 검사
```

---

## 💡 설계 철학

### 1️⃣ 선언적 (Declarative)

```json
// 코드가 아닌 데이터로 UI 정의
{
  "parameters": [
    {
      "id": "source",
      "type": "select",
      "options": [...]
    }
  ]
}
```

### 2️⃣ 조건부 (Conditional)

```json
// 조건에 따라 파라미터 표시/숨김
{
  "condition": {
    "parameter": "source",
    "operator": "equals",
    "value": "huggingface"
  }
}
```

### 3️⃣ 동적 (Dynamic)

```json
// API에서 옵션 로드
{
  "dynamicOptions": true,
  "apiEndpoint": "/model/local-models"
}
```

---

## ✅ 구현 완료 항목

- ✅ ModelLoaderNode.json 정의
- ✅ DatasetLoaderNode.json 정의
- ✅ nodeLoader.ts 유틸리티
- ✅ NodeForm.tsx 컴포넌트
- ✅ ModelLoaderNode.tsx 재구현
- ✅ DatasetLoaderNode.tsx 재구현
- ✅ 조건부 파라미터 처리
- ✅ 동적 옵션 로드
- ✅ 파일 업로드 처리
- ✅ 유효성 검사

---

## 🐛 주의사항

### 파일 경로

JSON 파일 위치: `frontend/src/@nodes/`

- `@` 접두사는 절대 경로 임포트를 위함
- `vite.config.ts`에 설정 필요:

```typescript
resolve: {
  alias: {
    '@nodes': resolve(__dirname, './src/@nodes')
  }
}
```

### 동적 import

JSON 파일 로드는 ES 모듈 임포트 사용:

```typescript
const module = await import(`../@nodes/${fileName}.json`);
```

---

## 🎓 배운 개념

1. **선언적 UI**: 데이터로 UI 정의
2. **조건부 렌더링**: 파라미터 조건 처리
3. **동적 옵션**: API에서 데이터 로드
4. **유효성 검사**: 클라이언트 측 검증
5. **상태 관리**: Zustand와의 연동

---

## 📚 다음 읽을 것

- `EDITOR_IMPLEMENTATION_GUIDE.md` - 에디터 상세 가이드
- `EDITOR_QUICK_START.md` - 에디터 빠른 시작
- `EDITOR_IMPLEMENTATION_SUMMARY.md` - 전체 요약

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 모델 로더 & 데이터셋 완성, 다른 노드 준비 중
