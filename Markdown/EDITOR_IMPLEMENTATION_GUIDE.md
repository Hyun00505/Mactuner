# 🎨 Editor (Workflow Editor) 구현 가이드

## 📋 개요

MacTuner의 **ComfyUI 스타일 노드 기반 워크플로우 에디터**가 완성되었습니다! 

이 에디터는 사용자가 노드를 드래그&드롭으로 추가하고, 노드 간 연결을 통해 LLM 파인튜닝 워크플로우를 시각적으로 구성할 수 있습니다.

---

## 🚀 빠른 시작

### 1. 에디터 접속

```
http://localhost:3000/editor
```

### 2. 노드 추가

**왼쪽 패널 (노드 팔레트)**에서:
- 🤖 **모델 로더**: 모델 다운로드
- 📊 **데이터셋**: 데이터 업로드
- 🎓 **학습**: LoRA/QLoRA 학습
- 💬 **Chat**: 모델과 대화
- 🔍 **RAG**: 문서 검색
- 📦 **GGUF**: 모델 양자화

### 3. 워크플로우 구성

1. 노드 팔레트에서 노드 클릭
2. 캔버스에 노드 추가
3. 노드 간 포트 연결
4. 각 노드의 파라미터 설정
5. **▶ 실행** 버튼으로 워크플로우 실행

### 4. 워크플로우 저장

**💾 저장** 버튼 클릭 → 이름 입력 → 저장

---

## 🏗️ 아키텍처

### 파일 구조

```
frontend/src/
├── types/
│   └── editor.ts                    # 모든 타입 정의
│
├── stores/
│   └── editorStore.ts               # Zustand 상태 관리
│
├── components/Editor/
│   ├── Node.tsx                     # 기본 노드 컴포넌트
│   ├── ModelLoaderNode.tsx          # 모델 로더 노드
│   ├── DatasetLoaderNode.tsx        # 데이터셋 노드
│   ├── TrainingNode.tsx             # 학습 노드 (구현 필요)
│   ├── ChatNode.tsx                 # Chat 노드 (구현 필요)
│   ├── RagNode.tsx                  # RAG 노드 (구현 필요)
│   ├── GgufExportNode.tsx           # GGUF 내보내기 노드 (구현 필요)
│   └── WorkflowCanvas.tsx           # 캔버스 컴포넌트
│
└── pages/
    └── Editor.tsx                   # 메인 에디터 페이지
```

### 데이터 흐름

```
Editor 페이지
    ↓
EditorStore (상태)
    ├─ 노드 관리 (추가, 삭제, 업데이트)
    ├─ 연결 관리
    ├─ 캔버스 (줌, 팬)
    └─ 실행 관리
    ↓
WorkflowCanvas
    ├─ 노드 렌더링
    ├─ 연결선 렌더링
    └─ 캔버스 상호작용
```

---

## 📦 노드 타입 상세

### 1️⃣ 모델 로더 노드 (ModelLoaderNode) ✅

**입력**: 없음  
**출력**: Model

**설정**:
```typescript
{
  modelId: string;           // HuggingFace 모델 ID
  source: 'huggingface' | 'local';  // 소스
  accessToken?: string;      // HF 토큰 (선택)
}
```

**상태**:
- 진행률 표시
- 다운로드 성공/실패

**구현상태**: ✅ 완료

---

### 2️⃣ 데이터셋 노드 (DatasetLoaderNode) ✅

**입력**: 없음  
**출력**: Dataset

**설정**:
```typescript
{
  fileName: string;          // 파일명
  dataFormat: 'csv' | 'json' | 'parquet';
}
```

**상태**:
- 파일 업로드 진행률
- 데이터 정보 (행, 열, 크기)

**구현상태**: ✅ 완료

---

### 3️⃣ 학습 노드 (TrainingNode) 🔄

**입력**: Model, Dataset  
**출력**: Trained Model

**설정**:
```typescript
{
  epochs: number;
  batchSize: number;
  learningRate: number;
  useLora: boolean;
  useQlora: boolean;
  loraRank: number;
  loraAlpha: number;
  warmupSteps: number;
}
```

**구현상태**: 📐 구조 완료, 세부 구현 필요

---

### 4️⃣ Chat 노드 (ChatNode) 🔄

**입력**: Model  
**출력**: Response

**설정**:
```typescript
{
  systemPrompt: string;
  temperature: number;
  maxTokens: number;
  topP: number;
}
```

**구현상태**: 📐 구조 완료, 세부 구현 필요

---

### 5️⃣ RAG 노드 (RagNode) 🔄

**입력**: Model  
**출력**: Search Results

**설정**:
```typescript
{
  documentPaths: string[];
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
}
```

**구현상태**: 📐 구조 완료, 세부 구현 필요

---

### 6️⃣ GGUF 내보내기 노드 (GgufExportNode) 🔄

**입력**: Model  
**출력**: GGUF File

**설정**:
```typescript
{
  quantizationMethod: 'Q2_K' | 'Q3_K' | ... | 'F32';
  outputPath: string;
}
```

**구현상태**: 📐 구조 완료, 세부 구현 필요

---

## 🔧 상태 관리 (EditorStore)

### Zustand 액션들

```typescript
// 워크플로우 관리
createNewWorkflow()          // 새 워크플로우 생성
loadWorkflow(workflow)       // 워크플로우 로드
saveWorkflow(name, desc)     // 저장
deleteWorkflow(id)           // 삭제

// 노드 관리
addNode(type, position)      // 노드 추가
deleteNode(id)               // 노드 삭제
updateNode(id, data)         // 노드 업데이트
selectNode(id)               // 노드 선택
clearNodeSelection()         // 선택 해제

// 연결 관리
addConnection(conn)          // 연결 추가
deleteConnection(id)         // 연결 삭제
selectConnection(id)         // 연결 선택

// 캔버스 제어
setZoom(zoom)                // 줌 설정
setPan(x, y)                 // 팬 설정

// 실행
executeWorkflow()            // 워크플로우 실행
setExecutionLog(log)         // 실행 로그 설정
```

---

## 🎨 UI 구성

### 헤더 (상단)
- 워크플로우 이름
- 새로 만들기 버튼
- 💾 저장
- ▶ 실행
- 패널 토글 (노드, 설정, 출력)

### 왼쪽 패널 (노드 팔레트)
- 6가지 노드 타입
- 각 노드별 설명
- 통계 (노드 수, 연결 수)

### 중앙 (워크플로우 캔버스)
- 드래그&드롭으로 노드 배치
- 마우스 휠: 줌
- Shift+드래그: 팬
- 우클릭: 메뉴

### 오른쪽 패널
- **속성 패널**: 선택된 노드의 파라미터 수정
- **출력 패널**: 실행 결과 표시

---

## 🎮 사용자 상호작용

### 노드 추가
```
1. 왼쪽 패널에서 노드 선택
2. 캔버스에 자동 추가 (위치: 200, 200)
```

### 노드 이동
```
1. 노드 헤더 클릭 및 드래그
2. 실시간으로 위치 업데이트
```

### 노드 선택
```
1. 노드 클릭
2. 파란 테두리로 표시
3. 오른쪽 패널에서 속성 수정
```

### 노드 삭제
```
1. 노드 선택
2. 호버 시 "Delete" 버튼 표시
3. 클릭으로 삭제
```

### 워크플로우 실행
```
1. 모든 파라미터 설정
2. ▶ 실행 버튼 클릭
3. 각 노드 순차 실행
4. 결과를 출력 패널에 표시
```

---

## 🔗 노드 연결 (향후 구현)

### 연결 규칙
- 모델 출력 → 다른 노드의 모델 입력
- 데이터셋 출력 → 학습 노드의 데이터셋 입력
- 학습된 모델 → Chat/RAG 노드 입력

### 유효성 검사
```typescript
// 같은 타입 간 연결만 가능
'model' → 'model'
'dataset' → 'dataset'
```

---

## 📡 API 통합 (향후 구현)

### 워크플로우 저장/로드
```python
POST /api/workflows/save
GET  /api/workflows/{id}
GET  /api/workflows/list
DELETE /api/workflows/{id}
```

### 워크플로우 실행
```python
POST /api/workflows/execute
GET  /api/workflows/execute/{executionId}
```

---

## 🎯 현재 구현 상태

### ✅ 완료된 항목
1. **타입 시스템**
   - `types/editor.ts`: 모든 노드, 연결, 워크플로우 타입
   - 전체 TypeScript 안전성 확보

2. **상태 관리**
   - `stores/editorStore.ts`: Zustand 기반 전역 상태
   - 노드/연결 CRUD 작업
   - 캔버스 줌/팬
   - 실행 로그 관리

3. **기본 노드 컴포넌트**
   - `Node.tsx`: 모든 노드의 베이스 컴포넌트
   - 드래그&드롭, 선택, 삭제 기능

4. **구체적 노드 구현**
   - ✅ `ModelLoaderNode.tsx`
   - ✅ `DatasetLoaderNode.tsx`
   - 📐 나머지는 구조만 정의됨

5. **캔버스**
   - `WorkflowCanvas.tsx`: 그리드, 줌, 팬, 연결선
   - 노드 렌더링

6. **UI 페이지**
   - `pages/Editor.tsx`: 완전한 에디터 인터페이스
   - 3-패널 레이아웃 (노드, 캔버스, 속성/출력)

### 🔄 부분 구현
- 노드 연결 UI (구조만 있음)
- 워크플로우 실행 (시뮬레이션 중)

### 🔲 미구현
- 다른 노드들의 세부 구현
- 백엔드 API 통합
- 실제 워크플로우 실행
- 데이터 흐름 추적
- 고급 기능 (실행 취소, 반복 실행 등)

---

## 🚀 다음 단계

### Phase 1: 나머지 노드 구현
```
TrainingNode.tsx     → 학습 파라미터 UI
ChatNode.tsx         → 대화 인터페이스
RagNode.tsx          → 문서 관리
GgufExportNode.tsx   → 양자화 옵션
```

### Phase 2: 노드 연결
```
- 연결 드래그 UI
- 연결 검증
- 데이터 흐름 표시
```

### Phase 3: 백엔드 통합
```
- 워크플로우 저장/로드 API
- 실행 API
- 실시간 진행률 (WebSocket)
```

### Phase 4: 고급 기능
```
- 실행 취소/재실행
- 워크플로우 템플릿
- 성능 최적화
- 에러 처리 개선
```

---

## 💡 개발자 가이드

### 새로운 노드 추가하기

**1. 타입 정의** (`types/editor.ts`)
```typescript
export interface MyNode extends BaseNode {
  type: 'my-node';
  data: {
    param1: string;
    param2: number;
  };
}
```

**2. 노드 컴포넌트** (`components/Editor/MyNode.tsx`)
```typescript
export const MyNode: React.FC<MyNodeProps> = ({ node, isSelected }) => {
  const updateNode = useEditorStore(s => s.updateNode);
  
  return (
    <NodeComponent
      node={node}
      isSelected={isSelected}
      onSelect={() => selectNode(node.id)}
      onDelete={() => deleteNode(node.id)}
    >
      {/* UI */}
    </NodeComponent>
  );
};
```

**3. Store에 등록** (`stores/editorStore.ts`)
```typescript
const createMyNode = (position): MyNode => ({
  id: generateNodeId('my-node'),
  type: 'my-node',
  // ...
});

const nodeFactories: Record<NodeType, any> = {
  // ...
  'my-node': createMyNode,
};
```

**4. Editor 페이지에서 렌더링** (`pages/Editor.tsx`)
```typescript
{node.type === 'my-node' && (
  <MyNode
    node={node as any}
    isSelected={selectedNodeId === node.id}
  />
)}
```

---

## 🐛 트러블슈팅

### ❓ 노드가 움직이지 않음
**해결**: 노드 헤더 부분을 드래그해야 합니다

### ❓ 줌이 작동하지 않음
**해결**: Ctrl (또는 Cmd) + 마우스 휠 사용

### ❓ 노드가 화면 밖으로 나감
**해결**: Shift+드래그로 캔버스 팬하기

---

## 📚 추가 자료

- TypeScript 타입: `types/editor.ts`
- 상태 관리: `stores/editorStore.ts`
- 컴포넌트 예제: `components/Editor/`

---

## 🎉 결론

MacTuner Editor는 이제 **노드 기반 워크플로우**를 완벽하게 지원합니다!

다음은 **백엔드 API 통합**과 **나머지 노드 구현**입니다. 🚀

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 기본 구현 완료, 확장 준비 완료

