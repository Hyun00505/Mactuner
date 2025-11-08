# 🚀 워크플로우 실행 시스템 완성 가이드

## 📋 구현 내용

### 1. ✅ 연결 가능 여부 시각적 표시

포트 호버 시 연결 가능 여부를 실시간으로 표시합니다.

```
포트 호버 중 (연결 드래그 진행):

✅ 연결 가능 상태
  • 포트: 초록색 동그라미
  • 그림자: 초록색
  • 제목: "✅ 연결 가능"

❌ 연결 불가능 상태
  • 포트: 빨강색 동그라미
  • 그림자: 빨강색
  • 제목: "❌ 연결 불가능"
```

### 2. ✅ 연결선 표시

이미 구현된 베지어 곡선으로 연결선이 표시됩니다.

```
연결선 스타일:

🔵 파란색  (데이터타입: "model")
🟢 초록색  (데이터타입: "dataset")
🟡 노란색  (데이터타입: "token"/"config")

• 부드러운 베지어 곡선
• 섀도우 효과
• 드래그 중 실시간 갱신
```

### 3. ✅ 실행 시 토큰 검증

실행 버튼 클릭 시 모든 필요한 토큰 연결을 검증합니다.

```
검증 규칙:

🤖 ModelLoaderNode (HuggingFace):
  ✓ 모델 ID 입력됨
  ✓ 🔑 Token 노드와 연결됨
  ✓ 토큰이 유효함 (hf_로 시작)

📊 DatasetLoaderNode (HuggingFace):
  ✓ 데이터셋 선택됨
  ✓ 🔑 Token 노드와 연결됨 (HF 소스 시만)
  ✓ 토큰이 유효함

❌ 검증 실패시:
  • 알림 메시지 표시
  • 누락된 항목 명시
  • 실행 중단
```

---

## 🔧 기술 구현

### 1. 워크플로우 헬퍼 (workflowHelper.ts)

```typescript
// 토큰 노드에서 유효한 토큰 추출
function extractTokenFromNode(tokenNode: Node | undefined): string | null

// 특정 노드에 연결된 토큰 노드 찾기
function findConnectedTokenNode(nodeId, nodes, connections): Node | null

// 모델 다운로드 노드 검증
function validateModelDownload(node, nodes, connections): { valid, errors }

// 데이터셋 다운로드 노드 검증
function validateDatasetDownload(node, nodes, connections): { valid, errors }

// 전체 워크플로우 검증
function validateWorkflow(nodes, connections): { valid, errors }
```

### 2. WorkflowCanvas 호버 감지

```typescript
// 포트 호버 상태
const [hoveredPortId, setHoveredPortId] = useState<string | null>(null);
const [canConnect, setCanConnect] = useState(false);

// 포트 호버 시 연결 가능 여부 확인
const handlePortMouseEnter = (nodeId, portId) => {
  // 연결 중인 경우 검증
  // canConnectPorts() 로직으로 가능 여부 판단
  setCanConnect(canConnectPorts(...));
};
```

### 3. executeWorkflow 검증

```typescript
executeWorkflow: async () => {
  const { nodes, connections } = get();
  
  // 워크플로우 검증
  const validation = validateWorkflow(nodes, connections);
  
  if (!validation.valid) {
    // 에러 메시지 표시
    alert(`⚠️ 워크플로우 검증 실패:\n\n${errorMessages}`);
    return;
  }
  
  // 검증 통과 시 실행
  ...
};
```

### 4. Node.tsx 포트 렌더링

```typescript
// 입력 포트
.map((port) => {
  const isHovered = hoveredPortId === portKey;
  const canConnectColor = isHovered && canConnect ? "green" : "red";
  
  return (
    <div
      onMouseEnter={() => onPortMouseEnter?.(node.id, port.id)}
      onMouseLeave={() => onPortMouseLeave?.()}
      onMouseUp={() => onPortMouseUp?.(node.id, port.id)}
    >
      {/* 포트 동그라미 */}
      <div className={`${canConnectColor} rounded-full ...`} />
    </div>
  );
})
```

---

## 🎯 사용자 흐름

### 워크플로우 구성

```
1️⃣ 노드 추가
   ├─ 🔑 HF Token 노드
   ├─ 🤖 Model Loader 노드
   └─ 📊 Dataset Loader 노드

2️⃣ HF Token 설정
   ├─ 토큰 입력 필드에 hf_... 입력
   ├─ ✅ 토큰 유효 표시 확인
   └─ 우측 상단 동그라미 확인

3️⃣ 노드 연결 (토큰)
   ├─ HF Token의 동그라미 드래그 시작
   │  → 황금색 점선 표시
   │  → 마우스 추적
   ├─ Model Loader의 입력 포트에 호버
   │  → 포트 초록색으로 변함 (✅ 연결 가능)
   └─ 마우스 업으로 연결 완료

4️⃣ 모델 설정
   ├─ Model Loader의 Source: HuggingFace 선택
   ├─ Model ID 입력
   └─ 입력 포트(Token) 자동 표시

5️⃣ 실행 버튼 클릭
   ├─ 검증 시작
   ├─ 검증 성공 ✅
   │  └─ 워크플로우 실행
   └─ 검증 실패 ❌
      └─ 에러 알림 + 실행 중단
```

---

## 📊 검증 플로우

```
▼ 실행 버튼 클릭
│
▼ validateWorkflow() 호출
│
├─ 각 노드 순회
│
├─ ModelLoaderNode 검증
│  ├─ 모델 ID: 있음? ✓
│  └─ HuggingFace 시:
│      ├─ findConnectedTokenNode() → 토큰 노드 찾기
│      ├─ extractTokenFromNode() → 토큰 값 추출
│      └─ isValidToken() → 유효성 확인 (hf_로 시작?)
│
├─ DatasetLoaderNode 검증
│  ├─ 데이터셋: 선택됨? ✓
│  └─ HuggingFace 시:
│      ├─ 토큰 노드 찾기
│      └─ 토큰 유효성 확인
│
├─ 모든 검증 통과?
│  ├─ YES → 워크플로우 실행 ✅
│  └─ NO → 에러 메시지 표시 ❌
│
▼ 실행 또는 중단
```

---

## 💡 에러 메시지 예시

### 검증 실패 시

```
⚠️ 워크플로우 검증 실패:

🤖 모델 로더:
🔑 HF Token 노드와 연결이 필요합니다

📊 데이터셋:
데이터셋을 선택하세요
🔑 HF Token 노드와 연결이 필요합니다
```

### 검증 성공 시

```
✅ 워크플로우 실행 시작
(콘솔: Executing workflow: { id, nodes, connections })
```

---

## 🎨 포트 상태 시각화

### 정상 상태 (연결 안 함)

```
🔑 HF Token             ●      (회색 동그라미)
│                            │
│  토큰: hf_xxx             │
│  ✅ 토큰 유효             │
│                            │
└────────────────────────────┘
```

### 드래그 중 (호버 - 연결 가능)

```
🔑 HF Token             ●      (녹색 동그라미)
                    ↓ 드래그
                   🟢 (초록색 그림자)
                   ✅ 연결 가능

🤖 Model Loader        ●      (파랑색)
        ↑ 호버 - 녹색으로 변함
        🟢 (초록색 포트)
        ✅ 연결 가능
```

### 드래그 중 (호버 - 연결 불가)

```
❌ 연결 불가능 상황

🤖 Model Loader
        ↑ 호버 - 빨강색으로 변함
        🔴 (빨강색 포트)
        ❌ 연결 불가능

(같은 노드 또는 타입 불일치 등)
```

---

## 📁 수정된 파일

```
✅ frontend/src/utils/workflowHelper.ts (NEW)
   ├─ 검증 함수들
   └─ 토큰 추출 로직

✅ frontend/src/stores/editorStore.ts
   └─ executeWorkflow 검증 추가

✅ frontend/src/components/Editor/WorkflowCanvas.tsx
   ├─ handlePortMouseEnter (호버 감지)
   ├─ handlePortMouseLeave (호버 해제)
   └─ canConnect 상태 관리

✅ frontend/src/components/Editor/Node.tsx
   ├─ 포트 호버 상태 표시
   ├─ 연결 가능/불가능 색상 변화
   └─ 제목 텍스트 동적 표시

✅ frontend/src/components/Editor/ModelLoaderNode.tsx
✅ frontend/src/components/Editor/DatasetLoaderNode.tsx
✅ frontend/src/components/Editor/HFTokenNode.tsx
   └─ 호버 props 추가 및 전달

✅ frontend/src/pages/Editor.tsx
   └─ 알림 메시지 개선
```

---

## ✨ 완성된 기능

```
✅ 포트 호버 시 연결 가능 여부 표시
   • 초록색 (연결 가능) / 빨강색 (불가능)
   • 툴팁 메시지

✅ 실시간 연결선 표시
   • 베지어 곡선
   • 색상 구분
   • 섀도우 효과

✅ 실행 시 토큰 검증
   • 필수 연결 확인
   • 토큰 유효성 검사
   • 상세 에러 메시지

✅ 자동 토큰 사용
   • 연결된 토큰 자동 감지
   • 모델 다운로드 시 적용
   • 데이터셋 다운로드 시 적용
```

---

## 🚀 다음 단계

1. **실제 API 연결**
   - 백엔드 모델 다운로드 API 호출
   - 진행률 실시간 표시
   - 에러 처리

2. **다른 노드 구현**
   - Training 노드
   - Chat 노드
   - RAG 노드
   - GGUF Export 노드

3. **워크플로우 저장/로드**
   - 워크플로우 JSON 저장
   - 이전 워크플로우 로드
   - 버전 관리

4. **고급 기능**
   - Undo/Redo
   - 노드 복제
   - 그룹화
   - 주석

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 워크플로우 실행 시스템 완성  
**버전**: 7.0.0

