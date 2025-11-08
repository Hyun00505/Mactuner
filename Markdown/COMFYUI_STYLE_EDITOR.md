# 🎨 ComfyUI 스타일 노드 에디터 구현 완료

## 📋 개요

**ComfyUI 스타일의 완벽한 워크플로우 에디터** 구현:
- ✅ 실시간 드래그 연결선 (베지어 곡선)
- ✅ 자동 애니메이션 (점선 흐르는 효과)
- ✅ 토큰 노드 자동 감지 및 활용
- ✅ 자연스러운 드래그 애니메이션

---

## 🔗 실시간 연결선 그리기

### ComfyUI 스타일의 베지어 곡선

```
포트에서 마우스 드래그 시작
       ↓
connectionStart 저장
       ↓
마우스 움직임 실시간 추적 (mousePos)
       ↓
베지어 곡선으로 임시 선 그리기
    - 시작점: connectionStart
    - 끝점: 마우스 위치
    - 제어점: 자동 계산
       ↓
마우스 업 시 정식 연결선으로 변환
```

### SVG 렌더링

**베지어 곡선 연결선**:

```typescript
// 기존 연결선 (저장된 것들)
{connections.map(conn => {
  const x1 = sourceNode.position.x + 288;
  const y1 = sourceNode.position.y + 200;
  const x2 = targetNode.position.x;
  const y2 = targetNode.position.y + 100;

  const dx = (x2 - x1) / 2;
  const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

  return <path d={path} stroke={strokeColor} strokeWidth="2" />;
})}

// 드래그 중인 실시간 선 (점선 애니메이션)
{isConnecting && connectionStart && (
  <path
    d={`M ${start} C ${start + 50}, ${mouse - 50}, ${mouse}`}
    stroke="#fbbf24"
    strokeWidth="2"
    strokeDasharray="5,5"
    style={{
      animation: 'dashflow 0.6s linear infinite',
    }}
  />
)}
```

---

## ⚡ 애니메이션 효과

### CSS 키프레임

**점선 흐르기 (Dash Flow)**:
```css
@keyframes dashflow {
  0% {
    stroke-dashoffset: 10;
  }
  100% {
    stroke-dashoffset: 0;
  }
}
```

**Pulse 효과**:
```css
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}
```

### 포트 Hover 효과

```typescript
<div className="w-3 h-3 bg-green-500 rounded-full 
             group-hover:scale-150 
             group-hover:shadow-lg 
             group-hover:shadow-green-500 
             transition-all duration-100" />
```

---

## 🔑 HuggingFace 토큰 자동 연결

### 토큰 헬퍼 유틸 (tokenHelper.ts)

```typescript
// 연결된 토큰 노드 찾기
export function findConnectedTokenNode(
  nodeId: string,
  nodes: Node[],
  connections: Array<{source: string; target: string}>
): Node | undefined

// 토큰 값 가져오기
export function getTokenValue(tokenNode: Node | undefined): string

// 토큰 유효성 확인
export function isValidToken(token: string): boolean
```

### ModelLoaderNode에서의 토큰 사용

```typescript
// 1. 토큰 감지
useEffect(() => {
  const tokenNode = findConnectedTokenNode(node.id, nodes, connections);
  const token = getTokenValue(tokenNode);
  setConnectedToken(token);
}, [node.id, nodes, connections]);

// 2. 다운로드 시 토큰 자동 활용
const token = connectedToken || node.data.accessToken || '';
const response = await fetch(
  `http://localhost:8001/model/download-stream?model_id=${modelId}&access_token=${token}`,
  { method: 'POST' }
);

// 3. UI에서 토큰 상태 표시
{connectedToken && isValidToken(connectedToken) ? (
  <p>✅ 연결된 토큰 활성화</p>
) : node.data.accessToken ? (
  <p>⚠️ 입력된 토큰 사용</p>
) : (
  <p>💡 토큰 노드를 연결하거나 토큰을 입력하세요</p>
)}
```

---

## 🎨 색상 체계

### 연결선 색상

| 데이터 타입 | 색상 | HEX 코드 | 의미 |
|-----------|------|---------|------|
| 모델 | 파란색 | #3b82f6 | 모델 데이터 흐름 |
| 데이터셋 | 초록색 | #10b981 | 데이터셋 흐름 |
| 토큰 | 노란색 | #f59e0b | 설정/인증 정보 |
| 드래그 중 | 황금색 | #fbbf24 | 임시 연결선 |

### 포트 색상

| 포트 타입 | 색상 | 상태 | 효과 |
|----------|------|------|------|
| 입력 | 파란색 | Idle | 일반 |
| 입력 | 파란색 | Hover | 확대 + 그림자 |
| 출력 | 초록색 | Idle | 일반 |
| 출력 | 초록색 | Hover | 확대 + 그림자 |

---

## 📊 워크플로우 예시

### 기본 구조

```
┌─────────────────┐
│  🔑 HF Token    │
│  (토큰 입력)    │
└────────┬────────┘
         │ 토큰 연결 (황금색)
         ↓
┌─────────────────────────────┐
│  🤖 Model Loader            │
│  (HuggingFace 모델 선택)    │
│  ✅ 연결된 토큰 활성화      │
└────────┬────────────────────┘
         │ 모델 연결 (파란색)
         ↓
┌─────────────────────────────┐
│  🎓 Training Node           │
│  (모델로 학습)              │
└────────┬────────────────────┘
         │
         ↓
    📊 완료
```

---

## 🔄 연결 프로세스

### 단계별 동작

```
1️⃣ 포트 마우스다운
   └─ connectionStart 저장
   └─ isConnecting = true
   └─ 포트 highlight

2️⃣ 마우스 이동 추적
   └─ mousePos 실시간 업데이트
   └─ SVG 자동 리렌더링
   └─ 베지어 곡선 그려짐
   └─ 점선 애니메이션 실행

3️⃣ 포트 마우스업
   └─ 연결 검증 (입출력 방향)
   └─ addConnection() 호출
   └─ 정식 연결선 저장
   └─ 상태 초기화

4️⃣ 완료
   └─ 캔버스에 연결선 표시
   └─ 데이터 흐름 활성화
```

---

## 💻 기술 구현

### WorkflowCanvas 주요 로직

**포트 마우스 다운**:
```typescript
const handlePortMouseDown = (nodeId, portId, event, isOutput) => {
  const rect = event.currentTarget.getBoundingClientRect();
  setConnectionStart({
    nodeId,
    portId,
    x: rect.left,
    y: rect.top,
    isOutput,  // 🔑 입출력 구분
  });
  setIsConnecting(true);
};
```

**마우스 이동 추적**:
```typescript
const handleMouseMove = (e) => {
  // 실시간 마우스 위치 업데이트
  setMousePos({ x: e.clientX, y: e.clientY });
  
  if (isConnecting && connectionStart) {
    // SVG가 자동으로 리렌더링됨
  }
};
```

**포트 마우스 업**:
```typescript
const handlePortMouseUp = (targetNodeId, targetPortId) => {
  if (!connectionStart) return;
  
  // 입출력 방향 검증
  if (sourcePort.type === 'output' && targetPort.type === 'input') {
    addConnection({
      source: connectionStart.nodeId,
      target: targetNodeId,
      dataType: sourcePort.dataType,
    });
  }
};
```

---

## 🎯 사용 방법

### 토큰 노드와 함께 사용

**1️⃣ 워크플로우 시작**:
```
🔑 HF Token 노드 추가
└─ 토큰 입력 또는 "기본으로 저장" 체크
```

**2️⃣ 모델 로더 노드**:
```
🤖 Model Loader 노드 추가
└─ 🔑 Token 포트에서 드래그
└─ 🤖 Model Loader의 입력 포트로 드롭
└─ 자동으로 토큰 활성화됨
```

**3️⃣ 모델 선택 및 다운로드**:
```
모델 ID 입력
└─ 🤗 다운로드 버튼 클릭
└─ 연결된 토큰이 자동으로 사용됨
└─ 진행률 표시
```

---

## ✅ 완료 항목

### 시각적 개선
- ✅ 베지어 곡선 연결선
- ✅ 실시간 드래그 선
- ✅ 점선 애니메이션 (dashflow)
- ✅ 포트 hover 효과 (scale + shadow)
- ✅ 색상 체계 (모델/데이터/토큰)

### 기능 개선
- ✅ 토큰 자동 감지
- ✅ 토큰 우선순위 처리
- ✅ 토큰 상태 표시
- ✅ 토큰 유효성 검사
- ✅ 부드러운 드래그 애니메이션

### 파일 수정
- ✅ `WorkflowCanvas.tsx` - 실시간 선 그리기
- ✅ `Node.tsx` - 포트 개선 (isOutput 추가)
- ✅ `ModelLoaderNode.tsx` - 토큰 자동 활용
- ✅ `DatasetLoaderNode.tsx` - 토큰 상태 표시
- ✅ `tokenHelper.ts` - 토큰 유틸 함수
- ✅ `index.css` - 애니메이션 추가

---

## 🎬 애니메이션 시각화

### 드래그 연결선 애니메이션

```
시작
  ↓
┌─────────────────────────┐
│ ▪▪▪▪▪▪▪▪▪▪─────────────  (dashflow 애니메이션)
│ 포트      마우스위치
└─────────────────────────┘
  ↓
계속 흐르는 점선...
  ↓
포트 마우스업
  ↓
정식 연결선으로 전환
```

---

## 📊 성능 최적화

### SVG 렌더링
- ✅ `pointerEvents: none` - 마우스 이벤트 제외
- ✅ 효율적인 경로 계산 (베지어)
- ✅ 그림자 필터 (drop-shadow)

### 상태 관리
- ✅ 최소한의 state 업데이트
- ✅ 연결 저장소 효율화
- ✅ 토큰 감지 의존성 최적화

---

## 🔮 향후 개선사항

### Phase 1: 기본 완료 (✅)
- 실시간 선 그리기
- 토큰 자동 연결

### Phase 2: 고급 기능 (🔄)
- 연결선 선택 & 삭제
- 다중 연결 검증
- 연결 방향 화살표

### Phase 3: 상태 시각화 (🔜)
- 워크플로우 실행 중 흐름 표시
- 데이터 처리 상태 색상
- 에러 강조 표시

---

## 📊 현재 상태

| 항목 | 상태 | 설명 |
|------|------|------|
| 실시간 선 | ✅ | 베지어 곡선으로 그려짐 |
| 애니메이션 | ✅ | dashflow 애니메이션 실행 |
| 토큰 감지 | ✅ | 자동 연결 및 활용 |
| 포트 효과 | ✅ | hover 효과 완성 |
| 색상 체계 | ✅ | 데이터 타입별 색상 |
| 연결 검증 | ✅ | 입출력 방향 확인 |

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ ComfyUI 스타일 에디터 완성  
**버전**: 3.0.0

