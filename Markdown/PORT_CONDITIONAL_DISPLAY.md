# 🔌 조건부 포트 표시 시스템

## 📋 개요

포트(동그라미)를 조건에 따라 동적으로 표시하거나 숨기는 시스템을 구현했습니다.

```
사용자 입력 → 조건 확인 → 포트 표시/숨김
```

---

## 🎯 구현 원리

### 1. JSON 정의에 `visible` 속성 추가

```json
{
  "inputs": [
    {
      "id": "in-token",
      "name": "Token",
      "type": "token",
      "visible": "download"  // ← 조건 지정
    }
  ],
  "outputs": [
    {
      "id": "out-token",
      "name": "Token",
      "type": "token",
      "visible": "has-token"  // ← 조건 지정
    }
  ]
}
```

### 2. Node.tsx에서 조건 필터링

```typescript
// 입력 포트 (상단)
.filter(p => {
  if ((p as any).visible === "download") {
    const data = node.data as any;
    return data?.source === "huggingface" || data?.source === "upload";
  }
  return true;
})

// 출력 포트 (헤더)
.filter(p => {
  if ((p as any).visible === "has-token") {
    const data = node.data as any;
    return data?.tokenInput && data.tokenInput.startsWith("hf_");
  }
  return true;
})
```

---

## 📊 각 노드의 포트 표시 조건

### 🔑 HFTokenNode

```
┌──────────────────────────────┐
│ 🔑 HF Token    Token: ●      │  ← 우측 상단 동그라미
├──────────────────────────────┤
│                              │
│ 🔑 토큰 입력                  │
│ [입력 필드]                   │
│                              │
│ ✅ 토큰 유효                 │
│ hf_xxx...xxx                 │
│                              │
│ ☑ ⭐ 기본 토큰으로 저장      │
│                              │
│ (사용처 설명)                │
└──────────────────────────────┘
```

**출력 포트 (`out-token`)**:
- `visible: "has-token"`
- **표시 조건**: `tokenInput`이 있고 `hf_`로 시작
- **위치**: 헤더 우측 상단 (회색 동그라미)
- **표시 안 함**: 토큰을 입력하지 않았을 때

```typescript
return data?.tokenInput && data.tokenInput.startsWith("hf_");
```

---

### 🤖 ModelLoaderNode

```
┌──────────────────────────────┐
│ 🤖 Model Loader             │
├──────────────────────────────┤
│ Token: ●    ← 입력 포트      │  (HuggingFace 선택 시만)
├──────────────────────────────┤
│ Source: [HuggingFace/Local] │
│ Model ID: [텍스트]           │
│ ...                          │
├──────────────────────────────┤
│ Model ●                      │
└──────────────────────────────┘
```

**입력 포트 (`in-token`)**:
- `visible: "download"`
- **표시 조건**: `source === "huggingface"`
- **위치**: 상단 입력 포트
- **표시 안 함**: `source === "local"`일 때

```typescript
if ((p as any).visible === "download") {
  const data = node.data as any;
  return data?.source === "huggingface" || data?.source === "upload";
}
```

**출력 포트 (`out-model`)**:
- 항상 표시 (조건 없음)

---

### 📊 DatasetLoaderNode

```
┌──────────────────────────────┐
│ 📊 Dataset Loader           │
├──────────────────────────────┤
│ Token: ●  ← 입력 포트        │  (HuggingFace 선택 시만)
├──────────────────────────────┤
│ Source: [Local/HF/Upload]   │
│ Dataset: [선택/입력]         │
│ ...                          │
├──────────────────────────────┤
│ Dataset ●                    │
└──────────────────────────────┘
```

**입력 포트 (`in-token`)**:
- `visible: "download"`
- **표시 조건**: `source === "huggingface"` 또는 `source === "upload"`
- **표시 안 함**: `source === "local"`일 때

---

## 🔌 포트 위치

### 우측 상단 (헤더) - 조건부 포트

```
┌─────────────────────────────────┐
│ 🔑 HF Token          [●]        │  ← 회색 동그라미 (2.5x2.5px)
│                                 │
│ (콘텐츠)                       │
│                                 │
└─────────────────────────────────┘
```

**특징**:
- 크기: 2.5x2.5px (작음)
- 색상: 회색 (`bg-gray-400`)
- 호버: 어두워짐 + 확대 (`hover:scale-125 hover:bg-gray-600`)
- 조건부 표시

### 상단 입력 - 입력 포트

```
┌─────────────────────────────────┐
│ 🤖 Model Loader                 │
├─────────────────────────────────┤
│  Token ●                        │  ← 입력 포트 (HuggingFace 시만)
├─────────────────────────────────┤
│ (콘텐츠)                       │
```

**특징**:
- 위치: `pt-3 pb-1`
- 조건부 표시
- 드래그로 연결 가능

### 하단 출력 - 주요 출력 포트

```
│ (콘텐츠)                       │
├─────────────────────────────────┤
│                            Model ●  ← 출력 포트
└─────────────────────────────────┘
```

**특징**:
- 위치: `py-3` 하단
- 대부분 항상 표시
- 크기: 3x3px (큼)
- 색상: 초록색

---

## 💡 사용자 경험

### 시나리오 1: HFTokenNode에 토큰 입력

```
1. 토큰 입력 필드에 "hf_..." 입력
   ↓
2. ✅ 토큰 유효 표시됨
   ↓
3. 우측 상단에 회색 동그라미(●) 나타남
   ↓
4. 동그라미를 드래그해서 ModelLoader와 연결!
```

### 시나리오 2: ModelLoaderNode - HuggingFace 선택

```
1. Source를 "HuggingFace"로 선택
   ↓
2. 상단에 입력 포트(Token) 나타남
   ↓
3. HFTokenNode와 연결 가능
```

### 시나리오 3: ModelLoaderNode - Local 선택

```
1. Source를 "Local"로 선택
   ↓
2. 상단의 입력 포트(Token) 사라짐 (토큰 필요 없음)
   ↓
3. 로컬 모델 자동 로드
```

---

## 🔧 구현 세부사항

### JSON 정의 (HFTokenNode.json)

```json
{
  "outputs": [
    {
      "id": "out-token",
      "name": "Token",
      "type": "token",
      "visible": "has-token"  // ← 조건 지정
    }
  ]
}
```

### React 필터링 (Node.tsx)

```typescript
// 헤더의 출력 포트
{node.ports
  .filter((p) => p.type === "output")
  .filter((p) => {
    if ((p as any).visible === "has-token") {
      const data = node.data as any;
      // 토큰이 유효하면 표시
      return data?.tokenInput && data.tokenInput.startsWith("hf_");
    }
    return true;
  })
  .map((port) => (
    <div
      className="w-2.5 h-2.5 rounded-full bg-gray-400 cursor-crosshair hover:scale-125 hover:bg-gray-600"
      onMouseDown={(e) => {
        e.stopPropagation();
        onPortMouseDown?.(node.id, port.id, e, true);
      }}
    />
  ))}
```

---

## 📋 조건 종류

| Condition | 의미 | 사용처 |
|-----------|------|--------|
| `"download"` | 다운로드 필요 | ModelLoader, DatasetLoader 입력 |
| `"has-token"` | 토큰 입력됨 | HFTokenNode 출력 |
| 없음 | 항상 표시 | 대부분의 출력 포트 |

---

## 🎨 포트 스타일

### 헤더 포트 (조건부)

```typescript
className="w-2.5 h-2.5 rounded-full bg-gray-400 cursor-crosshair hover:scale-125 hover:bg-gray-600 transition-all duration-100"
```

- 작은 동그라미 (2.5x2.5px)
- 회색 배경
- 호버 시 확대 + 색상 변경

### 입력/출력 포트 (기존)

```typescript
// 입력 포트
className="w-3 h-3 bg-blue-500 rounded-full"

// 출력 포트
className="w-3 h-3 bg-green-500 rounded-full"
```

- 일반 크기 (3x3px)
- 색상 구분

---

## ✅ 완성된 기능

```
✅ 조건부 포트 표시
✅ JSON 정의 기반 조건
✅ React 필터링 로직
✅ HFTokenNode 헤더 포트
✅ ModelLoaderNode 조건 포트
✅ DatasetLoaderNode 조건 포트
✅ 직관적인 UI/UX
```

---

## 🚀 사용 방법

### 워크플로우 구성

```
1️⃣ 🔑 HF Token 노드 추가
   ├─ 토큰 입력
   └─ 우측 상단 동그라미 나타남 ✓

2️⃣ 🤖 Model Loader 노드 추가
   ├─ Source: HuggingFace
   └─ 상단 입력 포트 나타남 ✓

3️⃣ 연결!
   ├─ HF Token의 우측 상단 동그라미 드래그
   └─ Model Loader의 상단 입력 포트에 연결

4️⃣ 완료! 🎉
   └─ 모델 다운로드 시 토큰 자동 사용
```

---

## 📊 조건 플로우

```
사용자 입력 (tokenInput 변경)
     ↓
useEffect 감지 (React state 업데이트)
     ↓
Node.tsx 재렌더링
     ↓
filter 로직 실행
     ↓
포트 표시/숨김 결정
     ↓
UI 업데이트 ✓
```

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 조건부 포트 표시 완료  
**버전**: 6.0.0

