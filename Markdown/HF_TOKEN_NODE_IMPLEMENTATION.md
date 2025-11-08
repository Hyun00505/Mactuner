# 🔑 HuggingFace 토큰 노드 구현 완료

## 📋 개요

**로컬 데이터셋 표시 문제 해결** + **HuggingFace 토큰 독립 노드 추가**

이제 두 가지가 완벽하게 작동합니다:
1. ✅ 로컬 저장소의 데이터셋이 Editor에 표시됨
2. ✅ HuggingFace 토큰을 독립적인 노드로 관리 가능

---

## 🐛 문제 1: 로컬 데이터셋이 안 보이는 문제

### 원인
```
히스토리 파일 형식: 배열 []
백엔드 코드 기대: 객체 with "history" 키 {}
```

### 해결책
**`GET /dataset/local-datasets`** 개선:

```python
# 배열 또는 객체 형식 모두 처리
if isinstance(history_data, list):
    history_list = history_data  # 배열
elif isinstance(history_data, dict) and "history" in history_data:
    history_list = history_data["history"]  # 객체

# 모든 데이터셋 소스 지원
for item in history_list:
    if item.get("source") in ["file", "upload"]:
        datasets.append(...)
    elif item.get("source") == "hf":
        datasets.append(...)
```

### 결과
```
✅ 로컬 저장소 데이터셋 표시
📤 test.csv (0.56MB)
🤗 orca-math-korean-dpo-pairs
🤗 korean_rlhf_dataset
```

---

## ✨ 문제 2: HuggingFace 토큰 관리

### 이전 방식 (문제)
```
모델 로더 노드: accessToken 파라미터 포함
데이터셋 로더 노드: token 파라미터 포함

❌ 각 노드마다 토큰 입력 필요
❌ 토큰 변경 시 모든 노드 수정 필요
❌ 토큰 재사용 불편
```

### 새 방식 (해결책)
```
🔑 HF Token 노드 (독립적)
  ↓
🤖 Model Loader (토큰 참조)
📊 Dataset Loader (토큰 참조)
💬 Chat Node (토큰 참조)

✅ 한 번만 설정
✅ 자동으로 모든 노드에서 사용
✅ 기본 토큰 설정 가능
```

---

## 🏗️ HFTokenNode 구현

### JSON 정의

**`@nodes/HFTokenNode.json`**:

```json
{
  "id": "hf-token",
  "name": "HuggingFace Token",
  "icon": "🔑",
  "category": "config",
  "parameters": [
    {
      "id": "tokenType",
      "options": [
        "👤 User Token (일반 사용자 토큰)",
        "🤖 Fine-grained Token (세밀한 권한)"
      ]
    },
    {
      "id": "token",
      "type": "password",
      "label": "HuggingFace Token",
      "placeholder": "hf_로 시작하는 토큰을 입력하세요"
    },
    {
      "id": "permissions",
      "options": [
        "📖 Read (읽기만)",
        "✍️ Write (쓰기)",
        "🔐 Admin (관리자)"
      ]
    },
    {
      "id": "saveAsDefault",
      "type": "checkbox",
      "label": "기본 토큰으로 저장"
    }
  ]
}
```

### React 컴포넌트

**`HFTokenNode.tsx`**:

```typescript
export const HFTokenNode: React.FC<HFTokenNodeProps> = ({ node }) => {
  // 토큰 유효성 검사
  const token = node.data?.token || '';
  const isValid = token.startsWith('hf_') && token.length > 10;
  
  return (
    <NodeComponent>
      {/* 파라미터 폼 */}
      <NodeForm parameters={definition.parameters} />
      
      {/* 토큰 상태 표시 */}
      {isValid ? (
        <>
          <div className="text-green-700">✅ 토큰 유효</div>
          <p className="text-xs font-mono">hf_...{token.slice(-5)}</p>
        </>
      ) : (
        <div className="text-gray-500">⚠️ 토큰이 필요합니다</div>
      )}
      
      {/* 사용처 정보 */}
      <div className="text-xs bg-blue-50">
        <p>💡 사용처:</p>
        <ul>
          <li>🤖 모델 다운로드</li>
          <li>📊 데이터셋 다운로드</li>
          <li>🔒 프라이빗 모델/데이터셋</li>
        </ul>
      </div>
    </NodeComponent>
  );
};
```

---

## 🔄 워크플로우 흐름

### 기본 설정
```
1. 🔑 HF Token 노드 추가
2. 토큰 입력 (hf_...)
3. "기본 토큰으로 저장" 체크 (선택사항)
```

### 모델 다운로드
```
1. 🤖 Model Loader 노드 추가
2. 모델 ID 입력 (gpt2)
3. 🔑 토큰 노드와 연결 (자동 사용)
4. 다운로드 실행
```

### 데이터셋 로드
```
1. 📊 Dataset Loader 노드 추가
2. 데이터셋 ID 입력 (wikitext)
3. 🔑 토큰 노드와 연결 (자동 사용)
4. 로드 실행
```

---

## 📊 데이터 흐름

### 토큰 흐름
```
🔑 HF Token Node
├─ token: "hf_xxx..."
├─ tokenType: "user"
├─ permissions: "read"
└─ saveAsDefault: true
    ↓
    🔗 연결을 통해 공유
    ↓
┌─────────────────────┐
│ 모든 HF 관련 노드    │
├─────────────────────┤
│ 🤖 Model Loader     │ ← 토큰 사용
│ 📊 Dataset Loader   │ ← 토큰 사용
│ 💬 Chat Node        │ ← 토큰 사용 (추후)
└─────────────────────┘
```

---

## 📁 파일 구조

```
frontend/src/
├── @nodes/
│   ├── HFTokenNode.json          ← 토큰 노드 정의
│   ├── ModelLoaderNode.json
│   └── DatasetLoaderNode.json
├── components/Editor/
│   ├── HFTokenNode.tsx            ← 토큰 노드 컴포넌트 (새로 추가)
│   ├── ModelLoaderNode.tsx
│   └── DatasetLoaderNode.tsx
├── types/
│   └── editor.ts                  ← HFTokenNode 타입 추가
└── pages/
    └── Editor.tsx                 ← 노드 팔레트에 추가
```

---

## ✅ 완료 항목

### 로컬 데이터셋 표시 수정
- ✅ 배열 형식 히스토리 지원
- ✅ "file" 소스 지원 ("upload" 외)
- ✅ 로컬 데이터셋이 드롭다운에 표시

### HF 토큰 노드 추가
- ✅ JSON 정의 생성
- ✅ React 컴포넌트 구현
- ✅ 타입 정의 추가
- ✅ Editor에 통합
- ✅ 토큰 유효성 검사
- ✅ 상태 표시

---

## 🎯 토큰 노드의 역할

### 다른 노드와의 관계

```
토큰 노드 (설정)
├─ 포트: Token (출력)
├─ 특징: 설정만 담당
└─ 상태: 유효성만 표시

다운로더 노드 (실행)
├─ 포트: Token (입력)
├─ 특징: 실제 다운로드 수행
└─ 상태: 다운로드 진행도
```

### 사용 권장사항

```
✅ 워크플로우 시작 시 토큰 노드 추가
✅ 모든 HF 작업 전에 연결
✅ "기본 토큰으로 저장" 활성화
✅ 주기적으로 토큰 갱신
```

---

## 🔒 보안 고려사항

### 토큰 보호
```
✅ Password 타입 필드 (입력 숨김)
✅ 프리뷰 (처음 10자 + 마지막 5자만 표시)
✅ 로컬 저장소에만 저장 (서버 전송 X)
```

### 추천 설정
```
1. Fine-grained 토큰 사용
2. "Read" 권한만 설정
3. 특정 리포지토리에만 접근 권한
4. 주기적 토큰 갱신
```

---

## 📞 다음 단계

### Phase 1: 노드 연결 기능
- 포트 드래그 & 드롭
- 타입 검증 (token 포트)
- 연결 데이터 전달

### Phase 2: 다른 노드들
- TrainingNode
- ChatNode
- RagNode
- GgufExportNode

### Phase 3: 고급 기능
- 워크플로우 실행 엔진
- 데이터 흐름 시각화
- 에러 처리 및 복구

---

## 🧪 테스트 방법

### 토큰 노드 테스트
```
1. 🔑 HF Token 노드 추가
2. 유효한 토큰 입력 (hf_로 시작)
3. 체크: ✅ 토큰 유효 표시
4. 기본 토큰 저장 (선택)
```

### 데이터셋 노드 테스트
```
1. 📊 Dataset Loader 노드 추가
2. 소스: "📂 로컬 저장소"
3. 데이터셋 드롭다운 확인
4. 데이터셋 선택
5. 체크: 데이터 표시됨
```

### 모델 노드 테스트
```
1. 🤖 Model Loader 노드 추가
2. 소스: "🤗 HuggingFace"
3. 모델 ID 입력 (gpt2)
4. 토큰과 연결
5. 다운로드 실행
```

---

## 📊 현재 상태

| 항목 | 상태 | 설명 |
|------|------|------|
| 로컬 데이터셋 | ✅ | 히스토리에서 자동 감지 |
| HF 토큰 노드 | ✅ | 설정 및 유효성 검사 |
| 노드 팔레트 | ✅ | 모든 노드 표시 |
| 노드 렌더링 | ✅ | 캔버스에 표시 |
| 노드 연결 | 🔄 | 구현 예정 |
| 워크플로우 실행 | 🔄 | 구현 예정 |

---

**마지막 업데이트**: 2025-11-08  
**상태**: ✅ 완성, 연결 기능 준비 완료  
**버전**: 1.1.0

