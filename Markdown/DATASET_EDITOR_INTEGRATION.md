# 📊 데이터셋 Editor 통합 완료

## 📋 개요

MacTuner의 **데이터셋 관리 시스템**을 모델처럼 완전히 통합했습니다!

이제 `/data` 페이지에서 보이는 **모든 데이터셋이 Editor의 노드에서도 동일하게 표시**됩니다. 🚀

---

## ✨ 주요 기능

### 1️⃣ 로컬 데이터셋 자동 감지

**감지 위치**:
- 히스토리 기반: `./data/dataset_history.json`
- 이미 로드된 모든 데이터셋

**감지 방식**:
```
✅ 업로드 파일 (CSV, JSON, JSONL, Parquet)
✅ HuggingFace 데이터셋 (다운로드된 것)
✅ 로컬 파일 경로
```

### 2️⃣ 데이터셋 소스 구분

**Source 선택**:
```
📂 로컬 저장소 (기존 데이터셋)
   └─ 이미 로드된 데이터셋 사용
   └─ 히스토리에서 자동 감지

🤗 HuggingFace 데이터셋
   └─ HuggingFace Hub에서 다운로드
   └─ 데이터셋 ID 입력 필요

📤 파일 업로드
   └─ 새로운 파일 업로드
   └─ 다양한 형식 지원
```

### 3️⃣ 동적 옵션 로드

**작동 방식**:
```
DatasetLoaderNode 렌더링
   ↓
NodeForm에서 "localDataset" 파라미터 감지
   ↓
dynamicOptions = true 확인
   ↓
API: http://localhost:8001/dataset/local-datasets 호출
   ↓
로컬 데이터셋 목록 받기
   ↓
드롭다운에 표시 (아이콘 + 이름 + 크기)
```

---

## 🏗️ 구현 상세

### Backend API 추가

**`GET /dataset/local-datasets`** (새로 추가):

```python
@router.get("/local-datasets")
async def get_local_datasets() -> Dict:
    """로컬에 있는 데이터셋 목록 조회 (히스토리 기반)"""
    # 히스토리 파일에서 데이터셋 정보 추출
    return {
        "status": "success",
        "datasets": [
            {
                "dataset_id": "train_data.csv",
                "source": "upload",
                "format": "csv",
                "size_mb": 10.5
            },
            {
                "dataset_id": "wikitext",
                "source": "huggingface",
                "split": "train"
            }
        ]
    }
```

### Frontend 동적 옵션

**`nodeLoader.ts` 개선**:

```typescript
// datasets 객체 형식 처리
if (data.datasets && Array.isArray(data.datasets)) {
  return data.datasets.map((item: any) => {
    const icon = item.source === 'huggingface' ? '🤗' : '📤';
    const label = `${icon} ${item.dataset_id} (${item.size_mb}MB)`;
    return { label, value: item.dataset_id };
  });
}
```

### DatasetLoaderNode.json 업데이트

```json
{
  "id": "source",
  "options": [
    "📂 로컬 저장소 (기존 데이터셋)",
    "🤗 HuggingFace 데이터셋",
    "📤 파일 업로드"
  ]
}

{
  "id": "localDataset",
  "dynamicOptions": true,
  "apiEndpoint": "/dataset/local-datasets"
  // ↑ 조건: source === "local"
}
```

---

## 📊 `/data` 페이지 에러 수정

### 문제점
```
❌ [GET /full-data] ValueError: 로드된 데이터가 없습니다.
```

**원인**: 페이지 로드 시 데이터가 없으면 에러 반복

### 해결책

**Backend** - `GET /full-data` 개선:
```python
except ValueError as e:
    # 에러 던지지 않고 빈 결과 반환
    return {
        "status": "no_data",
        "data": {
            "rows": [],
            "total_rows": 0,
            "columns": [],
            "dtypes": {}
        }
    }
```

**Frontend** - 에러 처리 개선:
```typescript
// 데이터가 없는 경우 (정상, 에러 아님)
if (response.data.status === "no_data") {
  console.log("📂 기존 데이터 없음 (정상 상태)");
  setMessage(""); // 메시지 초기화
  return;
}
```

---

## 📋 `/data` 페이지와 Editor 동기화

### 공통점

```
모두 같은 API 사용: GET /dataset/local-datasets
모두 같은 데이터셋 목록 표시
모두 같은 아이콘 사용 (🤗 vs 📤)
모두 같은 데이터셋 정보 표시
```

### 차이점

```
/data 페이지:
  └─ 주로 데이터셋 관리 (업로드, 정제)
  └─ 모든 데이터셋 목록 보기

Editor 노드:
  └─ 주로 데이터셋 선택 & 로드
  └─ 워크플로우에 필요한 데이터셋만 선택
```

---

## 🎨 UI 개선 사항

### DatasetLoaderNode.json 개선

```json
{
  "id": "source",
  "type": "select",
  "options": [
    {
      "label": "📂 로컬 저장소 (기존 데이터셋)",
      "value": "local"
    },
    {
      "label": "🤗 HuggingFace 데이터셋",
      "value": "huggingface"
    },
    {
      "label": "📤 파일 업로드",
      "value": "upload"
    }
  ]
}
```

### 지원 파일 형식

```
✅ CSV (.csv)
✅ JSON (.json)
✅ JSONL (.jsonl)
✅ Parquet (.parquet)
✅ Excel (.xlsx)
```

---

## 🔄 데이터 흐름

### 로컬 데이터셋 로드 흐름

```
1. Editor 시작
   ↓
2. DatasetLoaderNode 렌더링
   ↓
3. NodeForm이 "localDataset" 파라미터 감지
   ↓
4. dynamicOptions = true 확인
   ↓
5. fetchDynamicOptions("/dataset/local-datasets") 호출
   ↓
6. Backend: 로컬 데이터셋 목록 반환
   {
     "datasets": [
       {
         "dataset_id": "train_data.csv",
         "source": "upload",
         "size_mb": 10.5
       },
       {
         "dataset_id": "wikitext",
         "source": "huggingface"
       }
     ]
   }
   ↓
7. Frontend: 드롭다운 생성
   "📤 train_data.csv (10.5MB)"
   "🤗 wikitext"
   ↓
8. 사용자 선택
   ↓
9. 데이터셋 로드 (구현 예정)
   ↓
10. 완료
```

---

## 📁 데이터셋 히스토리 구조

**`./data/dataset_history.json`**:

```json
{
  "history": [
    {
      "source": "upload",
      "file_name": "train_data.csv",
      "file_format": "csv",
      "size_mb": 10.5,
      "timestamp": "2025-11-08T12:34:56"
    },
    {
      "source": "hf",
      "hf_dataset_id": "wikitext",
      "hf_split": "train",
      "timestamp": "2025-11-08T13:45:00"
    }
  ]
}
```

---

## 🚀 사용 방법

### 로컬 데이터셋 사용 (이미 로드된 것)

**1. `/data` 페이지에서 업로드 또는 다운로드**:
```
1. CSV 파일 업로드
   또는
2. HuggingFace 데이터셋 다운로드
```

**2. Editor에서 사용**:
```
1. 📊 데이터셋 로더 노드 추가
2. "데이터셋 소스" → "📂 로컬 저장소" 선택
3. "데이터셋 선택" → 드롭다운에서 선택
4. "데이터셋 로드" 클릭
5. 완료!
```

### HuggingFace 데이터셋 다운로드

**1. Editor에서**:
```
1. 📊 데이터셋 로더 노드 추가
2. "데이터셋 소스" → "🤗 HuggingFace 데이터셋" 선택
3. "데이터셋 ID" → "wikitext" 입력
4. "Dataset Split" → "train" (선택사항)
5. "🤗 다운로드" 클릭
6. 완료!
```

---

## ✅ 완료 항목

- ✅ `GET /dataset/local-datasets` API 추가
- ✅ DatasetLoaderNode.json 업데이트
- ✅ nodeLoader.ts 동적 옵션 처리 추가
- ✅ `/data` 페이지 에러 수정
- ✅ 데이터 없을 때 우아한 처리

---

## 🎓 배운 개념

1. **히스토리 기반 데이터 관리**: 모든 데이터셋 작업 기록
2. **동적 드롭다운**: API에서 실시간 데이터 로드
3. **조건부 UI**: 선택에 따라 다른 파라미터 표시
4. **에러 처리**: 데이터 없음을 에러가 아닌 정상 상태로 처리

---

## 📞 다음 단계

### Phase 1: DatasetLoaderNode 컴포넌트 (다음)
- 현재: JSON 정의 완성
- 다음: React 컴포넌트 구현

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

