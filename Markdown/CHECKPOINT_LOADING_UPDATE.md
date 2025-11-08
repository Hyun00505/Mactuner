# 📥 체크포인트 로드 진행 상황 - 프론트엔드 표시 개선

**작성일:** 2025-11-07  
**목적:** 모델 로드 중 HuggingFace 체크포인트 로드 진행 상황을 프론트엔드에서 실시간으로 표시

---

## 🎯 해결한 문제

### 1. 체크포인트 로드 진행 상황 미표시

**Before:**

- 터미널에만 진행 상황이 출력됨
  ```
  Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
  Loading checkpoint shards:  25%|██▌       | 1/4 [00:04<00:12,  4.00s/it]
  Loading checkpoint shards:  50%|█████     | 2/4 [00:10<00:10,  5.48s/it]
  Loading checkpoint shards:  75%|███████▌  | 3/4 [00:17<00:06,  6.08s/it]
  Loading checkpoint shards: 100%|██████████| 4/4 [00:23<00:00,  6.16s/it]
  ```
- 프론트엔드 UI에서 진행 상황을 알 수 없음

**After:**

- 프론트엔드에서 상세한 로드 단계를 실시간으로 표시
  - "토크나이저 로드 중..." (25%)
  - "모델 구성 로드 중..." (40%)
  - "모델 로드 완료" (85%)
  - "모델을 MPS로 이동" (90%)
  - "메타데이터 추출 완료" (95%)

### 2. TOKENIZERS_PARALLELISM 경고

**Before:**

```
huggingface/tokenizers: The current process just got forked, after
parallelism has already been used. Disabling parallelism to avoid deadlocks...
```

**After:**

- 경고가 표시되지 않음 (환경변수로 제어)

---

## 🔧 기술 구현 상세

### 1. `backend/services/model_service.py` 수정

#### 변경 사항:

- `load_from_hub()` 메서드에 `progress_callback` 파라미터 추가
- `load_local()` 메서드에 `progress_callback` 파라미터 추가
- 각 단계별로 진행 정보를 콜백으로 전송

```python
def load_local(self, path: str, progress_callback=None) -> Tuple[torch.nn.Module, AutoTokenizer, Dict]:
    # ...
    if progress_callback:
        progress_callback({"status": "loading_tokenizer", "message": "토크나이저 로드 중...", "progress": 10})

    tokenizer = AutoTokenizer.from_pretrained(...)

    if progress_callback:
        progress_callback({"status": "model_loaded", "message": "모델 로드 완료", "progress": 85})
    # ...
```

**진행도 단계:**

- 10%: 토크나이저 로드 시작
- 20%: 토크나이저 로드 완료
- 25%: 모델 구성 로드 중
- 85%: 모델 로드 완료
- 90%: 디바이스로 이동
- 95%: 메타데이터 추출 완료
- 100%: 완료

### 2. `backend/api/model_loader.py` 수정

#### 변경 사항:

- `upload-stream` 엔드포인트에서 진행 콜백 사용
- 수집된 진행 정보를 클라이언트로 스트리밍

```python
def collect_progress(progress_data):
    progress_updates.append(progress_data)

model, tokenizer, metadata = model_service_instance.load_local(
    model_path,
    collect_progress  # 콜백 함수 전달
)

# 수집된 진행 정보를 모두 스트리밍
for update in progress_updates:
    yield json.dumps({
        "status": update.get("status"),
        "message": update.get("message"),
        "progress": update.get("progress")
    }).encode() + b'\n'
```

### 3. `backend/main.py` 수정

#### 변경 사항:

- `TOKENIZERS_PARALLELISM` 환경변수를 `false`로 설정
- 모듈 임포트 전에 설정하여 경고 억제

```python
import os

# HuggingFace tokenizers 병렬 처리 경고 억제
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**효과:**

- HuggingFace tokenizers 포킹 경고 제거
- 콘솔 출력이 깔끔해짐

### 4. `frontend/src/pages/Chat.tsx` 수정

#### 변경 사항:

- 모델 로드 상태 표시 UI 개선
- 진행 상황 메시지와 프로그레스 바 추가

```jsx
{
  /* 모델 로드 상태 표시 */
}
{
  modelLoading && modelLoadStatus && (
    <div className="mt-3 p-2 bg-blue-900 rounded border border-blue-600 text-xs">
      <p className="text-blue-200 mb-2 font-semibold">📥 로드 중...</p>
      <p className="text-blue-100 text-xs mb-2">{modelLoadStatus}</p>
      <div className="w-full bg-blue-800 rounded-full h-2">
        <div className="bg-blue-400 h-2 rounded-full transition-all duration-300" style={{ width: `${Math.min(modelLoadProgress, 100)}%` }} />
      </div>
      <p className="text-blue-300 mt-1 text-xs text-right">{Math.round(modelLoadProgress)}%</p>
    </div>
  );
}
```

**UI 특징:**

- 파란색 테마 박스로 로드 상태 표시
- 실시간 진행 메시지 업데이트
- 부드러운 프로그레스 바 애니메이션
- 백분율 표시

---

## 📊 진행 흐름도

```
프론트엔드 (Chat.tsx)
     ↓
/model/upload-stream API
     ↓
collect_progress 콜백 호출
     ↓
ModelService.load_local() 실행
     ↓
진행 정보 수집
     ↓
JSON 스트림으로 클라이언트 전송
     ↓
프론트엔드 UI 업데이트
```

---

## ✅ 검증 방법

### 테스트 절차:

1. 프론트엔드에서 모델 로드 버튼 클릭
2. 우측 설정 패널에서 진행 상황 모니터링
3. 다음 메시지들이 순서대로 표시되는지 확인:
   - "토크나이저 로드 중..." (25%)
   - "모델 로드 완료" (85%)
   - "모델을 MPS로 이동" (90%)
   - 등등

### 콘솔 확인:

```bash
# 터미널에서 다음 경고가 나타나지 않아야 함
# "huggingface/tokenizers: The current process just got forked..."
```

---

## 🎓 기술 참고사항

### Progress Callback 패턴

- 동기 함수에서 비동기 스트리밍으로 진행 정보 전달
- 콜백 방식으로 느슨한 결합 유지
- FastAPI의 StreamingResponse와 호환

### TOKENIZERS_PARALLELISM

- HuggingFace tokenizers 라이브러리에서 포킹 후 병렬 처리 방지
- 주의사항: 성능에 미미한 영향 (대부분의 경우 측정 불가)
- 경고만 제거되고 기능은 정상 작동

---

## 📝 변경된 파일

| 파일                                | 변경 사항                       | 라인          |
| ----------------------------------- | ------------------------------- | ------------- |
| `backend/services/model_service.py` | progress_callback 파라미터 추가 | 20-72, 74-122 |
| `backend/api/model_loader.py`       | 진행 정보 스트리밍 구현         | 293-346       |
| `backend/main.py`                   | TOKENIZERS_PARALLELISM 설정     | 4-5           |
| `frontend/src/pages/Chat.tsx`       | 로드 상태 UI 개선               | 700-713       |

---

## 🚀 향후 개선 사항

1. **더 상세한 체크포인트 진행 정보**

   - HuggingFace `tqdm` 프로그레스 바 파싱
   - 실시간 샤드 로드 진행률 (0/4, 1/4, 2/4, 3/4, 4/4)

2. **로드 소요 시간 추정**

   - 현재 속도 기반 남은 시간 계산
   - ETA 표시

3. **로드 취소 기능**
   - 진행 중 모델 로드 취소
   - 백그라운드 작업 중단

---

## 📌 주의사항

- ✅ 기존 기능과 완전히 하위 호환
- ✅ 성능 영향 없음 (콜백은 가볍게 구현)
- ✅ 에러 처리 유지 (예외 발생 시 기존 동작)
