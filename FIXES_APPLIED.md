# Fixes Applied - Max Tokens and Response Issues

## 문제점
1. **응답 생성 실패**: 모델이 "I'm sorry, I didn't understand your question"를 반복 출력
2. **Max Tokens 설정 불가**: 프론트엔드에서 max_tokens 값을 조정할 수 없음
3. **파라미터 전달 오류**: 백엔드로 repeat_penalty, n_gpu_layers 등 파라미터가 제대로 전달되지 않음

## 해결 방법

### 1. 프론트엔드 (Frontend)

#### `/frontend/src/utils/api.ts`
- **변경 사항**:
  - `chatAPI.chat()` 함수 개선: 모든 파라미터에 기본값 설정
  - `maintain_history` 파라미터 추가
  - 파라미터 명확성 향상

```typescript
chat: (message: string, top_p?: number, temperature?: number, max_tokens?: number, repeat_penalty?: number, n_gpu_layers?: number) =>
  api.post('/chat/chat', { 
    message, 
    top_p: top_p || 0.9,
    temperature: temperature || 0.7,
    max_tokens: max_tokens || 1024,
    repeat_penalty: repeat_penalty || 1.1,
    n_gpu_layers: n_gpu_layers || 35,
    maintain_history: true
  })
```

#### `/frontend/src/pages/Chat.tsx`
- **변경 사항**:
  1. **Max Tokens 슬라이더 추가**
     - 범위: 256 ~ 8192 토큰
     - 슬라이더와 입력창 동시 제공
     - 저장소에 자동 저장

  2. **UI 개선**:
     - Max Tokens과 Max Response Length 구분 표시
     - 각 파라미터별 설명 추가
     - Request Parameters에 모든 파라미터 표시

  3. **디버그 정보 강화**:
     - 응답이 비어있는지 표시
     - 각 파라미터 설명 추가 (creativity, diversity 등)

### 2. 백엔드 (Backend)

#### `/backend/api/chat_interface.py`
- **변경 사항**:
  - ChatRequest 모델에 `repeat_penalty`, `n_gpu_layers` 필드 추가
  - GGUF 모델 처리 시 새로운 파라미터 전달
  - `max_length` → `max_tokens` 명칭 통일

#### `/backend/services/llama_cpp_service.py`
- **변경 사항**:
  1. **chat() 메서드**:
     - `repeat_penalty` 파라미터 추가
     - `n_gpu_layers` 파라미터 추가 (향후 동적 GPU 레이어 조정 가능)
     - repeat_penalty를 llama.cpp 호출에 포함

  2. **generate() 메서드**:
     - `repeat_penalty` 파라미터 추가

  3. **응답 처리 개선**:
     - `_cleanup_response()` 메서드 강화
     - 빈 응답 감지 및 에러 메시지 반환
     - 불완전한 코드 블록 자동 완성
     - 최소 길이 검증 (3자 이상)

## 효과

### 응답 품질 향상
- Repeat penalty 조정으로 반복 문제 해결
- Max tokens 증가로 더 완전한 응답 생성
- 빈 응답에 대한 폴백 처리

### 사용자 경험 개선
- Max tokens 직관적으로 조정 가능
- 모든 파라미터 한눈에 확인
- 요청/응답 정보 상세 표시
- localStorage에 자동 저장

## 사용 방법

### Max Tokens 조정
1. Settings 패널의 **"펼치기"** 버튼 클릭
2. **"Max Tokens"** 슬라이더 조정 또는 입력
3. 범위: 256 ~ 8192
   - 낮은 값 (256-512): 빠른 응답, 짧은 답변
   - 중간값 (1024-2048): 균형잡힌 응답
   - 높은 값 (4096-8192): 상세한 답변, 긴 응답 시간

### Repeat Penalty 조정
1. **"Repeat Penalty"** 슬라이더로 조정
2. 범위: 0.0 ~ 2.0
   - 1.0: 페널티 없음
   - 1.1-1.3: 권장값 (약한 페널티)
   - 1.5-2.0: 강한 페널티

### Request Parameters 확인
1. 요청 파라미터 확인: **"Request Parameters"** 섹션 클릭
2. 응답 정보 확인: **"Response Info"** 섹션 클릭 (응답 후)

## 테스트 체크리스트

- [ ] Max Tokens 슬라이더 동작 확인
- [ ] 높은 Max Tokens로 상세한 응답 생성
- [ ] Repeat Penalty 조정으로 반복 문제 해결
- [ ] Request Parameters 정확히 표시
- [ ] Response Info에 응답 통계 표시
- [ ] 설정 localStorage 저장/복구
- [ ] 빈 응답에 폴백 메시지 표시

## 백엔드 API 변경

### /chat/chat POST 요청 형식
```json
{
  "message": "your question",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024,
  "repeat_penalty": 1.1,
  "n_gpu_layers": 35,
  "maintain_history": true
}
```

### 응답 형식
```json
{
  "status": "success",
  "data": {
    "response": "answer text",
    "tokens_used": 250
  }
}
```

