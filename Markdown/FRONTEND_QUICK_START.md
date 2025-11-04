# 🎨 MacTuner 프론트엔드 빠른 시작

## 🚀 프론트엔드 실행 (3단계)

### Step 1️⃣: 디렉토리 이동
```bash
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner/frontend
```

### Step 2️⃣: 의존성 설치
```bash
npm install
```

### Step 3️⃣: 개발 서버 시작
```bash
npm run dev
```

---

## 📱 브라우저에서 확인

서버 시작 후 다음 주소로 접속:
```
http://localhost:5173
```

---

## 🎯 현재 구현된 페이지

✅ **Dashboard (대시보드)**
- 최근 워크플로우 표시
- 빠른 시작 버튼
- 통계 정보

✅ **Header (네비게이션)**
- 메뉴 네비게이션
- 로고
- 반응형 디자인

---

## 🛠️ 백엔드와 연결

백엔드가 실행 중이어야 합니다:

```bash
# 다른 터미널에서
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000
```

---

## 📂 프론트엔드 구조

```
frontend/
├── src/
│   ├── components/     # 재사용 가능한 컴포넌트
│   │   └── Header.tsx
│   ├── pages/          # 페이지 컴포넌트
│   │   └── Dashboard.tsx
│   ├── stores/         # Zustand 상태 관리
│   │   └── workflowStore.ts
│   ├── types/          # TypeScript 타입
│   │   └── index.ts
│   ├── App.tsx         # 메인 App 컴포넌트
│   ├── main.tsx        # 진입점
│   └── index.css       # 글로벌 스타일
├── package.json        # 의존성
├── vite.config.ts      # Vite 설정
└── tailwind.config.js  # Tailwind CSS 설정
```

---

## 🎨 기술 스택

- **React 18**: UI 프레임워크
- **TypeScript**: 타입 안전성
- **Vite**: 빌드 도구 (빠른 개발)
- **Tailwind CSS**: 스타일링
- **Zustand**: 상태 관리
- **React Router**: 라우팅
- **Socket.IO**: 실시간 통신 (준비 중)

---

## 🔗 API 연결

### API 기본 주소 설정
`src/utils/api.ts` 또는 `src/config.ts`에서:

```typescript
const API_BASE_URL = 'http://localhost:8000/api';
```

### 사용 예시
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000'
});

// 모델 헬스 체크
const health = await api.get('/model/health');
```

---

## 🐛 문제 해결

### 포트 충돌
```bash
# 다른 포트 사용
npm run dev -- --port 3000
```

### 의존성 문제
```bash
# node_modules 제거 후 재설치
rm -rf node_modules package-lock.json
npm install
```

### 타입스크립트 에러
```bash
# 캐시 정리
npm run build
```

---

## 📝 다음 단계

- [ ] npm install 실행
- [ ] npm run dev로 서버 시작
- [ ] http://localhost:5173 방문
- [ ] Dashboard 페이지 확인
- [ ] 백엔드 API 연결
- [ ] 노드 에디터 구현 시작

---

**프론트엔드 개발을 시작할 준비가 되었습니다!** 🚀✨

