# 🎨 MacTuner Frontend 구현 가이드

## 📊 프론트엔드 구조

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.tsx              ✅ (구현 완료)
│   │   ├── NodePalette.tsx         (구현 예정)
│   │   ├── Nodes/
│   │   │   ├── ModelLoaderNode.tsx
│   │   │   ├── DatasetNode.tsx
│   │   │   ├── TrainingNode.tsx
│   │   │   ├── ChatNode.tsx
│   │   │   ├── RAGNode.tsx
│   │   │   └── GGUFNode.tsx
│   │   └── Canvas/
│   │       ├── WorkflowCanvas.tsx
│   │       ├── NodeRenderer.tsx
│   │       └── ConnectionRenderer.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx           ✅ (구현 완료)
│   │   ├── Editor.tsx              (구현 예정)
│   │   ├── Monitor.tsx             (구현 예정)
│   │   └── History.tsx             (구현 예정)
│   ├── hooks/
│   │   ├── useWorkflow.ts
│   │   ├── useNode.ts
│   │   └── useWebSocket.ts
│   ├── stores/
│   │   ├── workflowStore.ts        ✅ (구현 완료)
│   │   ├── uiStore.ts
│   │   └── apiStore.ts
│   ├── types/
│   │   └── index.ts                ✅ (구현 완료)
│   ├── utils/
│   │   ├── api.ts
│   │   └── socket.ts
│   ├── App.tsx                     ✅ (구현 완료)
│   ├── main.tsx                    ✅ (구현 완료)
│   └── index.css                   ✅ (구현 완료)
├── package.json                    ✅ (구현 완료)
├── vite.config.ts                  ✅ (구현 완료)
├── tsconfig.json                   ✅ (구현 완료)
├── tailwind.config.js              ✅ (구현 완료)
├── postcss.config.js               ✅ (구현 완료)
├── index.html                      ✅ (구현 완료)
└── .env.example                    (예정)
```

## 🚀 설치 및 실행

### 1. 의존성 설치 (네트워크 필요)
```bash
cd frontend
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

### 3. 빌드
```bash
npm run build
```

### 4. 미리보기
```bash
npm run preview
```

## 📋 구현 계획

### Phase 3.1: 기초 UI (완료 ✅)
- [x] Header 컴포넌트
- [x] Dashboard 페이지
- [x] 라우팅 설정
- [x] Tailwind CSS 설정
- [x] TypeScript 타입 정의
- [x] Zustand 상태 관리

### Phase 3.2: 노드 시스템 (예정)
```typescript
// ModelLoaderNode.tsx 예시
export const ModelLoaderNode: React.FC<NodeProps> = ({ node }) => {
  const [modelId, setModelId] = useState("");
  const [loading, setLoading] = useState(false);

  const handleDownload = async () => {
    setLoading(true);
    try {
      const response = await api.post("/model/download", {
        model_id: modelId,
      });
      // 다운로드 완료
      console.log(response.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-blue-100 p-4 rounded-lg border-2 border-blue-500">
      <h3 className="font-bold text-blue-900">📥 Model Loader</h3>
      <input
        type="text"
        value={modelId}
        onChange={(e) => setModelId(e.target.value)}
        placeholder="gpt2"
        className="w-full mt-2 p-2 border rounded"
      />
      <button
        onClick={handleDownload}
        disabled={loading}
        className="w-full mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "다운로드 중..." : "다운로드"}
      </button>
    </div>
  );
};
```

### Phase 3.3: 실시간 통신 (예정)
```typescript
// useWebSocket.ts 예시
export const useWebSocket = () => {
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const newSocket = io("http://localhost:8000", {
      transports: ["websocket"],
    });

    newSocket.on("training:progress", (data) => {
      console.log("Training progress:", data);
      // 진행률 업데이트
    });

    newSocket.on("training:complete", (data) => {
      console.log("Training completed:", data);
      // 완료 처리
    });

    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  return socket;
};
```

### Phase 3.4: 통합 (예정)
```typescript
// Editor.tsx 예시
export const Editor: React.FC = () => {
  const { nodes, addNode, deleteNode } = useWorkflowStore();
  const socket = useWebSocket();

  return (
    <div className="flex h-screen">
      {/* 노드 팔레트 */}
      <div className="w-48 bg-gray-100 p-4 border-r">
        <h3 className="font-bold mb-4">노드 팔레트</h3>
        <div className="space-y-2">
          <button
            onClick={() =>
              addNode({
                id: `model_${Date.now()}`,
                type: "model",
                position: { x: 100, y: 100 },
                data: {},
                connections: { input: null, output: null },
              })
            }
            className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            + Model Loader
          </button>
          {/* 다른 노드들... */}
        </div>
      </div>

      {/* 캔버스 */}
      <div className="flex-1 bg-white relative">
        {nodes.map((node) => (
          <div
            key={node.id}
            className="absolute w-48 bg-white border-2 rounded shadow-lg p-4"
            style={{
              left: `${node.position.x}px`,
              top: `${node.position.y}px`,
            }}
          >
            {/* 노드 렌더링 */}
            <button
              onClick={() => deleteNode(node.id)}
              className="absolute top-2 right-2 text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      {/* 속성 패널 */}
      <div className="w-64 bg-gray-100 p-4 border-l">
        <h3 className="font-bold mb-4">속성</h3>
        {/* 선택된 노드의 속성 편집 */}
      </div>
    </div>
  );
};
```

## 🎨 디자인 시스템

### 색상 팔레트
```css
--primary:      #0284c7 (파란색)
--secondary:    #10b981 (초록색)
--warning:      #f59e0b (주황색)
--error:        #ef4444 (빨간색)
--neutral:      #6b7280 (회색)
```

### 노드 색상
```
Model:    🔵 파란색 (from-blue-500)
Dataset:  🟢 초록색 (from-green-500)
Training: 🟠 주황색 (from-orange-500)
Chat:     🔵 밝은 파란색 (from-cyan-500)
RAG:      🟣 보라색 (from-purple-500)
GGUF:     🟨 노란색 (from-yellow-500)
```

## 🔗 API 연동

### API 클라이언트
```typescript
// utils/api.ts
import axios from "axios";

export const api = axios.create({
  baseURL: process.env.VITE_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error);
    return Promise.reject(error);
  }
);
```

### 사용 예시
```typescript
// Dashboard.tsx에서
const loadModels = async () => {
  try {
    const { data } = await api.get("/model/local-models");
    console.log("Models:", data);
  } catch (error) {
    console.error("Failed to load models:", error);
  }
};
```

## 📱 반응형 디자인

### 브레이크포인트
```css
sm: 640px
md: 768px
lg: 1024px
xl: 1280px
2xl: 1536px
```

### 모바일 우선 접근
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* 모바일: 1열, 태블릿: 2열, 데스크톱: 3열 */}
</div>
```

## 🧪 테스트

### 단위 테스트 (예정)
```bash
npm run test
```

### E2E 테스트 (예정)
```bash
npm run test:e2e
```

## 📦 배포

### 빌드
```bash
npm run build
```

### Docker 배포
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## 🎯 다음 단계

### 즉시 (1주)
- [ ] 나머지 페이지 구현 (Editor, Monitor, History)
- [ ] 노드 컴포넌트 작성
- [ ] 캔버스 렌더링

### 진행 중 (2주)
- [ ] Socket.IO 통합
- [ ] 실시간 진행률 표시
- [ ] 워크플로우 저장/로드

### 최종 (1주)
- [ ] 통합 테스트
- [ ] 성능 최적화
- [ ] 배포

## 📞 참고 자료

- [Vite 공식 문서](https://vitejs.dev)
- [React 공식 문서](https://react.dev)
- [TypeScript 공식 문서](https://www.typescriptlang.org)
- [Tailwind CSS 문서](https://tailwindcss.com)
- [Zustand 문서](https://github.com/pmndrs/zustand)

---

**🎉 프론트엔드 기초 구현 완료!**

**이제 에디터와 노드 시스템을 구현할 준비가 되었습니다!** 🚀

