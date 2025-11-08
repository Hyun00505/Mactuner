import React, { useState, useEffect, useRef } from "react";
import { chatAPI } from "../utils/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PrismAsyncLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Message {
  id: string;
  sender: "user" | "assistant";
  content: string;
  timestamp: string;
}

interface Model {
  model_id: string;
  path: string;
  size_gb: number;
  source: string;
  model_type?: string;
}

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isGgufModel, setIsGgufModel] = useState(false);
  const [modelSizeGb, setModelSizeGb] = useState(0);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelLoading, setModelLoading] = useState<string | null>(null);
  const [currentModelName, setCurrentModelName] = useState<string>("");
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [modelLoadStatus, setModelLoadStatus] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful, friendly AI assistant. Keep your responses concise and natural. Remember information from the conversation. Do not repeat yourself. Provide direct, useful answers without unnecessary elaboration."
  );
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [maxResponseLength, setMaxResponseLength] = useState(8000);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [nGpuLayers, setNGpuLayers] = useState(35);
  const [repeatPenalty, setRepeatPenalty] = useState(1.1);
  const [debugMode, setDebugMode] = useState(false);
  const [debugInfo, setDebugInfo] = useState<string | null>(null);
  const [showParameters, setShowParameters] = useState(false);
  const [showAllSettings, setShowAllSettings] = useState(false);
  const [settingsPanelWidth, setSettingsPanelWidth] = useState(384); // w-96 = 384px
  const [isResizing, setIsResizing] = useState(false);
  const [modelCustomNames, setModelCustomNames] = useState<Record<string, string>>({});
  const [editingModelName, setEditingModelName] = useState<string | null>(null);
  const [newModelName, setNewModelName] = useState("");
  const [autoMaxTokens, setAutoMaxTokens] = useState(true); // Auto mode for max tokens
  const [autoMinTokens, setAutoMinTokens] = useState(512); // Auto mode min tokens
  const [autoMaxTokensValue, setAutoMaxTokensValue] = useState(4096); // Auto mode max tokens
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [copiedCodeId, setCopiedCodeId] = useState<string | null>(null);
  const settingsPanelRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // 전체 메시지 복사
  const copyToClipboard = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // 코드만 추출해서 복사
  const copyCodeBlock = async (code: string, codeId: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCodeId(codeId);
      setTimeout(() => setCopiedCodeId(null), 2000);
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  };

  // 코드 블록 렌더러
  const CodeBlockRenderer = ({ node, inline, className, children, ...props }: any) => {
    const match = /language-(\w+)/.exec(className || "");
    const language = match ? match[1] : "text";
    const code = String(children).replace(/\n$/, "");
    const codeId = `code-${Math.random().toString(36).substr(2, 9)}`;

    if (inline) {
      return <code className="bg-gray-800 px-2 py-1 rounded text-red-400">{children}</code>;
    }

    return (
      <div className="relative group my-3 rounded-lg overflow-hidden border border-gray-600">
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition z-10">
          <button
            onClick={() => copyCodeBlock(code, codeId)}
            className={`px-3 py-1 rounded text-xs font-medium transition ${
              copiedCodeId === codeId
                ? "bg-green-600 text-white"
                : "bg-gray-700 text-gray-200 hover:bg-gray-600"
            }`}
            title="Copy code"
          >
            {copiedCodeId === codeId ? "✓ Copied" : "📋 Copy"}
          </button>
        </div>
        <SyntaxHighlighter
          language={language}
          style={dracula}
          customStyle={{
            margin: 0,
            borderRadius: 0,
          }}
          {...props}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    );
  };


  // GPU layers 자동 추천 함수
  const calculateRecommendedGpuLayers = (modelSizeGb: number): number => {
    // Mac의 메모리 제약을 고려한 추천값
    // Q4 양자화 기준 (약 1GB ≈ 1-2 layers)
    if (modelSizeGb <= 1) {
      return 50; // 1B 모델 - 거의 모든 layer GPU에서 실행
    } else if (modelSizeGb <= 3) {
      return 40; // 3B 모델
    } else if (modelSizeGb <= 7) {
      return 30; // 7B 모델
    } else if (modelSizeGb <= 13) {
      return 20; // 13B 모델
    } else if (modelSizeGb <= 33) {
      return 10; // 33B 모델
    } else {
      return 5; // 70B 이상 - 제한적
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchModels();
    fetchHistory();
  }, []);

  useEffect(() => {
    // localStorage에서 저장된 데이터 로드
    const savedModel = localStorage.getItem("selectedModel");
    const savedWidth = localStorage.getItem("settingsPanelWidth");
    const savedNames = localStorage.getItem("modelCustomNames");
    const savedLocalModels = localStorage.getItem("localModels");
    const savedMaxTokens = localStorage.getItem("maxTokens");
    const savedMaxResponseLength = localStorage.getItem("maxResponseLength");
    const savedNGpuLayers = localStorage.getItem("nGpuLayers");
    const savedRepeatPenalty = localStorage.getItem("repeatPenalty");
    const savedAutoMaxTokens = localStorage.getItem("autoMaxTokens");
    const savedAutoMinTokens = localStorage.getItem("autoMinTokens");
    const savedAutoMaxTokensValue = localStorage.getItem("autoMaxTokensValue");

    if (savedModel) setSelectedModel(savedModel);
    if (savedWidth) setSettingsPanelWidth(parseInt(savedWidth));
    if (savedNames) {
      try {
        setModelCustomNames(JSON.parse(savedNames));
      } catch (e) {
        console.error("Failed to parse saved model names:", e);
      }
    }
    if (savedLocalModels) {
      try {
        // setLocalModels(JSON.parse(savedLocalModels)); // This line was removed
      } catch (e) {
        console.error("Failed to parse saved local models:", e);
      }
    }
    if (savedMaxTokens) setMaxTokens(parseInt(savedMaxTokens));
    if (savedMaxResponseLength) setMaxResponseLength(parseInt(savedMaxResponseLength));
    if (savedNGpuLayers) setNGpuLayers(parseInt(savedNGpuLayers));
    if (savedRepeatPenalty) setRepeatPenalty(parseFloat(savedRepeatPenalty));
    if (savedAutoMaxTokens !== null) setAutoMaxTokens(JSON.parse(savedAutoMaxTokens));
    if (savedAutoMinTokens) setAutoMinTokens(parseInt(savedAutoMinTokens));
    if (savedAutoMaxTokensValue) setAutoMaxTokensValue(parseInt(savedAutoMaxTokensValue));
  }, []);

  // 모델 선택 저장
  useEffect(() => {
    localStorage.setItem("selectedModel", selectedModel);
  }, [selectedModel]);

  // 설정창 너비 저장
  useEffect(() => {
    localStorage.setItem("settingsPanelWidth", settingsPanelWidth.toString());
  }, [settingsPanelWidth]);

  // 모델 이름 저장
  useEffect(() => {
    localStorage.setItem("modelCustomNames", JSON.stringify(modelCustomNames));
  }, [modelCustomNames]);

  // Max Tokens 저장
  useEffect(() => {
    localStorage.setItem("maxTokens", maxTokens.toString());
  }, [maxTokens]);

  // Max Response Length 저장
  useEffect(() => {
    localStorage.setItem("maxResponseLength", maxResponseLength.toString());
  }, [maxResponseLength]);

  // N GPU Layers 저장
  useEffect(() => {
    localStorage.setItem("nGpuLayers", nGpuLayers.toString());
  }, [nGpuLayers]);

  // Repeat Penalty 저장
  useEffect(() => {
    localStorage.setItem("repeatPenalty", repeatPenalty.toString());
  }, [repeatPenalty]);

  // Auto Max Tokens 저장
  useEffect(() => {
    localStorage.setItem("autoMaxTokens", JSON.stringify(autoMaxTokens));
  }, [autoMaxTokens]);

  // Auto Min/Max Tokens 저장
  useEffect(() => {
    localStorage.setItem("autoMinTokens", autoMinTokens.toString());
  }, [autoMinTokens]);

  useEffect(() => {
    localStorage.setItem("autoMaxTokensValue", autoMaxTokensValue.toString());
  }, [autoMaxTokensValue]);

  // 동적 max_tokens 계산 (auto 모드)
  const calculateAutoMaxTokens = (userMessageLength: number): number => {
    // 사용자 메시지 길이에 따라 동적으로 결정
    // 설정된 범위 내에서 메시지 길이에 비례하게 조정
    const range = autoMaxTokensValue - autoMinTokens;
    
    if (userMessageLength < 20) {
      return autoMinTokens; // 매우 짧은 질문 - 최소값
    } else if (userMessageLength < 50) {
      return Math.round(autoMinTokens + range * 0.25); // 짧은 질문
    } else if (userMessageLength < 100) {
      return Math.round(autoMinTokens + range * 0.5); // 중간 질문
    } else if (userMessageLength < 200) {
      return Math.round(autoMinTokens + range * 0.75); // 긴 질문
    } else {
      return autoMaxTokensValue; // 매우 긴 질문 - 최대값
    }
  };

  // 리사이즈 핸들러
  const handleMouseDown = () => {
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = Math.max(300, Math.min(600, window.innerWidth - e.clientX));
      setSettingsPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const fetchModels = async () => {
    try {
      setLoadingModels(true);
      const response = await fetch("http://localhost:8001/model/local-models");
      const data = await response.json();
      // 데이터셋 제외, 완전한 모델만 (local_folder 소스도 포함)
      const validModels = (data.models || []).filter((m: Model) => m.source === "huggingface" || m.source === "local" || m.source === "local_folder");
      setModels(validModels);
    } catch (error) {
      console.error("Failed to fetch models", error);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleLoadModel = async (modelPath: string, modelId: string) => {
    try {
      setModelLoading(modelId);
      setModelLoadProgress(0);

      // 스트리밍으로 실제 진행 상황 받기
      const response = await fetch("http://localhost:8001/model/upload-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath }),
      });

      if (!response.ok) {
        throw new Error("모델 로드 실패");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("스트림을 읽을 수 없습니다");
      }

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        // 마지막 불완전한 라인 제외
        buffer = lines[lines.length - 1];

        // 완전한 라인들 처리
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line) {
            try {
              const data = JSON.parse(line);
              console.log("Progress update:", data);

              if (data.progress !== undefined) {
                setModelLoadProgress(data.progress);
              }

              if (data.message) {
                setModelLoadStatus(data.message);
              }

              if (data.status === "completed") {
                setSelectedModel(modelId);
                setCurrentModelName(modelId);

                // GGUF 모델 여부 저장
                if (data.is_gguf !== undefined) {
                  setIsGgufModel(data.is_gguf);
                }

                // 모델 크기 저장 및 GPU layers 자동 계산
                if (data.model_size !== undefined) {
                  setModelSizeGb(data.model_size);

                  // GPU layers 자동 추천 계산 (모델 크기 기반)
                  // Mac의 경우 메모리 제한을 고려한 추천값
                  const recommendedLayers = calculateRecommendedGpuLayers(data.model_size);
                  setNGpuLayers(recommendedLayers);
                }

                // Chat 초기화
                try {
                  const initResponse = await fetch("http://localhost:8001/chat/initialize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ system_prompt: systemPrompt }),
                  });
                  if (!initResponse.ok) {
                    console.warn("Chat initialization returned:", initResponse.status);
                  }
                } catch (err) {
                  console.error("Chat initialization failed:", err);
                }

                // 초기화 메시지 추가
                const initMessage: Message = {
                  id: "0",
                  sender: "assistant",
                  content: `✅ ${modelId} 모델이 로드되었습니다!\n이제 대화를 시작할 수 있습니다.`,
                  timestamp: new Date().toLocaleTimeString(),
                };
                setMessages([initMessage]);
              }

              if (data.status === "error") {
                throw new Error(data.message || "모델 로드 실패");
              }
            } catch (parseErr) {
              console.error("JSON parse error:", parseErr);
            }
          }
        }
      }
    } catch (error: any) {
      console.error("Model loading error:", error);
      alert(`❌ 모델 로드 실패: ${error.message}`);
    } finally {
      setModelLoading(null);
      setModelLoadProgress(0);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await chatAPI.history();
      const formattedMessages: Message[] = (response.data.messages || []).map((msg: any, idx: number) => ({
        id: String(idx),
        sender: msg.role === "user" ? "user" : "assistant",
        content: msg.content,
        timestamp: new Date().toLocaleTimeString(),
      }));
      setMessages(formattedMessages);
    } catch (error) {
      console.error("Failed to fetch chat history", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    if (!selectedModel) {
      alert("먼저 모델을 선택해주세요");
      return;
    }

    const userMessage: Message = {
      id: String(messages.length),
      sender: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages([...messages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      console.log("Sending message:", input);
      // Auto mode: 메시지 길이에 따라 동적으로 max_tokens 결정
      const effectiveMaxTokens = autoMaxTokens ? calculateAutoMaxTokens(input.length) : maxTokens;
      console.log(`Auto mode: ${autoMaxTokens}, Input length: ${input.length}, Effective max_tokens: ${effectiveMaxTokens}`);
      const response = await chatAPI.chat(input, topP, temperature, effectiveMaxTokens, repeatPenalty, nGpuLayers);
      console.log("Chat API Response:", response);

      // 응답 처리 (여러 형식 지원)
      let responseText = "";
      if (response.data?.data?.response) {
        responseText = response.data.data.response;
      } else if (response.data?.response) {
        responseText = response.data.response;
      } else {
        responseText = JSON.stringify(response.data);
      }

      if (!responseText || responseText.trim() === "") {
        responseText = "(응답 생성 중 오류가 발생했거나 응답이 비어있습니다)";
      }

      // 디버그 정보 생성 및 저장
      const effectiveTokensUsed = autoMaxTokens ? calculateAutoMaxTokens(input.length) : maxTokens;
      const debugInfoStr = `Request Parameters:
  • Temperature: ${temperature.toFixed(2)} (creativity: 0=deterministic, 2=creative)
  • Top P: ${topP.toFixed(2)} (diversity: 0=focused, 1=diverse)
  • Max Tokens: ${autoMaxTokens ? `🤖 AUTO (${effectiveTokensUsed})` : maxTokens}
  • Repeat Penalty: ${repeatPenalty.toFixed(2)} (avoid repetition: 1.0=none, 2.0=strong)
  ${isGgufModel ? `• GPU Layers: ${nGpuLayers} (Metal GPU acceleration)` : ""}
  • Model: ${selectedModel} ${isGgufModel ? "(GGUF)" : "(HuggingFace)"}

Response Details:
  • Characters: ${responseText.length}
  • Lines: ${responseText.split("\n").length}
  • Words: ${responseText.split(/\s+/).length}
  • Estimated Tokens: ~${Math.ceil(responseText.split(/\s+/).length * 1.3)}
  • Empty Response: ${responseText.trim().length === 0 ? "YES ⚠️" : "No ✓"}
  • Timestamp: ${new Date().toLocaleTimeString()}`;

      setDebugInfo(debugInfoStr);

      const assistantMessage: Message = {
        id: String(messages.length + 1),
        sender: "assistant",
        content: responseText,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        id: String(messages.length + 1),
        sender: "assistant",
        content: `❌ 오류: ${error.response?.data?.detail || error.message || "알 수 없는 오류"}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      await chatAPI.clear_history();
      setMessages([]);
    } catch (error) {
      console.error("Failed to clear history", error);
    }
  };

  const handleRenameModel = (modelId: string) => {
    setEditingModelName(modelId);
    setNewModelName(modelCustomNames[modelId] || modelId.split("/").pop() || modelId);
  };

  const handleSaveModelName = (modelId: string) => {
    if (newModelName.trim()) {
      setModelCustomNames({
        ...modelCustomNames,
        [modelId]: newModelName,
      });
    }
    setEditingModelName(null);
    setNewModelName("");
  };

  const getDisplayModelName = (modelId: string) => {
    return modelCustomNames[modelId] || modelId.split("/").pop() || modelId;
  };

  // 캐시 모델과 로컬 모델을 합침
  const allAvailableModels = [...models];
  const uniqueModels = Array.from(new Map(allAvailableModels.map((model) => [model.model_id, model])).values());

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* 헤더 */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-2xl font-bold text-white">💬 LLM Chat Interface</h1>
      </div>

      {/* 메인 컨텐츠 (2컬럼) */}
      <div className="flex-1 flex overflow-hidden">
        {/* 왼쪽: 채팅 */}
        <div className="flex-1 flex flex-col border-r border-gray-700">
          {/* 메시지 창 */}
          <div className="flex-1 overflow-y-auto bg-gray-800 p-4">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-400 flex-col">
                <p className="text-lg mb-2">대화를 시작해보세요! 🚀</p>
                <p className="text-sm">우측 패널에서 모델을 선택하면 준비 완료!</p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`group relative max-w-xs lg:max-w-2xl px-4 py-2 rounded-lg ${msg.sender === "user" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-100"}`}>
                      {/* 복사 버튼 (Assistant 메시지만) */}
                      {msg.sender === "assistant" && (
                        <button
                          onClick={() => copyToClipboard(msg.content, msg.id)}
                          className={`absolute -top-8 right-0 px-2 py-1 rounded text-xs font-medium transition opacity-0 group-hover:opacity-100 ${
                            copiedMessageId === msg.id
                              ? "bg-green-600 text-white"
                              : "bg-gray-600 text-gray-200 hover:bg-gray-500"
                          }`}
                          title="Copy message"
                        >
                          {copiedMessageId === msg.id ? "✓ Copied" : "📋 Copy"}
                        </button>
                      )}

                      {/* 메시지 내용 - 적응형 높이 */}
                      <div className="markdown-content max-h-96 overflow-y-auto">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            code: CodeBlockRenderer as any,
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                      <p className={`text-xs mt-1 ${msg.sender === "user" ? "text-blue-100" : "text-gray-400"}`}>{msg.timestamp}</p>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-700 text-gray-100 px-4 py-2 rounded-lg">
                      <p>작성 중... ⏳</p>
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* 메시지 입력 */}
          <div className="bg-gray-700 border-t border-gray-600 p-4 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSend()}
              placeholder="메시지를 입력하세요..."
              className="flex-1 bg-gray-600 text-white px-4 py-2 rounded border border-gray-500 focus:outline-none focus:border-blue-500"
              disabled={!selectedModel || loading}
            />
            <button onClick={handleSend} disabled={!selectedModel || loading} className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? "작성 중..." : "전송"}
            </button>
          </div>
        </div>

        {/* 오른쪽: 설정 패널 */}
        <div ref={settingsPanelRef} className="bg-gray-800 border-l border-gray-700 overflow-y-auto flex flex-col" style={{ width: `${settingsPanelWidth}px` }}>
          {/* 리사이즈 핸들 */}
          <div
            onMouseDown={handleMouseDown}
            className={`absolute left-0 top-0 w-1 h-full bg-blue-500 cursor-col-resize hover:bg-blue-400 transition ${isResizing ? "bg-blue-400" : ""}`}
            style={{ left: "-2px" }}
          />

          {/* 설정 헤더 */}
          <div className="bg-gray-700 border-b border-gray-600 px-4 py-3 flex items-center justify-between sticky top-0">
            <h2 className="text-lg font-bold text-white">⚙️ Settings</h2>
            <button onClick={() => setShowAllSettings(!showAllSettings)} className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700">
              {showAllSettings ? "접기" : "펼치기"}
            </button>
          </div>

          {/* 설정 컨텐츠 */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* 모델 선택 */}
            <div className="bg-gray-700 rounded-lg p-3">
              <h3 className="text-sm font-bold text-white mb-2">🤖 모델 선택</h3>
              {currentModelName && <div className="mb-2 p-2 bg-green-900 border border-green-600 rounded text-sm text-green-100">✅ {getDisplayModelName(currentModelName)}</div>}
              <div className="flex flex-col gap-2 max-h-40 overflow-y-auto">
                {loadingModels ? (
                  <p className="text-xs text-gray-400">로딩 중...</p>
                ) : uniqueModels.length === 0 ? (
                  <p className="text-xs text-gray-400">모델 없음</p>
                ) : (
                  uniqueModels.map((model) => (
                    <div key={model.model_id} className="flex items-center gap-2">
                      <button
                        onClick={() => handleLoadModel(model.path, model.model_id)}
                        disabled={modelLoading !== null}
                        className={`flex-1 px-2 py-1 rounded text-xs transition ${
                          selectedModel === model.model_id ? "bg-blue-600 text-white" : "bg-gray-600 text-gray-200 hover:bg-gray-500"
                        } disabled:opacity-50`}
                      >
                        {modelLoading === model.model_id ? `⏳ ${Math.min(Math.round(modelLoadProgress), 100)}%` : getDisplayModelName(model.model_id)}
                      </button>
                      <button onClick={() => handleRenameModel(model.model_id)} className="px-2 py-1 bg-gray-600 text-gray-200 rounded hover:bg-gray-500 text-xs" title="이름 변경">
                        ✎
                      </button>
                    </div>
                  ))
                )}
              </div>

              {/* 이름 변경 다이얼로그 */}
              {editingModelName && (
                <div className="mt-3 p-2 bg-gray-600 rounded border border-gray-500">
                  <input
                    type="text"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    placeholder="모델 이름"
                    className="w-full px-2 py-1 bg-gray-700 text-white rounded text-xs mb-2 border border-gray-500"
                  />
                  <div className="flex gap-2">
                    <button onClick={() => handleSaveModelName(editingModelName)} className="flex-1 px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700">
                      저장
                    </button>
                    <button onClick={() => setEditingModelName(null)} className="flex-1 px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700">
                      취소
                    </button>
                  </div>
                </div>
              )}

              {/* 모델 로드 상태 표시 */}
              {modelLoading && modelLoadStatus && (
                <div className="mt-3 p-2 bg-blue-900 rounded border border-blue-600 text-xs">
                  <p className="text-blue-200 mb-2 font-semibold">📥 로드 중...</p>
                  <p className="text-blue-100 text-xs mb-2">{modelLoadStatus}</p>
                  <div className="w-full bg-blue-800 rounded-full h-2">
                    <div
                      className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(modelLoadProgress, 100)}%` }}
                    />
                  </div>
                  <p className="text-blue-300 mt-1 text-xs text-right">{Math.round(modelLoadProgress)}%</p>
                </div>
              )}
            </div>

            {/* 파라미터 설정 */}
            {showAllSettings && (
              <>
                {/* Temperature */}
                <div className="bg-gray-700 rounded-lg p-3">
                  <label className="text-sm font-bold text-white block mb-2">
                    Temperature: <span className="text-blue-400">{temperature.toFixed(2)}</span>
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                    disabled={!selectedModel}
                  />
                </div>

                {/* Top P */}
                <div className="bg-gray-700 rounded-lg p-3">
                  <label className="text-sm font-bold text-white block mb-2">
                    Top P: <span className="text-blue-400">{topP.toFixed(2)}</span>
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                    className="w-full bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                    disabled={!selectedModel}
                  />
                </div>

                {/* Max Tokens */}
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-bold text-white">
                      Max Tokens: <span className={autoMaxTokens ? "text-yellow-400" : "text-green-400"}>{autoMaxTokens ? "🤖 AUTO" : maxTokens}</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoMaxTokens}
                        onChange={(e) => setAutoMaxTokens(e.target.checked)}
                        className="w-4 h-4"
                        disabled={!selectedModel}
                      />
                      <span className="text-xs text-gray-300">Auto</span>
                    </label>
                  </div>

                  {/* Manual Mode */}
                  <div className={`flex gap-2 mb-3 ${autoMaxTokens ? "opacity-30 pointer-events-none" : "opacity-100"}`}>
                    <input
                      type="range"
                      min="256"
                      max="8192"
                      step="256"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="flex-1"
                      disabled={!selectedModel || autoMaxTokens}
                    />
                    <input
                      type="number"
                      min="256"
                      max="8192"
                      step="256"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-20 bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-green-500 text-sm"
                      disabled={!selectedModel || autoMaxTokens}
                    />
                  </div>
                  <div className={`flex justify-between text-xs text-gray-400 mb-3 ${autoMaxTokens ? "opacity-30" : "opacity-100"}`}>
                    <span>256</span>
                    <span>8192</span>
                  </div>

                  {/* Auto Mode Settings */}
                  {autoMaxTokens && (
                    <div className="bg-gray-800 rounded p-2 mb-2 border border-yellow-500">
                      <p className="text-xs font-bold text-yellow-400 mb-2">⚙️ Auto Mode 범위 설정</p>
                      <div className="space-y-2">
                        {/* Min Tokens */}
                        <div className="flex items-center gap-2">
                          <label className="text-xs text-gray-300 w-12">최소:</label>
                          <input
                            type="range"
                            min="256"
                            max={autoMaxTokensValue - 256}
                            step="256"
                            value={autoMinTokens}
                            onChange={(e) => setAutoMinTokens(parseInt(e.target.value))}
                            className="flex-1"
                            disabled={!selectedModel}
                          />
                          <input
                            type="number"
                            min="256"
                            max={autoMaxTokensValue - 256}
                            step="256"
                            value={autoMinTokens}
                            onChange={(e) => setAutoMinTokens(parseInt(e.target.value))}
                            className="w-16 bg-gray-700 text-white px-1 py-0.5 rounded border border-gray-500 text-xs"
                            disabled={!selectedModel}
                          />
                        </div>

                        {/* Max Tokens */}
                        <div className="flex items-center gap-2">
                          <label className="text-xs text-gray-300 w-12">최대:</label>
                          <input
                            type="range"
                            min={autoMinTokens + 256}
                            max="8192"
                            step="256"
                            value={autoMaxTokensValue}
                            onChange={(e) => setAutoMaxTokensValue(parseInt(e.target.value))}
                            className="flex-1"
                            disabled={!selectedModel}
                          />
                          <input
                            type="number"
                            min={autoMinTokens + 256}
                            max="8192"
                            step="256"
                            value={autoMaxTokensValue}
                            onChange={(e) => setAutoMaxTokensValue(parseInt(e.target.value))}
                            className="w-16 bg-gray-700 text-white px-1 py-0.5 rounded border border-gray-500 text-xs"
                            disabled={!selectedModel}
                          />
                        </div>
                      </div>
                      <p className="text-xs text-yellow-300 mt-2">💡 메시지 길이에 따라 {autoMinTokens}~{autoMaxTokensValue} 사이에서 자동 조정</p>
                    </div>
                  )}

                  {autoMaxTokens ? (
                    <p className="text-xs text-yellow-400">✨ Auto Mode: 메시지 길이에 따라 자동으로 조정됩니다</p>
                  ) : (
                    <p className="text-xs text-gray-400">💡 더 높은 값 = 더 길고 상세한 응답 (생성 시간 증가, 메모리 사용 증가)</p>
                  )}
                </div>

                {/* Max Response Length - 선택적 (필요시만 사용) */}
                <details className="bg-gray-700 rounded-lg p-3">
                  <summary className="text-sm font-bold text-white cursor-pointer hover:text-gray-300">
                    📏 Max Response Length (고급 설정)
                  </summary>
                  <div className="mt-3 pt-3 border-t border-gray-600 space-y-2">
                    <input
                      type="number"
                      min="100"
                      max="5000"
                      step="100"
                      value={maxResponseLength}
                      onChange={(e) => setMaxResponseLength(parseInt(e.target.value))}
                      className="w-full bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                      disabled={!selectedModel}
                    />
                    <p className="text-xs text-gray-400">💡 UI에 표시할 응답의 최대 길이 (일반적으로 필요 없음)</p>
                  </div>
                </details>

                {/* N GPU Layers (GGUF만) */}
                {isGgufModel && (
                  <div className="bg-gray-700 rounded-lg p-3 border border-green-500">
                    <label className="text-sm font-bold text-white block mb-2">
                      🎮 GPU Layers (GGUF): <span className="text-green-400">{nGpuLayers}</span>
                      {modelSizeGb > 0 && <span className="text-xs text-gray-400 ml-2">(모델: {modelSizeGb.toFixed(2)}GB)</span>}
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="1"
                      value={nGpuLayers}
                      onChange={(e) => setNGpuLayers(parseInt(e.target.value))}
                      className="w-full bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-green-500 text-sm"
                      disabled={!selectedModel}
                    />
                    <p className="text-xs text-gray-400 mt-1">💡 0 = CPU only | 높을수록 GPU 사용 | 자동으로 추천값 설정됨</p>
                  </div>
                )}

                {/* System Prompt */}
                <div className="bg-gray-700 rounded-lg p-3">
                  <button onClick={() => setShowPromptEditor(!showPromptEditor)} className="w-full text-sm font-bold text-white mb-2 hover:text-blue-400 flex items-center justify-between">
                    <span>🎭 System Prompt</span>
                    <span>{showPromptEditor ? "▼" : "▶"}</span>
                  </button>
                  {showPromptEditor && (
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      className="w-full bg-gray-600 text-white px-2 py-2 rounded border border-gray-500 focus:outline-none focus:border-blue-500 text-xs resize-none"
                      rows={4}
                    />
                  )}
                </div>

                {/* 요청 파라미터 Expander */}
                <div className="bg-gray-700 rounded-lg border border-gray-600">
                  <button onClick={() => setShowParameters(!showParameters)} className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-600 transition">
                    <span className="text-sm font-bold text-white">{showParameters ? "▼" : "▶"} Request Parameters</span>
                  </button>
                  {showParameters && (
                    <div className="bg-gray-800 p-3 border-t border-gray-600">
                      <pre className="bg-gray-900 text-gray-100 p-2 rounded font-mono text-xs overflow-x-auto">{`{
  "message": "...",
  "temperature": ${temperature.toFixed(2)},
  "top_p": ${topP.toFixed(2)},
  "max_tokens": ${autoMaxTokens ? `auto (${autoMinTokens}-${autoMaxTokensValue})` : maxTokens},
  "auto_mode": ${autoMaxTokens},
  ${autoMaxTokens ? `"auto_min_tokens": ${autoMinTokens},\n  "auto_max_tokens": ${autoMaxTokensValue},\n  ` : ""}
  "repeat_penalty": ${repeatPenalty.toFixed(2)},
  "n_gpu_layers": ${nGpuLayers},
  "maintain_history": true,
  "model": "${selectedModel || "N/A"}",
  "model_type": "${isGgufModel ? "GGUF" : "HuggingFace"}",
  "request_type": "chat.completion"
}`}</pre>
                    </div>
                  )}
                </div>

                {/* 응답 정보 Expander */}
                {debugInfo && (
                  <div className="bg-gray-700 rounded-lg border border-green-600">
                    <button onClick={() => setDebugMode(!debugMode)} className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-600 transition">
                      <span className="text-sm font-bold text-white">{debugMode ? "▼" : "▶"} Response Info</span>
                    </button>
                    {debugMode && (
                      <div className="bg-gray-800 p-3 border-t border-green-600">
                        <pre className="bg-gray-900 text-green-400 p-2 rounded font-mono text-xs overflow-x-auto">{debugInfo}</pre>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          {/* 하단 버튼 */}
          <div className="bg-gray-700 border-t border-gray-600 p-3 space-y-2">
            <button onClick={handleClear} className="w-full bg-red-600 text-white px-3 py-2 rounded hover:bg-red-700 text-sm font-medium">
              🗑️ 대화 초기화
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
