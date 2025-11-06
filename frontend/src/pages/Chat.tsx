import React, { useState, useEffect, useRef } from "react";
import { chatAPI } from "../utils/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
  const settingsPanelRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
      const response = await chatAPI.chat(input, topP, temperature, maxTokens, repeatPenalty, nGpuLayers);
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

      // 최대 응답 길이 제한
      if (responseText.length > maxResponseLength) {
        responseText = responseText.substring(0, maxResponseLength) + "...(길이 초과로 생략)";
      }

      // 디버그 정보 생성 및 저장
      const debugInfoStr = `Request Parameters:
  • Temperature: ${temperature.toFixed(2)} (creativity: 0=deterministic, 2=creative)
  • Top P: ${topP.toFixed(2)} (diversity: 0=focused, 1=diverse)
  • Max Tokens: ${maxTokens} (max response length in tokens)
  • Repeat Penalty: ${repeatPenalty.toFixed(2)} (avoid repetition: 1.0=none, 2.0=strong)
  ${isGgufModel ? `• GPU Layers: ${nGpuLayers} (Metal GPU acceleration)` : ""}
  • Model: ${selectedModel} ${isGgufModel ? "(GGUF)" : "(HuggingFace)"}

Response Details:
  • Characters: ${responseText.length} / ${maxResponseLength}
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
                    <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${msg.sender === "user" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-100"}`}>
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
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
                  <label className="text-sm font-bold text-white block mb-2">
                    Max Tokens: <span className="text-green-400">{maxTokens}</span>
                  </label>
                  <div className="flex gap-2">
                    <input type="range" min="256" max="8192" step="256" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value))} className="flex-1" disabled={!selectedModel} />
                    <input
                      type="number"
                      min="256"
                      max="8192"
                      step="256"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-20 bg-gray-600 text-white px-2 py-1 rounded border border-gray-500 focus:outline-none focus:border-green-500 text-sm"
                      disabled={!selectedModel}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>256</span>
                    <span>8192</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">💡 더 높은 값 = 더 길고 상세한 응답 (생성 시간 증가, 메모리 사용 증가)</p>
                </div>

                {/* Max Length */}
                <div className="bg-gray-700 rounded-lg p-3">
                  <label className="text-sm font-bold text-white block mb-2">
                    Max Response Length: <span className="text-blue-400">{maxResponseLength}</span>
                  </label>
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
                  <p className="text-xs text-gray-400 mt-1">응답을 UI에 표시할 때의 최대 길이</p>
                </div>

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
  "max_tokens": ${maxTokens},
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
