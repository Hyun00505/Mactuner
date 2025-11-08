/**
 * Editor 페이지
 * ComfyUI 스타일의 노드 기반 워크플로우 에디터
 */

import React, { useState, useRef, useEffect } from "react";
import { useEditorStore } from "../stores/editorStore";
import { WorkflowCanvas } from "../components/Editor/WorkflowCanvas";
import { WorkflowGuide } from "../components/Editor/WorkflowGuide";
import { TrainingProgressPanel } from "../components/Editor/TrainingProgressPanel";
import { NodeType } from "../types/editor";
import { workflowToNotebook, downloadNotebook } from "../utils/workflowToNotebook";

export const Editor: React.FC = () => {
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showWorkflowGuide, setShowWorkflowGuide] = useState(false);
  const [showWorkflowList, setShowWorkflowList] = useState(false);
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDescription, setWorkflowDescription] = useState("");
  const [nodeSearchQuery, setNodeSearchQuery] = useState("");
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    config: true,
    input: true,
    process: true,
    output: true,
    utility: true,
  });
  const consoleRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [copied, setCopied] = useState(false);

  const {
    currentWorkflow,
    nodes,
    connections,
    showNodePalette,
    showOutputPanel,
    isDirty,
    isExecuting,
    executionLog,
    addNode,
    executeWorkflow,
    saveWorkflow,
    createNewWorkflow,
    createExampleWorkflow,
    loadWorkflowById,
    listWorkflows,
    toggleNodePalette,
    toggleOutputPanel,
    exportWorkflowToJSON,
    exportWorkflowToFile,
    importWorkflowFromFile,
  } = useEditorStore();

  // 콘솔 자동 스크롤
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [executionLog]);

  // 콘솔 출력 내용을 텍스트로 변환
  const getConsoleText = (): string => {
    if (!executionLog) return "";

    let text = "═══════════════════════════════════════════════════════════\n";
    text += "Workflow Execution Log\n";
    text += "═══════════════════════════════════════════════════════════\n\n";
    text += "[INFO]\n";
    text += `▶ Execution started at: ${new Date(executionLog.startTime).toLocaleString("ko-KR")}\n`;
    text += `▶ Workflow ID: ${executionLog.workflowId}\n`;
    text += `▶ Total nodes: ${executionLog.nodeExecutions.length}\n\n`;
    text += "─────────────────────────────────────────────────────────────\n\n";

    executionLog.nodeExecutions.forEach((nodeExec, idx) => {
      const node = nodes.find((n) => n.id === nodeExec.nodeId);
      const nodeName = node ? `${node.type} (${node.id})` : nodeExec.nodeId;
      const status = nodeExec.status === "error" ? "❌ ERROR" : nodeExec.status === "completed" ? "✅ SUCCESS" : nodeExec.status === "running" ? "⏳ RUNNING" : "⏸ PENDING";
      
      text += `[${idx + 1}] ${status} - ${nodeName}\n`;
      
      if (nodeExec.startTime) {
        text += `Started: ${new Date(nodeExec.startTime).toLocaleTimeString("ko-KR")}\n`;
      }
      if (nodeExec.endTime) {
        text += `Ended: ${new Date(nodeExec.endTime).toLocaleTimeString("ko-KR")}${nodeExec.duration ? ` (${nodeExec.duration}ms)` : ""}\n`;
      }
      
      if (nodeExec.error) {
        text += `\nError Details:\n${nodeExec.error}\n`;
      }
      
      if (nodeExec.trainingLogs && nodeExec.trainingLogs.length > 0) {
        text += "\n📊 학습 진행 상황:\n";
        nodeExec.trainingLogs.forEach((log) => {
          text += `[${new Date(log.timestamp).toLocaleTimeString("ko-KR")}] ${log.message}`;
          if (log.data?.loss !== undefined) text += ` | Loss: ${log.data.loss.toFixed(4)}`;
          if (log.data?.step !== undefined) text += ` | Step: ${log.data.step}`;
          if (log.data?.progress !== undefined) text += ` | Progress: ${log.data.progress}%`;
          text += "\n";
        });
      }
      
      if (nodeExec.outputs && Object.keys(nodeExec.outputs).length > 0) {
        text += `\nOutput:\n${JSON.stringify(nodeExec.outputs, null, 2)}\n`;
      }
      
      text += "\n";
    });

    text += "─────────────────────────────────────────────────────────────\n\n";
    
    if (executionLog.endTime) {
      const result = executionLog.status === "error" ? "❌ FAILED" : "✅ COMPLETED";
      text += `[RESULT]\n${result} Execution ${executionLog.status === "error" ? "FAILED" : "COMPLETED"} at: ${new Date(executionLog.endTime).toLocaleString("ko-KR")}\n`;
      if (executionLog.totalDuration) {
        text += `⏱️ Total duration: ${executionLog.totalDuration}ms\n`;
      }
    }

    text += "\n═══════════════════════════════════════════════════════════\n";

    return text;
  };

  // 복사 버튼 클릭 핸들러
  const handleCopyConsole = async () => {
    const text = getConsoleText();
    if (text) {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error("복사 실패:", err);
      }
    }
  };

  const nodeTypes: Array<{ type: NodeType; label: string; icon: string; description: string; category: string }> = [
    // 설정 노드
    { type: "hf-token", label: "HF 토큰", icon: "🔑", description: "토큰 설정 (필수)", category: "config" },
    { type: "device-selector", label: "디바이스 선택", icon: "🖥️", description: "CPU/MPS/CUDA 선택", category: "config" },

    // 입력 노드 (모델)
    { type: "local-model-loader", label: "로컬 모델", icon: "📂", description: "로컬 모델 로드", category: "input" },
    { type: "hf-model-downloader", label: "HF 모델 다운로드", icon: "🤗", description: "HF에서 모델 다운로드", category: "input" },

    // 입력 노드 (데이터셋)
    { type: "local-dataset-loader", label: "로컬 데이터셋", icon: "📂", description: "로컬 데이터셋 로드", category: "input" },
    { type: "hf-dataset-downloader", label: "HF 데이터셋 다운로드", icon: "🤗", description: "HF에서 데이터셋 다운로드", category: "input" },

    // 전처리 노드
    { type: "dataset-preprocessor", label: "데이터 전처리", icon: "🔧", description: "토큰화 및 포맷 변환", category: "process" },
    { type: "dataset-splitter", label: "데이터 분할", icon: "✂️", description: "Train/Val/Test 분할", category: "process" },
    { type: "dataset-filter", label: "데이터 필터", icon: "🔍", description: "품질 필터링", category: "process" },

    // 설정 노드 (학습)
    { type: "training-config", label: "학습 설정", icon: "⚙️", description: "하이퍼파라미터 설정", category: "config" },
    { type: "lora-config", label: "LoRA 설정", icon: "🎯", description: "LoRA 파라미터", category: "config" },
    { type: "qlora-config", label: "QLoRA 설정", icon: "⚡", description: "QLoRA 파라미터", category: "config" },

    // 학습 노드
    { type: "training", label: "학습 실행", icon: "🎓", description: "LoRA/QLoRA 학습", category: "process" },

    // 평가/관리 노드
    { type: "model-evaluator", label: "모델 평가", icon: "📊", description: "성능 평가", category: "process" },
    { type: "checkpoint-manager", label: "체크포인트 관리", icon: "💾", description: "체크포인트 관리", category: "process" },
    { type: "model-saver", label: "모델 저장", icon: "💿", description: "모델 저장", category: "output" },

    // 유틸리티 노드
    { type: "chat", label: "채팅", icon: "💬", description: "모델과 대화", category: "utility" },
    { type: "rag", label: "RAG", icon: "🔍", description: "문서 검색", category: "utility" },
    { type: "gguf-export", label: "GGUF 내보내기", icon: "📦", description: "모델 양자화", category: "output" },
  ];

  const handleAddNode = (type: NodeType) => {
    addNode(type, { x: 200, y: 200 });
  };

  const handleSaveWorkflow = async () => {
    if (!workflowName.trim()) {
      alert("워크플로우 이름을 입력하세요");
      return;
    }
    await saveWorkflow(workflowName, workflowDescription);
    setShowSaveModal(false);
    setWorkflowName("");
    setWorkflowDescription("");
  };

  const handleExecute = async () => {
    if (nodes.length === 0) {
      alert("❌ 최소 1개 이상의 노드가 필요합니다");
      return;
    }
    await executeWorkflow();
  };

  const handleExportJSON = () => {
    exportWorkflowToJSON();
  };

  const handleSaveToFile = async () => {
    try {
      await exportWorkflowToFile();
    } catch (error: any) {
      console.error("Failed to save workflow:", error);
    }
  };

  const handleLoadFromFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await importWorkflowFromFile(file);
      alert("워크플로우를 성공적으로 불러왔습니다!");
    } catch (error: any) {
      alert(`워크플로우 불러오기 실패: ${error.message}`);
    } finally {
      // 파일 입력 초기화
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleExportToNotebook = () => {
    try {
      if (nodes.length === 0) {
        alert("워크플로우에 노드가 없습니다. 노드를 추가한 후 다시 시도하세요.");
        return;
      }

      const notebook = workflowToNotebook(nodes, connections);
      const filename = `${currentWorkflow?.name || "workflow"}_${Date.now()}.ipynb`;
      downloadNotebook(notebook, filename);
      alert(`✅ Jupyter Notebook 파일이 다운로드되었습니다: ${filename}`);
    } catch (error: any) {
      console.error("Notebook 생성 실패:", error);
      alert(`Notebook 생성 실패: ${error.message}`);
    }
  };

  return (
    <div className="h-screen bg-gray-900 flex flex-col overflow-hidden">
      {/* 헤더 */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-white">🎨 Workflow Editor</h1>
          <span className="text-sm text-gray-400">{currentWorkflow?.name || "새 워크플로우"}</span>
          {isDirty && <span className="text-xs bg-yellow-500 text-white px-2 py-1 rounded">수정됨</span>}
        </div>

        {/* 액션 버튼 */}
        <div className="flex items-center gap-3">
          <button onClick={() => setShowWorkflowGuide(true)} className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-semibold transition-colors">
            📖 가이드
          </button>

          <button onClick={() => setShowWorkflowList(true)} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold transition-colors">
            📚 워크플로우 목록
          </button>

          <button onClick={createExampleWorkflow} className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm font-semibold transition-colors">
            ⭐ 예시 워크플로우
          </button>

          <button onClick={createNewWorkflow} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-semibold transition-colors">
            📄 새로 만들기
          </button>

          <button
            onClick={handleSaveToFile}
            disabled={!isDirty}
            className={`
              px-4 py-2 rounded-lg text-sm font-semibold transition-colors
              ${isDirty ? "bg-blue-600 hover:bg-blue-700 text-white" : "bg-gray-700 text-gray-500 cursor-not-allowed"}
            `}
          >
            💾 저장
          </button>

          <button
            onClick={handleExecute}
            disabled={isExecuting || nodes.length === 0}
            className={`
              px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2
              ${isExecuting || nodes.length === 0 ? "bg-gray-700 text-gray-500 cursor-not-allowed" : "bg-green-600 hover:bg-green-700 text-white"}
            `}
          >
            {isExecuting ? <>⏳ 실행 중...</> : <>▶ 실행</>}
          </button>

          <button
            onClick={handleExportToNotebook}
            disabled={nodes.length === 0}
            className={`
              px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2
              ${nodes.length === 0 ? "bg-gray-700 text-gray-500 cursor-not-allowed" : "bg-orange-600 hover:bg-orange-700 text-white"}
            `}
            title="워크플로우를 Google Colab용 Jupyter Notebook으로 변환"
          >
            📓 학습 파이프라인 ipynb로 추출하기
          </button>

          <button
            onClick={toggleNodePalette}
            className={`
              px-4 py-2 rounded-lg text-sm font-semibold transition-colors
              ${showNodePalette ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300"}
            `}
          >
            📦 노드
          </button>

          <button
            onClick={toggleOutputPanel}
            className={`
              px-4 py-2 rounded-lg text-sm font-semibold transition-colors
              ${showOutputPanel ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300"}
            `}
          >
            📤 출력
          </button>

          <button onClick={handleExportJSON} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-semibold transition-colors">
            💾 JSON 저장
          </button>

          <input type="file" ref={fileInputRef} accept=".json" onChange={handleLoadFromFile} className="hidden" id="import-json-input" />
          <label htmlFor="import-json-input" className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-sm font-semibold transition-colors cursor-pointer">
            📂 JSON 불러오기
          </label>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="flex-1 flex gap-4 p-4 overflow-hidden min-h-0">
        {/* 왼쪽 패널: 노드 팔레트 */}
        {showNodePalette && (
          <div className="w-80 bg-gray-800 rounded-lg shadow-lg overflow-hidden flex flex-col border border-gray-700">
            <div className="bg-gray-700 px-4 py-3 border-b border-gray-600">
              <h2 className="text-sm font-bold text-white mb-2">📦 노드 팔레트</h2>
              {/* 검색 입력 */}
              <input
                type="text"
                value={nodeSearchQuery}
                onChange={(e) => setNodeSearchQuery(e.target.value)}
                placeholder="🔍 노드 검색..."
                className="w-full px-3 py-2 bg-gray-600 border border-gray-500 rounded text-white text-sm placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
              {(() => {
                // 검색어로 필터링
                const filteredNodes = nodeSearchQuery
                  ? nodeTypes.filter(
                      (node) =>
                        node.label.toLowerCase().includes(nodeSearchQuery.toLowerCase()) ||
                        node.description.toLowerCase().includes(nodeSearchQuery.toLowerCase()) ||
                        node.type.toLowerCase().includes(nodeSearchQuery.toLowerCase())
                    )
                  : nodeTypes;

                // 카테고리별로 노드 그룹화
                const categories: Record<string, Array<{ type: NodeType; label: string; icon: string; description: string; category: string }>> = {};
                filteredNodes.forEach((node) => {
                  if (!categories[node.category]) {
                    categories[node.category] = [];
                  }
                  categories[node.category].push(node);
                });

                const categoryLabels: Record<string, string> = {
                  config: "⚙️ 설정",
                  input: "📥 입력",
                  process: "🔄 처리",
                  output: "📤 출력",
                  utility: "🛠️ 유틸리티",
                };

                if (Object.keys(categories).length === 0) {
                  return (
                    <div className="text-center py-12 text-gray-500">
                      <p className="text-sm">검색 결과가 없습니다</p>
                      <p className="text-xs mt-1">다른 키워드로 검색해보세요</p>
                    </div>
                  );
                }

                return Object.entries(categories).map(([category, nodes]) => {
                  const isExpanded = expandedCategories[category] ?? true;

                  return (
                    <div key={category} className="space-y-2">
                      <button
                        onClick={() => setExpandedCategories((prev) => ({ ...prev, [category]: !isExpanded }))}
                        className="w-full flex items-center justify-between px-2 py-2 hover:bg-gray-700 rounded-lg transition-colors group"
                      >
                        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider group-hover:text-gray-300">{categoryLabels[category] || category}</h3>
                        <span className={`text-gray-500 text-xs transition-transform ${isExpanded ? "rotate-90" : ""}`}>▶</span>
                      </button>
                      {isExpanded && (
                        <div className="space-y-1.5 pl-2">
                          {nodes.map(({ type, label, icon, description }) => (
                            <button
                              key={type}
                              onClick={() => handleAddNode(type)}
                              className="
                                w-full p-2.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-left
                                transition-colors border border-gray-600 hover:border-blue-500
                                group
                              "
                            >
                              <div className="flex items-center gap-2.5">
                                <span className="text-xl">{icon}</span>
                                <div className="flex-1 min-w-0">
                                  <p className="font-semibold text-white text-sm truncate">{label}</p>
                                  <p className="text-xs text-gray-400 truncate">{description}</p>
                                </div>
                                <span className="text-sm opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">➕</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                });
              })()}
            </div>

            {/* 통계 */}
            <div className="bg-gray-700 border-t border-gray-600 px-4 py-3 space-y-1 text-xs text-gray-300">
              <p>📍 노드: {nodes.length}</p>
              <p>🔗 연결: {connections.length}</p>
            </div>
          </div>
        )}

        {/* 중앙: 워크플로우 캔버스 */}
        <div className="flex-1 bg-gray-800 rounded-lg shadow-lg border border-gray-700 overflow-hidden">
          <WorkflowCanvas />
        </div>

        {/* 오른쪽 패널: 콘솔 출력 (전체 화면) */}
        <div className="w-96 flex flex-col min-h-0">
          {/* 출력 패널 - 전체 화면 */}
          <div className="bg-black rounded-lg shadow-lg border border-gray-700 overflow-hidden flex flex-col h-full">
            <div className="bg-gray-900 px-4 py-2 border-b border-gray-700 flex justify-between items-center flex-shrink-0">
              <h2 className="text-sm font-bold text-green-400 font-mono">📤 Console Output</h2>
              <div className="flex items-center gap-2">
                {executionLog && (
                  <span className={`text-xs font-mono ${executionLog.status === "error" ? "text-red-400" : executionLog.status === "completed" ? "text-green-400" : "text-yellow-400"}`}>
                    [{executionLog.status.toUpperCase()}]
                  </span>
                )}
                {executionLog && (
                  <button
                    onClick={handleCopyConsole}
                    className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors flex items-center gap-1"
                    title="콘솔 출력 복사"
                  >
                    {copied ? (
                      <>
                        <span>✓</span>
                        <span>복사됨</span>
                      </>
                    ) : (
                      <>
                        <span>📋</span>
                        <span>복사</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto overflow-x-auto p-4 text-xs font-mono text-green-400 space-y-1 bg-black" style={{ minHeight: 0, scrollbarWidth: "thin", scrollbarColor: "#4b5563 #1f2937" }} ref={consoleRef}>
              {/* 학습 진행 상황 패널 (학습이 시작되면 위에 표시) */}
              <TrainingProgressPanel executionLog={executionLog} nodes={nodes} />

              {executionLog ? (
                <>
                  {/* 헤더 */}
                  <div className="text-gray-500 mb-2">
                    <p>═══════════════════════════════════════════════════════════</p>
                    <p>Workflow Execution Log</p>
                    <p>═══════════════════════════════════════════════════════════</p>
                  </div>

                  {/* 시작 정보 */}
                  <div className="text-green-400">
                    <p className="text-gray-500">[INFO]</p>
                    <p>▶ Execution started at: {new Date(executionLog.startTime).toLocaleString("ko-KR")}</p>
                    <p>▶ Workflow ID: {executionLog.workflowId}</p>
                    <p>▶ Total nodes: {executionLog.nodeExecutions.length}</p>
                  </div>

                  <div className="text-gray-500 my-2">─────────────────────────────────────────────────────────────</div>

                  {/* 노드별 실행 로그 */}
                  {executionLog.nodeExecutions.map((nodeExec, idx) => {
                    const node = nodes.find((n) => n.id === nodeExec.nodeId);
                    const nodeName = node ? `${node.type} (${node.id})` : nodeExec.nodeId;
                    const isError = nodeExec.status === "error";
                    const isCompleted = nodeExec.status === "completed";
                    const isRunning = nodeExec.status === "running";

                    return (
                      <div key={nodeExec.nodeId} className={`mb-3 ${isError ? "text-red-400" : isCompleted ? "text-green-400" : isRunning ? "text-yellow-400" : "text-gray-400"}`}>
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">[{idx + 1}]</span>
                          <div className="flex-1">
                            <p className="font-semibold">
                              {isError ? "❌ ERROR" : isCompleted ? "✅ SUCCESS" : isRunning ? "⏳ RUNNING" : "⏸ PENDING"} - {nodeName}
                            </p>

                            {nodeExec.startTime && <p className="text-gray-500 text-[10px] mt-1">Started: {new Date(nodeExec.startTime).toLocaleTimeString("ko-KR")}</p>}

                            {nodeExec.endTime && (
                              <p className="text-gray-500 text-[10px]">
                                Ended: {new Date(nodeExec.endTime).toLocaleTimeString("ko-KR")}
                                {nodeExec.duration && ` (${nodeExec.duration}ms)`}
                              </p>
                            )}

                            {/* 에러 메시지 */}
                            {isError && nodeExec.error && (
                              <div className="mt-2 p-2 bg-red-900 bg-opacity-30 border border-red-700 rounded text-red-300">
                                <p className="font-semibold text-red-400">Error Details:</p>
                                <pre className="whitespace-pre-wrap text-[10px] mt-1">{nodeExec.error}</pre>
                              </div>
                            )}

                            {/* 학습 진행 로그 (Training 노드인 경우) */}
                            {nodeExec.trainingLogs && nodeExec.trainingLogs.length > 0 && (
                              <div className="mt-2 space-y-2">
                                <p className="text-xs text-blue-400 font-semibold">📊 학습 진행 상황:</p>

                                {/* 학습 곡선 (Loss) */}
                                {nodeExec.trainingLogs.some((log) => log.data?.loss !== undefined) && (
                                  <div className="mt-2 p-2 bg-blue-900 bg-opacity-20 border border-blue-700 rounded">
                                    <p className="text-xs text-blue-300 font-semibold mb-1">📈 Loss 곡선:</p>
                                    <div className="flex items-end gap-1 h-20">
                                      {nodeExec.trainingLogs
                                        .filter((log) => log.data?.loss !== undefined)
                                        .map((log, idx) => {
                                          const loss = log.data.loss;
                                          const maxLoss = Math.max(...nodeExec.trainingLogs!.filter((l) => l.data?.loss !== undefined).map((l) => l.data.loss));
                                          const height = maxLoss > 0 ? (loss / maxLoss) * 100 : 0;
                                          return (
                                            <div
                                              key={idx}
                                              className="flex-1 bg-blue-500 hover:bg-blue-400 transition-colors rounded-t"
                                              style={{ height: `${Math.max(height, 5)}%` }}
                                              title={`Step ${log.data?.step || idx}: Loss ${loss.toFixed(4)}`}
                                            />
                                          );
                                        })}
                                    </div>
                                    <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                                      <span>Step 0</span>
                                      <span>Step {nodeExec.trainingLogs.filter((l) => l.data?.step !== undefined).length}</span>
                                    </div>
                                  </div>
                                )}

                                {/* 학습 로그 목록 */}
                                <div className="space-y-1 max-h-40 overflow-y-auto">
                                  {nodeExec.trainingLogs.map((log, logIdx) => (
                                    <div key={logIdx} className="text-xs text-blue-300 pl-2 border-l-2 border-blue-600">
                                      <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString("ko-KR")}]</span> {log.message}
                                      {log.data?.loss !== undefined && <span className="text-yellow-400 ml-2">Loss: {log.data.loss.toFixed(4)}</span>}
                                      {log.data?.step !== undefined && <span className="text-cyan-400 ml-2">Step: {log.data.step}</span>}
                                      {log.data?.progress !== undefined && <span className="text-green-400 ml-2">Progress: {log.data.progress}%</span>}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* 출력 데이터 (성공 시) */}
                            {isCompleted && nodeExec.outputs && Object.keys(nodeExec.outputs).length > 0 && (
                              <div className="mt-2 p-2 bg-green-900 bg-opacity-20 border border-green-700 rounded text-green-300">
                                <p className="font-semibold text-green-400 text-[10px]">Output:</p>
                                <pre className="whitespace-pre-wrap text-[10px] mt-1 overflow-x-auto">{JSON.stringify(nodeExec.outputs, null, 2)}</pre>
                              </div>
                            )}

                            {/* 입력 데이터 (디버깅용) */}
                            {nodeExec.inputData && Object.keys(nodeExec.inputData).length > 0 && (
                              <details className="mt-1">
                                <summary className="text-gray-500 text-[10px] cursor-pointer hover:text-gray-400">[View Input Data]</summary>
                                <pre className="whitespace-pre-wrap text-[10px] mt-1 text-gray-600 overflow-x-auto">{JSON.stringify(nodeExec.inputData, null, 2)}</pre>
                              </details>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  <div className="text-gray-500 my-2">─────────────────────────────────────────────────────────────</div>

                  {/* 종료 정보 */}
                  {executionLog.endTime && (
                    <div className={`${executionLog.status === "error" ? "text-red-400" : "text-green-400"}`}>
                      <p className="text-gray-500">[RESULT]</p>
                      <p>
                        {executionLog.status === "error" ? "❌" : "✅"} Execution {executionLog.status === "error" ? "FAILED" : "COMPLETED"} at:{" "}
                        {new Date(executionLog.endTime).toLocaleString("ko-KR")}
                      </p>
                      {executionLog.totalDuration && <p>⏱️ Total duration: {executionLog.totalDuration}ms</p>}
                    </div>
                  )}

                  {executionLog.status === "running" && (
                    <div className="text-yellow-400">
                      <p className="text-gray-500">[STATUS]</p>
                      <p>⏳ Execution in progress...</p>
                    </div>
                  )}

                  <div className="text-gray-500 mt-2">
                    <p>═══════════════════════════════════════════════════════════</p>
                  </div>
                </>
              ) : (
                <div className="text-gray-600">
                  <p className="text-gray-500">[INFO]</p>
                  <p>Ready to execute workflow...</p>
                  <p className="text-gray-700 mt-2">Type 'help' for available commands (coming soon)</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 저장 모달 */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4 border border-gray-700">
            <div className="bg-gray-700 px-6 py-4 border-b border-gray-600">
              <h2 className="text-lg font-bold text-white">워크플로우 저장</h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-semibold text-white mb-2">이름 *</label>
                <input
                  type="text"
                  value={workflowName}
                  onChange={(e) => setWorkflowName(e.target.value)}
                  placeholder="워크플로우 이름"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-white mb-2">설명</label>
                <textarea
                  value={workflowDescription}
                  onChange={(e) => setWorkflowDescription(e.target.value)}
                  placeholder="워크플로우 설명 (선택사항)"
                  rows={3}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
                />
              </div>
            </div>

            <div className="bg-gray-700 px-6 py-4 border-t border-gray-600 flex gap-3 justify-end">
              <button onClick={() => setShowSaveModal(false)} className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg text-sm font-semibold transition-colors">
                취소
              </button>
              <button onClick={handleSaveWorkflow} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-semibold transition-colors">
                저장
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 워크플로우 가이드 모달 */}
      {showWorkflowGuide && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="max-h-[90vh] overflow-y-auto">
            <WorkflowGuide onClose={() => setShowWorkflowGuide(false)} />
          </div>
        </div>
      )}

      {/* 워크플로우 목록 모달 */}
      {showWorkflowList && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg shadow-lg border border-gray-700 w-full max-w-2xl max-h-[80vh] flex flex-col">
            <div className="bg-gray-700 px-6 py-4 border-b border-gray-600 flex justify-between items-center">
              <h2 className="text-xl font-bold text-white">📚 워크플로우 목록</h2>
              <button onClick={() => setShowWorkflowList(false)} className="text-gray-400 hover:text-white transition-colors text-xl">
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              {listWorkflows().length === 0 ? (
                <div className="text-center text-gray-400 py-12">
                  <p className="text-lg mb-2">저장된 워크플로우가 없습니다</p>
                  <p className="text-sm">새 워크플로우를 만들거나 예시 워크플로우를 불러오세요</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {listWorkflows().map((workflow) => (
                    <div key={workflow.id} className="bg-gray-700 rounded-lg p-4 border border-gray-600 hover:border-blue-500 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h3 className="text-white font-semibold text-lg">{workflow.name}</h3>
                          {workflow.description && <p className="text-gray-400 text-sm mt-1">{workflow.description}</p>}
                        </div>
                        <div className="text-xs text-gray-400">{new Date(workflow.updatedAt).toLocaleDateString("ko-KR")}</div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-400 mb-3">
                        <span>📍 노드: {workflow.nodes?.length || 0}</span>
                        <span>🔗 연결: {workflow.connections?.length || 0}</span>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            loadWorkflowById(workflow.id);
                            setShowWorkflowList(false);
                          }}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-semibold transition-colors"
                        >
                          불러오기
                        </button>
                        <button
                          onClick={async () => {
                            if (confirm("정말 삭제하시겠습니까?")) {
                              await useEditorStore.getState().deleteWorkflow(workflow.id);
                              setShowWorkflowList(false);
                              setShowWorkflowList(true); // 목록 새로고침
                            }
                          }}
                          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-semibold transition-colors"
                        >
                          삭제
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Editor;
