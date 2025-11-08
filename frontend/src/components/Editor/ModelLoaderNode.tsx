/**
 * 모델 로더 노드
 * JSON 정의 기반 동적 파라미터 폼
 */

import React, { useState, useEffect } from "react";
import { 
  ModelLoaderNode as ModelLoaderNodeType,
  LocalModelLoaderNode,
  HFModelDownloaderNode,
  Node
} from "../../types/editor";
import { useEditorStore } from "../../stores/editorStore";
import { loadNodeDefinition, NodeDefinition } from "../../utils/nodeLoader";
import { findConnectedTokenNode, getTokenValue, isValidToken } from "../../utils/tokenHelper";
import NodeComponent from "./Node";
import { NodeForm } from "./NodeForm";

interface ModelLoaderNodeProps {
  node: LocalModelLoaderNode | HFModelDownloaderNode | ModelLoaderNodeType | Node;
  isSelected: boolean;
  onPortMouseDown?: (nodeId: string, portId: string, event: React.MouseEvent) => void;
  onPortMouseEnter?: (nodeId: string, portId: string) => void;
  onPortMouseLeave?: () => void;
  onPortMouseUp?: (nodeId: string, portId: string) => void;
  hoveredPortId?: string | null;
  canConnect?: boolean;
}

export const ModelLoaderNode: React.FC<ModelLoaderNodeProps> = ({ node, isSelected, onPortMouseDown, onPortMouseEnter, onPortMouseLeave, onPortMouseUp, hoveredPortId, canConnect }) => {
  const selectNode = useEditorStore((s) => s.selectNode);
  const deleteNode = useEditorStore((s) => s.deleteNode);
  const updateNode = useEditorStore((s) => s.updateNode);

  const { nodes, connections } = useEditorStore();
  const [definition, setDefinition] = useState<NodeDefinition | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [connectedToken, setConnectedToken] = useState<string>("");

  // 노드 정의 로드 - 노드 타입에 따라 올바른 정의 로드
  useEffect(() => {
    const load = async () => {
      // 노드 타입에 따라 올바른 정의 로드
      const nodeType = node.type;
      const def = await loadNodeDefinition(nodeType);
      setDefinition(def);
    };
    load();
  }, [node.type]);

  // 연결된 토큰 노드 감지
  useEffect(() => {
    const tokenNode = findConnectedTokenNode(node.id, nodes, connections);
    const token = getTokenValue(tokenNode);
    setConnectedToken(token);
  }, [node.id, nodes, connections]);

  const handleParameterChange = (parameterId: string, value: any) => {
    updateNode(node.id, {
      data: { ...node.data, [parameterId]: value },
    });
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    setProgress(0);

    try {
      updateNode(node.id, { status: "running" });

      // 소스에 따라 다르게 처리
      const isLocal = node.data.source === "local";
      const modelId = isLocal ? node.data.localModel : node.data.modelId;

      if (!modelId) {
        throw new Error(`${isLocal ? "로컬 모델을" : "모델 ID를"} 선택하세요`);
      }

      if (isLocal) {
        // 로컬 모델: 스트림으로 로드 (진행률 표시)
        const response = await fetch(`http://localhost:8001/model/upload-stream?model_path=${encodeURIComponent(modelId)}`, { method: "POST" });

        if (!response.body) throw new Error("스트림을 사용할 수 없습니다");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n").filter((line) => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.progress !== undefined) {
                setProgress(data.progress);
              }
              if (data.status === "completed") {
                updateNode(node.id, {
                  status: "completed",
                  data: {
                    ...node.data,
                    loadedModel: modelId,
                    loadedSource: "local",
                  },
                });
              } else if (data.status === "error") {
                throw new Error(data.message || "로컬 모델 로드 중 오류 발생");
              }
            } catch (e) {
              // JSON 파싱 오류 무시
            }
          }
        }
      } else {
        // HuggingFace: 다운로드
        // 연결된 토큰 사용 (없으면 빈 문자열)
        const response = await fetch(`http://localhost:8001/model/download-stream?model_id=${modelId}&access_token=${connectedToken || ""}`, { method: "POST" });

        if (!response.body) throw new Error("스트림을 사용할 수 없습니다");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n").filter((line) => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.progress !== undefined) {
                setProgress(data.progress);
              }
              if (data.status === "completed") {
                updateNode(node.id, {
                  status: "completed",
                  data: {
                    ...node.data,
                    loadedModel: modelId,
                    loadedSource: "huggingface",
                  },
                });
              } else if (data.status === "error") {
                throw new Error(data.message || "다운로드 중 오류 발생");
              }
            } catch (e) {
              // JSON 파싱 오류 무시
            }
          }
        }
      }
    } catch (error) {
      console.error("Failed to load model:", error);
      updateNode(node.id, {
        status: "error",
        error: error instanceof Error ? error.message : "모델 로드 실패",
      });
    } finally {
      setIsDownloading(false);
    }
  };

  if (!definition) {
    return (
      <NodeComponent node={node} isSelected={isSelected} onSelect={() => selectNode(node.id)} onDelete={() => deleteNode(node.id)} onPortMouseDown={onPortMouseDown}>
        <div className="text-sm text-gray-500">로딩 중...</div>
      </NodeComponent>
    );
  }

  return (
    <NodeComponent
      node={node}
      isSelected={isSelected}
      onSelect={() => selectNode(node.id)}
      onDelete={() => deleteNode(node.id)}
      onPortMouseDown={onPortMouseDown}
      onPortMouseEnter={onPortMouseEnter}
      onPortMouseLeave={onPortMouseLeave}
      onPortMouseUp={onPortMouseUp}
      hoveredPortId={hoveredPortId}
      canConnect={canConnect}
      nodeName={definition.name}
      nodeIcon={definition.icon}
    >
      {/* 파라미터 폼 */}
      <NodeForm parameters={definition.parameters} values={node.data} onChange={handleParameterChange} />

      {/* 로드된 모델 정보 */}
      {node.data.loadedModel && !isDownloading && (
        <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded">
          <p className="text-xs font-semibold text-green-700 mb-1">✓ 로드됨</p>
          <p className="text-xs text-gray-700">{node.data.loadedModel}</p>
          <p className="text-xs text-gray-500">출처: {node.data.loadedSource === "huggingface" ? "🤗 HuggingFace" : "📂 로컬"}</p>
        </div>
      )}

      {/* 진행률 */}
      {isDownloading && (
        <div className="mt-3 space-y-2">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-gray-700">{node.data.source === "local" ? "📂 로컬 모델 로드 중..." : "🤗 다운로드 중..."}</span>
            <span className="text-gray-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-blue-500 h-2 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* 토큰 상태 표시 (다운로드 시만) */}
      {node.data.source === "huggingface" && (
        <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
          {connectedToken && isValidToken(connectedToken) ? (
            <p className="text-xs text-blue-700">✅ 토큰 노드에서 연결됨</p>
          ) : (
            <p className="text-xs text-gray-500">💡 🔑 토큰 노드와 연결하면 자동으로 사용됩니다</p>
          )}
        </div>
      )}

      {/* 로드 버튼 */}
      <button
        onClick={handleDownload}
        disabled={isDownloading || (!node.data.modelId && !node.data.localModel)}
        className={`
          w-full px-3 py-2 rounded text-sm font-semibold text-white transition-colors mt-3
          ${isDownloading || (!node.data.modelId && !node.data.localModel) ? "bg-gray-400 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"}
        `}
      >
        {isDownloading ? <>⏳ {node.data.source === "local" ? "로드 중..." : "다운로드 중..."}</> : <>{node.data.source === "local" ? "📂 로컬 모델 로드" : "🤗 HuggingFace에서 다운로드"}</>}
      </button>
    </NodeComponent>
  );
};

export default ModelLoaderNode;
