/**
 * HuggingFace 토큰 노드
 * 모델 다운로드 및 데이터셋 로딩에 필요한 토큰 설정
 */

import React, { useState, useEffect } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { loadNodeDefinition, NodeDefinition } from "../../utils/nodeLoader";
import NodeComponent from "./Node";

interface HFTokenNodeProps {
  node: any;
  isSelected: boolean;
  onPortMouseDown?: (nodeId: string, portId: string, event: React.MouseEvent) => void;
  onPortMouseEnter?: (nodeId: string, portId: string) => void;
  onPortMouseLeave?: () => void;
  onPortMouseUp?: (nodeId: string, portId: string) => void;
  hoveredPortId?: string | null;
  canConnect?: boolean;
}

export const HFTokenNode: React.FC<HFTokenNodeProps> = ({ node, isSelected, onPortMouseDown, onPortMouseEnter, onPortMouseLeave, onPortMouseUp, hoveredPortId, canConnect }) => {
  const selectNode = useEditorStore((s) => s.selectNode);
  const deleteNode = useEditorStore((s) => s.deleteNode);
  const updateNode = useEditorStore((s) => s.updateNode);

  const [definition, setDefinition] = useState<NodeDefinition | null>(null);
  const [tokenValid, setTokenValid] = useState(false);
  const [tokenPreview, setTokenPreview] = useState("");

  // 노드 정의 로드
  useEffect(() => {
    const load = async () => {
      const def = await loadNodeDefinition("hf-token");
      setDefinition(def);
    };
    load();
  }, []);

  // 토큰 유효성 검사
  useEffect(() => {
    const token = node.data?.token || "";
    if (token.length > 0) {
      // HF 토큰은 hf_로 시작해야 함
      const isValid = token.startsWith("hf_") && token.length > 10;
      setTokenValid(isValid);

      // 토큰 미리보기 (일부만 표시)
      if (token.length > 10) {
        setTokenPreview(`${token.substring(0, 10)}...${token.substring(token.length - 5)}`);
      }
    } else {
      setTokenValid(false);
      setTokenPreview("");
    }
  }, [node.data?.token]);

  const handleParameterChange = (parameterId: string, value: any) => {
    // tokenInput이 변경되면 token도 함께 업데이트
    const updates: any = { [parameterId]: value };
    if (parameterId === "tokenInput" && value) {
      updates.token = value;
    }

    updateNode(node.id, {
      data: { ...node.data, ...updates },
    });
  };

  // 정의 없으면 기본값 사용
  const nodeDefinition = definition;

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
      nodeName={definition?.name}
      nodeIcon={definition?.icon}
    >
      {/* 토큰 입력 */}
      <div className="p-3 space-y-2">
        <div>
          <label className="block text-xs font-semibold text-gray-700 mb-1">🔑 토큰 입력</label>
          <input
            type="password"
            value={node.data?.tokenInput || ""}
            onChange={(e) => handleParameterChange("tokenInput", e.target.value)}
            placeholder="hf_로 시작하는 토큰"
            className="w-full px-2.5 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            💡 발급받기:{" "}
            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
              HF Settings
            </a>
          </p>
        </div>

        {/* 상태 표시 */}
        <div className={`p-2 rounded text-xs leading-tight ${tokenValid ? "bg-green-50 border border-green-200" : "bg-gray-50 border border-gray-200"}`}>
          {tokenValid ? (
            <>
              <p className="font-semibold text-green-700 mb-1">✅ 토큰 유효</p>
              <p className="text-gray-600 font-mono break-all">{tokenPreview}</p>
            </>
          ) : (
            <>
              <p className="font-semibold text-gray-600 mb-1">⚠️ 토큰 미입력</p>
              <p className="text-gray-500">hf_로 시작하는 토큰을 입력하세요</p>
            </>
          )}
        </div>

        {/* 선택사항 */}
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={node.data?.saveAsDefault || false} onChange={(e) => handleParameterChange("saveAsDefault", e.target.checked)} className="w-3 h-3" />
            <span className="text-gray-700">⭐ 기본 토큰으로 저장</span>
          </label>
        </div>

      </div>
    </NodeComponent>
  );
};

export default HFTokenNode;
