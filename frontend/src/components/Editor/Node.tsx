/**
 * 기본 노드 컴포넌트
 * 모든 노드의 베이스 역할
 */

import React, { useState } from "react";
import { Node } from "../../types/editor";
import { useEditorStore } from "../../stores/editorStore";

interface NodeProps {
  node: Node;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onDrag?: (x: number, y: number) => void;
  onPortMouseDown?: (nodeId: string, portId: string, event: React.MouseEvent, isOutput: boolean) => void;
  onPortMouseEnter?: (nodeId: string, portId: string) => void;
  onPortMouseLeave?: () => void;
  onPortMouseUp?: (nodeId: string, portId: string) => void;
  hoveredPortId?: string | null;
  canConnect?: boolean;
  children?: React.ReactNode;
  nodeName?: string; // 노드의 실제 이름 (JSON 정의에서)
  nodeIcon?: string; // 노드의 실제 아이콘 (JSON 정의에서)
}

const getNodeIcon = (type: string): string => {
  const icons: Record<string, string> = {
    "hf-token": "🔑",
    "model-loader": "🤖",
    "dataset-loader": "📊",
    training: "🎓",
    chat: "💬",
    rag: "🔍",
    "gguf-export": "📦",
  };
  return icons[type] || "📍";
};

const getNodeColor = (type: string): string => {
  const colors: Record<string, string> = {
    "hf-token": "from-yellow-500 to-yellow-600",
    "model-loader": "from-blue-500 to-blue-600",
    "dataset-loader": "from-green-500 to-green-600",
    training: "from-orange-500 to-orange-600",
    chat: "from-purple-500 to-purple-600",
    rag: "from-pink-500 to-pink-600",
    "gguf-export": "from-red-500 to-red-600",
  };
  return colors[type] || "from-gray-500 to-gray-600";
};

const getNodeLabel = (type: string): string => {
  const labels: Record<string, string> = {
    "hf-token": "HF Token",
    "model-loader": "Model Loader",
    "dataset-loader": "Dataset Loader",
    training: "Training",
    chat: "Chat",
    rag: "RAG",
    "gguf-export": "GGUF Export",
  };
  return labels[type] || "Node";
};

const getStatusColor = (status: string): string => {
  const statusColors: Record<string, string> = {
    idle: "bg-gray-400",
    pending: "bg-yellow-400",
    running: "bg-blue-500",
    completed: "bg-green-500",
    error: "bg-red-500",
  };
  return statusColors[status] || "bg-gray-400";
};

export const NodeComponent: React.FC<NodeProps> = ({
  node,
  isSelected,
  onSelect,
  onDelete,
  onDrag,
  onPortMouseDown,
  onPortMouseEnter,
  onPortMouseLeave,
  onPortMouseUp,
  hoveredPortId,
  canConnect,
  children,
  nodeName,
  nodeIcon,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const updateNode = useEditorStore((s) => s.updateNode);
  const nodeRef = React.useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("[data-port]")) return;
    if ((e.target as HTMLElement).closest("button")) return;
    if ((e.target as HTMLElement).closest("input")) return;
    if ((e.target as HTMLElement).closest("select")) return;
    if ((e.target as HTMLElement).closest("textarea")) return;

    setIsDragging(true);
    setDragOffset({
      x: e.clientX - node.position.x,
      y: e.clientY - node.position.y,
    });

    e.preventDefault();
    e.stopPropagation();
  };

  // Document 레벨에서 마우스 이동 감지
  React.useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newX = e.clientX - dragOffset.x;
      const newY = e.clientY - dragOffset.y;

      updateNode(node.id, {
        position: { x: Math.round(newX), y: Math.round(newY) },
      });

      onDrag?.(newX, newY);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, dragOffset, node.id, updateNode, onDrag]);

  return (
    <div
      ref={nodeRef}
      className={`
        absolute rounded-xl shadow-lg transition-all duration-150 cursor-move
        ${isSelected ? "ring-2 ring-blue-500 shadow-xl" : "shadow-lg"}
        ${isDragging ? "opacity-80 shadow-2xl" : "opacity-100"}
        bg-white select-none flex flex-col
      `}
      style={{
        left: `${node.position.x}px`,
        top: `${node.position.y}px`,
        pointerEvents: isDragging ? "none" : "auto",
        minWidth: "260px",
        maxWidth: "520px",
      }}
      onMouseDown={handleMouseDown}
      onClick={onSelect}
    >
      {/* 헤더 */}
      <div
        className={`
          bg-gradient-to-r ${getNodeColor(node.type)} text-white
          px-3 py-2.5 rounded-t-xl flex items-center justify-between
          cursor-grab active:cursor-grabbing
        `}
      >
        <div className="flex items-center gap-2 flex-1">
          <span className="text-lg">{nodeIcon || getNodeIcon(node.type)}</span>
          <div>
            <p className="font-semibold text-sm">{nodeName || getNodeLabel(node.type)}</p>
            <p className="text-xs opacity-90">ID: {node.id}</p>
          </div>
        </div>

        {/* 상태 표시 */}
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${getStatusColor(node.status)}`} />
          {node.progress !== undefined && <span className="text-xs bg-white bg-opacity-20 px-2 py-0.5 rounded">{node.progress}%</span>}
        </div>
      </div>

      {/* 메인 콘텐츠 영역 (포트 + 컨텐츠) */}
      <div className="flex flex-row items-stretch">

      {/* 포트 입력 (왼쪽 세로) */}
      <div className="flex flex-col gap-1.5 px-2 py-3 border-r border-gray-200 min-w-[72px]">
        {node.ports
          .filter((p) => p.type === "input")
          .filter((p) => {
            // visible 속성이 있으면 source 조건 확인
            if ((p as any).visible === "download") {
              const data = node.data as any;
              return data?.source === "huggingface" || data?.source === "upload";
            }
            return true;
          })
          .map((port) => {
            const portKey = `${node.id}-${port.id}`;
            const isHovered = hoveredPortId === portKey;
            
            // 포트 데이터타입에 따른 색상
            let defaultColor = "bg-blue-400";
            let hoverShadow = "group-hover:shadow-blue-400";
            if ((port as any).dataType === "token" || (port as any).dataType === "config") {
              defaultColor = "bg-yellow-400";
              hoverShadow = "group-hover:shadow-yellow-400";
            } else if ((port as any).dataType === "dataset") {
              defaultColor = "bg-green-400";
              hoverShadow = "group-hover:shadow-green-400";
            } else if ((port as any).dataType === "model") {
              defaultColor = "bg-blue-400";
              hoverShadow = "group-hover:shadow-blue-400";
            }
            
            // 호버 상태에 따른 색상
            const bgColor = isHovered && canConnect 
              ? "bg-green-500 animate-pulse" 
              : isHovered && !canConnect 
              ? "bg-red-500" 
              : defaultColor;
            const shadowColor = isHovered && canConnect 
              ? "group-hover:shadow-green-500" 
              : isHovered && !canConnect 
              ? "group-hover:shadow-red-500" 
              : hoverShadow;
            
            return (
              <div
                key={port.id}
                data-port={port.id}
                data-node-id={node.id}
                className="flex items-center gap-2 cursor-crosshair group relative z-10"
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onPortMouseDown?.(node.id, port.id, e, false);
                }}
                onMouseEnter={() => onPortMouseEnter?.(node.id, port.id)}
                onMouseLeave={() => onPortMouseLeave?.()}
                onMouseUp={() => onPortMouseUp?.(node.id, port.id)}
                title={`${isHovered && canConnect ? "✅ 여기에 연결하면 됩니다!" : isHovered && !canConnect ? "❌ 연결 불가능" : "클릭 & 드래그로 연결"}`}
              >
                <div className={`w-4 h-4 ${bgColor} rounded-full group-hover:scale-150 group-hover:shadow-lg ${shadowColor} transition-all duration-100 flex-shrink-0`} />
                <span className="text-xs text-gray-600 whitespace-nowrap">{port.name}</span>
              </div>
            );
          })}
      </div>

      {/* 컨텐츠 */}
      <div className="flex-1 bg-gray-50">
        <div className="max-h-[440px] overflow-y-auto">
          <div className="px-4 py-3 w-full max-w-[380px] mx-auto">
            {children}
          </div>
        </div>
      </div>

      {/* 포트 출력 (오른쪽 세로) */}
      <div className="flex flex-col gap-1.5 px-2 py-3 border-l border-gray-200 min-w-[72px]">
        {node.ports
          .filter((p) => p.type === "output")
          .filter((p) => {
            // visible 조건 확인
            if ((p as any).visible === "has-token") {
              const data = node.data as any;
              return data?.tokenInput && data.tokenInput.startsWith("hf_");
            }
            // visible이 없거나 다른 조건이면 항상 표시
            return true;
          })
          .map((port) => {
            const portKey = `${node.id}-${port.id}`;
            const isHovered = hoveredPortId === portKey;
            
            // 포트 데이터타입에 따른 색상
            let defaultColor = "bg-yellow-400";
            let hoverColor = "hover:bg-yellow-500";
            let shadowColor = "hover:shadow-yellow-400";
            
            if ((port as any).dataType === "token" || (port as any).dataType === "config") {
              defaultColor = "bg-yellow-400";
              hoverColor = "hover:bg-yellow-500";
              shadowColor = "hover:shadow-yellow-400";
            } else if ((port as any).dataType === "dataset") {
              defaultColor = "bg-green-400";
              hoverColor = "hover:bg-green-500";
              shadowColor = "hover:shadow-green-400";
            } else if ((port as any).dataType === "model") {
              defaultColor = "bg-blue-400";
              hoverColor = "hover:bg-blue-500";
              shadowColor = "hover:shadow-blue-400";
            }
            
            return (
              <div
                key={`output-${port.id}`}
                data-port={port.id}
                data-node-id={node.id}
                className={`flex items-center gap-2 cursor-crosshair group relative z-10 ${
                  isHovered && canConnect 
                    ? "bg-green-50 rounded px-1 py-0.5" 
                    : ""
                }`}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onPortMouseDown?.(node.id, port.id, e, true);
                }}
                onMouseEnter={() => onPortMouseEnter?.(node.id, port.id)}
                onMouseLeave={() => onPortMouseLeave?.()}
                onMouseUp={() => onPortMouseUp?.(node.id, port.id)}
                title={isHovered && canConnect ? "✅ 여기에 연결하면 됩니다!" : port.name || "출력 포트"}
              >
                <div className={`w-4 h-4 ${isHovered && canConnect 
                  ? "bg-green-500 animate-pulse hover:shadow-lg hover:shadow-green-500" 
                  : `${defaultColor} ${hoverColor} hover:shadow-lg ${shadowColor}`
                } rounded-full group-hover:scale-150 transition-all duration-100 flex-shrink-0`} />
                <span className="text-xs text-gray-600 whitespace-nowrap">{port.name}</span>
              </div>
            );
          })}
      </div>
      </div>

      {/* 삭제 버튼 */}
      <div className="absolute -top-8 right-0 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="px-2 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600"
        >
          Delete
        </button>
      </div>

      {/* 에러 메시지 */}
      {node.error && <div className="px-4 py-2 bg-red-100 text-red-700 text-xs rounded-b-lg border-t border-red-300">{node.error}</div>}
    </div>
  );
};

export default NodeComponent;
