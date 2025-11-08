/**
 * 워크플로우 캔버스
 * 노드를 표시하고 연결하는 캔버스
 */

import React, { useRef, useState, useEffect } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { Node } from "../../types/editor";
import { ModelLoaderNode } from "./ModelLoaderNode";
import { DatasetLoaderNode } from "./DatasetLoaderNode";
import { HFTokenNode } from "./HFTokenNode";
import { GenericNode } from "./GenericNode";
import { checkPortConnection } from "../../utils/connectionRules";

const DEFAULT_CANVAS_DIMENSIONS = {
  width: 4000,
  height: 3000,
  offsetX: 0,
  offsetY: 0,
};

const NODE_WIDTH_ESTIMATE = 360;
const NODE_HEIGHT_ESTIMATE = 320;
const CANVAS_MARGIN = 600;

interface WorkflowCanvasProps {
  onNodePaletteToggle?: () => void;
}

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({ onNodePaletteToggle }) => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectStart, setSelectStart] = useState({ x: 0, y: 0 });
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionStart, setConnectionStart] = useState<{
    nodeId: string;
    portId: string;
    x: number;
    y: number;
    isOutput: boolean;
    element?: HTMLElement;
  } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [hoveredPortId, setHoveredPortId] = useState<string | null>(null);
  const [canConnect, setCanConnect] = useState(false);
  const { nodes, connections, selectedNodeId, zoom, panX, panY, canvasMode, setZoom, setPan, setCanvasMode, selectNode, clearNodeSelection, addConnection } = useEditorStore();
  const [portPositions, setPortPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [canvasDimensions, setCanvasDimensions] = useState(DEFAULT_CANVAS_DIMENSIONS);
  const transformWithOffsets = React.useMemo(
    () => `translate(${panX}px, ${panY}px) scale(${zoom}) translate(${canvasDimensions.offsetX}px, ${canvasDimensions.offsetY}px)`,
    [panX, panY, zoom, canvasDimensions.offsetX, canvasDimensions.offsetY]
  );

  // 포트 위치 업데이트 (줌/팬 변경 시) - 정확한 계산 및 실시간 업데이트
  useEffect(() => {
    let animationFrameId: number;
    let resizeObserver: ResizeObserver | null = null;
    let mutationObserver: MutationObserver | null = null;
    let updateTimeout: NodeJS.Timeout | null = null;

    const updatePortPositions = () => {
      if (!canvasRef.current) return;

      const positions: Record<string, { x: number; y: number }> = {};
      const canvasRect = canvasRef.current.getBoundingClientRect();

      nodes.forEach((node) => {
        node.ports.forEach((port) => {
          const portElement = document.querySelector(`[data-node-id="${node.id}"][data-port="${port.id}"]`) as HTMLElement;

          if (portElement) {
            // 포트 내부의 원형 div 찾기 (실제 포트 시각적 중심)
            const portCircle = portElement.querySelector('div[class*="rounded-full"]') as HTMLElement;
            const targetElement = portCircle || portElement;

            const elementRect = targetElement.getBoundingClientRect();

            // 요소의 중심 좌표 (화면 좌표)
            const screenX = elementRect.left + elementRect.width / 2;
            const screenY = elementRect.top + elementRect.height / 2;

            // 포트의 SVG 좌표 계산
            // SVG와 노드 컨테이너에 translate(panX, panY) scale(zoom) translate(offsetX, offsetY)가 적용되어 있음
            // transform origin이 '0 0'이므로:
            // 화면 좌표 = ((SVG 좌표 + offset) * zoom) + pan + canvasRect.origin
            // 따라서: SVG 좌표 = (화면 좌표 - canvasRect.origin - pan) / zoom - offset
            const svgX = (screenX - canvasRect.left - panX) / zoom - canvasDimensions.offsetX;
            const svgY = (screenY - canvasRect.top - panY) / zoom - canvasDimensions.offsetY;

            positions[`${node.id}-${port.id}`] = { x: svgX, y: svgY };
          } else {
            // 폴백: 노드 위치 기반 계산 (노드의 SVG 좌표 + 포트의 상대 위치)
            const isInput = port.type === "input";
            const portIndex = node.ports.filter((p) => p.type === port.type).findIndex((p) => p.id === port.id);

            const headerHeight = 60;
            const portHeight = 32;
            const nodeWidth = 280;

            let x: number, y: number;
            if (isInput) {
              x = node.position.x;
              y = node.position.y + headerHeight + 8 + portIndex * portHeight;
            } else {
              x = node.position.x + nodeWidth;
              y = node.position.y + headerHeight + 8 + portIndex * portHeight;
            }

            positions[`${node.id}-${port.id}`] = { x, y };
          }
        });
      });

      setPortPositions(positions);
    };

    // 디바운스된 업데이트 함수
    const scheduleUpdate = (immediate = false) => {
      if (updateTimeout) {
        clearTimeout(updateTimeout);
        updateTimeout = null;
      }

      if (immediate) {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
        // 즉시 업데이트를 위해 requestAnimationFrame을 두 번 사용하여 렌더링 완료 보장
        animationFrameId = requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            updatePortPositions();
          });
        });
      } else {
        updateTimeout = setTimeout(() => {
          if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
          }
          animationFrameId = requestAnimationFrame(() => {
            updatePortPositions();
          });
        }, 16); // 약 60fps
      }
    };

    // 초기 업데이트
    scheduleUpdate(true);

    // ResizeObserver로 노드 크기 변경 감지
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        scheduleUpdate();
      });

      // 모든 노드 요소 관찰
      nodes.forEach((node) => {
        const nodeElement = document.querySelector(`[data-node-id="${node.id}"]`)?.closest(".group") as HTMLElement;
        if (nodeElement) {
          resizeObserver?.observe(nodeElement);
        }
      });
    }

    // MutationObserver로 DOM 변경 감지 (노드 위치 변경 포함)
    if (typeof MutationObserver !== "undefined" && canvasRef.current) {
      mutationObserver = new MutationObserver(() => {
        scheduleUpdate();
      });

      mutationObserver.observe(canvasRef.current, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["style", "class"],
      });
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (updateTimeout) {
        clearTimeout(updateTimeout);
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      if (mutationObserver) {
        mutationObserver.disconnect();
      }
    };
  }, [nodes, zoom, panX, panY, canvasDimensions.offsetX, canvasDimensions.offsetY]);

  // 포트의 실제 위치를 가져오는 함수
  const getPortPosition = (nodeId: string, portId: string): { x: number; y: number } | null => {
    const key = `${nodeId}-${portId}`;
    return portPositions[key] || null;
  };
  // 포트 위치 강제 업데이트 함수
  const forceUpdatePortPositions = () => {
    if (!canvasRef.current) return;

    const positions: Record<string, { x: number; y: number }> = {};
    const canvasRect = canvasRef.current.getBoundingClientRect();

    nodes.forEach((node) => {
      node.ports.forEach((port) => {
        const portElement = document.querySelector(`[data-node-id="${node.id}"][data-port="${port.id}"]`) as HTMLElement;

        if (portElement) {
          // 포트 내부의 원형 div 찾기 (실제 포트 시각적 중심)
          const portCircle = portElement.querySelector('div[class*="rounded-full"]') as HTMLElement;
          const targetElement = portCircle || portElement;

          const elementRect = targetElement.getBoundingClientRect();

          // 요소의 중심 좌표 (화면 좌표)
          const screenX = elementRect.left + elementRect.width / 2;
          const screenY = elementRect.top + elementRect.height / 2;

          // 포트의 SVG 좌표 계산
          // SVG와 노드 컨테이너에 translate(panX, panY) scale(zoom) translate(offsetX, offsetY)가 적용되어 있음
          // 화면 좌표 = ((SVG 좌표 + offset) * zoom) + pan + canvasRect.origin
          // 따라서: SVG 좌표 = (화면 좌표 - canvasRect.origin - pan) / zoom - offset
          const svgX = (screenX - canvasRect.left - panX) / zoom - canvasDimensions.offsetX;
          const svgY = (screenY - canvasRect.top - panY) / zoom - canvasDimensions.offsetY;

          positions[`${node.id}-${port.id}`] = { x: svgX, y: svgY };
        } else {
          // 폴백: 노드 위치 기반 계산 (노드의 SVG 좌표 + 포트의 상대 위치)
          const isInput = port.type === "input";
          const portIndex = node.ports.filter((p) => p.type === port.type).findIndex((p) => p.id === port.id);

          const headerHeight = 60;
          const portHeight = 32;
          const nodeWidth = 280;

          let x: number, y: number;
          if (isInput) {
            x = node.position.x;
            y = node.position.y + headerHeight + 8 + portIndex * portHeight;
          } else {
            x = node.position.x + nodeWidth;
            y = node.position.y + headerHeight + 8 + portIndex * portHeight;
          }

          positions[`${node.id}-${port.id}`] = { x, y };
        }
      });
    });

    setPortPositions(positions);
  };

  useEffect(() => {
    if (nodes.length === 0) {
      setCanvasDimensions(DEFAULT_CANVAS_DIMENSIONS);
      return;
    }

    let minX = 0;
    let minY = 0;
    let maxX = 0;
    let maxY = 0;

    const includePoint = (x: number, y: number) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    };

    nodes.forEach((node) => {
      includePoint(node.position.x, node.position.y);
      includePoint(node.position.x + NODE_WIDTH_ESTIMATE, node.position.y + NODE_HEIGHT_ESTIMATE);
    });

    Object.values(portPositions).forEach((pos) => {
      includePoint(pos.x, pos.y);
    });

    if (isConnecting) {
      if (connectionStart) {
        includePoint(connectionStart.x, connectionStart.y);
      }
      includePoint(mousePos.x, mousePos.y);
    }

    const offsetX = minX < 0 ? Math.abs(minX) + CANVAS_MARGIN : 0;
    const offsetY = minY < 0 ? Math.abs(minY) + CANVAS_MARGIN : 0;

    const nextWidth = Math.max(DEFAULT_CANVAS_DIMENSIONS.width, Math.ceil(maxX + CANVAS_MARGIN + offsetX));

    const nextHeight = Math.max(DEFAULT_CANVAS_DIMENSIONS.height, Math.ceil(maxY + CANVAS_MARGIN + offsetY));

    setCanvasDimensions((prev) => {
      if (prev.width === nextWidth && prev.height === nextHeight && prev.offsetX === offsetX && prev.offsetY === offsetY) {
        return prev;
      }
      return { width: nextWidth, height: nextHeight, offsetX, offsetY };
    });
  }, [nodes, portPositions, isConnecting, connectionStart, mousePos]);

  const handleZoomIn = () => {
    const newZoom = Math.min(3, zoom * 1.2);
    setZoom(newZoom);
    // 줌 변경 후 포트 위치 즉시 업데이트 (렌더링 완료 대기)
    setTimeout(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          forceUpdatePortPositions();
        });
      });
    }, 0);
  };

  const handleZoomOut = () => {
    const newZoom = Math.max(0.1, zoom * 0.8);
    setZoom(newZoom);
    // 줌 변경 후 포트 위치 즉시 업데이트 (렌더링 완료 대기)
    setTimeout(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          forceUpdatePortPositions();
        });
      });
    }, 0);
  };

  const handleZoomReset = () => {
    setZoom(1);
    setPan(0, 0);
    // 줌 리셋 후 포트 위치 즉시 업데이트 (렌더링 완료 대기)
    setTimeout(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          forceUpdatePortPositions();
        });
      });
    }, 0);
  };

  // 자동 레이아웃 함수 (노드들을 더 잘 보이도록 배치) - 완전히 개선된 알고리즘
  const handleAutoLayout = () => {
    if (nodes.length === 0) return;

    const { updateNode } = useEditorStore.getState();
    const horizontalSpacing = 600; // 노드 간 가로 간격 증가
    const verticalSpacing = 350; // 노드 간 세로 간격 증가
    const startX = 500;
    const startY = 400;

    // 노드가 2개이고 연결이 있는 경우 특별 처리
    if (nodes.length === 2 && connections.length > 0) {
      const conn = connections[0];
      const sourceNode = nodes.find((n) => n.id === conn.source);
      const targetNode = nodes.find((n) => n.id === conn.target);

      if (sourceNode && targetNode) {
        // 소스 노드를 왼쪽에, 타겟 노드를 오른쪽에 수평 정렬
        updateNode(sourceNode.id, {
          position: { x: Math.round(startX), y: Math.round(startY) },
        });
        updateNode(targetNode.id, {
          position: { x: Math.round(startX + horizontalSpacing), y: Math.round(startY) },
        });
        setTimeout(() => forceUpdatePortPositions(), 200);
        return;
      }
    }

    // 노드가 적을 때 (3개 이하) 간단한 레이아웃 사용
    if (nodes.length <= 3 && connections.length === 0) {
      // 연결이 없으면 간단한 그리드 배치
      const cols = Math.ceil(Math.sqrt(nodes.length));
      nodes.forEach((node, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        const x = startX + col * horizontalSpacing;
        const y = startY + row * verticalSpacing;
        updateNode(node.id, {
          position: { x: Math.round(x), y: Math.round(y) },
        });
      });
      setTimeout(() => forceUpdatePortPositions(), 200);
      return;
    }

    // 노드를 레이어별로 분류 (위상 정렬)
    const layers: Node[][] = [];
    const nodeDepths = new Map<string, number>();
    const processed = new Set<string>();

    // 각 노드의 깊이 계산 (의존성 기반)
    const calculateDepth = (nodeId: string): number => {
      if (nodeDepths.has(nodeId)) {
        return nodeDepths.get(nodeId)!;
      }

      if (processed.has(nodeId)) {
        return 0; // 순환 참조 방지
      }
      processed.add(nodeId);

      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return 0;

      // 이 노드에 연결된 입력 노드들의 최대 깊이 + 1
      const inputConnections = connections.filter((c) => c.target === nodeId);
      if (inputConnections.length === 0) {
        nodeDepths.set(nodeId, 0);
        return 0;
      }

      const depths = inputConnections.map((c) => calculateDepth(c.source));
      const maxDepth = depths.length > 0 ? Math.max(...depths) : 0;
      const depth = maxDepth + 1;
      nodeDepths.set(nodeId, depth);
      return depth;
    };

    // 모든 노드의 깊이 계산
    nodes.forEach((node) => {
      calculateDepth(node.id);
    });

    // 깊이별로 노드 그룹화
    const depths = Array.from(nodeDepths.values());
    const maxDepth = depths.length > 0 ? Math.max(...depths) : 0;

    // 모든 노드가 깊이 0인 경우 (연결이 없거나 모두 독립적)
    if (maxDepth === 0 && depths.length === nodes.length) {
      // 수평으로 배치
      nodes.forEach((node, index) => {
        const x = startX + index * horizontalSpacing;
        const y = startY;
        updateNode(node.id, {
          position: { x: Math.round(x), y: Math.round(y) },
        });
      });
      setTimeout(() => forceUpdatePortPositions(), 200);
      return;
    }

    // 레이어 생성
    for (let i = 0; i <= maxDepth; i++) {
      layers[i] = nodes.filter((n) => nodeDepths.get(n.id) === i);
    }

    // 각 레이어의 노드들을 배치 (중앙 정렬, 수직 정렬)
    layers.forEach((layer, layerIndex) => {
      if (layer.length === 0) return;

      const x = startX + layerIndex * horizontalSpacing;

      // 레이어의 노드들을 수직으로 중앙 정렬
      const totalHeight = (layer.length - 1) * verticalSpacing;
      const startYForLayer = startY - totalHeight / 2;

      layer.forEach((node, nodeIndex) => {
        const y = startYForLayer + nodeIndex * verticalSpacing;
        updateNode(node.id, {
          position: { x: Math.round(x), y: Math.round(y) },
        });
      });
    });

    // 레이아웃 완료 후 포트 위치 업데이트
    setTimeout(() => {
      forceUpdatePortPositions();
    }, 300);
  };

  // 워크플로우 로드 시 자동 레이아웃 적용
  useEffect(() => {
    if (nodes.length > 0 && connections.length > 0) {
      // 노드들이 불규칙하게 배치되어 있는지 확인
      const positions = nodes.map((n) => n.position);
      const minX = Math.min(...positions.map((p) => p.x));
      const maxX = Math.max(...positions.map((p) => p.x));
      const minY = Math.min(...positions.map((p) => p.y));
      const maxY = Math.max(...positions.map((p) => p.y));

      // 노드들이 너무 가까이 있거나 불규칙하게 배치되어 있으면 자동 레이아웃 적용
      const width = maxX - minX;
      const height = maxY - minY;
      const avgSpacing = Math.min(width / Math.max(nodes.length, 1), height / Math.max(nodes.length, 1));

      // 평균 간격이 너무 작으면 (300px 미만) 자동 레이아웃 적용
      if (avgSpacing < 300) {
        const timer = setTimeout(() => {
          handleAutoLayout();
        }, 500);
        return () => clearTimeout(timer);
      }
    }
    return undefined;
  }, [nodes.length, connections.length]); // 초기 로드 시에만 실행

  // 캔버스 클릭 (선택 해제)
  const handleCanvasClick = (e: React.MouseEvent) => {
    if (e.target === canvasRef.current) {
      clearNodeSelection();
    }
  };

  // 팬 동작
  const handleMouseDown = (e: React.MouseEvent) => {
    // 팬 모드일 때만 팬 동작
    if (canvasMode === "pan" && e.button === 0) {
      setIsSelecting(true);
      setSelectStart({ x: e.clientX, y: e.clientY });
    } else if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
      // 우클릭 또는 Shift+좌클릭 (기본 팬 동작)
      setIsSelecting(true);
      setSelectStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isSelecting) {
      const dx = e.clientX - selectStart.x;
      const dy = e.clientY - selectStart.y;
      const newPanX = panX + dx;
      const newPanY = panY + dy;
      setPan(newPanX, newPanY);
      setSelectStart({ x: e.clientX, y: e.clientY });

      // 팬 변경 시 포트 위치 즉시 업데이트 (디바운스)
      // 팬 중에는 매번 업데이트하지 않고 마지막에 한 번만 업데이트
    }

    // 연결 중일 때 마우스 위치 업데이트 (Canvas 좌표 기준)
    if (isConnecting && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      // Canvas 좌표를 SVG 좌표로 변환
      const svgX = (canvasX - panX) / zoom - canvasDimensions.offsetX;
      const svgY = (canvasY - panY) / zoom - canvasDimensions.offsetY;

      setMousePos({ x: svgX, y: svgY });
    }
  };

  // 줌/팬 변경 시 포트 위치 강제 업데이트
  useEffect(() => {
    // 줌이나 팬이 변경되면 포트 위치를 즉시 업데이트
    // 여러 번의 requestAnimationFrame을 사용하여 렌더링 완료 보장
    let frameId1: number;
    let frameId2: number;

    frameId1 = requestAnimationFrame(() => {
      frameId2 = requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          forceUpdatePortPositions();
        });
      });
    });

    return () => {
      if (frameId1) cancelAnimationFrame(frameId1);
      if (frameId2) cancelAnimationFrame(frameId2);
    };
  }, [zoom, panX, panY, nodes.length, canvasDimensions.offsetX, canvasDimensions.offsetY]);

  const handleMouseUp = () => {
    setIsSelecting(false);
    // 연결 중인 경우는 port mouseUp에서 처리하므로 여기서 자동 종료하지 않음
  };

  // 포트 마우스 다운
  const handlePortMouseDown = (nodeId: string, portId: string, event: React.MouseEvent, isOutput: boolean) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const portElement = event.currentTarget as HTMLElement;
    const rect = portElement.getBoundingClientRect();
    const canvasRect = canvasRef.current?.getBoundingClientRect() || { left: 0, top: 0 };

    // 포트의 중앙 위치 (Canvas 좌표)
    const centerX = rect.left - canvasRect.left + rect.width / 2;
    const centerY = rect.top - canvasRect.top + rect.height / 2;

    // Canvas 좌표를 SVG 좌표로 변환
    const svgX = (centerX - panX) / zoom - canvasDimensions.offsetX;
    const svgY = (centerY - panY) / zoom - canvasDimensions.offsetY;

    setConnectionStart({
      nodeId,
      portId,
      x: svgX,
      y: svgY,
      isOutput,
      element: portElement,
    });
    setIsConnecting(true);
    event.stopPropagation();
  };

  // 포트 호버 (연결 가능 여부 확인)
  const handlePortMouseEnter = (nodeId: string, portId: string) => {
    const portKey = `${nodeId}-${portId}`;
    setHoveredPortId(portKey);

    if (!isConnecting || !connectionStart) {
      return;
    }

    const sourceNode = nodes.find((n) => n.id === connectionStart.nodeId);
    const targetNode = nodes.find((n) => n.id === nodeId);
    if (!sourceNode || !targetNode) {
      setCanConnect(false);
      return;
    }

    const sourcePort = sourceNode.ports.find((p) => p.id === connectionStart.portId);
    const targetPort = targetNode.ports.find((p) => p.id === portId);
    if (!sourcePort || !targetPort) {
      setCanConnect(false);
      return;
    }

    // 연결 규칙 확인
    const result = checkPortConnection(sourceNode, sourcePort, targetNode, targetPort);
    setCanConnect(result.canConnect);
  };

  const handlePortMouseLeave = () => {
    setHoveredPortId(null);
    setCanConnect(false);
  };

  // 포트 마우스 업 (연결 완료)
  const handlePortMouseUp = (targetNodeId: string, targetPortId: string) => {
    if (!connectionStart || !isConnecting) return;
    if (connectionStart.nodeId === targetNodeId) return;

    const sourceNode = nodes.find((n) => n.id === connectionStart.nodeId);
    const targetNode = nodes.find((n) => n.id === targetNodeId);
    if (!sourceNode || !targetNode) return;

    const sourcePort = sourceNode.ports.find((p) => p.id === connectionStart.portId);
    const targetPort = targetNode.ports.find((p) => p.id === targetPortId);
    if (!sourcePort || !targetPort) return;

    // 입출력 방향 검증
    if (sourcePort.type === "output" && targetPort.type === "input") {
      addConnection({
        id: `conn-${Date.now()}`,
        source: connectionStart.nodeId,
        target: targetNodeId,
        sourcePort: connectionStart.portId,
        targetPort: targetPortId,
        dataType: sourcePort.dataType,
      });
    } else if (sourcePort.type === "input" && targetPort.type === "output") {
      addConnection({
        id: `conn-${Date.now()}`,
        source: targetNodeId,
        target: connectionStart.nodeId,
        sourcePort: targetPortId,
        targetPort: connectionStart.portId,
        dataType: targetPort.dataType,
      });
    }

    setIsConnecting(false);
    setConnectionStart(null);
    setHoveredPortId(null);
  };

  return (
    <div
      ref={canvasRef}
      className="relative w-full h-full bg-gradient-to-br from-gray-50 to-gray-100 overflow-hidden workflow-canvas"
      onContextMenu={(e) => e.preventDefault()}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={(e) => {
        handleMouseUp();
        // 모든 포트 마우스 업 처리
        const port = (e.target as HTMLElement).closest("[data-port]");
        if (port && isConnecting && connectionStart) {
          // 포트 위에서 마우스 업 시
          const nodeId = (e.currentTarget as HTMLElement).id;
          // 노드 ID가 필요하므로 다르게 처리
        }
        // 연결 중 포트 없는 곳에서 업한 경우
        if (isConnecting && !port) {
          setIsConnecting(false);
          setConnectionStart(null);
          setHoveredPortId(null);
        }
      }}
      onMouseLeave={() => {
        handleMouseUp();
        if (isConnecting) {
          setIsConnecting(false);
          setConnectionStart(null);
          setHoveredPortId(null);
        }
      }}
      onClick={handleCanvasClick}
      style={{
        userSelect: "none",
        cursor: isSelecting ? "grab" : "default",
      }}
    >
      {/* 그리드 배경 및 연결선 */}
      <svg
        className="absolute"
        style={{
          left: 0,
          top: 0,
          width: `${canvasDimensions.width}px`,
          height: `${canvasDimensions.height}px`,
          transform: transformWithOffsets,
          transformOrigin: "0 0",
          zIndex: 1, // 노드 뒤에 렌더링되지만 연결선은 보이도록
          pointerEvents: "none", // 연결선은 클릭 불가능
          overflow: "visible",
        }}
      >
        {/* 그리드 선 */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* 베지어 곡선 연결선 - 노드 간 데이터 흐름 시각화 */}
        {connections.length > 0 &&
          connections.map((conn) => {
            const sourceNode = nodes.find((n) => n.id === conn.source);
            const targetNode = nodes.find((n) => n.id === conn.target);

            if (!sourceNode || !targetNode) return null;

            // 소스 포트 찾기 (출력 포트)
            const sourcePort = sourceNode.ports.find((p) => p.id === conn.sourcePort);
            const targetPort = targetNode.ports.find((p) => p.id === conn.targetPort);

            if (!sourcePort || !targetPort) return null;

            // 포트 위치 계산 - 실제 DOM에서 포트 위치 가져오기
            const sourcePos = getPortPosition(conn.source, conn.sourcePort);
            const targetPos = getPortPosition(conn.target, conn.targetPort);

            // 포트 위치 계산 - 실제 위치 우선, 없으면 계산된 위치 사용
            let x1: number, y1: number, x2: number, y2: number;

            // 포트 위치를 직접 계산 (더 정확함) - 렌더링 시점에 항상 최신 위치 사용
            const getPortSVGPosition = (nodeId: string, portId: string): { x: number; y: number } | null => {
              if (!canvasRef.current) return null;

              const portElement = document.querySelector(`[data-node-id="${nodeId}"][data-port="${portId}"]`) as HTMLElement;
              if (!portElement) return null;

              // 포트 내부의 원형 div 찾기 (실제 포트 시각적 중심)
              const portCircle = portElement.querySelector('div[class*="rounded-full"]') as HTMLElement;
              const targetElement = portCircle || portElement;

              const canvasRect = canvasRef.current.getBoundingClientRect();
              const elementRect = targetElement.getBoundingClientRect();

              // 요소의 중심 좌표 (화면 좌표)
              const screenX = elementRect.left + elementRect.width / 2;
              const screenY = elementRect.top + elementRect.height / 2;

              // SVG 좌표로 변환
              // SVG와 노드 컨테이너에 translate(panX, panY) scale(zoom) translate(offsetX, offsetY)가 적용되어 있음
              // transform origin이 '0 0'이므로:
              // 화면 좌표 = ((SVG 좌표 + offset) * zoom) + pan + canvasRect.origin
              // 따라서: SVG 좌표 = (화면 좌표 - canvasRect.origin - pan) / zoom - offset
              const svgX = (screenX - canvasRect.left - panX) / zoom - canvasDimensions.offsetX;
              const svgY = (screenY - canvasRect.top - panY) / zoom - canvasDimensions.offsetY;

              return { x: svgX, y: svgY };
            };

            const sourcePosDirect = getPortSVGPosition(conn.source, conn.sourcePort);
            const targetPosDirect = getPortSVGPosition(conn.target, conn.targetPort);

            if (sourcePosDirect) {
              x1 = sourcePosDirect.x;
              y1 = sourcePosDirect.y;
            } else if (sourcePos) {
              x1 = sourcePos.x;
              y1 = sourcePos.y;
            } else {
              // 폴백: 노드 위치 기반 계산 (정확한 SVG 좌표)
              const sourceOutputPorts = sourceNode.ports.filter((p) => p.type === "output");
              const sourcePortIndex = sourceOutputPorts.findIndex((p) => p.id === conn.sourcePort);
              const headerHeight = 60;
              const portHeight = 32;
              const nodeWidth = 280;
              x1 = sourceNode.position.x + nodeWidth;
              y1 = sourceNode.position.y + headerHeight + 8 + sourcePortIndex * portHeight;
            }

            if (targetPosDirect) {
              x2 = targetPosDirect.x;
              y2 = targetPosDirect.y;
            } else if (targetPos) {
              x2 = targetPos.x;
              y2 = targetPos.y;
            } else {
              // 폴백: 노드 위치 기반 계산 (정확한 SVG 좌표)
              const targetInputPorts = targetNode.ports.filter((p) => p.type === "input");
              const targetPortIndex = targetInputPorts.findIndex((p) => p.id === conn.targetPort);
              const headerHeight = 60;
              const portHeight = 32;
              x2 = targetNode.position.x;
              y2 = targetNode.position.y + headerHeight + 8 + targetPortIndex * portHeight;
            }

            // 베지어 곡선 제어점 (부드러운 곡선)
            const dx = Math.abs(x2 - x1) * 0.5;
            const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

            // 데이터 타입별 색상 (연결 규칙에 따라)
            let strokeColor = "#6b7280"; // 기본 회색
            if (sourcePort && targetPort) {
              const result = checkPortConnection(sourceNode, sourcePort, targetNode, targetPort);
              switch (result.color) {
                case "blue":
                  strokeColor = "#2563eb"; // 파란색 (모델)
                  break;
                case "green":
                  strokeColor = "#059669"; // 초록색 (데이터셋)
                  break;
                case "yellow":
                  strokeColor = "#d97706"; // 노란색 (토큰/설정)
                  break;
                case "purple":
                  strokeColor = "#9333ea"; // 보라색 (설정)
                  break;
                default:
                  // 데이터 타입 기반 폴백
                  const dataType = (sourcePort as any).dataType || conn.dataType;
                  strokeColor = dataType === "model" ? "#2563eb" : dataType === "dataset" ? "#059669" : dataType === "token" || dataType === "config" ? "#d97706" : "#6b7280";
              }
            } else {
              // 포트 정보가 없을 때 데이터 타입 기반
              const dataType = conn.dataType;
              strokeColor = dataType === "model" ? "#2563eb" : dataType === "dataset" ? "#059669" : dataType === "token" || dataType === "config" ? "#d97706" : "#6b7280";
            }

            return (
              <g key={conn.id} style={{ opacity: 1 }}>
                {/* 연결선 배경 (더 두껍게) */}
                <path
                  d={path}
                  stroke="rgba(0,0,0,0.15)"
                  strokeWidth="6"
                  fill="none"
                  style={{
                    pointerEvents: "none",
                    opacity: 1,
                  }}
                />
                {/* 메인 연결선 (두껍고 명확하게) */}
                <path
                  d={path}
                  stroke={strokeColor}
                  strokeWidth="4"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{
                    filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.3))",
                    pointerEvents: "none",
                    opacity: 1,
                  }}
                />
                {/* 시작점 원 (출력 포트) */}
                <circle
                  cx={x1}
                  cy={y1}
                  r="6"
                  fill={strokeColor}
                  stroke="white"
                  strokeWidth="2"
                  style={{
                    filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.4))",
                    pointerEvents: "none",
                    opacity: 1,
                  }}
                />
                {/* 끝점 원 (입력 포트) */}
                <circle
                  cx={x2}
                  cy={y2}
                  r="6"
                  fill={strokeColor}
                  stroke="white"
                  strokeWidth="2"
                  style={{
                    filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.4))",
                    pointerEvents: "none",
                    opacity: 1,
                  }}
                />
              </g>
            );
          })}

        {/* 드래그 중인 연결선 (실시간) */}
        {isConnecting && connectionStart && (
          <g pointerEvents="none">
            <path
              d={`M ${connectionStart.x} ${connectionStart.y} C ${connectionStart.x + (mousePos.x - connectionStart.x) * 0.3} ${connectionStart.y}, ${
                mousePos.x - (mousePos.x - connectionStart.x) * 0.3
              } ${mousePos.y}, ${mousePos.x} ${mousePos.y}`}
              stroke="#fbbf24"
              strokeWidth="2.5"
              fill="none"
              strokeDasharray="6,3"
              style={{
                filter: "drop-shadow(0 0 4px rgba(251,191,36,0.6))",
                animation: "dashflow 0.6s linear infinite",
              }}
            />
            {/* 끝점 동그라미 */}
            <circle
              cx={mousePos.x}
              cy={mousePos.y}
              r="5"
              fill="#fbbf24"
              style={{
                filter: "drop-shadow(0 0 3px rgba(251,191,36,0.9))",
                animation: "pulse 1.2s ease-in-out infinite",
              }}
            />
          </g>
        )}
      </svg>

      {/* 노드 렌더링 */}
      <div
        className="relative"
        style={{
          width: `${canvasDimensions.width}px`,
          height: `${canvasDimensions.height}px`,
          transform: transformWithOffsets,
          transformOrigin: "0 0",
          zIndex: 10, // 연결선 위에 렌더링
        }}
      >
        {nodes.map((node) => (
          <div key={node.id} className="group">
            {node.type === "hf-token" && (
              <HFTokenNode
                node={node as any}
                isSelected={selectedNodeId === node.id}
                onPortMouseDown={handlePortMouseDown}
                onPortMouseEnter={handlePortMouseEnter}
                onPortMouseLeave={handlePortMouseLeave}
                onPortMouseUp={handlePortMouseUp}
                hoveredPortId={hoveredPortId}
                canConnect={canConnect}
              />
            )}
            {(node.type === "model-loader" || node.type === "local-model-loader" || node.type === "hf-model-downloader") && (
              <ModelLoaderNode
                node={node as any}
                isSelected={selectedNodeId === node.id}
                onPortMouseDown={handlePortMouseDown}
                onPortMouseEnter={handlePortMouseEnter}
                onPortMouseLeave={handlePortMouseLeave}
                onPortMouseUp={handlePortMouseUp}
                hoveredPortId={hoveredPortId}
                canConnect={canConnect}
              />
            )}
            {(node.type === "dataset-loader" || node.type === "local-dataset-loader" || node.type === "hf-dataset-downloader") && (
              <DatasetLoaderNode
                node={node as any}
                isSelected={selectedNodeId === node.id}
                onPortMouseDown={handlePortMouseDown}
                onPortMouseEnter={handlePortMouseEnter}
                onPortMouseLeave={handlePortMouseLeave}
                onPortMouseUp={handlePortMouseUp}
                hoveredPortId={hoveredPortId}
                canConnect={canConnect}
              />
            )}
            {/* 새로운 노드 타입들은 GenericNode로 렌더링 */}
            {(node.type === "device-selector" ||
              node.type === "dataset-preprocessor" ||
              node.type === "dataset-splitter" ||
              node.type === "dataset-filter" ||
              node.type === "training-config" ||
              node.type === "lora-config" ||
              node.type === "qlora-config" ||
              node.type === "model-evaluator" ||
              node.type === "checkpoint-manager" ||
              node.type === "model-saver" ||
              node.type === "training" ||
              node.type === "chat" ||
              node.type === "rag" ||
              node.type === "gguf-export") && (
              <GenericNode
                node={node}
                isSelected={selectedNodeId === node.id}
                onPortMouseDown={handlePortMouseDown}
                onPortMouseEnter={handlePortMouseEnter}
                onPortMouseLeave={handlePortMouseLeave}
                onPortMouseUp={handlePortMouseUp}
                hoveredPortId={hoveredPortId}
                canConnect={canConnect}
              />
            )}
          </div>
        ))}
      </div>

      {/* 헬퍼 텍스트 */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center text-gray-400">
            <p className="text-lg font-semibold mb-2">🎨 워크플로우 캔버스</p>
            <p className="text-sm">왼쪽 패널에서 노드를 드래그하여 추가하세요</p>
            <p className="text-xs mt-2 opacity-50">우측 상단 버튼: 줌 | Shift+드래그: 팬 | 우클릭: 메뉴</p>
          </div>
        </div>
      )}

      {/* 줌 컨트롤 버튼 */}
      <div className="absolute top-4 right-4 bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-2 flex flex-col gap-2 z-20">
        <button onClick={handleZoomIn} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm font-semibold transition-colors" title="줌 인">
          ➕
        </button>
        <button onClick={handleZoomOut} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm font-semibold transition-colors" title="줌 아웃">
          ➖
        </button>
        <div className="border-t border-gray-600 my-1"></div>
        <button onClick={handleZoomReset} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs font-semibold transition-colors" title="줌 리셋">
          🔄
        </button>
        {nodes.length > 0 && (
          <>
            <div className="border-t border-gray-600 my-1"></div>
            <button
              onClick={handleAutoLayout}
              className="px-3 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded text-xs font-bold transition-all shadow-lg hover:shadow-xl animate-pulse"
              title="자동 레이아웃 - 노드들을 정렬합니다"
            >
              📐 정렬
            </button>
          </>
        )}
        <div className="text-xs text-gray-400 text-center pt-1">{(zoom * 100).toFixed(0)}%</div>
      </div>

      {/* 줌/팬 정보 */}
      <div className="absolute bottom-4 right-4 bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-3 text-xs space-y-1 text-gray-300">
        <p>📍 노드: {nodes.length}</p>
        <p>🔗 연결: {connections.length}</p>
      </div>

      {/* 캔버스 모드 전환 버튼 (하단 중앙) */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-2 flex gap-2 z-20">
        <button
          onClick={() => setCanvasMode("select")}
          className={`px-4 py-2 rounded text-sm font-semibold transition-all ${canvasMode === "select" ? "bg-blue-600 text-white shadow-lg" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          title="선택 모드 (노드 선택 및 드래그)"
        >
          <span className="mr-2">👆</span>
          <span>선택</span>
        </button>
        <button
          onClick={() => setCanvasMode("pan")}
          className={`px-4 py-2 rounded text-sm font-semibold transition-all ${canvasMode === "pan" ? "bg-blue-600 text-white shadow-lg" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          title="팬 모드 (캔버스 이동)"
        >
          <span className="mr-2">✋</span>
          <span>이동</span>
        </button>
      </div>

      {/* 캔버스 커서 스타일 */}
      <style>{`
        .workflow-canvas {
          cursor: ${canvasMode === "pan" ? "grab" : "default"};
        }
        .workflow-canvas:active {
          cursor: ${canvasMode === "pan" ? "grabbing" : "default"};
        }
      `}</style>
    </div>
  );
};

export default WorkflowCanvas;
