/**
 * 범용 노드 컴포넌트
 * JSON 정의 기반으로 동적으로 노드를 렌더링합니다
 */

import React, { useState, useEffect } from "react";
import { Node } from "../../types/editor";
import { useEditorStore } from "../../stores/editorStore";
import { loadNodeDefinition, NodeDefinition } from "../../utils/nodeLoader";
import NodeComponent from "./Node";
import { NodeForm } from "./NodeForm";
import { datasetAPI } from "../../utils/api";

interface GenericNodeProps {
  node: Node;
  isSelected: boolean;
  onPortMouseDown?: (nodeId: string, portId: string, event: React.MouseEvent, isOutput: boolean) => void;
  onPortMouseEnter?: (nodeId: string, portId: string) => void;
  onPortMouseLeave?: () => void;
  onPortMouseUp?: (nodeId: string, portId: string) => void;
  hoveredPortId?: string | null;
  canConnect?: boolean;
}

export const GenericNode: React.FC<GenericNodeProps> = ({ node, isSelected, onPortMouseDown, onPortMouseEnter, onPortMouseLeave, onPortMouseUp, hoveredPortId, canConnect }) => {
  const selectNode = useEditorStore((s) => s.selectNode);
  const deleteNode = useEditorStore((s) => s.deleteNode);
  const updateNode = useEditorStore((s) => s.updateNode);
  const { connections, nodes } = useEditorStore((s) => ({
    connections: s.currentWorkflow?.connections || [],
    nodes: s.currentWorkflow?.nodes || [],
  }));

  const [definition, setDefinition] = useState<NodeDefinition | null>(null);
  const [columnOptions, setColumnOptions] = useState<Record<string, Array<{ label: string; value: string }>>>({});

  // 연결된 데이터셋 노드 찾기
  const datasetConnection = React.useMemo(() => {
    const conn = connections.find((conn) => conn.target === node.id && conn.targetPort === "in-dataset");
    console.log("🔍 [GenericNode] Looking for connection:", {
      nodeId: node.id,
      connections: connections.map((c) => ({
        source: c.source,
        target: c.target,
        targetPort: c.targetPort,
      })),
      found: conn,
    });
    return conn;
  }, [connections, node.id]);

  const sourceDatasetNode = React.useMemo(() => {
    if (!datasetConnection) return null;
    return nodes.find((n) => n.id === datasetConnection.source);
  }, [nodes, datasetConnection]);

  // 데이터셋 노드 상태 추적 (리렌더링 트리거)
  const datasetNodeStatus = React.useMemo(() => {
    return sourceDatasetNode?.status;
  }, [sourceDatasetNode]);

  const datasetNodeDataInfo = React.useMemo(() => {
    return sourceDatasetNode ? (sourceDatasetNode.data as any).dataInfo : null;
  }, [sourceDatasetNode]);

  // 노드 정의 로드
  useEffect(() => {
    const load = async () => {
      const def = await loadNodeDefinition(node.type);
      setDefinition(def);
    };
    load();
  }, [node.type]);

  // 데이터셋 컬럼 정보 로드 (dataset-preprocessor 노드인 경우)
  useEffect(() => {
    if (node.type !== "dataset-preprocessor" || !definition) return;

    const loadColumns = async () => {
      console.log("🔍 [GenericNode] Checking for dataset connection...");
      console.log("🔍 [GenericNode] All connections:", connections);
      console.log("🔍 [GenericNode] Current node ID:", node.id);

      if (!datasetConnection) {
        // 연결이 없어도 API를 통해 컬럼 정보 가져오기 시도
        console.log("⚠️ [GenericNode] No dataset connection found, trying API...");

        try {
          const response = await datasetAPI.info();
          console.log("📊 [GenericNode] API response (no connection):", response.data);

          if (response.data?.status === "success" && response.data?.data?.columns) {
            const columns = response.data.data.columns;
            if (Array.isArray(columns) && columns.length > 0) {
              const options = columns.map((col: string) => ({ label: col, value: col }));
              console.log("✅ [GenericNode] Loaded columns from API (no connection):", options);
              setColumnOptions({
                inputColumns: options,
                outputColumns: options,
                textColumn: options,
                userColumn: options,
                assistantColumn: options,
                systemColumn: options,
              });
              return;
            }
          }
        } catch (error) {
          console.error("❌ [GenericNode] Failed to load columns from API:", error);
        }

        setColumnOptions({});
        return;
      }

      console.log("✅ [GenericNode] Found dataset connection:", datasetConnection);

      if (!sourceDatasetNode) {
        console.log("⚠️ [GenericNode] Source node not found:", datasetConnection.source);
        console.log(
          "🔍 [GenericNode] Available nodes:",
          nodes.map((n) => ({ id: n.id, type: n.type }))
        );
        setColumnOptions({});
        return;
      }

      console.log("📊 [GenericNode] Source node status:", sourceDatasetNode.status);
      console.log("📊 [GenericNode] Source node data:", sourceDatasetNode.data);
      console.log("📊 [GenericNode] dataInfo:", datasetNodeDataInfo);

      // 데이터셋 노드가 로드 완료되었는지 확인
      const isDatasetLoaded = sourceDatasetNode.status === "completed" || datasetNodeDataInfo;

      if (!isDatasetLoaded) {
        // 데이터셋이 아직 로드되지 않음
        console.log("⚠️ [GenericNode] Dataset not loaded yet. Status:", sourceDatasetNode.status);
        setColumnOptions({});
        return;
      }

      // dataInfo에 컬럼 이름이 있으면 직접 사용 (더 빠름)
      const columnNames = (sourceDatasetNode.data as any).dataInfo?.columnNames;
      console.log("🔍 [GenericNode] Column names from dataInfo:", columnNames);

      if (Array.isArray(columnNames) && columnNames.length > 0) {
        console.log("✅ [GenericNode] Using columns from node dataInfo:", columnNames);
        const options = columnNames.map((col: string) => ({ label: col, value: col }));
        setColumnOptions({
          inputColumns: options,
          outputColumns: options,
          textColumn: options,
          userColumn: options,
          assistantColumn: options,
          systemColumn: options,
        });
        return;
      }

      // dataInfo에 컬럼이 없으면 API 호출
      try {
        // 데이터셋 정보 가져오기
        console.log("📡 [GenericNode] Fetching dataset info from API...");
        const response = await datasetAPI.info();
        console.log("📊 [GenericNode] Dataset info response:", response.data);

        if (response.data?.status === "success" && response.data?.data?.columns) {
          const columns = response.data.data.columns;
          if (Array.isArray(columns) && columns.length > 0) {
            const options = columns.map((col: string) => ({ label: col, value: col }));

            console.log("✅ [GenericNode] Loaded columns:", options);

            // 모든 컬럼 선택 파라미터에 옵션 설정
            setColumnOptions({
              inputColumns: options,
              outputColumns: options,
              textColumn: options,
              userColumn: options,
              assistantColumn: options,
              systemColumn: options,
            });
          } else {
            console.warn("⚠️ [GenericNode] Dataset columns array is empty");
            setColumnOptions({});
          }
        } else if (response.data?.status === "no_data") {
          console.warn("⚠️ [GenericNode] No dataset loaded yet");
          setColumnOptions({});
        } else {
          console.warn("⚠️ [GenericNode] Dataset columns not available yet", response.data);
          setColumnOptions({});
        }
      } catch (error) {
        console.error("❌ [GenericNode] Failed to load dataset columns:", error);
        setColumnOptions({});
      }
    };

    // 즉시 실행
    loadColumns();
  }, [node.type, node.id, definition, datasetConnection, sourceDatasetNode, datasetNodeStatus, datasetNodeDataInfo]);

  const handleParameterChange = (parameterId: string, value: any) => {
    updateNode(node.id, {
      data: { ...node.data, [parameterId]: value },
    });
  };

  // 파라미터에 동적 옵션 적용 (모든 hooks는 early return 전에 호출되어야 함)
  const parametersWithOptions = React.useMemo(() => {
    if (!definition) return [];
    return (
      definition.parameters?.map((param) => {
        if (param.dynamicSource === "dataset-columns" && columnOptions[param.id]) {
          return { ...param, options: columnOptions[param.id] };
        }
        return param;
      }) || []
    );
  }, [definition, columnOptions]);

  if (!definition) {
    return (
      <div className="bg-gray-200 rounded-lg p-4 min-w-[288px]">
        <div className="text-sm text-gray-500">로딩 중...</div>
      </div>
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
      <NodeForm parameters={parametersWithOptions} values={node.data} onChange={handleParameterChange} />
    </NodeComponent>
  );
};

export default GenericNode;
