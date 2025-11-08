/**
 * 데이터셋 로더 노드
 * JSON 정의 기반 동적 파라미터 폼
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  DatasetLoaderNode as DatasetLoaderNodeType,
  LocalDatasetLoaderNode,
  HFDatasetDownloaderNode,
  Node
} from "../../types/editor";
import { useEditorStore } from "../../stores/editorStore";
import { loadNodeDefinition, NodeDefinition } from "../../utils/nodeLoader";
import { findConnectedTokenNode, getTokenValue, isValidToken } from "../../utils/tokenHelper";
import { datasetAPI } from "../../utils/api";
import NodeComponent from "./Node";
import { NodeForm } from "./NodeForm";

interface DatasetLoaderNodeProps {
  node: LocalDatasetLoaderNode | HFDatasetDownloaderNode | DatasetLoaderNodeType | Node;
  isSelected: boolean;
  onPortMouseDown?: (nodeId: string, portId: string, event: React.MouseEvent) => void;
  onPortMouseEnter?: (nodeId: string, portId: string) => void;
  onPortMouseLeave?: () => void;
  onPortMouseUp?: (nodeId: string, portId: string) => void;
  hoveredPortId?: string | null;
  canConnect?: boolean;
}

export const DatasetLoaderNode: React.FC<DatasetLoaderNodeProps> = ({ node, isSelected, onPortMouseDown, onPortMouseEnter, onPortMouseLeave, onPortMouseUp, hoveredPortId, canConnect }) => {
  const selectNode = useEditorStore((s) => s.selectNode);
  const deleteNode = useEditorStore((s) => s.deleteNode);
  const updateNode = useEditorStore((s) => s.updateNode);
  const { nodes, connections } = useEditorStore();

  const [definition, setDefinition] = useState<NodeDefinition | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [connectedToken, setConnectedToken] = useState<string>("");
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const previousDatasetPathRef = useRef<string>("");
  const previousDatasetIdRef = useRef<string>("");

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

  const handleParameterChange = (parameterId: string, value: any) => {
    updateNode(node.id, {
      data: { ...node.data, [parameterId]: value },
    });
  };

  // 연결된 토큰 노드 감지
  useEffect(() => {
    const tokenNode = findConnectedTokenNode(node.id, nodes, connections);
    const token = getTokenValue(tokenNode);
    setConnectedToken(token);
  }, [node.id, nodes, connections]);

  const handleLoadLocalDataset = useCallback(async (datasetPath: string) => {
    if (!datasetPath || isLoading) return;
    
    setIsLoading(true);
    setProgress(0);

    try {
      // 데이터셋 로드
      const response = await datasetAPI.loadById(datasetPath);
      
      if (response.data.status !== "success") {
        throw new Error(response.data.message || "데이터셋 로드 실패");
      }

      // 진행률 시뮬레이션
      for (let i = 0; i <= 100; i += 20) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        setProgress(i);
      }

      // 데이터셋 정보 조회
      const infoResponse = await datasetAPI.info();
      console.log('📊 [DatasetLoaderNode] Dataset info after load:', infoResponse.data);
      const datasetInfo = infoResponse.data.data;

      if (!datasetInfo || !datasetInfo.columns) {
        console.warn('⚠️ [DatasetLoaderNode] Dataset info missing columns:', datasetInfo);
      }

      updateNode(node.id, {
        status: "completed",
        data: {
          ...node.data,
          datasetPath: datasetPath,
          dataInfo: {
            rows: datasetInfo?.shape?.rows || 0,
            columns: datasetInfo?.columns?.length || 0,
            columnNames: datasetInfo?.columns || [], // 컬럼 이름도 저장
            size: `${(datasetInfo?.size_mb || 0).toFixed(2)} MB`,
          },
        },
      });
      
      console.log('✅ [DatasetLoaderNode] Dataset loaded successfully. Columns:', datasetInfo?.columns);
    } catch (error: any) {
      console.error("Failed to load dataset:", error);
      updateNode(node.id, {
        status: "error",
        error: error.response?.data?.detail || error.message || "데이터셋 로드 실패",
      });
    } finally {
      setIsLoading(false);
      setProgress(0);
    }
  }, [node.id, isLoading, updateNode]);

  const handleDownloadHFDataset = useCallback(async (datasetId: string) => {
    if (!datasetId || isLoading) return;
    if (!connectedToken || !isValidToken(connectedToken)) {
      updateNode(node.id, {
        status: "error",
        error: "HF 토큰이 필요합니다. 토큰 노드를 연결해주세요.",
      });
      return;
    }

    setIsLoading(true);
    setProgress(0);

    try {
      // HF 데이터셋 다운로드
      const response = await datasetAPI.downloadHF(datasetId, connectedToken, "train");
      
      if (response.data.status !== "success") {
        throw new Error(response.data.message || "데이터셋 다운로드 실패");
      }

      // 진행률 시뮬레이션
      for (let i = 0; i <= 100; i += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setProgress(i);
      }

      // 데이터셋 정보 조회
      const infoResponse = await datasetAPI.info();
      console.log('📊 [DatasetLoaderNode] Dataset info after download:', infoResponse.data);
      const datasetInfo = infoResponse.data.data;

      if (!datasetInfo || !datasetInfo.columns) {
        console.warn('⚠️ [DatasetLoaderNode] Dataset info missing columns:', datasetInfo);
      }

      updateNode(node.id, {
        status: "completed",
        data: {
          ...node.data,
          datasetId: datasetId,
          dataInfo: {
            rows: datasetInfo?.shape?.rows || 0,
            columns: datasetInfo?.columns?.length || 0,
            columnNames: datasetInfo?.columns || [], // 컬럼 이름도 저장
            size: `${(datasetInfo?.size_mb || 0).toFixed(2)} MB`,
          },
        },
      });
      
      console.log('✅ [DatasetLoaderNode] Dataset downloaded successfully. Columns:', datasetInfo?.columns);
    } catch (error: any) {
      console.error("Failed to download HF dataset:", error);
      updateNode(node.id, {
        status: "error",
        error: error.response?.data?.detail || error.message || "데이터셋 다운로드 실패",
      });
    } finally {
      setIsLoading(false);
      setProgress(0);
    }
  }, [node.id, isLoading, connectedToken, updateNode]);

  // Local Dataset Loader: datasetPath 변경 시 자동 로드
  useEffect(() => {
    if (node.type === "local-dataset-loader") {
      const datasetPath = (node.data as any).datasetPath;
      if (datasetPath && datasetPath !== previousDatasetPathRef.current && !isLoading) {
        previousDatasetPathRef.current = datasetPath;
        handleLoadLocalDataset(datasetPath);
      }
    }
  }, [(node.data as any).datasetPath, node.type, isLoading, handleLoadLocalDataset]);

  // HF Dataset Downloader: datasetId 변경 시 자동 다운로드 및 로드
  useEffect(() => {
    if (node.type === "hf-dataset-downloader") {
      const datasetId = (node.data as any).datasetId;
      if (datasetId && datasetId !== previousDatasetIdRef.current && !isLoading) {
        // 토큰이 연결되어 있을 때만 자동 다운로드
        if (connectedToken && isValidToken(connectedToken)) {
          previousDatasetIdRef.current = datasetId;
          handleDownloadHFDataset(datasetId);
        }
      }
    }
  }, [(node.data as any).datasetId, node.type, connectedToken, isLoading, handleDownloadHFDataset]);

  const handleFileSelect = async (file: File) => {
    setIsUploading(true);
    setProgress(0);

    try {
      // FormData로 파일 업로드
      const formData = new FormData();
      formData.append("file", file);
      formData.append("data_format", (node.data as any).dataFormat || "csv");

      const response = await fetch("http://localhost:8001/dataset/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("파일 업로드 실패");

      // 진행률 시뮬레이션
      for (let i = 0; i <= 100; i += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setProgress(i);
      }

      // 데이터셋 정보 조회
      const infoResponse = await datasetAPI.info();
      const datasetInfo = infoResponse.data.data;

      updateNode(node.id, {
        status: "completed",
        data: {
          ...node.data,
          filePath: file.name,
          dataInfo: {
            rows: datasetInfo.shape?.rows || 0,
            columns: datasetInfo.columns?.length || 0,
            size: `${(datasetInfo.size_mb || 0).toFixed(2)} MB`,
          },
        },
      });
    } catch (error) {
      console.error("Failed to upload dataset:", error);
      updateNode(node.id, {
        status: "error",
        error: "파일 업로드 실패",
      });
    } finally {
      setIsUploading(false);
    }
  };

  if (!definition) {
    return (
      <NodeComponent node={node} isSelected={isSelected} onSelect={() => selectNode(node.id)} onDelete={() => deleteNode(node.id)} onPortMouseDown={onPortMouseDown}>
        <div className="text-sm text-gray-500">로딩 중...</div>
      </NodeComponent>
    );
  }

  const isLocalDatasetLoader = node.type === "local-dataset-loader";
  const isHFDatasetDownloader = node.type === "hf-dataset-downloader";
  const datasetPath = (node.data as any).datasetPath;
  const datasetId = (node.data as any).datasetId;
  const canLoad = (isLocalDatasetLoader && datasetPath) || (isHFDatasetDownloader && datasetId && connectedToken && isValidToken(connectedToken));

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
      {/* 숨겨진 파일 입력 */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelect(file);
        }}
        accept=".csv,.json,.parquet,.jsonl"
        className="hidden"
      />

      {/* 파라미터 폼 */}
      <NodeForm parameters={definition.parameters} values={node.data} onChange={handleParameterChange} onFileSelect={(_, file) => handleFileSelect(file)} />

      {/* 토큰 상태 표시 (HF 다운로더만) */}
      {isHFDatasetDownloader && (
        <div className="mt-2 p-2 bg-green-50 border border-green-200 rounded">
          {connectedToken && isValidToken(connectedToken) ? (
            <p className="text-xs text-green-700">✅ 토큰 노드에서 연결됨</p>
          ) : (
            <p className="text-xs text-gray-500">💡 🔑 토큰 노드와 연결하면 자동으로 사용됩니다</p>
          )}
        </div>
      )}

      {/* 데이터셋 정보 표시 */}
      {(node.data as any).dataInfo && !isLoading && !isUploading && (
        <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded">
          <p className="text-xs font-semibold text-gray-700 mb-1">
            ✓ {isLocalDatasetLoader ? `데이터셋: ${datasetPath}` : isHFDatasetDownloader ? `데이터셋: ${datasetId}` : `파일: ${(node.data as any).filePath}`}
          </p>
          {(node.data as any).dataInfo && (
            <div className="space-y-0.5 text-xs text-gray-600">
              <p>📊 행: {(node.data as any).dataInfo.rows?.toLocaleString()}</p>
              <p>📋 열: {(node.data as any).dataInfo.columns}</p>
              <p>💾 크기: {(node.data as any).dataInfo.size}</p>
            </div>
          )}
        </div>
      )}

      {/* 로딩/업로드 진행률 */}
      {(isLoading || isUploading) && (
        <div className="mt-3 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-gray-700">
              {isLoading ? (isLocalDatasetLoader ? "로드 중..." : "다운로드 중...") : "업로드 중..."}
            </span>
            <span className="text-gray-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-green-500 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* 에러 표시 */}
      {(node.status === "error" && (node as any).error) && (
        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded">
          <p className="text-xs text-red-700">❌ {(node as any).error}</p>
        </div>
      )}

      {/* 수동 불러오기 버튼 */}
      {canLoad && !isLoading && !isUploading && (
        <button
          onClick={() => {
            if (isLocalDatasetLoader) {
              handleLoadLocalDataset(datasetPath);
            } else if (isHFDatasetDownloader) {
              handleDownloadHFDataset(datasetId);
            }
          }}
          className="w-full px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm font-semibold transition-colors mt-3"
        >
          🔄 불러오기
        </button>
      )}

      {/* 파일 선택 버튼 (조건부) */}
      {(node.data as any).source === "upload" && !isUploading && (
        <button onClick={() => fileInputRef.current?.click()} className="w-full px-3 py-2 bg-green-500 hover:bg-green-600 text-white rounded text-sm font-semibold transition-colors mt-3">
          📂 파일 선택
        </button>
      )}
    </NodeComponent>
  );
};

export default DatasetLoaderNode;
