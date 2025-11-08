/**
 * 워크플로우 헬퍼 함수들
 * 노드 검증, 토큰 추출 등
 */

import { Node, Connection } from "../types/editor";

/**
 * 토큰 노드에서 유효한 토큰 값 추출
 */
export function extractTokenFromNode(tokenNode: Node | undefined): string | null {
  if (!tokenNode) return null;

  const data = tokenNode.data as any;
  const token = data?.tokenInput || data?.token || "";

  // 유효한 토큰인지 확인 (hf_로 시작)
  if (token && token.startsWith("hf_") && token.length > 10) {
    return token;
  }

  return null;
}

/**
 * 특정 노드에 연결된 토큰 노드 찾기
 */
export function findConnectedTokenNode(
  nodeId: string,
  nodes: Node[],
  connections: Connection[]
): Node | null {
  // 이 노드에 입력으로 연결된 연결선 찾기
  const incomingConnection = connections.find(
    (conn) => conn.target === nodeId && (conn as any).targetPort?.includes("token")
  );

  if (!incomingConnection) return null;

  // 연결 출발지 노드 찾기
  const sourceNode = nodes.find((n) => n.id === incomingConnection.source);
  if (!sourceNode || sourceNode.type !== "hf-token") return null;

  return sourceNode;
}

/**
 * 모델 다운로드 노드 검증
 */
export function validateModelDownload(
  node: Node,
  nodes: Node[],
  connections: Connection[]
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const data = node.data as any;

  // 모델 ID 확인
  if (!data?.modelId && data?.source === "huggingface") {
    errors.push("모델 ID를 입력하세요");
  }

  // HuggingFace 모델 다운로드 시 토큰 확인
  if (data?.source === "huggingface") {
    const tokenNode = findConnectedTokenNode(node.id, nodes, connections);
    const token = extractTokenFromNode(tokenNode);

    if (!token) {
      errors.push("🔑 HF Token 노드와 연결이 필요합니다");
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * 데이터셋 다운로드 노드 검증
 */
export function validateDatasetDownload(
  node: Node,
  nodes: Node[],
  connections: Connection[]
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const data = node.data as any;

  // 데이터셋 선택 확인
  if (!data?.localDataset && !data?.datasetId && data?.source !== "upload") {
    errors.push("데이터셋을 선택하세요");
  }

  // HuggingFace 데이터셋 다운로드 시 토큰 확인
  if (data?.source === "huggingface") {
    const tokenNode = findConnectedTokenNode(node.id, nodes, connections);
    const token = extractTokenFromNode(tokenNode);

    if (!token) {
      errors.push("🔑 HF Token 노드와 연결이 필요합니다");
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * 전체 워크플로우 검증
 */
export function validateWorkflow(
  nodes: Node[],
  connections: Connection[]
): { valid: boolean; errors: Record<string, string[]> } {
  const errors: Record<string, string[]> = {};

  for (const node of nodes) {
    if (node.type === "model-loader") {
      const validation = validateModelDownload(node, nodes, connections);
      if (!validation.valid) {
        errors[node.id] = validation.errors;
      }
    } else if (node.type === "dataset-loader") {
      const validation = validateDatasetDownload(node, nodes, connections);
      if (!validation.valid) {
        errors[node.id] = validation.errors;
      }
    }
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}

/**
 * 워크플로우의 모든 토큰 노드를 찾기
 */
export function getAllTokenNodes(nodes: Node[]): Node[] {
  return nodes.filter((n) => n.type === "hf-token");
}

/**
 * 포트가 다른 포트와 연결 가능한지 확인
 */
export function canConnectPorts(
  sourceNode: Node,
  sourcePortId: string,
  targetNode: Node,
  targetPortId: string
): boolean {
  // 같은 노드는 연결 불가
  if (sourceNode.id === targetNode.id) return false;

  const sourcePort = sourceNode.ports.find((p) => p.id === sourcePortId);
  const targetPort = targetNode.ports.find((p) => p.id === targetPortId);

  if (!sourcePort || !targetPort) return false;

  // 출력 → 입력만 가능
  if (sourcePort.type === "output" && targetPort.type === "input") {
    return true;
  }

  // 입력 ← 출력도 가능
  if (sourcePort.type === "input" && targetPort.type === "output") {
    return true;
  }

  return false;
}

/**
 * 포트 연결 가능 여부를 나타내는 메시지
 */
export function getConnectionStatusMessage(
  sourceNode: Node,
  sourcePortId: string,
  targetNode: Node | null,
  targetPortId: string | null
): string | null {
  if (!targetNode || !targetPortId) {
    return null;
  }

  if (!canConnectPorts(sourceNode, sourcePortId, targetNode, targetPortId)) {
    return "❌ 연결할 수 없습니다";
  }

  return "✅ 연결 가능";
}

