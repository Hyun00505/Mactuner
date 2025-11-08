/**
 * HuggingFace 토큰 헬퍼
 * 토큰 노드 검색 및 토큰 값 조회
 */

import { Node } from '../types/editor';

/**
 * 워크플로우에서 HF 토큰 노드 찾기
 */
export function findHFTokenNode(nodes: Node[]): Node | undefined {
  return nodes.find(node => node.type === 'hf-token');
}

/**
 * 현재 노드에 연결된 토큰 노드 찾기
 */
export function findConnectedTokenNode(
  nodeId: string,
  nodes: Node[],
  connections: Array<{ source: string; target: string }>
): Node | undefined {
  // 이 노드의 입력 연결 찾기
  const incomingConnections = connections.filter(conn => conn.target === nodeId);
  
  for (const conn of incomingConnections) {
    const sourceNode = nodes.find(n => n.id === conn.source);
    if (sourceNode?.type === 'hf-token') {
      return sourceNode;
    }
  }

  return undefined;
}

/**
 * 토큰 값 가져오기
 */
export function getTokenValue(tokenNode: Node | undefined): string {
  return tokenNode?.data?.token || '';
}

/**
 * 토큰이 유효한지 확인
 */
export function isValidToken(token: string): boolean {
  return token.length > 0 && token.startsWith('hf_');
}

/**
 * 토큰이 기본 토큰인지 확인
 */
export function isDefaultToken(tokenNode: Node | undefined): boolean {
  return tokenNode?.data?.saveAsDefault === true;
}

