/**
 * 노드 연결 규칙 유틸리티
 * LLM 파인튜닝 워크플로우에 맞는 연결 규칙 정의
 */

import { Node, Port, Connection } from '../types/editor';

// 포트 타입별 연결 가능 여부
export function canConnectPorts(
  sourcePort: Port,
  targetPort: Port,
  sourceNode: Node,
  targetNode: Node
): { canConnect: boolean; reason?: string; color?: string } {
  // 같은 노드는 연결 불가
  if (sourceNode.id === targetNode.id) {
    return { canConnect: false, reason: '같은 노드는 연결할 수 없습니다', color: 'red' };
  }

  // 입출력 방향 검증
  if (sourcePort.type === 'output' && targetPort.type === 'input') {
    // 출력 → 입력: 정상 연결
    return checkDataTypeCompatibility(sourcePort.dataType, targetPort.dataType, sourceNode, targetNode);
  } else if (sourcePort.type === 'input' && targetPort.type === 'output') {
    // 입력 ← 출력: 역방향 연결 (허용)
    return checkDataTypeCompatibility(targetPort.dataType, sourcePort.dataType, targetNode, sourceNode);
  } else {
    return { canConnect: false, reason: '출력 포트는 입력 포트에만 연결할 수 있습니다', color: 'red' };
  }
}

// 데이터 타입 호환성 검사
function checkDataTypeCompatibility(
  sourceDataType: string,
  targetDataType: string,
  sourceNode: Node,
  targetNode: Node
): { canConnect: boolean; reason?: string; color?: string } {
  // 정확히 일치하는 경우
  if (sourceDataType === targetDataType) {
    return { canConnect: true, color: getDataTypeColor(sourceDataType) };
  }

  // 특수 규칙: config 타입은 어디든 연결 가능
  if (sourceDataType === 'config' || targetDataType === 'config') {
    return { canConnect: true, color: 'purple' };
  }

  // 워크플로우 규칙에 따른 연결 가능 여부
  const workflowRules = getWorkflowConnectionRules(sourceNode.type, targetNode.type);
  if (workflowRules.canConnect) {
    return { canConnect: true, color: workflowRules.color || getDataTypeColor(sourceDataType) };
  }

  return { 
    canConnect: false, 
    reason: `데이터 타입이 호환되지 않습니다 (${sourceDataType} → ${targetDataType})`,
    color: 'red' 
  };
}

// 데이터 타입별 색상
function getDataTypeColor(dataType: string): string {
  switch (dataType) {
    case 'model':
      return 'blue';
    case 'dataset':
      return 'green';
    case 'token':
    case 'config':
      return 'yellow';
    case 'text':
      return 'gray';
    default:
      return 'gray';
  }
}

// 워크플로우 연결 규칙
function getWorkflowConnectionRules(
  sourceType: string,
  targetType: string
): { canConnect: boolean; color?: string; reason?: string } {
  // LLM 파인튜닝 워크플로우 규칙

  // 1. 모델 로더 → 학습, 평가, 채팅, RAG, 저장, 내보내기
  if (['local-model-loader', 'hf-model-downloader', 'model-loader'].includes(sourceType)) {
    if (['training', 'model-evaluator', 'chat', 'rag', 'model-saver', 'gguf-export'].includes(targetType)) {
      return { canConnect: true, color: 'blue' };
    }
  }

  // 2. 데이터셋 로더 → 전처리, 필터, 분할
  if (['local-dataset-loader', 'hf-dataset-downloader', 'dataset-loader'].includes(sourceType)) {
    if (['dataset-preprocessor', 'dataset-filter', 'dataset-splitter'].includes(targetType)) {
      return { canConnect: true, color: 'green' };
    }
  }

  // 3. 전처리 → 분할, 필터, 학습
  if (sourceType === 'dataset-preprocessor') {
    if (['dataset-splitter', 'dataset-filter', 'training'].includes(targetType)) {
      return { canConnect: true, color: 'green' };
    }
  }

  // 4. 필터 → 분할, 학습
  if (sourceType === 'dataset-filter') {
    if (['dataset-splitter', 'training'].includes(targetType)) {
      return { canConnect: true, color: 'green' };
    }
  }

  // 5. 분할 → 학습, 평가
  if (sourceType === 'dataset-splitter') {
    if (['training', 'model-evaluator'].includes(targetType)) {
      return { canConnect: true, color: 'green' };
    }
  }

  // 6. 설정 노드 → 학습
  if (['training-config', 'lora-config', 'qlora-config', 'device-selector'].includes(sourceType)) {
    if (targetType === 'training') {
      return { canConnect: true, color: 'purple' };
    }
  }

  // 7. 학습 → 평가, 체크포인트 관리, 저장
  if (sourceType === 'training') {
    if (['model-evaluator', 'checkpoint-manager', 'model-saver'].includes(targetType)) {
      return { canConnect: true, color: 'blue' };
    }
  }

  // 8. 체크포인트 관리 → 평가, 저장, 내보내기
  if (sourceType === 'checkpoint-manager') {
    if (['model-evaluator', 'model-saver', 'gguf-export', 'chat', 'rag'].includes(targetType)) {
      return { canConnect: true, color: 'blue' };
    }
  }

  // 9. 토큰 → HF 다운로더
  if (sourceType === 'hf-token') {
    if (['hf-model-downloader', 'hf-dataset-downloader'].includes(targetType)) {
      return { canConnect: true, color: 'yellow' };
    }
  }

  // 10. 모델 → 전처리 (토크나이저용)
  if (['local-model-loader', 'hf-model-downloader', 'model-loader'].includes(sourceType)) {
    if (targetType === 'dataset-preprocessor') {
      return { canConnect: true, color: 'blue', reason: '토크나이저용 (선택사항)' };
    }
  }

  return { canConnect: false };
}

// 포트 호버 시 연결 가능 여부 확인
export function checkPortConnection(
  sourceNode: Node,
  sourcePort: Port,
  targetNode: Node,
  targetPort: Port
): { canConnect: boolean; color: string; reason?: string } {
  const result = canConnectPorts(sourcePort, targetPort, sourceNode, targetNode);
  return {
    canConnect: result.canConnect,
    color: result.color || 'gray',
    reason: result.reason,
  };
}

