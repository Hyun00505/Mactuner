/**
 * 노드 정의 로더
 * JSON 파일에서 노드 템플릿을 로드합니다
 */

export interface ParameterOption {
  label: string;
  value: string | number | boolean;
}

export interface ParameterCondition {
  parameter: string;
  operator: "equals" | "not_equals" | "contains" | "greater_than" | "less_than";
  value: any;
}

export interface NodeParameter {
  id: string;
  name: string;
  type: "text" | "number" | "select" | "checkbox" | "textarea" | "password" | "file" | "multiselect" | "checkbox-group";
  label: string;
  value: any;
  placeholder?: string;
  required: boolean;
  description?: string;
  options?: ParameterOption[];
  dynamicOptions?: boolean;
  dynamicSource?: string;
  apiEndpoint?: string;
  condition?: ParameterCondition;
  visibleWhen?: Record<string, any>; // Format에 따른 조건부 표시
  min?: number;
  max?: number;
  step?: number;
}

export interface NodePort {
  id: string;
  name: string;
  type: "input" | "output";
  dataType: string;
}

export interface NodeDefinition {
  id: string;
  name: string;
  icon: string;
  category: "input" | "process" | "output";
  description: string;
  color: string;
  inputs: NodePort[];
  outputs: NodePort[];
  parameters: NodeParameter[];
}

// 노드 정의 맵
const nodeDefinitions: Map<string, NodeDefinition> = new Map();

/**
 * 모든 노드 정의 로드
 */
export async function loadAllNodeDefinitions(): Promise<Map<string, NodeDefinition>> {
  const nodeIds = ["ModelLoaderNode", "DatasetLoaderNode", "TrainingNode", "ChatNode", "RagNode", "GgufExportNode"];

  for (const nodeId of nodeIds) {
    try {
      const module = await import(`../@nodes/${nodeId}.json`);
      const definition = module.default || module;
      nodeDefinitions.set(definition.id, definition);
    } catch (error) {
      console.warn(`Failed to load node definition: ${nodeId}`, error);
    }
  }

  return nodeDefinitions;
}

/**
 * 특정 노드 정의 로드
 */
export async function loadNodeDefinition(nodeType: string): Promise<NodeDefinition | null> {
  if (nodeDefinitions.has(nodeType)) {
    return nodeDefinitions.get(nodeType) || null;
  }

  try {
    // 노드 ID에서 파일명 생성
    let fileName: string;

    // 특수 케이스 처리
    const specialCases: Record<string, string> = {
      "hf-token": "HFTokenNode",
      "device-selector": "DeviceSelectorNode",
      "hf-model-downloader": "HFModelDownloaderNode",
      "hf-dataset-downloader": "HFDatasetDownloaderNode",
      "local-model-loader": "LocalModelLoaderNode",
      "local-dataset-loader": "LocalDatasetLoaderNode",
      "dataset-preprocessor": "DatasetPreprocessorNode",
      "dataset-splitter": "DatasetSplitterNode",
      "dataset-filter": "DatasetFilterNode",
      "training-config": "TrainingConfigNode",
      "lora-config": "LoRAConfigNode",
      "qlora-config": "QLoRAConfigNode",
      "model-evaluator": "ModelEvaluatorNode",
      "checkpoint-manager": "CheckpointManagerNode",
      "model-saver": "ModelSaverNode",
      training: "TrainingNode",
      chat: "ChatNode",
      rag: "RagNode",
      "gguf-export": "GgufExportNode",
    };

    if (specialCases[nodeType]) {
      fileName = specialCases[nodeType];
    } else {
      // 일반적인 변환: model-loader → ModelLoaderNode
      fileName =
        nodeType
          .split("-")
          .map((part) => {
            return part.charAt(0).toUpperCase() + part.slice(1);
          })
          .join("") + "Node";
    }

    const module = await import(`../@nodes/${fileName}.json`);
    const definition = module.default || module;
    nodeDefinitions.set(nodeType, definition);
    return definition;
  } catch (error) {
    console.error(`Failed to load node definition: ${nodeType}`, error);
    return null;
  }
}

/**
 * 모든 노드 정의 가져오기
 */
export function getNodeDefinition(nodeType: string): NodeDefinition | null {
  return nodeDefinitions.get(nodeType) || null;
}

/**
 * 모든 노드 정의 목록
 */
export function getAllNodeDefinitions(): NodeDefinition[] {
  return Array.from(nodeDefinitions.values());
}

/**
 * 파라미터가 조건을 만족하는지 확인
 */
export function checkParameterCondition(condition: ParameterCondition, parameterValues: Record<string, any>): boolean {
  const paramValue = parameterValues[condition.parameter];

  switch (condition.operator) {
    case "equals":
      return paramValue === condition.value;
    case "not_equals":
      return paramValue !== condition.value;
    case "contains":
      return String(paramValue).includes(String(condition.value));
    case "greater_than":
      return Number(paramValue) > Number(condition.value);
    case "less_than":
      return Number(paramValue) < Number(condition.value);
    default:
      return true;
  }
}

/**
 * 표시할 파라미터 필터링
 */
export function getVisibleParameters(parameters: NodeParameter[] | undefined, parameterValues: Record<string, any>): NodeParameter[] {
  if (!parameters || !Array.isArray(parameters)) {
    return [];
  }
  return parameters.filter((param) => {
    // 기존 condition 체크
    if (param.condition) {
      if (!checkParameterCondition(param.condition, parameterValues)) {
        return false;
      }
    }

    // visibleWhen 체크 (Format에 따른 조건부 표시)
    if (param.visibleWhen) {
      for (const [key, value] of Object.entries(param.visibleWhen)) {
        if (parameterValues[key] !== value) {
          return false;
        }
      }
    }

    return true;
  });
}

/**
 * 노드 기본값 생성
 */
export function createNodeDefaultValues(definition: NodeDefinition): Record<string, any> {
  const values: Record<string, any> = {};

  if (!definition.parameters || !Array.isArray(definition.parameters)) {
    return values;
  }

  for (const param of definition.parameters) {
    values[param.id] = param.value;
  }

  return values;
}

/**
 * 파라미터 유효성 검사
 */
export function validateNodeParameters(parameters: NodeParameter[], values: Record<string, any>): { valid: boolean; errors: Record<string, string> } {
  const errors: Record<string, string> = {};

  for (const param of parameters) {
    // 필수 파라미터 확인
    if (param.required && !values[param.id]) {
      errors[param.id] = `${param.label} is required`;
    }

    // 숫자 범위 확인
    if (param.type === "number" && values[param.id] !== undefined) {
      const value = Number(values[param.id]);
      if (param.min !== undefined && value < param.min) {
        errors[param.id] = `Must be at least ${param.min}`;
      }
      if (param.max !== undefined && value > param.max) {
        errors[param.id] = `Must be at most ${param.max}`;
      }
    }

    // 파일 확인
    if (param.type === "file" && param.required && !values[param.id]) {
      errors[param.id] = `${param.label} is required`;
    }
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}

/**
 * 파라미터를 API 요청 형식으로 변환
 */
export function parametersToApiRequest(definition: NodeDefinition, values: Record<string, any>): Record<string, any> {
  const request: Record<string, any> = {
    nodeType: definition.id,
    parameters: {},
  };

  for (const param of definition.parameters) {
    request.parameters[param.id] = values[param.id];
  }

  return request;
}

/**
 * 동적 옵션 가져오기 (API에서)
 */
export async function fetchDynamicOptions(apiEndpoint: string): Promise<ParameterOption[]> {
  try {
    const response = await fetch(`http://localhost:8001${apiEndpoint}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();

    // API 응답 형식에 따라 변환
    // 1. 배열 형식
    if (Array.isArray(data)) {
      return data.map((item: any) => ({
        label: typeof item === "string" ? item : item.name || item.label,
        value: typeof item === "string" ? item : item.id || item.value,
      }));
    }

    // 2. models 객체 형식 (백엔드 local-models 응답)
    if (data.models && Array.isArray(data.models)) {
      return (
        data.models
          // 데이터셋 제외 (모델만 필터링)
          .filter((item: any) => {
            // 데이터셋 제외
            if (item.source === "huggingface_dataset") return false;
            // model_id가 "datasets/"로 시작하면 제외
            if (item.model_id && item.model_id.startsWith("datasets/")) return false;
            return true;
          })
          .map((item: any) => {
            // 소스 구분
            const icon = item.source === "huggingface" ? "🤗" : "📂";
            const label = `${icon} ${item.model_id}${item.size_gb ? ` (${item.size_gb}GB)` : ""}`;
            return {
              label,
              value: item.model_id,
            };
          })
      );
    }

    // 2b. datasets 객체 형식 (백엔드 local-datasets 응답)
    if (data.datasets && Array.isArray(data.datasets)) {
      return data.datasets.map((item: any) => {
        // 소스 구분
        const icon = item.source === "huggingface" ? "🤗" : "📤";
        const sizeLabel = item.size_mb ? ` (${item.size_mb}MB)` : "";
        const label = `${icon} ${item.dataset_id}${sizeLabel}`;
        return {
          label,
          value: item.dataset_id,
        };
      });
    }

    // 3. items 객체 형식
    if (data.items && Array.isArray(data.items)) {
      return data.items.map((item: any) => ({
        label: item.name || item.label,
        value: item.id || item.value,
      }));
    }

    // 4. 단순 배열 문자열
    if (data.data && Array.isArray(data.data)) {
      return data.data.map((item: any) => ({
        label: typeof item === "string" ? item : item.name || item.label,
        value: typeof item === "string" ? item : item.id || item.value,
      }));
    }

    return [];
  } catch (error) {
    console.error("Failed to fetch dynamic options:", error);
    return [];
  }
}

/**
 * 노드 정의를 에디터 스토어 노드로 변환
 */
export function definitionToNodeData(definition: NodeDefinition, nodeId: string, position: { x: number; y: number }): any {
  return {
    id: nodeId,
    type: definition.id,
    position,
    data: createNodeDefaultValues(definition),
    ports: [...definition.inputs, ...definition.outputs],
    status: "idle",
  };
}
