/**
 * Editor 타입 정의
 * 노드 기반 워크플로우 시스템
 */

// ====================================
// 노드 타입
// ====================================

export type NodeType = 
  | 'hf-token'
  | 'device-selector'
  | 'local-model-loader'
  | 'hf-model-downloader'
  | 'local-dataset-loader'
  | 'hf-dataset-downloader'
  | 'dataset-preprocessor'
  | 'dataset-splitter'
  | 'dataset-filter'
  | 'training-config'
  | 'lora-config'
  | 'qlora-config'
  | 'training'
  | 'model-evaluator'
  | 'checkpoint-manager'
  | 'model-saver'
  | 'chat'
  | 'rag'
  | 'gguf-export';

export type NodeStatus = 'idle' | 'running' | 'completed' | 'error' | 'pending';

// 노드 포트 (입출력)
export interface Port {
  id: string;
  name: string;
  type: 'input' | 'output';
  dataType: 'model' | 'dataset' | 'text' | 'config' | 'token' | 'any';
}

// 노드 기본 인터페이스
export interface BaseNode {
  id: string;
  type: NodeType;
  position: { x: number; y: number };
  data: Record<string, any>;
  ports: Port[];
  status: NodeStatus;
  progress?: number;
  error?: string;
}

// HuggingFace 토큰 노드
export interface HFTokenNode extends BaseNode {
  type: 'hf-token';
  data: {
    token: string;
    tokenType: 'user' | 'fine-grained';
    tokenName?: string;
    permissions?: 'read' | 'write' | 'admin';
    saveAsDefault?: boolean;
  };
}

// 디바이스 선택 노드
export interface DeviceSelectorNode extends BaseNode {
  type: 'device-selector';
  data: {
    deviceType: 'auto' | 'cpu' | 'mps' | 'cuda';
  };
}

// 로컬 모델 로더 노드
export interface LocalModelLoaderNode extends BaseNode {
  type: 'local-model-loader';
  data: {
    modelPath: string;
    modelId?: string;
    loadStatus?: 'idle' | 'loading' | 'completed' | 'error';
    loadProgress?: number;
  };
}

// HuggingFace 모델 다운로더 노드
export interface HFModelDownloaderNode extends BaseNode {
  type: 'hf-model-downloader';
  data: {
    modelId: string;
    accessToken?: string;
    downloadStatus?: 'idle' | 'downloading' | 'completed' | 'error';
    downloadProgress?: number;
  };
}

// 로컬 데이터셋 로더 노드
export interface LocalDatasetLoaderNode extends BaseNode {
  type: 'local-dataset-loader';
  data: {
    datasetPath: string;
    datasetId?: string;
    dataFormat: 'csv' | 'json' | 'parquet';
    loadStatus?: 'idle' | 'loading' | 'completed' | 'error';
    loadProgress?: number;
    dataInfo?: {
      rows: number;
      columns: number;
      size: string;
    };
  };
}

// HuggingFace 데이터셋 다운로더 노드
export interface HFDatasetDownloaderNode extends BaseNode {
  type: 'hf-dataset-downloader';
  data: {
    datasetId: string;
    accessToken?: string;
    downloadStatus?: 'idle' | 'downloading' | 'completed' | 'error';
    downloadProgress?: number;
  };
}

// 모델 로더 노드 (레거시 - 제거 예정)
export interface ModelLoaderNode extends BaseNode {
  type: 'model-loader';
  data: {
    modelId: string;
    source: 'huggingface' | 'local';
    accessToken?: string;
    downloadStatus?: 'idle' | 'downloading' | 'completed' | 'error';
    downloadProgress?: number;
  };
}

// 데이터셋 노드 (레거시 - 제거 예정)
export interface DatasetLoaderNode extends BaseNode {
  type: 'dataset-loader';
  data: {
    filePath?: string;
    fileName?: string;
    dataFormat: 'csv' | 'json' | 'parquet';
    uploadStatus?: 'idle' | 'uploading' | 'completed' | 'error';
    uploadProgress?: number;
    dataInfo?: {
      rows: number;
      columns: number;
      size: string;
    };
  };
}

// 데이터셋 전처리 노드
export interface DatasetPreprocessorNode extends BaseNode {
  type: 'dataset-preprocessor';
  data: {
    tokenizerName?: string;
    maxLength: number;
    padding: 'max_length' | 'longest' | 'do_not_pad';
    truncation: boolean;
    format: 'instruction' | 'completion' | 'chat';
    template?: string;
    processStatus?: 'idle' | 'processing' | 'completed' | 'error';
    processedRows?: number;
  };
}

// 데이터셋 분할 노드
export interface DatasetSplitterNode extends BaseNode {
  type: 'dataset-splitter';
  data: {
    trainRatio: number;
    valRatio: number;
    testRatio: number;
    shuffle: boolean;
    seed?: number;
    splitStatus?: 'idle' | 'splitting' | 'completed' | 'error';
    trainSize?: number;
    valSize?: number;
    testSize?: number;
  };
}

// 데이터셋 필터 노드
export interface DatasetFilterNode extends BaseNode {
  type: 'dataset-filter';
  data: {
    filterType: 'length' | 'quality' | 'custom';
    minLength?: number;
    maxLength?: number;
    filterExpression?: string;
    filterStatus?: 'idle' | 'filtering' | 'completed' | 'error';
    filteredRows?: number;
    originalRows?: number;
  };
}

// 학습 설정 노드
export interface TrainingConfigNode extends BaseNode {
  type: 'training-config';
  data: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    warmupSteps: number;
    maxGradNorm: number;
    saveStrategy: 'epoch' | 'steps' | 'no';
    saveSteps?: number;
    evalStrategy: 'epoch' | 'steps' | 'no';
    evalSteps?: number;
    loggingSteps: number;
    outputDir: string;
  };
}

// LoRA 설정 노드
export interface LoRAConfigNode extends BaseNode {
  type: 'lora-config';
  data: {
    rank: number;
    alpha: number;
    dropout: number;
    targetModules: string[];
    bias: 'none' | 'all' | 'lora_only';
    taskType: 'CAUSAL_LM' | 'SEQ_2_SEQ_LM' | 'SEQ_CLS' | 'TOKEN_CLS';
  };
}

// QLoRA 설정 노드
export interface QLoRAConfigNode extends BaseNode {
  type: 'qlora-config';
  data: {
    bits: 4 | 8;
    quantType: 'nf4' | 'fp4';
    doubleQuant: boolean;
    computeDtype: 'float16' | 'bfloat16' | 'float32';
    loraRank: number;
    loraAlpha: number;
    loraDropout: number;
    targetModules: string[];
  };
}

// 모델 평가 노드
export interface ModelEvaluatorNode extends BaseNode {
  type: 'model-evaluator';
  data: {
    metrics: string[];
    evalBatchSize: number;
    evalStatus?: 'idle' | 'evaluating' | 'completed' | 'error';
    results?: Record<string, number>;
  };
}

// 체크포인트 관리 노드
export interface CheckpointManagerNode extends BaseNode {
  type: 'checkpoint-manager';
  data: {
    maxCheckpoints: number;
    keepBest: boolean;
    saveStrategy: 'all' | 'best' | 'last';
    checkpointDir: string;
    status?: 'idle' | 'managing' | 'completed' | 'error';
    checkpoints?: Array<{
      path: string;
      epoch: number;
      loss: number;
      timestamp: number;
    }>;
  };
}

// 모델 저장 노드
export interface ModelSaverNode extends BaseNode {
  type: 'model-saver';
  data: {
    savePath: string;
    saveFormat: 'pytorch' | 'safetensors' | 'both';
    includeTokenizer: boolean;
    includeConfig: boolean;
    saveStatus?: 'idle' | 'saving' | 'completed' | 'error';
    savedPath?: string;
  };
}

// 학습 노드
export interface TrainingNode extends BaseNode {
  type: 'training';
  data: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    useLora: boolean;
    useQlora: boolean;
    loraRank?: number;
    loraAlpha?: number;
    loraDropout?: number;
    warmupSteps: number;
    maxGradNorm: number;
    trainingStatus?: 'idle' | 'preparing' | 'running' | 'completed' | 'error';
    currentEpoch?: number;
    totalEpochs?: number;
    loss?: number;
    evalLoss?: number;
  };
}

// Chat 노드
export interface ChatNode extends BaseNode {
  type: 'chat';
  data: {
    systemPrompt: string;
    temperature: number;
    maxTokens: number;
    topP: number;
    repeatPenalty: number;
    messages: Array<{
      role: 'user' | 'assistant';
      content: string;
      timestamp: number;
    }>;
    chatStatus?: 'idle' | 'generating' | 'completed' | 'error';
  };
}

// RAG 노드
export interface RagNode extends BaseNode {
  type: 'rag';
  data: {
    documentPaths: string[];
    chunkSize: number;
    chunkOverlap: number;
    topK: number;
    searchQuery?: string;
    ragStatus?: 'idle' | 'loading' | 'searching' | 'completed' | 'error';
    documentsLoaded?: number;
    searchResults?: Array<{
      content: string;
      score: number;
      source: string;
    }>;
  };
}

// GGUF 내보내기 노드
export interface GgufExportNode extends BaseNode {
  type: 'gguf-export';
  data: {
    quantizationMethod: 'Q2_K' | 'Q3_K' | 'Q4_0' | 'Q4_K' | 'Q5_0' | 'Q5_K' | 'Q6_K' | 'Q8_0' | 'F16' | 'F32';
    outputPath: string;
    exportStatus?: 'idle' | 'converting' | 'completed' | 'error';
    exportProgress?: number;
    fileSize?: string;
  };
}

// 모든 노드 타입의 합집합
export type Node = 
  | HFTokenNode
  | DeviceSelectorNode
  | LocalModelLoaderNode
  | HFModelDownloaderNode
  | LocalDatasetLoaderNode
  | HFDatasetDownloaderNode
  | DatasetPreprocessorNode
  | DatasetSplitterNode
  | DatasetFilterNode
  | TrainingConfigNode
  | LoRAConfigNode
  | QLoRAConfigNode
  | TrainingNode
  | ModelEvaluatorNode
  | CheckpointManagerNode
  | ModelSaverNode
  | ChatNode
  | RagNode
  | GgufExportNode
  | ModelLoaderNode // 레거시
  | DatasetLoaderNode; // 레거시

// ====================================
// 연결 (Connection)
// ====================================

export interface Connection {
  id: string;
  source: string;  // 노드 ID
  sourcePort: string;  // 포트 ID
  target: string;  // 노드 ID
  targetPort: string;  // 포트 ID
  dataType: 'model' | 'dataset' | 'text' | 'config' | 'any';
}

// ====================================
// 워크플로우
// ====================================

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  nodes: Node[];
  connections: Connection[];
  createdAt: number;
  updatedAt: number;
  version: string;
}

// ====================================
// 실행 관련
// ====================================

export interface ExecutionContext {
  nodeId: string;
  inputData: Record<string, any>;
  outputs: Record<string, any>;
  status: NodeStatus;
  error?: string;
  startTime?: number;
  endTime?: number;
  duration?: number;
  trainingLogs?: Array<{
    timestamp: number;
    message: string;
    data?: any;
  }>;
}

export interface ExecutionLog {
  id: string;
  workflowId: string;
  startTime: number;
  endTime?: number;
  status: 'running' | 'completed' | 'error';
  nodeExecutions: ExecutionContext[];
  totalDuration?: number;
}

// ====================================
// UI 상태
// ====================================

export interface EditorState {
  // 워크플로우
  currentWorkflow: Workflow | null;
  workflows: Workflow[];
  
  // 노드 관리
  nodes: Node[];
  connections: Connection[];
  selectedNodeId: string | null;
  selectedConnectionId: string | null;
  
  // 캔버스
  zoom: number;
  panX: number;
  panY: number;
  canvasMode: 'select' | 'pan'; // 선택 모드 또는 팬 모드
  
  // UI
  showNodePalette: boolean;
  showPropertiesPanel: boolean;
  showOutputPanel: boolean;
  showExecutionLog: boolean;
  
  // 실행
  isExecuting: boolean;
  executionLog: ExecutionLog | null;
  
  // 상태
  isDirty: boolean;
}

// ====================================
// 노드 정의 (템플릿)
// ====================================

export interface NodeTemplate {
  id: string;
  name: string;
  type: NodeType;
  description: string;
  icon: string;
  category: 'input' | 'process' | 'output';
  ports: Port[];
  defaultData: Record<string, any>;
  color: string;
}

// ====================================
// API 요청/응답
// ====================================

export interface SaveWorkflowRequest {
  name: string;
  description?: string;
  workflow: Workflow;
}

export interface ExecuteWorkflowRequest {
  workflowId: string;
  nodes: Node[];
  connections: Connection[];
}

export interface ExecuteWorkflowResponse {
  id: string;
  workflowId: string;
  status: 'queued' | 'running' | 'completed' | 'error';
  results: Record<string, any>;
  logs: string[];
  duration: number;
}

// ====================================
// 도우미 함수 타입
// ====================================

export type NodeFactory = (position: { x: number; y: number }) => Node;

export interface NodeRegistry {
  [key in NodeType]: {
    template: NodeTemplate;
    factory: NodeFactory;
    component: React.ComponentType<any>;
  };
}

