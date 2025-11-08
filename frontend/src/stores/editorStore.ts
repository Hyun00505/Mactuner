/**
 * Editor Store (Zustand)
 * 워크플로우, 노드, 연결, 실행 상태를 관리합니다.
 */

import { create } from "zustand";
import {
  Workflow,
  Node,
  Connection,
  EditorState,
  ExecutionLog,
  NodeType,
  NodeStatus,
  HFTokenNode,
  DeviceSelectorNode,
  LocalModelLoaderNode,
  HFModelDownloaderNode,
  LocalDatasetLoaderNode,
  HFDatasetDownloaderNode,
  DatasetPreprocessorNode,
  DatasetSplitterNode,
  DatasetFilterNode,
  TrainingConfigNode,
  LoRAConfigNode,
  QLoRAConfigNode,
  TrainingNode,
  ModelEvaluatorNode,
  CheckpointManagerNode,
  ModelSaverNode,
  ModelLoaderNode,
  DatasetLoaderNode,
  ChatNode,
  RagNode,
  GgufExportNode,
} from "../types/editor";

interface EditorStore extends EditorState {
  // 워크플로우 관리
  createNewWorkflow: () => void;
  loadWorkflow: (workflow: Workflow) => void;
  loadWorkflowById: (id: string) => void;
  listWorkflows: () => Workflow[];
  saveWorkflow: (name: string, description?: string) => Promise<void>;
  deleteWorkflow: (id: string) => Promise<void>;
  createExampleWorkflow: () => Promise<void>;
  exportWorkflowToJSON: () => void;
  importWorkflowFromJSON: (file: File) => Promise<void>;
  exportWorkflowToFile: () => Promise<void>;
  importWorkflowFromFile: (file: File) => Promise<void>;

  // 노드 관리
  addNode: (type: NodeType, position: { x: number; y: number }) => void;
  deleteNode: (id: string) => void;
  updateNode: (id: string, data: Partial<Node>) => void;
  selectNode: (id: string) => void;
  clearNodeSelection: () => void;

  // 연결 관리
  addConnection: (connection: Connection) => void;
  deleteConnection: (id: string) => void;
  selectConnection: (id: string) => void;
  clearConnectionSelection: () => void;

  // 캔버스 관리
  setZoom: (zoom: number) => void;
  setPan: (x: number, y: number) => void;
  setCanvasMode: (mode: 'select' | 'pan') => void;

  // UI 토글
  toggleNodePalette: () => void;
  togglePropertiesPanel: () => void;
  toggleOutputPanel: () => void;
  toggleExecutionLog: () => void;

  // 실행 관리
  executeWorkflow: () => Promise<void>;
  setExecutionLog: (log: ExecutionLog) => void;
  clearExecutionLog: () => void;

  // 상태 관리
  setDirty: (dirty: boolean) => void;
  resetWorkflow: () => void;
}

// 기본 노드 ID 생성
const generateNodeId = (type: NodeType) => {
  return `${type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// 연결 ID 생성
const generateConnectionId = () => {
  return `conn-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// 기본 노드 생성 함수들
const createHFTokenNode = (position: { x: number; y: number }): HFTokenNode => ({
  id: generateNodeId("hf-token"),
  type: "hf-token",
  position,
  data: {
    token: "",
    tokenType: "user",
    tokenName: "My Token",
    permissions: "read",
    saveAsDefault: false,
  },
  ports: [{ id: "out-token", name: "Token", type: "output", dataType: "config" }],
  status: "idle",
});

const createDeviceSelectorNode = (position: { x: number; y: number }): DeviceSelectorNode => ({
  id: generateNodeId("device-selector"),
  type: "device-selector",
  position,
  data: {
    deviceType: "auto",
  },
  ports: [{ id: "out-device", name: "Device", type: "output", dataType: "config" }],
  status: "idle",
});

const createModelLoaderNode = (position: { x: number; y: number }): ModelLoaderNode => ({
  id: generateNodeId("model-loader"),
  type: "model-loader",
  position,
  data: {
    modelId: "gpt2",
    source: "huggingface",
    accessToken: "",
    downloadStatus: "idle",
    downloadProgress: 0,
  },
  ports: [{ id: "out-model", name: "Model", type: "output", dataType: "model" }],
  status: "idle",
});

const createDatasetLoaderNode = (position: { x: number; y: number }): DatasetLoaderNode => ({
  id: generateNodeId("dataset-loader"),
  type: "dataset-loader",
  position,
  data: {
    fileName: "",
    dataFormat: "csv",
    uploadStatus: "idle",
    uploadProgress: 0,
    dataInfo: { rows: 0, columns: 0, size: "0 KB" },
  },
  ports: [{ id: "out-dataset", name: "Dataset", type: "output", dataType: "dataset" }],
  status: "idle",
});

const createLocalModelLoaderNode = (position: { x: number; y: number }): LocalModelLoaderNode => ({
  id: generateNodeId("local-model-loader"),
  type: "local-model-loader",
  position,
  data: {
    modelPath: "",
    modelId: "",
    loadStatus: "idle",
    loadProgress: 0,
  },
  ports: [{ id: "out-model", name: "Model", type: "output", dataType: "model" }],
  status: "idle",
});

const createHFModelDownloaderNode = (position: { x: number; y: number }): HFModelDownloaderNode => ({
  id: generateNodeId("hf-model-downloader"),
  type: "hf-model-downloader",
  position,
  data: {
    modelId: "",
    accessToken: "",
    downloadStatus: "idle",
    downloadProgress: 0,
  },
  ports: [
    { id: "in-token", name: "Token", type: "input", dataType: "token" },
    { id: "out-model", name: "Model", type: "output", dataType: "model" },
  ],
  status: "idle",
});

const createLocalDatasetLoaderNode = (position: { x: number; y: number }): LocalDatasetLoaderNode => ({
  id: generateNodeId("local-dataset-loader"),
  type: "local-dataset-loader",
  position,
  data: {
    datasetPath: "",
    datasetId: "",
    dataFormat: "csv",
    loadStatus: "idle",
    loadProgress: 0,
    dataInfo: { rows: 0, columns: 0, size: "0 KB" },
  },
  ports: [{ id: "out-dataset", name: "Dataset", type: "output", dataType: "dataset" }],
  status: "idle",
});

const createHFDatasetDownloaderNode = (position: { x: number; y: number }): HFDatasetDownloaderNode => ({
  id: generateNodeId("hf-dataset-downloader"),
  type: "hf-dataset-downloader",
  position,
  data: {
    datasetId: "",
    accessToken: "",
    downloadStatus: "idle",
    downloadProgress: 0,
  },
  ports: [
    { id: "in-token", name: "Token", type: "input", dataType: "token" },
    { id: "out-dataset", name: "Dataset", type: "output", dataType: "dataset" },
  ],
  status: "idle",
});

const createTrainingNode = (position: { x: number; y: number }): TrainingNode => ({
  id: generateNodeId("training"),
  type: "training",
  position,
  data: {
    epochs: 3,
    batchSize: 4,
    learningRate: 5e-5,
    useLora: true,
    useQlora: false,
    loraRank: 8,
    loraAlpha: 16,
    loraDropout: 0.1,
    warmupSteps: 500,
    maxGradNorm: 1.0,
    trainingStatus: "idle",
    currentEpoch: 0,
    totalEpochs: 3,
    loss: 0,
    evalLoss: 0,
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "in-train-dataset", name: "Train Dataset", type: "input", dataType: "dataset" },
    { id: "in-val-dataset", name: "Validation Dataset", type: "input", dataType: "dataset", optional: true },
    { id: "in-training-config", name: "Training Config", type: "input", dataType: "config", optional: true },
    { id: "in-lora-config", name: "LoRA Config", type: "input", dataType: "config", optional: true },
    { id: "in-qlora-config", name: "QLoRA Config", type: "input", dataType: "config", optional: true },
    { id: "in-device", name: "Device", type: "input", dataType: "config", optional: true },
    { id: "out-trained-model", name: "Trained Model", type: "output", dataType: "model" },
    { id: "out-checkpoints", name: "Checkpoints", type: "output", dataType: "config" },
  ],
  status: "idle",
});

const createChatNode = (position: { x: number; y: number }): ChatNode => ({
  id: generateNodeId("chat"),
  type: "chat",
  position,
  data: {
    systemPrompt: "You are a helpful AI assistant.",
    temperature: 0.7,
    maxTokens: 1024,
    topP: 0.9,
    repeatPenalty: 1.1,
    messages: [],
    chatStatus: "idle",
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "out-response", name: "Response", type: "output", dataType: "text" },
  ],
  status: "idle",
});

const createRagNode = (position: { x: number; y: number }): RagNode => ({
  id: generateNodeId("rag"),
  type: "rag",
  position,
  data: {
    documentPaths: [],
    chunkSize: 512,
    chunkOverlap: 50,
    topK: 5,
    searchQuery: "",
    ragStatus: "idle",
    documentsLoaded: 0,
    searchResults: [],
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "out-results", name: "Search Results", type: "output", dataType: "text" },
  ],
  status: "idle",
});

const createGgufExportNode = (position: { x: number; y: number }): GgufExportNode => ({
  id: generateNodeId("gguf-export"),
  type: "gguf-export",
  position,
  data: {
    quantizationMethod: "Q4_K",
    outputPath: "output/model.gguf",
    exportStatus: "idle",
    exportProgress: 0,
    fileSize: "0 MB",
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "out-gguf", name: "GGUF File", type: "output", dataType: "any" },
  ],
  status: "idle",
});

const createDatasetPreprocessorNode = (position: { x: number; y: number }): DatasetPreprocessorNode => ({
  id: generateNodeId("dataset-preprocessor"),
  type: "dataset-preprocessor",
  position,
  data: {
    tokenizerName: "",
    maxLength: 512,
    padding: "max_length",
    truncation: true,
    format: "instruction",
    template: "",
    processStatus: "idle",
    processedRows: 0,
  },
  ports: [
    { id: "in-dataset", name: "Dataset", type: "input", dataType: "dataset" },
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "out-dataset", name: "Processed Dataset", type: "output", dataType: "dataset" },
  ],
  status: "idle",
});

const createDatasetSplitterNode = (position: { x: number; y: number }): DatasetSplitterNode => ({
  id: generateNodeId("dataset-splitter"),
  type: "dataset-splitter",
  position,
  data: {
    trainRatio: 80,
    valRatio: 10,
    testRatio: 10,
    shuffle: true,
    seed: 42,
    splitStatus: "idle",
    trainSize: 0,
    valSize: 0,
    testSize: 0,
  },
  ports: [
    { id: "in-dataset", name: "Dataset", type: "input", dataType: "dataset" },
    { id: "out-train", name: "Train Dataset", type: "output", dataType: "dataset" },
    { id: "out-val", name: "Validation Dataset", type: "output", dataType: "dataset" },
    { id: "out-test", name: "Test Dataset", type: "output", dataType: "dataset" },
  ],
  status: "idle",
});

const createDatasetFilterNode = (position: { x: number; y: number }): DatasetFilterNode => ({
  id: generateNodeId("dataset-filter"),
  type: "dataset-filter",
  position,
  data: {
    filterType: "length",
    minLength: 10,
    maxLength: 2048,
    filterExpression: "",
    filterStatus: "idle",
    filteredRows: 0,
    originalRows: 0,
  },
  ports: [
    { id: "in-dataset", name: "Dataset", type: "input", dataType: "dataset" },
    { id: "out-dataset", name: "Filtered Dataset", type: "output", dataType: "dataset" },
  ],
  status: "idle",
});

const createTrainingConfigNode = (position: { x: number; y: number }): TrainingConfigNode => ({
  id: generateNodeId("training-config"),
  type: "training-config",
  position,
  data: {
    epochs: 3,
    batchSize: 4,
    learningRate: 5e-5,
    warmupSteps: 500,
    maxGradNorm: 1.0,
    saveStrategy: "epoch",
    saveSteps: 500,
    evalStrategy: "epoch",
    evalSteps: 500,
    loggingSteps: 10,
    outputDir: "./output",
  },
  ports: [{ id: "out-config", name: "Training Config", type: "output", dataType: "config" }],
  status: "idle",
});

const createLoRAConfigNode = (position: { x: number; y: number }): LoRAConfigNode => ({
  id: generateNodeId("lora-config"),
  type: "lora-config",
  position,
  data: {
    rank: 8,
    alpha: 16,
    dropout: 0.1,
    targetModules: ["q_proj", "v_proj"],
    bias: "none",
    taskType: "CAUSAL_LM",
  },
  ports: [{ id: "out-config", name: "LoRA Config", type: "output", dataType: "config" }],
  status: "idle",
});

const createQLoRAConfigNode = (position: { x: number; y: number }): QLoRAConfigNode => ({
  id: generateNodeId("qlora-config"),
  type: "qlora-config",
  position,
  data: {
    bits: 4,
    quantType: "nf4",
    doubleQuant: true,
    computeDtype: "float16",
    loraRank: 64,
    loraAlpha: 16,
    loraDropout: 0.1,
    targetModules: ["q_proj", "v_proj"],
  },
  ports: [{ id: "out-config", name: "QLoRA Config", type: "output", dataType: "config" }],
  status: "idle",
});

const createModelEvaluatorNode = (position: { x: number; y: number }): ModelEvaluatorNode => ({
  id: generateNodeId("model-evaluator"),
  type: "model-evaluator",
  position,
  data: {
    metrics: ["perplexity", "accuracy"],
    evalBatchSize: 8,
    evalStatus: "idle",
    results: {},
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "in-dataset", name: "Test Dataset", type: "input", dataType: "dataset" },
    { id: "out-results", name: "Evaluation Results", type: "output", dataType: "config" },
  ],
  status: "idle",
});

const createCheckpointManagerNode = (position: { x: number; y: number }): CheckpointManagerNode => ({
  id: generateNodeId("checkpoint-manager"),
  type: "checkpoint-manager",
  position,
  data: {
    maxCheckpoints: 5,
    keepBest: true,
    saveStrategy: "best",
    checkpointDir: "./checkpoints",
    status: "idle",
    checkpoints: [],
  },
  ports: [
    { id: "in-checkpoints", name: "Checkpoints", type: "input", dataType: "config" },
    { id: "out-best", name: "Best Checkpoint", type: "output", dataType: "model" },
    { id: "out-latest", name: "Latest Checkpoint", type: "output", dataType: "model" },
  ],
  status: "idle",
});

const createModelSaverNode = (position: { x: number; y: number }): ModelSaverNode => ({
  id: generateNodeId("model-saver"),
  type: "model-saver",
  position,
  data: {
    savePath: "./saved_models",
    saveFormat: "both",
    includeTokenizer: true,
    includeConfig: true,
    saveStatus: "idle",
    savedPath: "",
  },
  ports: [
    { id: "in-model", name: "Model", type: "input", dataType: "model" },
    { id: "out-path", name: "Saved Model Path", type: "output", dataType: "config" },
  ],
  status: "idle",
});

// 노드 팩토리 맵
const nodeFactories: Record<NodeType, any> = {
  "hf-token": createHFTokenNode,
  "device-selector": createDeviceSelectorNode,
  "local-model-loader": createLocalModelLoaderNode,
  "hf-model-downloader": createHFModelDownloaderNode,
  "local-dataset-loader": createLocalDatasetLoaderNode,
  "hf-dataset-downloader": createHFDatasetDownloaderNode,
  "dataset-preprocessor": createDatasetPreprocessorNode,
  "dataset-splitter": createDatasetSplitterNode,
  "dataset-filter": createDatasetFilterNode,
  "training-config": createTrainingConfigNode,
  "lora-config": createLoRAConfigNode,
  "qlora-config": createQLoRAConfigNode,
  training: createTrainingNode,
  "model-evaluator": createModelEvaluatorNode,
  "checkpoint-manager": createCheckpointManagerNode,
  "model-saver": createModelSaverNode,
  "model-loader": createModelLoaderNode, // 레거시
  "dataset-loader": createDatasetLoaderNode, // 레거시
  chat: createChatNode,
  rag: createRagNode,
  "gguf-export": createGgufExportNode,
};

// 기본 워크플로우
const createDefaultWorkflow = (): Workflow => ({
  id: `workflow-${Date.now()}`,
  name: "새 워크플로우",
  description: "",
  nodes: [],
  connections: [],
  createdAt: Date.now(),
  updatedAt: Date.now(),
  version: "1.0.0",
});

// 스토어 생성
export const useEditorStore = create<EditorStore>((set, get) => ({
  // 초기 상태
  currentWorkflow: createDefaultWorkflow(),
  workflows: [],
  nodes: [],
  connections: [],
  selectedNodeId: null,
  selectedConnectionId: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  canvasMode: 'select', // 기본값: 선택 모드
  showNodePalette: true,
  showPropertiesPanel: true,
  showOutputPanel: false,
  showExecutionLog: false,
  isExecuting: false,
  executionLog: null,
  isDirty: false,

  // 워크플로우 관리
  createNewWorkflow: () => {
    const newWorkflow = createDefaultWorkflow();
    set({
      currentWorkflow: newWorkflow,
      nodes: [],
      connections: [],
      selectedNodeId: null,
      selectedConnectionId: null,
      isDirty: false,
    });
  },

  loadWorkflow: (workflow: Workflow) => {
    set({
      currentWorkflow: workflow,
      nodes: workflow.nodes || [],
      connections: workflow.connections || [],
      selectedNodeId: null,
      selectedConnectionId: null,
      isDirty: false,
      zoom: 1,
      panX: 0,
      panY: 0,
    });
  },

  loadWorkflowById: (id: string) => {
    try {
      const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
      const workflow = workflows.find((w: Workflow) => w.id === id);
      if (workflow) {
        get().loadWorkflow(workflow);
      }
    } catch (error) {
      console.error("Failed to load workflow:", error);
    }
  },

  listWorkflows: (): Workflow[] => {
    try {
      return JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
    } catch (error) {
      console.error("Failed to list workflows:", error);
      return [];
    }
  },

  saveWorkflow: async (name: string, description?: string) => {
    const { currentWorkflow, nodes, connections } = get();
    if (!currentWorkflow) return;

    const updated = {
      ...currentWorkflow,
      name,
      description: description || "",
      nodes,
      connections,
      updatedAt: Date.now(),
    };

    try {
      // 백엔드 API에 저장
      const { workflowAPI } = await import("../utils/api");
      await workflowAPI.save(updated);

      // 로컬 스토리지에도 저장 (백업)
      const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
      const existingIndex = workflows.findIndex((w: Workflow) => w.id === updated.id);

      if (existingIndex >= 0) {
        workflows[existingIndex] = updated;
      } else {
        workflows.push(updated);
      }

      localStorage.setItem("mactuner_workflows", JSON.stringify(workflows));

      set({
        currentWorkflow: updated,
        workflows,
        isDirty: false,
      });
    } catch (error) {
      console.error("Failed to save workflow:", error);
      throw error;
    }
  },

  deleteWorkflow: async (id: string) => {
    try {
      const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
      const filtered = workflows.filter((w: Workflow) => w.id !== id);
      localStorage.setItem("mactuner_workflows", JSON.stringify(filtered));

      set({
        workflows: filtered,
      });

      // 현재 워크플로우가 삭제된 경우 새 워크플로우 생성
      if (get().currentWorkflow?.id === id) {
        get().createNewWorkflow();
      }
    } catch (error) {
      console.error("Failed to delete workflow:", error);
      throw error;
    }
  },

  createExampleWorkflow: async () => {
    const exampleWorkflow: Workflow = {
      id: `workflow_${Date.now()}`,
      name: "Korean Dialog Fine-tuning (gemma-3-1b-it)",
      description: "Korean_dialog 데이터셋을 google/gemma-3-1b-it 모델에 학습시키는 워크플로우",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      version: "1.0.0",
      nodes: [
        // 로컬 모델 로더
        {
          id: "node_model_1",
          type: "local-model-loader",
          position: { x: 100, y: 100 },
          data: { modelPath: "google/gemma-3-1b-it" },
          ports: [{ id: "out-model", name: "Model", type: "output", dataType: "model" }],
          status: "idle",
        },
        // 로컬 데이터셋 로더
        {
          id: "node_dataset_1",
          type: "local-dataset-loader",
          position: { x: 100, y: 250 },
          data: { datasetPath: "jungsungmoon/Korean_dialog" },
          ports: [{ id: "out-dataset", name: "Dataset", type: "output", dataType: "dataset" }],
          status: "idle",
        },
        // 데이터 전처리
        {
          id: "node_preprocessor_1",
          type: "dataset-preprocessor",
          position: { x: 400, y: 250 },
          data: {
            inputColumns: [],
            outputColumns: [],
            outputSeparator: "\\n",
            tokenizerName: "",
            maxLength: 512,
            padding: "max_length",
            truncation: true,
            format: "instruction",
            template: "",
          },
          ports: [
            { id: "in-dataset", name: "Dataset", type: "input", dataType: "dataset" },
            { id: "in-model", name: "Tokenizer Model", type: "input", dataType: "model", optional: true },
            { id: "out-dataset", name: "Processed Dataset", type: "output", dataType: "dataset" },
          ],
          status: "idle",
        },
        // 데이터 분할
        {
          id: "node_splitter_1",
          type: "dataset-splitter",
          position: { x: 700, y: 250 },
          data: {
            trainRatio: 80,
            valRatio: 10,
            testRatio: 10,
            shuffle: true,
            seed: 42,
          },
          ports: [
            { id: "in-dataset", name: "Dataset", type: "input", dataType: "dataset" },
            { id: "out-train", name: "Train Dataset", type: "output", dataType: "dataset" },
            { id: "out-val", name: "Validation Dataset", type: "output", dataType: "dataset" },
            { id: "out-test", name: "Test Dataset", type: "output", dataType: "dataset" },
          ],
          status: "idle",
        },
        // LoRA 설정
        {
          id: "node_lora_1",
          type: "lora-config",
          position: { x: 400, y: 100 },
          data: {
            rank: 8,
            alpha: 16,
            dropout: 0.1,
            targetModules: "q_proj,v_proj",
            bias: "none",
            taskType: "CAUSAL_LM",
          },
          ports: [{ id: "out-config", name: "LoRA Config", type: "output", dataType: "config" }],
          status: "idle",
        },
        // 학습 설정
        {
          id: "node_training_config_1",
          type: "training-config",
          position: { x: 700, y: 100 },
          data: {
            epochs: 3,
            batchSize: 4,
            learningRate: 5e-5,
            warmupSteps: 500,
            maxGradNorm: 1.0,
            saveStrategy: "epoch",
            saveSteps: 500,
            evalStrategy: "epoch",
            evalSteps: 500,
            loggingSteps: 10,
            outputDir: "./output",
          },
          ports: [{ id: "out-config", name: "Training Config", type: "output", dataType: "config" }],
          status: "idle",
        },
        // 학습 실행
        {
          id: "node_training_1",
          type: "training",
          position: { x: 1000, y: 175 },
          data: {
            useLora: true,
            useQlora: false,
          },
          ports: [
            { id: "in-model", name: "Model", type: "input", dataType: "model" },
            { id: "in-train-dataset", name: "Train Dataset", type: "input", dataType: "dataset" },
            { id: "in-val-dataset", name: "Validation Dataset", type: "input", dataType: "dataset", optional: true },
            { id: "in-training-config", name: "Training Config", type: "input", dataType: "config", optional: true },
            { id: "in-lora-config", name: "LoRA Config", type: "input", dataType: "config", optional: true },
            { id: "in-device", name: "Device", type: "input", dataType: "config", optional: true },
            { id: "out-trained-model", name: "Trained Model", type: "output", dataType: "model" },
            { id: "out-checkpoints", name: "Checkpoints", type: "output", dataType: "config" },
          ],
          status: "idle",
        },
        // 모델 저장
        {
          id: "node_saver_1",
          type: "model-saver",
          position: { x: 1300, y: 175 },
          data: {
            savePath: "./saved_models",
            saveFormat: "both",
            includeTokenizer: true,
            includeConfig: true,
          },
          ports: [
            { id: "in-model", name: "Model", type: "input", dataType: "model" },
            { id: "out-path", name: "Saved Model Path", type: "output", dataType: "config" },
          ],
          status: "idle",
        },
        // 디바이스 선택
        {
          id: "device-selector-1762596732655-btmqygcan",
          type: "device-selector",
          position: { x: 400, y: 400 },
          data: {
            deviceType: "auto",
          },
          ports: [{ id: "out-device", name: "Device", type: "output", dataType: "config" }],
          status: "idle",
        },
      ],
      connections: [
        {
          id: "conn_1",
          source: "node_model_1",
          target: "node_training_1",
          sourcePort: "out-model",
          targetPort: "in-model",
          dataType: "model",
        },
        {
          id: "conn_2",
          source: "node_dataset_1",
          target: "node_preprocessor_1",
          sourcePort: "out-dataset",
          targetPort: "in-dataset",
          dataType: "dataset",
        },
        {
          id: "conn_model_preprocessor",
          source: "node_model_1",
          target: "node_preprocessor_1",
          sourcePort: "out-model",
          targetPort: "in-model",
          dataType: "model",
        },
        {
          id: "conn_3",
          source: "node_preprocessor_1",
          target: "node_splitter_1",
          sourcePort: "out-dataset",
          targetPort: "in-dataset",
          dataType: "dataset",
        },
        {
          id: "conn_4",
          source: "node_splitter_1",
          target: "node_training_1",
          sourcePort: "out-train",
          targetPort: "in-train-dataset",
          dataType: "dataset",
        },
        {
          id: "conn_5",
          source: "node_splitter_1",
          target: "node_training_1",
          sourcePort: "out-val",
          targetPort: "in-val-dataset",
          dataType: "dataset",
        },
        {
          id: "conn_6",
          source: "node_training_config_1",
          target: "node_training_1",
          sourcePort: "out-config",
          targetPort: "in-training-config",
          dataType: "config",
        },
        {
          id: "conn_7",
          source: "node_lora_1",
          target: "node_training_1",
          sourcePort: "out-config",
          targetPort: "in-lora-config",
          dataType: "config",
        },
        {
          id: "conn_8",
          source: "node_training_1",
          target: "node_saver_1",
          sourcePort: "out-trained-model",
          targetPort: "in-model",
          dataType: "model",
        },
        {
          id: "conn_9",
          source: "device-selector-1762596732655-btmqygcan",
          target: "node_training_1",
          sourcePort: "out-device",
          targetPort: "in-device",
          dataType: "config",
        },
      ],
    };

    // 백엔드와 로컬 스토리지에 저장
    try {
      // 백엔드 API에 저장
      const { workflowAPI } = await import("../utils/api");
      await workflowAPI.save(exampleWorkflow);

      // 로컬 스토리지에도 저장
      const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
      workflows.push(exampleWorkflow);
      localStorage.setItem("mactuner_workflows", JSON.stringify(workflows));

      set({
        workflows,
      });
    } catch (error) {
      console.error("Failed to save example workflow:", error);
      // 백엔드 저장 실패해도 로컬 스토리지에는 저장
      try {
        const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
        workflows.push(exampleWorkflow);
        localStorage.setItem("mactuner_workflows", JSON.stringify(workflows));
      } catch (e) {
        console.error("Failed to save to localStorage:", e);
      }
    }

    // 워크플로우 로드
    get().loadWorkflow(exampleWorkflow);
  },

  exportWorkflowToJSON: () => {
    const { currentWorkflow, nodes, connections } = get();
    if (!currentWorkflow) {
      alert("저장할 워크플로우가 없습니다.");
      return;
    }

    const workflowData = {
      ...currentWorkflow,
      nodes,
      connections,
      exportedAt: Date.now(),
    };

    const jsonStr = JSON.stringify(workflowData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentWorkflow.name || "workflow"}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  importWorkflowFromJSON: async (file: File) => {
    try {
      const text = await file.text();
      const workflowData = JSON.parse(text) as Workflow;

      // 워크플로우 유효성 검사
      if (!workflowData.nodes || !Array.isArray(workflowData.nodes)) {
        throw new Error("유효하지 않은 워크플로우 파일입니다.");
      }

      if (!workflowData.connections || !Array.isArray(workflowData.connections)) {
        workflowData.connections = [];
      }

      // 새 ID 생성 (중복 방지)
      const newId = `workflow_${Date.now()}`;
      const workflow: Workflow = {
        ...workflowData,
        id: newId,
        createdAt: workflowData.createdAt || Date.now(),
        updatedAt: Date.now(),
      };

      // 워크플로우 로드
      get().loadWorkflow(workflow);

      // 로컬 스토리지에도 저장
      const workflows = JSON.parse(localStorage.getItem("mactuner_workflows") || "[]");
      workflows.push(workflow);
      localStorage.setItem("mactuner_workflows", JSON.stringify(workflows));

      set({
        workflows,
        isDirty: false,
      });

      return workflow;
    } catch (error) {
      console.error("Failed to import workflow:", error);
      throw error;
    }
  },

  exportWorkflowToFile: async () => {
    const { currentWorkflow, nodes, connections } = get();
    if (!currentWorkflow) {
      alert("저장할 워크플로우가 없습니다.");
      return;
    }

    const workflowData = {
      ...currentWorkflow,
      nodes,
      connections,
      updatedAt: Date.now(),
    };

    try {
      const { workflowAPI } = await import("../utils/api");
      const response = await workflowAPI.save(workflowData);

      alert(`워크플로우가 저장되었습니다: ${response.data.filename}`);

      // 워크플로우 목록 새로고침
      const listResponse = await workflowAPI.list();
      set({
        workflows: listResponse.data.workflows || [],
        isDirty: false,
      });
    } catch (error: any) {
      console.error("Failed to save workflow:", error);
      alert(`워크플로우 저장 실패: ${error.response?.data?.detail || error.message}`);
      throw error;
    }
  },

  importWorkflowFromFile: async (file: File) => {
    try {
      const { workflowAPI } = await import("../utils/api");
      const response = await workflowAPI.upload(file);

      const workflowData = response.data.workflow as Workflow;

      // 새 ID 생성 (중복 방지)
      const newId = `workflow_${Date.now()}`;
      const workflow: Workflow = {
        ...workflowData,
        id: newId,
        createdAt: workflowData.createdAt || Date.now(),
        updatedAt: Date.now(),
      };

      // 워크플로우 로드
      get().loadWorkflow(workflow);

      // 워크플로우 목록 새로고침
      const listResponse = await workflowAPI.list();
      set({
        workflows: listResponse.data.workflows || [],
        isDirty: false,
      });

      return workflow;
    } catch (error: any) {
      console.error("Failed to import workflow:", error);
      throw new Error(error.response?.data?.detail || error.message || "워크플로우 불러오기 실패");
    }
  },
  addNode: (type: NodeType, position: { x: number; y: number }) => {
    const factory = nodeFactories[type];
    if (!factory) return;

    const newNode = factory(position);
    set({
      nodes: [...get().nodes, newNode],
      isDirty: true,
    });
  },

  deleteNode: (id: string) => {
    const { nodes, connections } = get();
    set({
      nodes: nodes.filter((n) => n.id !== id),
      connections: connections.filter((c) => c.source !== id && c.target !== id),
      selectedNodeId: null,
      isDirty: true,
    });
  },

  updateNode: (id: string, data: Partial<Node>) => {
    set({
      nodes: get().nodes.map((n) => (n.id === id ? { ...n, ...data } : n)),
      isDirty: true,
    });
  },

  selectNode: (id: string) => {
    set({
      selectedNodeId: id,
      selectedConnectionId: null,
    });
  },

  clearNodeSelection: () => {
    set({ selectedNodeId: null });
  },

  // 연결 관리
  addConnection: (connection: Connection) => {
    set({
      connections: [...get().connections, connection],
      isDirty: true,
    });
  },

  deleteConnection: (id: string) => {
    set({
      connections: get().connections.filter((c) => c.id !== id),
      selectedConnectionId: null,
      isDirty: true,
    });
  },

  selectConnection: (id: string) => {
    set({
      selectedConnectionId: id,
      selectedNodeId: null,
    });
  },

  clearConnectionSelection: () => {
    set({ selectedConnectionId: null });
  },

  // 캔버스 관리
  setZoom: (zoom: number) => {
    set({ zoom: Math.max(0.1, Math.min(3, zoom)) });
  },

  setPan: (x: number, y: number) => {
    set({ panX: x, panY: y });
  },

  setCanvasMode: (mode: 'select' | 'pan') => {
    set({ canvasMode: mode });
  },

  // UI 토글
  toggleNodePalette: () => {
    set({ showNodePalette: !get().showNodePalette });
  },

  togglePropertiesPanel: () => {
    set({ showPropertiesPanel: !get().showPropertiesPanel });
  },

  toggleOutputPanel: () => {
    set({ showOutputPanel: !get().showOutputPanel });
  },

  toggleExecutionLog: () => {
    set({ showExecutionLog: !get().showExecutionLog });
  },

  // 실행 관리
  executeWorkflow: async () => {
    const { currentWorkflow, nodes, connections } = get();
    if (!currentWorkflow) {
      alert("⚠️ 워크플로우가 없습니다.");
      return;
    }

    if (nodes.length === 0) {
      alert("⚠️ 실행할 노드가 없습니다.");
      return;
    }

    // 워크플로우 검증
    const { validateWorkflow } = await import("../utils/workflowHelper");
    const validation = validateWorkflow(nodes, connections);

    if (!validation.valid) {
      // 에러 메시지 표시
      const errorMessages = Object.entries(validation.errors)
        .map(([nodeId, errors]) => {
          const node = nodes.find((n) => n.id === nodeId);
          return `${node?.type || "Unknown"} (${nodeId}):\n${errors.join("\n")}`;
        })
        .join("\n\n");

      const errorLog: ExecutionLog = {
        id: `exec-${Date.now()}`,
        workflowId: currentWorkflow.id,
        startTime: Date.now(),
        endTime: Date.now(),
        status: "error",
        nodeExecutions: [],
        totalDuration: 0,
      };

      set({
        executionLog: errorLog,
        isExecuting: false,
      });

      alert(`⚠️ 워크플로우 검증 실패:\n\n${errorMessages}`);
      return;
    }

    set({ isExecuting: true });

    const startTime = Date.now();
    const nodeExecutions: ExecutionContext[] = [];
    const nodeOutputs: Record<string, Record<string, any>> = {};

    try {
      // 노드를 위상 정렬하여 실행 순서 결정 (의존성 기반)
      const executedNodes = new Set<string>();
      const nodeDependencies = new Map<string, string[]>();

      // 각 노드의 의존성 계산
      nodes.forEach((node) => {
        const deps: string[] = [];
        connections.forEach((conn) => {
          if (conn.target === node.id) {
            deps.push(conn.source);
          }
        });
        nodeDependencies.set(node.id, deps);
      });

      // 위상 정렬 실행
      while (executedNodes.size < nodes.length) {
        let progress = false;

        for (const node of nodes) {
          if (executedNodes.has(node.id)) continue;

          const deps = nodeDependencies.get(node.id) || [];
          const allDepsExecuted = deps.every((dep) => executedNodes.has(dep));

          if (allDepsExecuted) {
            progress = true;
            executedNodes.add(node.id);

            // 노드 실행
            const nodeStartTime = Date.now();
            const nodeExec: ExecutionContext = {
              nodeId: node.id,
              inputData: {},
              outputs: {},
              status: "running",
              startTime: nodeStartTime,
            };

            // 입력 데이터 수집
            connections.forEach((conn) => {
              if (conn.target === node.id) {
                const sourceOutputs = nodeOutputs[conn.source] || {};
                nodeExec.inputData[conn.targetPort] = sourceOutputs[conn.sourcePort] || null;
              }
            });

            nodeExecutions.push(nodeExec);
            set({
              executionLog: {
                id: `exec-${Date.now()}`,
                workflowId: currentWorkflow.id,
                startTime,
                status: "running",
                nodeExecutions: [...nodeExecutions],
              },
            });

            // 실제 노드 실행 (백엔드 API 호출)
            try {
              const outputs: Record<string, any> = {};

              // 노드 타입별 실행 로직
              switch (node.type) {
                case "local-model-loader": {
                  const { modelAPI, api } = await import("../utils/api");
                  const modelPath = (node.data as any).modelPath;
                  if (!modelPath) {
                    throw new Error("모델 경로가 지정되지 않았습니다");
                  }

                  // 모델 경로가 HuggingFace 모델 ID 형식인지 확인
                  const isHuggingFaceId = modelPath.includes("/") && !modelPath.startsWith("/") && !modelPath.startsWith("./");

                  let actualModelPath = modelPath;

                  if (isHuggingFaceId) {
                    // HuggingFace 모델 ID인 경우, 로컬 모델 목록에서 실제 경로 찾기
                    try {
                      const localModelsResponse = await modelAPI.listLocal();
                      const localModels = localModelsResponse.data.models || [];
                      const foundModel = localModels.find((m: any) => m.model_id === modelPath || m.model_id?.includes(modelPath));

                      if (foundModel && foundModel.path) {
                        // 실제 경로를 찾았으면 해당 경로 사용
                        actualModelPath = foundModel.path;
                      }
                    } catch (error: any) {
                      // 로컬 모델 목록 조회 실패 시 모델 ID를 그대로 사용
                      console.warn("로컬 모델 목록 조회 실패, 모델 ID를 그대로 사용:", error);
                    }
                  }

                  // 로컬 모델 로드 API 호출
                  try {
                    await api.post("/model/upload", { model_path: actualModelPath });
                  } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || error.message || "모델 로드 실패";
                    throw new Error(`모델 로드 실패: ${errorMessage}`);
                  }

                  outputs["out-model"] = {
                    type: "model",
                    value: modelPath,
                    nodeId: node.id,
                  };
                  break;
                }

                case "hf-model-downloader": {
                  const { modelAPI } = await import("../utils/api");
                  const modelId = (node.data as any).modelId;
                  const token = nodeExec.inputData["in-token"]?.value || "";

                  if (!modelId) {
                    throw new Error("모델 ID가 지정되지 않았습니다");
                  }

                  // 모델 다운로드
                  await modelAPI.download(modelId, token);
                  outputs["out-model"] = {
                    type: "model",
                    value: modelId,
                    nodeId: node.id,
                  };
                  break;
                }

                case "local-dataset-loader": {
                  const { api } = await import("../utils/api");
                  const datasetPath = (node.data as any).datasetPath;
                  if (!datasetPath) {
                    throw new Error("데이터셋 경로가 지정되지 않았습니다");
                  }

                  // 데이터셋 ID로 데이터셋 로드
                  try {
                    const response = await api.post("/dataset/load-by-id", {
                      dataset_id: datasetPath,
                    });

                    if (response.data.status !== "success") {
                      throw new Error(response.data.message || "데이터셋 로드 실패");
                    }

                    outputs["out-dataset"] = {
                      type: "dataset",
                      value: datasetPath,
                      nodeId: node.id,
                      loaded: true, // 데이터셋이 로드되었음을 표시
                    };
                  } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || error.message || "데이터셋 로드 실패";
                    throw new Error(`데이터셋 로드 실패: ${errorMessage}`);
                  }
                  break;
                }

                case "dataset-preprocessor": {
                  // 데이터셋 전처리 노드: 데이터셋을 받아서 포맷 변환 정보를 출력으로 전달
                  const inputDataset = nodeExec.inputData["in-dataset"];
                  if (!inputDataset) {
                    throw new Error("데이터셋이 연결되지 않았습니다");
                  }

                  // 전처리 설정 추출
                  const format = (node.data as any).format || "instruction";
                  const template = (node.data as any).template || "";
                  const maxLength = (node.data as any).maxLength || 512;
                  const padding = (node.data as any).padding || "max_length";
                  const truncation = (node.data as any).truncation ?? true;
                  const inputColumns = (node.data as any).inputColumns || [];
                  const outputColumns = (node.data as any).outputColumns || [];
                  const outputSeparator = (node.data as any).outputSeparator || "\\n";

                  // 입력/출력 컬럼 검증
                  if (!Array.isArray(inputColumns) || inputColumns.length === 0) {
                    // 컬럼이 없으면 API에서 가져오기 시도
                    console.warn("⚠️ [dataset-preprocessor] 입력 컬럼이 없습니다. API에서 가져오기를 시도합니다.");
                    try {
                      const { datasetAPI } = await import("../utils/api");
                      const infoResponse = await datasetAPI.info();
                      if (infoResponse.data?.status === 'success' && infoResponse.data?.data?.columns) {
                        const columns = infoResponse.data.data.columns;
                        if (columns.length > 0) {
                          // 첫 번째 컬럼을 기본값으로 사용
                          inputColumns = [columns[0]];
                          console.log(`✅ [dataset-preprocessor] 기본 입력 컬럼 사용: ${inputColumns[0]}`);
                        }
                      }
                    } catch (e) {
                      console.error("Failed to get columns from API:", e);
                    }
                    
                    if (inputColumns.length === 0) {
                      throw new Error("입력 컬럼을 최소 하나 이상 선택해야 합니다");
                    }
                  }
                  
                  if (!Array.isArray(outputColumns) || outputColumns.length === 0) {
                    // 컬럼이 없으면 API에서 가져오기 시도
                    console.warn("⚠️ [dataset-preprocessor] 출력 컬럼이 없습니다. API에서 가져오기를 시도합니다.");
                    try {
                      const { datasetAPI } = await import("../utils/api");
                      const infoResponse = await datasetAPI.info();
                      if (infoResponse.data?.status === 'success' && infoResponse.data?.data?.columns) {
                        const columns = infoResponse.data.data.columns;
                        if (columns.length > 1) {
                          // 두 번째 컬럼을 기본값으로 사용
                          outputColumns = [columns[1]];
                        } else if (columns.length === 1) {
                          // 컬럼이 하나만 있으면 같은 컬럼 사용
                          outputColumns = [columns[0]];
                        }
                        if (outputColumns.length > 0) {
                          console.log(`✅ [dataset-preprocessor] 기본 출력 컬럼 사용: ${outputColumns[0]}`);
                        }
                      }
                    } catch (e) {
                      console.error("Failed to get columns from API:", e);
                    }
                    
                    if (outputColumns.length === 0) {
                      throw new Error("출력 컬럼을 최소 하나 이상 선택해야 합니다");
                    }
                  }

                  // 포맷 타입 매핑
                  let formatType = "causal_lm";
                  if (format === "instruction") {
                    formatType = "instruction";
                  } else if (format === "chat") {
                    formatType = "chat";
                  }

                  // 전처리된 데이터셋 정보를 출력으로 전달
                  outputs["out-dataset"] = {
                    type: "dataset",
                    value: inputDataset.value,
                    nodeId: node.id,
                    preprocessed: true,
                    formatType: formatType,
                    template: template,
                    maxLength: maxLength,
                    padding: padding,
                    truncation: truncation,
                    inputColumns: inputColumns,
                    outputColumns: outputColumns,
                    outputSeparator: outputSeparator,
                  };
                  break;
                }

                case "training-config": {
                  // 학습 설정 노드: 설정 데이터를 출력으로 전달
                  const configData = {
                    epochs: (node.data as any).epochs || 3,
                    batchSize: (node.data as any).batchSize || 4,
                    learningRate: (node.data as any).learningRate || 5e-5,
                    warmupSteps: (node.data as any).warmupSteps || 500,
                    maxGradNorm: (node.data as any).maxGradNorm || 1.0,
                    saveStrategy: (node.data as any).saveStrategy || "epoch",
                    saveSteps: (node.data as any).saveSteps,
                    evalStrategy: (node.data as any).evalStrategy || "epoch",
                    evalSteps: (node.data as any).evalSteps,
                    loggingSteps: (node.data as any).loggingSteps || 10,
                    outputDir: (node.data as any).outputDir || "./output",
                  };
                  outputs["out-config"] = {
                    type: "config",
                    value: configData,
                    nodeId: node.id,
                  };
                  break;
                }

                case "lora-config": {
                  // LoRA 설정 노드: 설정 데이터를 출력으로 전달
                  const configData = {
                    rank: (node.data as any).rank || 8,
                    alpha: (node.data as any).alpha || 16,
                    dropout: (node.data as any).dropout || 0.1,
                    targetModules: (node.data as any).targetModules || "q_proj,v_proj",
                    bias: (node.data as any).bias || "none",
                    taskType: (node.data as any).taskType || "CAUSAL_LM",
                  };
                  outputs["out-config"] = {
                    type: "config",
                    value: configData,
                    nodeId: node.id,
                  };
                  break;
                }

                case "qlora-config": {
                  // QLoRA 설정 노드: 설정 데이터를 출력으로 전달
                  const configData = {
                    bits: (node.data as any).bits || 4,
                    quantType: (node.data as any).quantType || "nf4",
                    doubleQuant: (node.data as any).doubleQuant ?? true,
                    computeDtype: (node.data as any).computeDtype || "float16",
                    loraRank: (node.data as any).loraRank || 64,
                    loraAlpha: (node.data as any).loraAlpha || 16,
                    loraDropout: (node.data as any).loraDropout || 0.1,
                    targetModules: (node.data as any).targetModules || "q_proj,v_proj",
                  };
                  outputs["out-config"] = {
                    type: "config",
                    value: configData,
                    nodeId: node.id,
                  };
                  break;
                }

                case "device-selector": {
                  const deviceType = (node.data as any).deviceType || "auto";
                  const { deviceAPI } = await import("../utils/api");

                  // 디바이스 선택
                  if (deviceType === "auto") {
                    await deviceAPI.autoSelect();
                  } else {
                    await deviceAPI.selectDevice(deviceType);
                  }

                  outputs["out-device"] = {
                    type: "config",
                    value: deviceType,
                    nodeId: node.id,
                  };
                  break;
                }

                case "training": {
                  const { trainingAPI, api } = await import("../utils/api");
                  const useLora = (node.data as any).useLora ?? true;
                  const useQlora = (node.data as any).useQlora ?? false;

                  // 입력 데이터 수집
                  const modelInput = nodeExec.inputData["in-model"];
                  const trainDataset = nodeExec.inputData["in-train-dataset"];
                  const valDataset = nodeExec.inputData["in-val-dataset"];
                  const trainingConfig = nodeExec.inputData["in-training-config"];
                  const loraConfig = nodeExec.inputData["in-lora-config"];
                  const qloraConfig = nodeExec.inputData["in-qlora-config"];
                  const deviceConfig = nodeExec.inputData["in-device"];

                  if (!modelInput) {
                    throw new Error("모델이 연결되지 않았습니다");
                  }
                  if (!trainDataset) {
                    throw new Error("학습 데이터셋이 연결되지 않았습니다");
                  }

                  // 디바이스 설정 (연결되어 있으면)
                  if (deviceConfig?.value) {
                    const { deviceAPI } = await import("../utils/api");
                    if (deviceConfig.value === "auto") {
                      await deviceAPI.autoSelect();
                    } else {
                      await deviceAPI.selectDevice(deviceConfig.value);
                    }
                  }

                  // 학습 준비 요청 구성
                  const prepareRequest: any = {
                    use_lora: useLora,
                    use_qlora: useQlora,
                  };

                  // LoRA 설정 추가
                  if (loraConfig?.value) {
                    prepareRequest.lora_config = {
                      rank: loraConfig.value.rank || 8,
                      alpha: loraConfig.value.alpha || 16,
                      dropout: loraConfig.value.dropout || 0.1,
                      target_modules: loraConfig.value.targetModules?.split(",") || ["q_proj", "v_proj"],
                    };
                  }

                  // 학습 설정 추가
                  if (trainingConfig?.value) {
                    prepareRequest.training_args = {
                      output_dir: trainingConfig.value.outputDir || "./output",
                      num_epochs: trainingConfig.value.epochs || 3,
                      batch_size: trainingConfig.value.batchSize || 4,
                      learning_rate: trainingConfig.value.learningRate || 5e-5,
                      warmup_steps: trainingConfig.value.warmupSteps || 500,
                      weight_decay: trainingConfig.value.weightDecay || 0.01,
                      gradient_accumulation_steps: trainingConfig.value.gradientAccumulationSteps || 1,
                      max_grad_norm: trainingConfig.value.maxGradNorm || 1.0,
                      save_strategy: trainingConfig.value.saveStrategy || "epoch",
                      eval_strategy: trainingConfig.value.evalStrategy || "epoch",
                    };
                  } else {
                    // 기본 학습 설정
                    prepareRequest.training_args = {
                      output_dir: "./output",
                      num_epochs: 3,
                      batch_size: 4,
                      learning_rate: 5e-5,
                      warmup_steps: 500,
                      weight_decay: 0.01,
                      gradient_accumulation_steps: 1,
                      max_grad_norm: 1.0,
                      save_strategy: "epoch",
                      eval_strategy: "epoch",
                    };
                  }

                  // 학습 준비 API 호출
                  try {
                    await api.post("/train/prepare", prepareRequest);
                  } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || error.message || "학습 준비 실패";
                    throw new Error(`학습 준비 실패: ${errorMessage}`);
                  }

                  // 학습 시작 (스트리밍)
                  try {
                    // 데이터셋 정보 가져오기
                    const { datasetAPI } = await import("../utils/api");
                    let datasetInfo;
                    try {
                      const infoResponse = await datasetAPI.info();
                      datasetInfo = infoResponse.data;
                    } catch (e) {
                      console.warn("데이터셋 정보를 가져올 수 없습니다. 기본값을 사용합니다.");
                      datasetInfo = null;
                    }

                    // 데이터셋 컬럼 자동 감지
                    let textColumn = "text";
                    let formatType = "causal_lm";
                    let inputColumn: string | null = null;
                    let outputColumn: string | null = null;
                    let template: string | null = null;
                    let maxLength = 512;

                    // 전처리된 데이터셋 정보 확인
                    if (trainDataset.preprocessed) {
                      formatType = trainDataset.formatType || "causal_lm";
                      template = trainDataset.template || null;
                      maxLength = trainDataset.maxLength || 512;
                      
                      // 사용자가 지정한 컬럼 사용
                      if (trainDataset.inputColumns && Array.isArray(trainDataset.inputColumns) && trainDataset.inputColumns.length > 0) {
                        // 여러 입력 컬럼이 있으면 첫 번째 컬럼 사용 (또는 합치기)
                        inputColumn = trainDataset.inputColumns[0];
                      }
                      if (trainDataset.outputColumns && Array.isArray(trainDataset.outputColumns) && trainDataset.outputColumns.length > 0) {
                        // 여러 출력 컬럼은 백엔드에서 합치므로 여기서는 첫 번째만 사용하지 않음
                        // outputColumns는 그대로 전달됨
                      }
                    }

                    // 전처리 정보가 없으면 자동 감지
                    if (!inputColumn || !outputColumn) {
                      if (datasetInfo && datasetInfo.columns && Array.isArray(datasetInfo.columns)) {
                        if (formatType === "instruction" || formatType === "chat") {
                          // Instruction/Chat 포맷: input/output 컬럼 찾기
                          const inputCols = datasetInfo.columns.filter((col: string) =>
                            ["instruction", "input", "question", "prompt", "query", "user"].some((possible) => col.toLowerCase().includes(possible))
                          );
                          const outputCols = datasetInfo.columns.filter((col: string) => ["output", "response", "answer", "reply", "assistant"].some((possible) => col.toLowerCase().includes(possible)));

                          if (inputCols.length > 0 && !inputColumn) {
                            inputColumn = inputCols[0];
                          }
                          if (outputCols.length > 0 && !outputColumn) {
                            outputColumn = outputCols[0];
                          }
                        } else {
                          // Causal LM: 단일 텍스트 컬럼 찾기
                          const possibleColumns = ["text", "content", "input", "instruction", "question", "prompt", "message"];
                          const foundColumn = datasetInfo.columns.find((col: string) => possibleColumns.some((possible) => col.toLowerCase().includes(possible)));
                          if (foundColumn) {
                            textColumn = foundColumn;
                          } else if (datasetInfo.columns.length > 0) {
                            textColumn = datasetInfo.columns[0];
                          }
                        }
                      }
                    }

                    // 데이터셋 준비 요청
                    const datasetRequest: any = {
                      text_column: textColumn,
                      label_column: null,
                      max_length: maxLength,
                      test_size: 0.1,
                      format_type: formatType,
                    };

                    // Instruction/Chat 포맷인 경우 input/output 컬럼 추가
                    if (formatType === "instruction" || formatType === "chat") {
                      // 여러 컬럼 지원
                      if (trainDataset.preprocessed) {
                        if (trainDataset.inputColumns && Array.isArray(trainDataset.inputColumns) && trainDataset.inputColumns.length > 0) {
                          datasetRequest.input_columns = trainDataset.inputColumns;
                        } else if (inputColumn) {
                          datasetRequest.input_column = inputColumn;
                        }
                        
                        if (trainDataset.outputColumns && Array.isArray(trainDataset.outputColumns) && trainDataset.outputColumns.length > 0) {
                          datasetRequest.output_columns = trainDataset.outputColumns;
                          if (trainDataset.outputSeparator) {
                            datasetRequest.output_separator = trainDataset.outputSeparator;
                          }
                        } else if (outputColumn) {
                          datasetRequest.output_column = outputColumn;
                        }
                      } else {
                        // 전처리 정보가 없으면 단일 컬럼 사용
                        if (inputColumn) {
                          datasetRequest.input_column = inputColumn;
                        }
                        if (outputColumn) {
                          datasetRequest.output_column = outputColumn;
                        }
                      }
                      
                      if (template) {
                        datasetRequest.template = template;
                      }
                    }

                    // 스트리밍 API 호출
                    const response = await fetch("http://localhost:8001/train/start-stream", {
                      method: "POST",
                      headers: {
                        "Content-Type": "application/json",
                      },
                      body: JSON.stringify(datasetRequest),
                    });

                    if (!response.ok) {
                      throw new Error(`학습 시작 실패: ${response.statusText}`);
                    }

                    // 스트리밍 응답 처리
                    const reader = response.body?.getReader();
                    const decoder = new TextDecoder();

                    if (reader) {
                      let buffer = "";
                      while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split("\n");
                        buffer = lines.pop() || "";

                        for (const line of lines) {
                          if (line.trim()) {
                            try {
                              const data = JSON.parse(line);

                              // 학습 진행 상황을 콘솔에 추가
                              let logMessage = `[학습] ${data.message || data.status}`;
                              if (data.loss !== undefined) {
                                logMessage += ` | Loss: ${data.loss.toFixed(4)}`;
                              }
                              if (data.step !== undefined) {
                                logMessage += ` | Step: ${data.step}`;
                              }

                              // 실행 로그에 추가
                              const currentLog = get().executionLog;
                              if (currentLog) {
                                const updatedExecutions = [...currentLog.nodeExecutions];
                                const trainingExec = updatedExecutions.find((e) => e.nodeId === node.id);
                                if (trainingExec) {
                                  trainingExec.trainingLogs = trainingExec.trainingLogs || [];
                                  trainingExec.trainingLogs.push({
                                    timestamp: Date.now(),
                                    message: logMessage,
                                    data: data,
                                  });

                                  set({
                                    executionLog: {
                                      ...currentLog,
                                      nodeExecutions: updatedExecutions,
                                    },
                                  });
                                }
                              }

                              // 노드 상태 업데이트
                              if (data.status === "completed") {
                                get().updateNode(node.id, { status: "completed", progress: 100 });
                              } else if (data.status === "training") {
                                get().updateNode(node.id, { status: "running", progress: data.progress || 0 });
                              }
                            } catch (e) {
                              console.error("스트리밍 데이터 파싱 실패:", e);
                            }
                          }
                        }
                      }
                    }
                  } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || error.message || "학습 시작 실패";
                    throw new Error(`학습 시작 실패: ${errorMessage}`);
                  }

                  outputs["out-trained-model"] = {
                    type: "model",
                    value: modelInput.value || "trained_model",
                    nodeId: node.id,
                  };
                  outputs["out-checkpoints"] = {
                    type: "config",
                    value: { checkpoints: [] },
                    nodeId: node.id,
                  };
                  break;
                }

                default:
                  // 다른 노드 타입은 시뮬레이션
                  await new Promise((resolve) => setTimeout(resolve, 300));
                  node.ports
                    .filter((p) => p.type === "output")
                    .forEach((port) => {
                      outputs[port.id] = {
                        type: port.dataType,
                        value: `Output from ${node.type}`,
                        nodeId: node.id,
                      };
                    });
              }

              nodeExec.outputs = outputs;
              nodeExec.status = "completed";
              nodeExec.endTime = Date.now();
              nodeExec.duration = nodeExec.endTime - nodeStartTime;

              nodeOutputs[node.id] = outputs;

              // 노드 상태 업데이트
              get().updateNode(node.id, { status: "completed" });
            } catch (error: any) {
              nodeExec.status = "error";
              nodeExec.error = error.message || String(error);
              nodeExec.endTime = Date.now();
              nodeExec.duration = nodeExec.endTime - nodeStartTime;

              get().updateNode(node.id, { status: "error", error: nodeExec.error });

              // 에러 발생 시 중단
              const endTime = Date.now();
              set({
                executionLog: {
                  id: `exec-${Date.now()}`,
                  workflowId: currentWorkflow.id,
                  startTime,
                  endTime,
                  status: "error",
                  nodeExecutions: [...nodeExecutions],
                  totalDuration: endTime - startTime,
                },
                isExecuting: false,
              });
              return;
            }
          }
        }

        if (!progress) {
          // 순환 의존성 또는 실행 불가능한 노드
          throw new Error("워크플로우에 순환 의존성이 있거나 실행할 수 없는 노드가 있습니다.");
        }
      }

      // 모든 노드 실행 완료
      const endTime = Date.now();
      const finalLog: ExecutionLog = {
        id: `exec-${Date.now()}`,
        workflowId: currentWorkflow.id,
        startTime,
        endTime,
        status: "completed",
        nodeExecutions,
        totalDuration: endTime - startTime,
      };

      set({
        executionLog: finalLog,
        isExecuting: false,
      });

      console.log("✅ 워크플로우 실행 완료", finalLog);
    } catch (error: any) {
      console.error("Failed to execute workflow:", error);
      const endTime = Date.now();
      set({
        executionLog: {
          id: `exec-${Date.now()}`,
          workflowId: currentWorkflow.id,
          startTime,
          endTime,
          status: "error",
          nodeExecutions,
          totalDuration: endTime - startTime,
        },
        isExecuting: false,
      });
    }
  },

  setExecutionLog: (log: ExecutionLog) => {
    set({ executionLog: log });
  },

  clearExecutionLog: () => {
    set({ executionLog: null });
  },

  // 상태 관리
  setDirty: (dirty: boolean) => {
    set({ isDirty: dirty });
  },

  resetWorkflow: () => {
    get().createNewWorkflow();
  },
}));
