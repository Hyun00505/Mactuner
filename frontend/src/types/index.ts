// API 타입들
export interface ModelInfo {
  model_id: string;
  source: "hub" | "local";
  model_type: string;
  num_parameters: number;
  estimated_memory_gb: number;
  device: string;
  dtype: string;
}

export interface DatasetInfo {
  shape: { rows: number; columns: number };
  size_mb: number;
  dtypes: Record<string, string>;
  columns: string[];
}

export interface TrainingConfig {
  output_dir: string;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
}

export interface NodeData {
  id: string;
  type: "model" | "dataset" | "training" | "chat" | "rag" | "gguf";
  position: { x: number; y: number };
  data: Record<string, any>;
  connections: { input: string[] | null; output: string[] | null };
}

export interface WorkflowData {
  id: string;
  name: string;
  nodes: NodeData[];
  createdAt: string;
  updatedAt: string;
}

export interface TrainingStatus {
  status: "idle" | "running" | "completed" | "error";
  epoch?: number;
  totalEpochs?: number;
  loss?: number;
  progress?: number;
  eta?: number;
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RAGSearchResult {
  rank: number;
  content: string;
  similarity: number;
  source: string;
}

export interface QuantizationMethod {
  name: string;
  quality: number;
  size: number;
  description: string;
}

// UI 상태
export interface UIState {
  theme: "light" | "dark";
  selectedNode: string | null;
  isTraining: boolean;
  showNodePalette: boolean;
}
