import axios from "axios";

const API_BASE_URL = "http://localhost:8001";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Device API (디바이스 선택 및 관리)
export const deviceAPI = {
  getAvailable: () => api.get("/device/devices/available"),
  selectDevice: (deviceType: string) => api.post(`/device/devices/select/${deviceType}`),
  getCurrent: () => api.get("/device/devices/current"),
  autoSelect: () => api.post("/device/devices/auto-select"),
  getMemory: () => api.get("/device/devices/memory"),
  clearCache: () => api.post("/device/devices/clear-cache"),
};

// API 엔드포인트들
export const modelAPI = {
  health: () => api.get("/model/health"),
  download: (model_id: string, access_token?: string) => api.post("/model/download", { model_id, access_token }),
  info: (model_id: string) => api.get(`/model/info/${model_id}`),
  current: () => api.get("/model/current"),
  listLocal: () => api.get("/model/local-models"),
  scanFolderModels: () => api.get("/model/folder-models"),
};

export const datasetAPI = {
  upload: (file: File, data_format: string) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("data_format", data_format);
    return api.post("/dataset/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  info: () => api.get("/dataset/info"),
  preview: (n_rows?: number) => api.get(`/dataset/preview?n_rows=${n_rows || 5}`),
  fullData: (limit?: number, offset?: number) => {
    // limit과 offset을 안전하게 처리
    const params = new URLSearchParams();
    if (limit !== undefined && limit !== null) {
      params.append('limit', String(limit));
    }
    if (offset !== undefined && offset !== null) {
      params.append('offset', String(offset));
    }
    const url = `/dataset/full-data${params.toString() ? '?' + params.toString() : ''}`;
    console.log("📡 [API] GET", url);
    return api.get(url).then(res => {
      console.log("📡 [API] 응답 받음:", res);
      return res;
    }).catch(err => {
      console.error("📡 [API] 에러:", err);
      throw err;
    });
  },
  clean: (operation: string, kwargs?: any) => api.post("/dataset/clean", { operation, kwargs }),
  statistics: () => api.get("/dataset/eda/statistics"),
  split: (test_size?: number) => api.post("/dataset/split", { test_size: test_size || 0.2 }),
  downloadHF: (dataset_id: string, hf_token?: string, split?: string, max_samples?: number) =>
    api.post("/dataset/download-hf", { dataset_id, hf_token, split: split || "train", max_samples }),
  loadById: (dataset_id: string) => api.post("/dataset/load-by-id", { dataset_id }),
  // 히스토리 및 캐시
  getHistory: () => api.get("/dataset/history"),
  reloadFromHistory: (index: number) => api.post(`/dataset/history/reload/${index}`),
  getCachedDatasets: () => api.get("/dataset/cached-datasets"),
  clearHistory: (deleteData: boolean = false) => api.post(`/dataset/history/clear?delete_data=${deleteData}`),
  deleteHistoryItem: (index: number, deleteData: boolean = false) => api.delete(`/dataset/history/${index}?delete_data=${deleteData}`),
};

export const trainingAPI = {
  configure_lora: (rank: number, alpha: number, dropout: number) => api.post("/train/config-lora", { rank, alpha, dropout }),
  start: () => api.post("/train/start"),
  status: () => api.get("/train/status"),
  history: () => api.get("/train/history"),
  save: (output_dir: string) => api.post("/train/save", { output_dir }),
};

export const chatAPI = {
  initialize: (system_prompt?: string) => api.post("/chat/initialize", { system_prompt }),
  chat: (message: string, top_p?: number, temperature?: number, max_tokens?: number, repeat_penalty?: number, n_gpu_layers?: number) =>
    api.post("/chat/chat", {
      message,
      top_p: top_p || 0.9,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 1024,
      repeat_penalty: repeat_penalty || 1.1,
      n_gpu_layers: n_gpu_layers || 35,
      maintain_history: true,
    }),
  history: () => api.get("/chat/history"),
  clear_history: () => api.post("/chat/history/clear"),
};

export const ragAPI = {
  initialize: () => api.post("/rag/initialize"),
  load_pdf: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/rag/load-pdf", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  load_text: (text: string) => api.post("/rag/load-text", { text }),
  search: (query: string, top_k?: number) => api.post("/rag/search", { query, top_k }),
  chat: (message: string) => api.post("/rag/chat", { message }),
  get_documents: () => api.get("/rag/documents/info"),
};

export const ggufAPI = {
  get_methods: () => api.get("/gguf/methods"),
  get_recommended: (model_size_gb: number) => api.get(`/gguf/methods/recommended?model_size_gb=${model_size_gb}`),
  convert: (model_path: string, method: string, output_path: string) => api.post("/gguf/convert", { model_path, method, output_path }),
  validate: (file_path: string) => api.post(`/gguf/validate/${file_path}`),
};

export const workflowAPI = {
  list: () => api.get("/workflow/list"),
  save: (workflow: any) => api.post("/workflow/save", { workflow }),
  load: (filename: string) => api.get(`/workflow/load/${filename}`),
  delete: (filename: string) => api.delete(`/workflow/delete/${filename}`),
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/workflow/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

export default api;
