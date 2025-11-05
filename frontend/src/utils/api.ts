import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API 엔드포인트들
export const modelAPI = {
  health: () => api.get('/model/health'),
  download: (model_id: string, access_token?: string) =>
    api.post('/model/download', { model_id, access_token }),
  info: (model_id: string) => api.get(`/model/info/${model_id}`),
  current: () => api.get('/model/current'),
  listLocal: () => api.get('/model/local-models'),
  scanFolderModels: () => api.get('/model/folder-models'),
};

export const datasetAPI = {
  upload: (file: File, data_format: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_format', data_format);
    return api.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  info: () => api.get('/dataset/info'),
  preview: (n_rows?: number) => api.get(`/dataset/preview?n_rows=${n_rows || 5}`),
  clean: (operation: string, kwargs?: any) =>
    api.post('/dataset/clean', { operation, kwargs }),
  statistics: () => api.get('/dataset/eda/statistics'),
  split: (test_size?: number) =>
    api.post('/dataset/split', { test_size: test_size || 0.2 }),
};

export const trainingAPI = {
  configure_lora: (rank: number, alpha: number, dropout: number) =>
    api.post('/train/config-lora', { rank, alpha, dropout }),
  start: () => api.post('/train/start'),
  status: () => api.get('/train/status'),
  history: () => api.get('/train/history'),
  save: (output_dir: string) => api.post('/train/save', { output_dir }),
};

export const chatAPI = {
  initialize: () => api.post('/chat/initialize'),
  chat: (message: string, top_p?: number, temperature?: number) =>
    api.post('/chat/chat', { message, top_p, temperature }),
  history: () => api.get('/chat/history'),
  clear_history: () => api.post('/chat/history/clear'),
};

export const ragAPI = {
  initialize: () => api.post('/rag/initialize'),
  load_pdf: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/rag/load-pdf', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  load_text: (text: string) =>
    api.post('/rag/load-text', { text }),
  search: (query: string, top_k?: number) =>
    api.post('/rag/search', { query, top_k }),
  chat: (message: string) =>
    api.post('/rag/chat', { message }),
  get_documents: () => api.get('/rag/documents/info'),
};

export const ggufAPI = {
  get_methods: () => api.get('/gguf/methods'),
  get_recommended: (model_size_gb: number) =>
    api.get(`/gguf/methods/recommended?model_size_gb=${model_size_gb}`),
  convert: (model_path: string, method: string, output_path: string) =>
    api.post('/gguf/convert', { model_path, method, output_path }),
  validate: (file_path: string) =>
    api.post(`/gguf/validate/${file_path}`),
};

export default api;
