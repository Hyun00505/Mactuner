const { contextBridge, ipcRenderer } = require("electron");

// React에서 접근 가능한 Electron API 노출
contextBridge.exposeInMainWorld("electronAPI", {
  // 파일 선택 (GGUF)
  selectFile: (filters) => ipcRenderer.invoke("select-file", filters),

  // 폴더 선택
  selectFolder: () => ipcRenderer.invoke("select-folder"),

  // 여러 파일 선택
  selectFiles: (filters) => ipcRenderer.invoke("select-files", filters),
});
