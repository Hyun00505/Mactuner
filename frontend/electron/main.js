const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const isDev = require("electron-is-dev");

let mainWindow;

// Electron 앱 준비 완료
app.on("ready", createWindow);

// 모든 창이 닫히면 앱 종료
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// 앱이 활성화되면 창 생성
app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const startUrl = isDev ? "http://localhost:5173" : `file://${path.join(__dirname, "../dist/index.html")}`;

  mainWindow.loadURL(startUrl);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// IPC 핸들러: 파일 선택
ipcMain.handle("select-file", async (event, filters) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: filters || [{ name: "All Files", extensions: ["*"] }],
  });

  if (!result.canceled) {
    return result.filePaths[0]; // 첫 번째 파일 경로 반환
  }
  return null;
});

// IPC 핸들러: 폴더 선택
ipcMain.handle("select-folder", async (event) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });

  if (!result.canceled) {
    return result.filePaths[0]; // 폴더 경로 반환
  }
  return null;
});

// IPC 핸들러: 파일 선택 (여러 파일)
ipcMain.handle("select-files", async (event, filters) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile", "multiSelections"],
    filters: filters || [{ name: "All Files", extensions: ["*"] }],
  });

  if (!result.canceled) {
    return result.filePaths; // 모든 파일 경로 반환
  }
  return [];
});
