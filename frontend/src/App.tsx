import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";
import { ModelDownload } from "./pages/ModelDownload";
import { DataProcessing } from "./pages/DataProcessing";
import { Chat } from "./pages/Chat";
import { Editor } from "./pages/Editor";

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/model" element={<ModelDownload />} />
        <Route path="/data" element={<DataProcessing />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/editor" element={<Editor />} />
        <Route path="/editor/:id" element={<Editor />} />
        <Route path="/history" element={<div className="p-8">히스토리 (구현 중)</div>} />
        <Route path="/api" element={<div className="p-8">API 문서 (구현 중)</div>} />
      </Routes>
    </Router>
  );
}

export default App;
