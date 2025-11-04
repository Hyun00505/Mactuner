import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";
import { ModelDownload } from "./pages/ModelDownload";
import { DataProcessing } from "./pages/DataProcessing";
import { Chat } from "./pages/Chat";

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/model" element={<ModelDownload />} />
        <Route path="/data" element={<DataProcessing />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/editor" element={<div className="p-8">에디터 (구현 중)</div>} />
        <Route path="/editor/:id" element={<div className="p-8">에디터 (구현 중)</div>} />
        <Route path="/history" element={<div className="p-8">히스토리 (구현 중)</div>} />
        <Route path="/api" element={<div className="p-8">API 문서 (구현 중)</div>} />
      </Routes>
    </Router>
  );
}

export default App;
