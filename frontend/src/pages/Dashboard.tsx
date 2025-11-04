import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [workflows] = useState([
    {
      id: "1",
      name: "Fine-tuning GPT-2",
      date: "2시간 전",
      status: "completed",
    },
    {
      id: "2",
      name: "RAG with PDF Document",
      date: "1일 전",
      status: "completed",
    },
  ]);

  const quickStartItems = [
    {
      icon: "📥",
      title: "모델 다운로드",
      description: "Hugging Face에서 모델 다운로드",
      color: "from-blue-500 to-blue-600",
    },
    {
      icon: "📊",
      title: "데이터 처리",
      description: "데이터 업로드 및 정제",
      color: "from-green-500 to-green-600",
    },
    {
      icon: "🎓",
      title: "학습",
      description: "LoRA/QLoRA 미세조정",
      color: "from-orange-500 to-orange-600",
    },
    {
      icon: "💬",
      title: "Chat",
      description: "학습된 모델과 대화",
      color: "from-cyan-500 to-cyan-600",
    },
    {
      icon: "🔍",
      title: "RAG",
      description: "문서 기반 검색",
      color: "from-purple-500 to-purple-600",
    },
    {
      icon: "📦",
      title: "GGUF",
      description: "양자화 및 배포",
      color: "from-yellow-500 to-yellow-600",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-12">
        {/* 타이틀 */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🍎 MacTuner Dashboard
          </h1>
          <p className="text-lg text-gray-600">
            LLM 파인튜닝을 위한 완벽한 플랫폼에 오신 것을 환영합니다
          </p>
        </div>

        {/* 최근 워크플로우 */}
        <section className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">📊 최근 워크플로우</h2>
            <button
              onClick={() => navigate("/editor")}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              + 새 워크플로우
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {workflows.map((workflow) => (
              <div
                key={workflow.id}
                className="bg-white p-6 rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer"
                onClick={() => navigate(`/editor/${workflow.id}`)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {workflow.name}
                  </h3>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      workflow.status === "completed"
                        ? "bg-green-100 text-green-800"
                        : "bg-yellow-100 text-yellow-800"
                    }`}
                  >
                    {workflow.status === "completed" ? "✓ 완료" : "진행 중"}
                  </span>
                </div>
                <p className="text-sm text-gray-500">{workflow.date}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 빠른 시작 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">🎯 빠른 시작</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {quickStartItems.map((item, index) => (
              <div
                key={index}
                onClick={() => {
                  if (index === 0) navigate('/model');
                  else if (index === 1) navigate('/data');
                  else if (index === 2) navigate('/editor');
                  else if (index === 3) navigate('/chat');
                  else if (index === 4) navigate('/editor');
                  else if (index === 5) navigate('/editor');
                }}
                className={`bg-gradient-to-br ${item.color} text-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-all cursor-pointer transform hover:scale-105`}
              >
                <div className="text-4xl mb-4">{item.icon}</div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-sm opacity-90">{item.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 통계 */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-3xl font-bold text-blue-600">5</div>
            <p className="text-gray-600 mt-2">총 모델</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-3xl font-bold text-green-600">3</div>
            <p className="text-gray-600 mt-2">실행 중인 작업</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-3xl font-bold text-orange-600">2h 15m</div>
            <p className="text-gray-600 mt-2">평균 학습 시간</p>
          </div>
        </section>
      </div>
    </div>
  );
};
