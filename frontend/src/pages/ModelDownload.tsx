import React, { useState } from 'react';

export const ModelDownload: React.FC = () => {
  const [modelId, setModelId] = useState('');
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [downloadedModels, setDownloadedModels] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDownload = async () => {
    if (!modelId.trim()) {
      setMessage('모델 ID를 입력해주세요');
      return;
    }

    try {
      setLoading(true);
      setProgress(0);
      setMessage('모델 다운로드 시작...');
      
      const API_URL = 'http://localhost:8001';
      const response = await fetch(
        `${API_URL}/model/download-stream?model_id=${modelId}&access_token=${token}`,
        { method: 'POST' }
      );

      if (!response.body) {
        throw new Error('Response body not available');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            setStatus(data.status || '');
            setProgress(data.progress || 0);
            setMessage(data.message || '다운로드 진행 중...');

            if (data.status === 'completed') {
              setMessage(`✅ ${modelId} 다운로드 완료!`);
              setModelId('');
              setToken('');
              await fetchLocalModels();
            } else if (data.status === 'error') {
              setMessage(`❌ 오류: ${data.message}`);
            }
          } catch (e) {
            // JSON 파싱 오류 무시
          }
        }
      }
    } catch (error: any) {
      setMessage(`❌ 오류: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchLocalModels = async () => {
    try {
      setRefreshing(true);
      const response = await fetch('http://localhost:8001/model/local-models');
      const data = await response.json();
      setDownloadedModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models', error);
    } finally {
      setRefreshing(false);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm(`정말 "${modelId}" 모델을 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.`)) {
      return;
    }

    try {
      setDeleting(modelId);
      const response = await fetch(
        `http://localhost:8001/model/delete/${encodeURIComponent(modelId)}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error('모델 삭제 실패');
      }

      const data = await response.json();
      setMessage(`✅ ${data.message}`);
      setSelectedModel(null);
      await fetchLocalModels();
    } catch (error: any) {
      setMessage(`❌ 오류: ${error.message}`);
    } finally {
      setDeleting(null);
    }
  };

  const handleOpenFolder = async (modelPath: string) => {
    try {
      const response = await fetch(
        'http://localhost:8001/model/open-folder',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: modelPath })
        }
      );

      if (!response.ok) {
        throw new Error('폴더 열기 실패');
      }

      const data = await response.json();
      console.log(data.message);
    } catch (error: any) {
      alert(`❌ 오류: ${error.message}`);
    }
  };

  React.useEffect(() => {
    fetchLocalModels();
  }, []);

  const getModelIcon = (source: string) => {
    return source === 'huggingface' ? '🤗' : '📂';
  };

  const formatSize = (sizeGb: number) => {
    if (sizeGb < 0.01) return '< 10 MB';
    if (sizeGb < 1) return `${(sizeGb * 1024).toFixed(0)} MB`;
    return `${sizeGb.toFixed(2)} GB`;
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">📥 모델 다운로드 및 관리</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* 다운로드 폼 */}
        <div className="lg:col-span-2">
          <div className="bg-white p-8 rounded-lg shadow">
            <h2 className="text-xl font-bold mb-6">새 모델 다운로드</h2>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hugging Face 모델 ID
              </label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="예: gpt2, meta-llama/Llama-3.2-1B"
                disabled={loading}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              />
              <p className="text-xs text-gray-500 mt-1">💡 Hugging Face Hub에서 원하는 모델의 ID를 찾아 입력하세요</p>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                액세스 토큰 (선택사항)
              </label>
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="개인 모델에 접근하려면 토큰 입력"
                disabled={loading}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              />
              <p className="text-xs text-gray-500 mt-1">🔐 https://huggingface.co/settings/tokens에서 토큰 생성</p>
            </div>

            <button
              onClick={handleDownload}
              disabled={loading || !modelId.trim()}
              className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
            >
              {loading ? '다운로드 중...' : '모델 다운로드'}
            </button>

            {/* 진행상황 표시 */}
            {loading && (
              <div className="mt-6 space-y-4">
                {/* 상태 메시지 */}
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">
                    상태: <span className="text-blue-600 font-semibold">{status}</span>
                  </p>
                </div>

                {/* 진행률 */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-600">진행률</span>
                    <span className="text-blue-600 font-bold text-lg">{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out shadow-lg"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* 단계별 진행상황 */}
                <div className="bg-gray-50 p-4 rounded-lg space-y-2 mt-4">
                  <p className="text-xs font-semibold text-gray-600 mb-3">📋 진행 단계</p>
                  <div className={`flex items-center text-sm ${progress >= 10 ? 'text-green-600' : 'text-gray-400'}`}>
                    <span className={`mr-3 ${progress >= 10 ? '✅' : '○'}`}></span>
                    <span>토크나이저 다운로드</span>
                  </div>
                  <div className={`flex items-center text-sm ${progress >= 25 ? 'text-green-600' : 'text-gray-400'}`}>
                    <span className={`mr-3 ${progress >= 25 ? '✅' : '○'}`}></span>
                    <span>모델 다운로드</span>
                  </div>
                  <div className={`flex items-center text-sm ${progress >= 90 ? 'text-green-600' : 'text-gray-400'}`}>
                    <span className={`mr-3 ${progress >= 90 ? '✅' : '○'}`}></span>
                    <span>모델 로드</span>
                  </div>
                  <div className={`flex items-center text-sm ${progress === 100 ? 'text-green-600' : 'text-gray-400'}`}>
                    <span className={`mr-3 ${progress === 100 ? '✅' : '○'}`}></span>
                    <span>완료</span>
                  </div>
                </div>
              </div>
            )}

            {message && !loading && (
              <div className={`mt-6 p-4 rounded-lg ${message.includes('✅') ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                {message}
              </div>
            )}
          </div>
        </div>

        {/* 빠른 정보 패널 */}
        <div className="space-y-4">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200">
            <h3 className="font-bold text-blue-900 mb-4">📊 캐시 통계</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-blue-700">총 모델 수:</span>
                <span className="font-bold text-blue-900">{downloadedModels.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700">총 용량:</span>
                <span className="font-bold text-blue-900">
                  {formatSize(downloadedModels.reduce((acc, m) => acc + (m.size_gb || 0), 0))}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700">Hugging Face:</span>
                <span className="font-bold text-blue-900">
                  {downloadedModels.filter(m => m.source === 'huggingface').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700">로컬:</span>
                <span className="font-bold text-blue-900">
                  {downloadedModels.filter(m => m.source === 'local').length}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={() => fetchLocalModels()}
            disabled={refreshing}
            className="w-full bg-gray-200 text-gray-700 py-2 rounded-lg font-medium hover:bg-gray-300 disabled:opacity-50 transition-colors flex items-center justify-center"
          >
            {refreshing ? '🔄 새로고침 중...' : '🔄 새로고침'}
          </button>
        </div>
      </div>

      {/* 모델 목록 */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold mb-6">💾 캐시된 모델 ({downloadedModels.length})</h2>
        
        {downloadedModels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {downloadedModels.map((model: any, index) => (
              <div
                key={index}
                onClick={() => setSelectedModel(selectedModel?.path === model.path ? null : model)}
                className="bg-white p-6 rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer border-l-4 border-blue-500"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{getModelIcon(model.source)}</span>
                    <div>
                      <h3 className="font-bold text-gray-800 break-all">{model.model_id}</h3>
                      <p className="text-xs text-gray-500 mt-1">
                        {model.source === 'huggingface' ? '🤗 Hugging Face' : '📂 로컬'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-blue-600 text-lg">{formatSize(model.size_gb)}</p>
                  </div>
                </div>

                <div className="space-y-1 text-sm text-gray-600 border-t pt-3">
                  {model.model_type && (
                    <p>
                      <span className="font-semibold text-gray-700">타입:</span>{' '}
                      <span className="bg-gray-100 px-2 py-1 rounded text-xs">{model.model_type}</span>
                    </p>
                  )}
                  {model.model_present !== undefined && (
                    <p>
                      <span className="font-semibold text-gray-700">모델:</span>{' '}
                      <span className={model.model_present ? 'text-green-600' : 'text-red-600'}>
                        {model.model_present ? '✅ 있음' : '❌ 없음'}
                      </span>
                    </p>
                  )}
                  {model.tokenizer_present !== undefined && (
                    <p>
                      <span className="font-semibold text-gray-700">토크나이저:</span>{' '}
                      <span className={model.tokenizer_present ? 'text-green-600' : 'text-red-600'}>
                        {model.tokenizer_present ? '✅ 있음' : '❌ 없음'}
                      </span>
                    </p>
                  )}
                  {model.config_present && (
                    <p>
                      <span className="font-semibold text-gray-700">설정:</span>{' '}
                      <span className="text-green-600">✅ 있음</span>
                    </p>
                  )}
                </div>

                {/* 추가 상세 정보 */}
                {selectedModel?.path === model.path && (
                  <div className="mt-4 pt-4 border-t space-y-2 text-sm">
                    {model.num_hidden_layers && (
                      <p>
                        <span className="font-semibold">레이어 수:</span> {model.num_hidden_layers}
                      </p>
                    )}
                    {model.hidden_size && (
                      <p>
                        <span className="font-semibold">은닉 크기:</span> {model.hidden_size}
                      </p>
                    )}
                    <p className="text-gray-500 break-all">
                      <span className="font-semibold">경로:</span> {model.path}
                    </p>

                    {/* 액션 버튼들 */}
                    <div className="flex gap-2 mt-4 pt-4 border-t">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOpenFolder(model.path);
                        }}
                        className="flex-1 bg-blue-50 text-blue-600 py-2 px-3 rounded hover:bg-blue-100 transition-colors font-medium text-xs flex items-center justify-center gap-1"
                      >
                        📂 폴더 열기
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteModel(model.model_id);
                        }}
                        disabled={deleting === model.model_id}
                        className="flex-1 bg-red-50 text-red-600 py-2 px-3 rounded hover:bg-red-100 transition-colors font-medium text-xs disabled:opacity-50 flex items-center justify-center gap-1"
                      >
                        {deleting === model.model_id ? '🗑️ 삭제 중...' : '🗑️ 삭제'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-gray-50 p-12 rounded-lg text-center">
            <p className="text-gray-500 text-lg mb-4">📭 캐시된 모델이 없습니다</p>
            <p className="text-gray-400 text-sm">위에서 모델을 다운로드하면 여기에 표시됩니다</p>
          </div>
        )}
      </div>
    </div>
  );
};
