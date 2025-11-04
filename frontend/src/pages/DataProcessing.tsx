import React, { useState, useRef } from 'react';
import { datasetAPI } from '../utils/api';

export const DataProcessing: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dataFormat, setDataFormat] = useState('csv');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [dataInfo, setDataInfo] = useState<any>(null);
  const [preview, setPreview] = useState<any[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('파일을 선택해주세요');
      return;
    }

    try {
      setLoading(true);
      setMessage('파일 업로드 중...');
      const response = await datasetAPI.upload(file, dataFormat);
      setMessage('✅ 파일 업로드 완료!');
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await fetchDataInfo();
    } catch (error: any) {
      setMessage(`❌ 오류: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchDataInfo = async () => {
    try {
      const response = await datasetAPI.info();
      setDataInfo(response.data);
      const previewResponse = await datasetAPI.preview(5);
      setPreview(previewResponse.data.data || []);
    } catch (error) {
      console.error('Failed to fetch data info', error);
    }
  };

  const handleClean = async (operation: string) => {
    try {
      setLoading(true);
      setMessage(`${operation} 작업 중...`);
      await datasetAPI.clean(operation);
      setMessage(`✅ ${operation} 작업 완료!`);
      await fetchDataInfo();
    } catch (error: any) {
      setMessage(`❌ 오류: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    fetchDataInfo();
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">📊 데이터 처리</h1>

      {/* 파일 업로드 */}
      <div className="bg-white p-8 rounded-lg shadow mb-8">
        <h2 className="text-xl font-bold mb-6">파일 업로드</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              파일 형식
            </label>
            <select
              value={dataFormat}
              onChange={(e) => setDataFormat(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            >
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
              <option value="jsonl">JSONL</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              파일 선택
            </label>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleUpload}
              disabled={loading || !file}
              className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? '업로드 중...' : '업로드'}
            </button>
          </div>
        </div>
        {message && (
          <div className={`mt-4 p-4 rounded-lg ${message.includes('✅') ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
            {message}
          </div>
        )}
      </div>

      {/* 데이터 정보 */}
      {dataInfo && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-blue-50 p-6 rounded-lg">
              <div className="text-3xl font-bold text-blue-600">{dataInfo.shape?.rows}</div>
              <p className="text-gray-600 mt-2">행 수</p>
            </div>
            <div className="bg-green-50 p-6 rounded-lg">
              <div className="text-3xl font-bold text-green-600">{dataInfo.shape?.columns}</div>
              <p className="text-gray-600 mt-2">열 수</p>
            </div>
            <div className="bg-orange-50 p-6 rounded-lg">
              <div className="text-3xl font-bold text-orange-600">{(dataInfo.size_mb || 0).toFixed(2)}</div>
              <p className="text-gray-600 mt-2">크기 (MB)</p>
            </div>
          </div>

          {/* 데이터 정제 */}
          <div className="bg-white p-8 rounded-lg shadow mb-8">
            <h2 className="text-xl font-bold mb-6">데이터 정제</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <button
                onClick={() => handleClean('missing_values')}
                className="p-4 bg-gray-100 hover:bg-gray-200 rounded-lg font-medium"
              >
                결측치 처리
              </button>
              <button
                onClick={() => handleClean('duplicates')}
                className="p-4 bg-gray-100 hover:bg-gray-200 rounded-lg font-medium"
              >
                중복 제거
              </button>
              <button
                onClick={() => handleClean('normalize_text')}
                className="p-4 bg-gray-100 hover:bg-gray-200 rounded-lg font-medium"
              >
                텍스트 정규화
              </button>
              <button
                onClick={() => handleClean('filter_by_length')}
                className="p-4 bg-gray-100 hover:bg-gray-200 rounded-lg font-medium"
              >
                길이 필터링
              </button>
            </div>
          </div>

          {/* 데이터 미리보기 */}
          {preview.length > 0 && (
            <div className="bg-white p-8 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-6">데이터 미리보기</h2>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      {Object.keys(preview[0] || {}).map((key) => (
                        <th key={key} className="border p-2 text-left font-semibold">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.map((row, idx) => (
                      <tr key={idx} className="border-b hover:bg-gray-50">
                        {Object.values(row as any).map((val, cidx) => (
                          <td key={cidx} className="border p-2 text-sm">
                            {String(val).substring(0, 50)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
