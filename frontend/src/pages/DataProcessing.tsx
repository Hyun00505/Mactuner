import React, { useState, useRef, useMemo, useEffect } from "react";
import { datasetAPI } from "../utils/api";

interface DataStats {
  total_rows: number;
  total_columns: number;
  memory_mb: number;
  missing_values: number;
  duplicates: number;
  columns: Array<{ name: string; dtype: string; missing_count: number }>;
}

interface SortConfig {
  column: string | null;
  direction: "asc" | "desc";
}

interface FilterConfig {
  column: string;
  operator: "contains" | "equals" | ">" | "<" | ">=";
  value: string;
}

export const DataProcessing: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dataFormat, setDataFormat] = useState("csv");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [dataInfo, setDataInfo] = useState<any>(null);
  const [preview, setPreview] = useState<any[]>([]);
  const [stats, setStats] = useState<DataStats | null>(null);

  // 검색/필터/정렬
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState<SortConfig>({ column: null, direction: "asc" });
  const [filters, setFilters] = useState<FilterConfig[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  
  // HuggingFace 다운로드
  const [hfDatasetId, setHfDatasetId] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [hfSplit, setHfSplit] = useState("train");
  const [hfMaxSamples, setHfMaxSamples] = useState<number | undefined>();
  const [showHFPanel, setShowHFPanel] = useState(false);
  
  // 히스토리 및 캐시
  const [history, setHistory] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [cachedDatasets, setCachedDatasets] = useState<any[]>([]);
  const [showCached, setShowCached] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteDataOption, setDeleteDataOption] = useState(false);
  const [deleteTargetIndex, setDeleteTargetIndex] = useState<number | null>(null);
  
  // 최소화
  const [minimizeUpload, setMinimizeUpload] = useState(false);
  const [minimizeHF, setMinimizeHF] = useState(false);
  
  // 전체 최소화
  const toggleAllMinimize = () => {
    const bothMinimized = minimizeUpload && minimizeHF;
    setMinimizeUpload(!bothMinimized);
    setMinimizeHF(!bothMinimized);
  };

  // 페이지 로드 시 기존 데이터 자동 불러오기
  useEffect(() => {
    const loadExistingData = async () => {
      try {
        console.log("📂 페이지 로드: 기존 데이터 확인 중...");
        const response = await datasetAPI.info();
        console.log("📂 기존 데이터 응답:", response);
        
        // 데이터가 없는 경우
        if (!response.data || response.data.status === "no_data") {
          console.log("📂 기존 데이터 없음 (정상)");
          return;
        }
        
        // 데이터가 있으면 자동 불러오기
        if (response.data.status === "success" && response.data.data) {
          setMessage("📂 기존 데이터 발견! 자동으로 로드하는 중...");
          
          // 데이터 정보 설정
          const infoData = response.data.data;
          setDataInfo(infoData);
          console.log("📂 데이터 정보 설정:", infoData);
          
          // 프리뷰 데이터 가져오기
          try {
            const previewResponse = await datasetAPI.preview(50);
            const previewData = previewResponse.data?.data || previewResponse.data;
            
            if (previewData && previewData.head && Array.isArray(previewData.head)) {
              setPreview(previewData.head);
              console.log("📋 프리뷰 로드:", previewData.head.length, "행");
            } else if (Array.isArray(previewData)) {
              setPreview(previewData);
              console.log("📋 프리뷰 로드:", previewData.length, "행");
            }
          } catch (previewError) {
            console.warn("프리뷰 로드 실패:", previewError);
          }
          
          // 통계 정보 가져오기
          try {
            const statsResponse = await datasetAPI.statistics();
            const statsData = statsResponse.data?.data || statsResponse.data;
            if (statsData) {
              setStats(statsData);
              console.log("📈 통계 정보 로드됨");
            }
          } catch (statsError) {
            console.warn("통계 로드 실패:", statsError);
          }
          
          setMessage("✅ 데이터 로드 완료!");
        }
      } catch (error) {
        console.log("📂 기존 데이터 확인 오류 (무시):", error);
        // 데이터가 없으면 무시 (정상 상태)
      }
    };
    
    loadExistingData();
  }, []); // 페이지 로드 시 한 번만 실행

  // 파일 형식 자동 감지
  const detectFileFormat = (filename: string): string => {
    const ext = filename.toLowerCase().split(".").pop() || "";
    if (ext === "csv") return "csv";
    if (ext === "json") return "json";
    if (ext === "jsonl") return "jsonl";
    if (ext === "xlsx" || ext === "xls") return "excel";
    return "csv"; // 기본값
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      // 자동으로 파일 형식 감지
      const detectedFormat = detectFileFormat(selectedFile.name);
      setDataFormat(detectedFormat);
      // 자동으로 업로드 실행
      setTimeout(() => {
        uploadFile(selectedFile, detectedFormat);
      }, 100);
    }
  };

  // 드래그앤드롭 핸들러
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add("border-blue-400", "bg-blue-50");
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");

    if (e.dataTransfer.files) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      // 자동으로 파일 형식 감지 및 업로드
      const detectedFormat = detectFileFormat(droppedFile.name);
      setDataFormat(detectedFormat);
      // 즉시 업로드
      uploadFile(droppedFile, detectedFormat);
    }
  };

  const uploadFile = async (uploadFile: File, format: string) => {
    if (!uploadFile) {
      setMessage("파일을 선택해주세요");
      return;
    }

    try {
      setLoading(true);
      setMessage("파일 업로드 중...");
      console.log("Uploading file:", uploadFile.name, "Format:", format);
      const response = await datasetAPI.upload(uploadFile, format);
      console.log("Upload response:", response);
      setMessage(`✅ 파일 업로드 완료! (${format.toUpperCase()})`);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await fetchDataInfo();
    } catch (error: any) {
      console.error("Upload error details:", error);
      const errorMsg = error.response?.data?.detail || error.message || "알 수 없는 오류";
      setMessage(`❌ 오류: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  // HuggingFace 데이터셋 다운로드
  const downloadHFDataset = async () => {
    if (!hfDatasetId) {
      setMessage("데이터셋 ID를 입력해주세요");
      return;
    }

    try {
      setLoading(true);
      setMessage("🔄 HuggingFace 데이터셋 다운로드 중...");
      console.log("Downloading HF dataset:", hfDatasetId);
      
      const response = await datasetAPI.downloadHF(hfDatasetId, hfToken || undefined, hfSplit, hfMaxSamples);
      console.log("Download response:", response);
      
      setMessage(`✅ HuggingFace 데이터셋 로드 완료!`);
      setHfDatasetId("");
      setShowHFPanel(false);
      
      // 데이터 정보 불러오기
      await fetchDataInfo();
      
      // 데이터 정보 영역으로 자동 스크롤
      setTimeout(() => {
        const infoSection = document.querySelector("h2.text-xl");
        if (infoSection) {
          infoSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 500);
    } catch (error: any) {
      console.error("Download error:", error);
      const errorMsg = error.response?.data?.detail || error.message || "알 수 없는 오류";
      setMessage(`❌ 오류: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchDataInfo = async () => {
    try {
      const response = await datasetAPI.info();
      console.log("📊 Data info response:", response);
      
      // 백엔드 응답 구조: { status: "success", data: { shape: {...}, file_info: {...}, ... } }
      const infoData = response.data?.data || response.data;
      setDataInfo(infoData);
      console.log("📊 Set dataInfo to:", infoData);
      
      // preview 데이터 가져오기
      const previewResponse = await datasetAPI.preview(50);
      console.log("📋 Preview response:", previewResponse);
      
      const previewData = previewResponse.data?.data || previewResponse.data;
      
      // preview는 { head: [...], tail: [...], ... } 구조이므로 head만 사용
      if (previewData && previewData.head && Array.isArray(previewData.head)) {
        setPreview(previewData.head);
        console.log("📋 Set preview (from head):", previewData.head.length, "rows");
      } else if (Array.isArray(previewData)) {
        setPreview(previewData);
        console.log("📋 Set preview (direct array):", previewData.length, "rows");
      } else {
        setPreview([]);
        console.log("📋 No preview data found");
      }
      
      // 통계 정보도 가져오기
      try {
        const statsResponse = await datasetAPI.statistics();
        console.log("📈 Statistics response:", statsResponse);
        const statsData = statsResponse.data?.data || statsResponse.data;
        setStats(statsData);
        console.log("📈 Set stats to:", statsData);
      } catch (statsError) {
        console.warn("Failed to fetch statistics", statsError);
      }
      
      // 히스토리 및 캐시 로드
      loadHistory();
      loadCachedDatasets();
    } catch (error) {
      console.error("Failed to fetch data info", error);
      setMessage("❌ 데이터 정보를 불러오지 못했습니다");
    }
  };
  
  const loadHistory = async () => {
    try {
      const response = await datasetAPI.getHistory();
      console.log("📚 History response:", response);
      const historyData = response.data?.data || response.data || [];
      setHistory(historyData);
    } catch (error) {
      console.warn("히스토리 로드 실패:", error);
    }
  };
  
  const loadCachedDatasets = async () => {
    try {
      const response = await datasetAPI.getCachedDatasets();
      console.log("💾 Cached datasets response:", response);
      const cachedData = response.data?.data || response.data || [];
      setCachedDatasets(cachedData);
    } catch (error) {
      console.warn("캐시 데이터셋 로드 실패:", error);
    }
  };
  
  const handleReloadFromHistory = async (index: number) => {
    try {
      setLoading(true);
      setMessage("⏳ 데이터셋 로드 중...");
      const response = await datasetAPI.reloadFromHistory(index);
      console.log("📚 Reload response:", response);
      setMessage("✅ 데이터셋 로드 완료!");
      await fetchDataInfo();
    } catch (error: any) {
      console.error("히스토리에서 로드 실패:", error);
      setMessage(`❌ 오류: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleClearHistory = async () => {
    try {
      setLoading(true);
      const response = await datasetAPI.deleteHistoryItem(deleteTargetIndex!, deleteDataOption);
      console.log("🗑️ Delete item response:", response);
      setMessage(response.data?.message || "✅ 삭제되었습니다!");
      setShowDeleteConfirm(false);
      setDeleteDataOption(false);
      setDeleteTargetIndex(null);
      await loadHistory(); // 히스토리 새로고침
    } catch (error: any) {
      console.error("항목 삭제 실패:", error);
      setMessage(`❌ 오류: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleDeleteHistoryItem = async (index: number, e: React.MouseEvent) => {
    e.stopPropagation(); // 클릭 전파 방지
    setDeleteTargetIndex(index);
    setDeleteDataOption(false); // 기본값: 히스토리만 삭제
    setShowDeleteConfirm(true);
  };

  // 검색/필터/정렬이 적용된 데이터
  const processedData = useMemo(() => {
    let result = [...preview];

    // 필터 적용
    filters.forEach((filter) => {
      result = result.filter((row) => {
        const cellValue = String(row[filter.column] || "").toLowerCase();
        const filterValue = filter.value.toLowerCase();

        switch (filter.operator) {
          case "contains":
            return cellValue.includes(filterValue);
          case "equals":
            return cellValue === filterValue;
          case ">":
            return parseFloat(cellValue) > parseFloat(filter.value);
          case "<":
            return parseFloat(cellValue) < parseFloat(filter.value);
          case ">=":
            return parseFloat(cellValue) >= parseFloat(filter.value);
          default:
            return true;
        }
      });
    });

    // 검색 적용
    if (searchTerm) {
      result = result.filter((row) => Object.values(row).some((val) => String(val).toLowerCase().includes(searchTerm.toLowerCase())));
    }

    // 정렬 적용
    if (sortConfig.column) {
      result.sort((a, b) => {
        const aVal = a[sortConfig.column!];
        const bVal = b[sortConfig.column!];

        const comparison = String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
        return sortConfig.direction === "asc" ? comparison : -comparison;
      });
    }

    return result;
  }, [preview, filters, searchTerm, sortConfig]);

  // 페이지 처리된 데이터
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    return processedData.slice(start, end);
  }, [processedData, currentPage, rowsPerPage]);

  const totalPages = Math.ceil(processedData.length / rowsPerPage);

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
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* 헤더 */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">📊 데이터 처리</h1>
          <p className="text-gray-400">CSV, Excel, JSON 파일을 업로드하고 분석, 정제하기</p>
        </div>

        {/* 히스토리 토글 */}
        {history.length > 0 && (
          <div className="mb-4">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              📚 {showHistory ? "히스토리 숨기기" : "히스토리 보기"} ({history.length})
            </button>
          </div>
        )}
        
        {/* 개별 항목 삭제 확인 다이얼로그 */}
        {showDeleteConfirm && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 border border-red-600 max-w-md">
              <h3 className="text-lg font-bold mb-4 text-white">🗑️ 항목 삭제</h3>
              <p className="text-gray-300 mb-4">어떤 항목을 삭제하시겠어요?</p>
              
              <div className="space-y-3 mb-6">
                <label className="flex items-center gap-3 p-3 border border-gray-600 rounded-lg cursor-pointer hover:bg-gray-700">
                  <input
                    type="radio"
                    checked={!deleteDataOption}
                    onChange={() => setDeleteDataOption(false)}
                    className="w-4 h-4"
                  />
                  <div>
                    <div className="font-medium text-white">📋 히스토리만 삭제</div>
                    <div className="text-xs text-gray-400">파일은 유지됩니다</div>
                  </div>
                </label>
                
                <label className="flex items-center gap-3 p-3 border border-red-600 rounded-lg cursor-pointer hover:bg-gray-700">
                  <input
                    type="radio"
                    checked={deleteDataOption}
                    onChange={() => setDeleteDataOption(true)}
                    className="w-4 h-4"
                  />
                  <div>
                    <div className="font-medium text-white">🗑️ 히스토리 + 파일 삭제</div>
                    <div className="text-xs text-gray-400">모두 삭제됩니다</div>
                  </div>
                </label>
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
                >
                  ❌ 취소
                </button>
                <button
                  onClick={handleClearHistory}
                  disabled={loading}
                  className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
                >
                  {loading ? "⏳ 삭제 중..." : "✓ 삭제"}
                </button>
              </div>
            </div>
          </div>
        )}
        
        {/* 데이터셋 히스토리 */}
        {showHistory && history.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-4 mb-8 border border-cyan-600 overflow-x-auto">
            <h3 className="text-lg font-bold mb-3">📚 최근 로드된 데이터셋</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {history.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-gray-700 rounded-lg p-3 border border-gray-600 hover:border-cyan-400 transition-colors cursor-pointer group relative"
                  onClick={() => handleReloadFromHistory(idx)}
                >
                  {/* 삭제 버튼 - 우측 상단 */}
                  <button
                    onClick={(e) => handleDeleteHistoryItem(idx, e)}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-red-600 hover:bg-red-700 text-white rounded p-1 text-xs"
                    title="이 항목 삭제"
                  >
                    🗑️
                  </button>
                  
                  <div className="text-sm font-bold text-cyan-300 truncate pr-6">
                    {item.source === "hf" ? "🤗" : "📁"} {item.filename}
                  </div>
                  
                  <div className="text-xs text-gray-400 mt-2 space-y-1">
                    <div>📊 {item.rows} 행 × {item.columns} 열</div>
                    <div>💾 {item.size_mb?.toFixed(2) || "N/A"} MB</div>
                    <div>🏷️ {item.format}</div>
                    {item.encoding && (
                      <div className="text-gray-500">
                        🔤 {item.encoding}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 파일 업로드 & HuggingFace 데이터셋 - 옆으로 배치 */}
        <div className="space-y-4 mb-8">
          {/* 전체 제어 버튼 - 왼쪽 */}
          <div className="flex justify-start mb-2">
            <button
              onClick={toggleAllMinimize}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs font-medium transition-colors"
            >
              {minimizeUpload && minimizeHF ? "▼ 모두 펼치기" : "▲ 모두 접기"}
            </button>
          </div>
          
          {/* 상단 헤더 행 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 파일 업로드 섹션 헤더 */}
            <h3 className="text-lg font-bold text-blue-400">📁 파일 업로드</h3>
            
            {/* HuggingFace 섹션 헤더 */}
            <h3 className="text-lg font-bold text-purple-400">🤗 HuggingFace</h3>
          </div>
          
          {/* 컨텐츠 행 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 파일 업로드 영역 - 드래그앤드롭 */}
            {!minimizeUpload && (
            <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className="bg-gray-800 rounded-lg p-6 border-2 border-dashed border-blue-600 hover:border-blue-400 transition-colors cursor-pointer flex flex-col"
          >
            <div className="text-center flex-1 flex flex-col">
              <h2 className="text-xl font-bold mb-2">📁 파일 업로드</h2>
              <p className="text-gray-400 text-sm mb-1">드래그앤드롭하거나</p>
              <p className="text-gray-500 text-xs mb-4">CSV, Excel, JSON, JSONL</p>

              {/* 드래그앤드롭 표시 영역 */}
              <div className="flex-1 flex items-center justify-center border-2 border-dashed border-gray-600 rounded-lg mb-4 hover:border-blue-400 transition-colors bg-gray-700">
                <div className="text-center">
                  <div className="text-4xl mb-2">📤</div>
                  <p className="text-gray-300 font-semibold text-sm">파일을 드래그</p>
                </div>
              </div>

              {/* 파일 선택 버튼 */}
              <input ref={fileInputRef} type="file" onChange={handleFileChange} accept=".csv,.json,.jsonl,.xlsx,.xls" className="hidden" id="file-input" />
              <label htmlFor="file-input" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-semibold cursor-pointer transition-colors text-sm mb-3 text-center">
                📂 파일 선택
              </label>

              {/* 선택된 파일 정보 */}
              {file && (
                <div className="p-3 bg-gray-700 rounded w-full text-left mb-3">
                  <p className="text-xs text-gray-300">
                    <span className="font-semibold">📄</span> {file.name}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    <span className="font-semibold">🎯</span> {dataFormat.toUpperCase()} • {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              )}

              {/* 상태 표시 및 버튼 */}
              <div className="flex gap-2 w-full">
                {loading && (
                  <div className="flex-1 flex items-center justify-center gap-1 text-blue-400 text-xs">
                    <div className="animate-spin text-sm">⏳</div>
                    <span>중...</span>
                  </div>
                )}
                {!loading && file && (
                  <button
                    onClick={() => {
                      setFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                    className="flex-1 bg-red-700 hover:bg-red-600 text-white py-1 rounded text-xs font-medium transition-colors"
                  >
                    ❌ 초기화
                  </button>
                )}
                <button onClick={() => setCurrentPage(1)} className="flex-1 bg-gray-700 hover:bg-gray-600 text-white py-1 rounded text-xs font-medium transition-colors">
                  🔄 새로
                </button>
              </div>

              {/* 메시지 */}
              {message && (
                <div className={`mt-3 p-2 rounded text-xs ${message.includes("✅") ? "bg-green-900 text-green-200 border border-green-700" : "bg-red-900 text-red-200 border border-red-700"}`}>{message}</div>
              )}
            </div>
          </div>
            )}
            
            {/* HuggingFace 데이터셋 다운로드 */}
            {!minimizeHF && (
          <div className="bg-gray-800 rounded-lg p-6 border-2 border-dashed border-purple-600 hover:border-purple-400 transition-colors flex flex-col">
            <div className="text-center flex-1 flex flex-col">
              <p className="text-xs text-yellow-400 mb-4">⚠️ 인터넷 연결 필수</p>

              <div className="flex gap-2 justify-center mb-3">
                <button
                  onClick={() => setShowHFPanel(!showHFPanel)}
                  className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded text-sm font-semibold transition-colors"
                >
                  {showHFPanel ? "▼" : "▶"} {showHFPanel ? "닫기" : "열기"}
                </button>
                <a
                  href="https://huggingface.co/datasets"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded text-sm font-semibold transition-colors"
                >
                  🔍 탐색
                </a>
              </div>

              {showHFPanel && (
                <div className="mt-3 space-y-2 bg-gray-700 p-4 rounded-lg flex-1 flex flex-col">
                  <div>
                    <label className="block text-xs font-medium text-gray-300 mb-1">📌 ID</label>
                    <input
                      type="text"
                      placeholder="wikitext 또는 username/dataset-name"
                      value={hfDatasetId}
                      onChange={(e) => setHfDatasetId(e.target.value)}
                      className="w-full bg-gray-600 text-white px-2 py-1 rounded text-xs border border-gray-500 focus:border-purple-500 focus:outline-none"
                    />
                    <div className="text-xs text-gray-400 mt-1 space-y-1">
                      <p>💡 예: wikitext, poperson1205/mrtydi-v1.1-korean</p>
                      <p className="text-yellow-400">⭐ 테스트: test, demo (인터넷 불필요)</p>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-300 mb-1">🔐 토큰 (선택)</label>
                    <input
                      type="password"
                      placeholder="토큰"
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      className="w-full bg-gray-600 text-white px-2 py-1 rounded text-xs border border-gray-500 focus:border-purple-500 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-1">📂 Split</label>
                      <input
                        type="text"
                        placeholder="train"
                        value={hfSplit}
                        onChange={(e) => setHfSplit(e.target.value)}
                        className="w-full bg-gray-600 text-white px-2 py-1 rounded text-xs border border-gray-500 focus:border-purple-500 focus:outline-none"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-1">📊 샘플</label>
                      <input
                        type="number"
                        placeholder="전체"
                        value={hfMaxSamples || ""}
                        onChange={(e) => setHfMaxSamples(e.target.value ? parseInt(e.target.value) : undefined)}
                        className="w-full bg-gray-600 text-white px-2 py-1 rounded text-xs border border-gray-500 focus:border-purple-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  <button
                    onClick={downloadHFDataset}
                    disabled={loading || !hfDatasetId}
                    className="mt-auto w-full bg-purple-600 hover:bg-purple-700 text-white py-1 rounded text-xs font-medium disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? "⏳ 중..." : "🚀 다운로드"}
                  </button>
                </div>
              )}
            </div>
            </div>
            )}
          </div>
        </div>

        {/* 데이터 정보 카드 */}
        {dataInfo && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gradient-to-br from-blue-900 to-blue-800 p-6 rounded-lg border border-blue-700">
                <div className="text-3xl font-bold text-blue-300">{dataInfo.shape?.rows}</div>
                <p className="text-gray-400 mt-2">총 행 수</p>
              </div>
              <div className="bg-gradient-to-br from-green-900 to-green-800 p-6 rounded-lg border border-green-700">
                <div className="text-3xl font-bold text-green-300">{dataInfo.shape?.columns}</div>
                <p className="text-gray-400 mt-2">열 수</p>
              </div>
              <div className="bg-gradient-to-br from-purple-900 to-purple-800 p-6 rounded-lg border border-purple-700">
                <div className="text-3xl font-bold text-purple-300">{(dataInfo.size_mb || 0).toFixed(2)}</div>
                <p className="text-gray-400 mt-2">크기 (MB)</p>
              </div>
              <div className="bg-gradient-to-br from-orange-900 to-orange-800 p-6 rounded-lg border border-orange-700">
                <div className="text-3xl font-bold text-orange-300">{stats?.missing_values || 0}</div>
                <p className="text-gray-400 mt-2">결측치</p>
              </div>
            </div>

            {/* 데이터 조작 패널 */}
            <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
              <h2 className="text-xl font-bold mb-6">🔧 데이터 조작</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {/* 검색 */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">🔍 검색</label>
                  <input
                    type="text"
                    placeholder="모든 컬럼에서 검색..."
                    value={searchTerm}
                    onChange={(e) => {
                      setSearchTerm(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>

                {/* 정렬 */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">🔄 정렬</label>
                  <select
                    value={sortConfig.column || ""}
                    onChange={(e) => setSortConfig((prev) => ({ ...prev, column: e.target.value || null }))}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="">열 선택...</option>
                    {preview.length > 0 &&
                      Object.keys(preview[0]).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                  </select>
                </div>

                {/* 정렬 방향 */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">순서</label>
                  <select
                    value={sortConfig.direction}
                    onChange={(e) => setSortConfig((prev) => ({ ...prev, direction: e.target.value as "asc" | "desc" }))}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="asc">⬆️ 오름차순</option>
                    <option value="desc">⬇️ 내림차순</option>
                  </select>
                </div>
              </div>

              {/* 필터 추가 */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => {
                    if (preview.length > 0) {
                      const firstCol = Object.keys(preview[0])[0];
                      setFilters([...filters, { column: firstCol, operator: "contains", value: "" }]);
                    }
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
                >
                  ➕ 필터 추가
                </button>
                {filters.length > 0 && (
                  <button onClick={() => setFilters([])} className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors">
                    ❌ 필터 초기화
                  </button>
                )}
                <div className="text-gray-400 text-sm flex items-center ml-auto">
                  검색 결과: <span className="text-green-400 font-bold ml-2">{processedData.length}</span> / <span className="text-gray-500 ml-1">{preview.length}</span> 행
                </div>
              </div>

              {/* 필터 UI */}
              {filters.map((filter, idx) => (
                <div key={idx} className="grid grid-cols-1 md:grid-cols-4 gap-2 mb-3 p-3 bg-gray-700 rounded-lg">
                  <select
                    value={filter.column}
                    onChange={(e) => {
                      const newFilters = [...filters];
                      newFilters[idx].column = e.target.value;
                      setFilters(newFilters);
                    }}
                    className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                  >
                    {preview.length > 0 &&
                      Object.keys(preview[0]).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                  </select>
                  <select
                    value={filter.operator}
                    onChange={(e) => {
                      const newFilters = [...filters];
                      newFilters[idx].operator = e.target.value as any;
                      setFilters(newFilters);
                    }}
                    className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="contains">포함</option>
                    <option value="equals">동일</option>
                    <option value=">">초과</option>
                    <option value="<">미만</option>
                    <option value=">=">이상</option>
                  </select>
                  <input
                    type="text"
                    value={filter.value}
                    onChange={(e) => {
                      const newFilters = [...filters];
                      newFilters[idx].value = e.target.value;
                      setFilters(newFilters);
                    }}
                    placeholder="필터 값..."
                    className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                  />
                  <button onClick={() => setFilters(filters.filter((_, i) => i !== idx))} className="px-4 py-2 bg-red-700 hover:bg-red-600 rounded-lg font-medium transition-colors">
                    🗑️ 제거
                  </button>
                </div>
              ))}
            </div>

            {/* 데이터 정제 패널 */}
            <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
              <h2 className="text-xl font-bold mb-6">🧹 데이터 정제</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <button onClick={() => handleClean("missing_values")} disabled={loading} className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors disabled:opacity-50">
                  ⚠️ 결측치 처리
                </button>
                <button onClick={() => handleClean("duplicates")} disabled={loading} className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors disabled:opacity-50">
                  🔁 중복 제거
                </button>
                <button onClick={() => handleClean("normalize_text")} disabled={loading} className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors disabled:opacity-50">
                  📝 텍스트 정규화
                </button>
                <button onClick={() => handleClean("filter_by_length")} disabled={loading} className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors disabled:opacity-50">
                  📏 길이 필터링
                </button>
              </div>
            </div>

            {/* 데이터 테이블 */}
            {preview && preview.length > 0 ? (
              <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 mb-8">
                <div className="p-6 border-b border-gray-700">
                  <h2 className="text-xl font-bold">📋 데이터 미리보기</h2>
                  <p className="text-gray-400 text-sm mt-1">
                    페이지 {currentPage} / {totalPages} ({paginatedData.length}개 행)
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-700 border-b border-gray-600">
                        {paginatedData.length > 0 &&
                          Object.keys(paginatedData[0]).map((key) => (
                            <th
                              key={key}
                              className="px-4 py-3 text-left font-semibold text-gray-300 whitespace-nowrap cursor-pointer hover:text-white"
                              onClick={() => setSortConfig({ column: key, direction: sortConfig.direction })}
                            >
                              {key}
                            </th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedData.map((row, idx) => (
                        <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700 transition-colors">
                          {Object.values(row as any).map((val, cidx) => (
                            <td key={cidx} className="px-4 py-3 text-sm text-gray-300">
                              <div className="max-w-xs truncate" title={String(val)}>
                                {val === null || val === undefined ? "∅" : String(val).substring(0, 100)}
                              </div>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* 페이지네이션 */}
                <div className="p-6 border-t border-gray-700 flex items-center justify-between">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    ⬅️ 이전
                  </button>
                  <div className="flex gap-2 items-center">
                    <input
                      type="number"
                      min="1"
                      max={totalPages}
                      value={currentPage}
                      onChange={(e) => setCurrentPage(Math.min(totalPages, Math.max(1, parseInt(e.target.value) || 1)))}
                      className="w-16 bg-gray-700 text-white px-3 py-2 rounded text-center border border-gray-600 focus:border-blue-500 focus:outline-none"
                    />
                    <span className="text-gray-400">/ {totalPages}</span>
                  </div>
                  <select
                    value={rowsPerPage}
                    onChange={(e) => {
                      setRowsPerPage(parseInt(e.target.value));
                      setCurrentPage(1);
                    }}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="10">10개/페이지</option>
                    <option value="25">25개/페이지</option>
                    <option value="50">50개/페이지</option>
                    <option value="100">100개/페이지</option>
                  </select>
                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    다음 ➜
                  </button>
                </div>

                {/* 데이터 내보내기 */}
                <div className="p-6 border-t border-gray-700 flex gap-4 flex-wrap">
                  <h3 className="w-full text-lg font-semibold mb-3">📥 데이터 내보내기</h3>
                  <button
                    onClick={() => {
                      const csv = generateCSV(paginatedData);
                      downloadFile(csv, "data.csv", "text/csv");
                    }}
                    className="flex-1 min-w-[150px] px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
                  >
                    📊 CSV로 내보내기
                  </button>
                  <button
                    onClick={() => {
                      const json = JSON.stringify(paginatedData, null, 2);
                      downloadFile(json, "data.json", "application/json");
                    }}
                    className="flex-1 min-w-[150px] px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
                  >
                    📋 JSON으로 내보내기
                  </button>
                  <button
                    onClick={() => {
                      const json = paginatedData.map((row) => JSON.stringify(row)).join("\n");
                      downloadFile(json, "data.jsonl", "text/plain");
                    }}
                    className="flex-1 min-w-[150px] px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors"
                  >
                    🔗 JSONL로 내보내기
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
                <p className="text-gray-400">📭 파일을 업로드하면 여기에 데이터가 표시됩니다</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// 유틸리티 함수들
function generateCSV(data: any[]): string {
  if (data.length === 0) return "";

  const headers = Object.keys(data[0]);
  const headerRow = headers.map((h) => `"${h}"`).join(",");

  const rows = data.map((row) =>
    headers
      .map((header) => {
        const value = row[header];
        const stringValue = String(value === null || value === undefined ? "" : value);
        return `"${stringValue.replace(/"/g, '""')}"`;
      })
      .join(",")
  );

  return [headerRow, ...rows].join("\n");
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}
