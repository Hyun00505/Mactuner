import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";

interface DataGridProps {
  data: any[];
  columns?: string[];
  title?: string;
  onRowSelect?: (rows: any[]) => void;
  showSearch?: boolean;
  showSort?: boolean;
  showFilter?: boolean;
  showExport?: boolean;
}

interface ColumnInfo {
  name: string;
  type: "text" | "number" | "date" | "boolean";
}

interface FilterRule {
  column: string;
  operator: "contains" | "equals" | ">" | "<" | ">=" | "<=" | "between";
  value: string | number;
  value2?: string | number;
}

const VIRTUAL_SCROLL_BUFFER = 10; // 버퍼 행 수
const ROW_HEIGHT = 40; // 각 행의 높이 (px)

export const DataGrid: React.FC<DataGridProps> = ({
  data,
  columns,
  title = "📊 데이터 그리드",
  onRowSelect,
  showSearch = true,
  showSort = true,
  showFilter = true,
  showExport = true,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState<{
    column: string | null;
    direction: "asc" | "desc";
  }>({ column: null, direction: "asc" });
  const [filters, setFilters] = useState<FilterRule[]>([]);
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [selectAll, setSelectAll] = useState(false);
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(
    new Set(columns || (data.length > 0 ? Object.keys(data[0]) : []))
  );
  const [scrollPosition, setScrollPosition] = useState({ x: 0, y: 0 });
  const [startIndex, setStartIndex] = useState(0);
  const gridRef = useRef<HTMLDivElement>(null);

  // 컬럼 정보 추출
  const columnInfo = useMemo(() => {
    if (data.length === 0) return {};
    
    const info: Record<string, ColumnInfo> = {};
    const firstRow = data[0];
    
    for (const col of Object.keys(firstRow)) {
      const value = firstRow[col];
      let type: ColumnInfo["type"] = "text";
      
      if (typeof value === "number") type = "number";
      else if (value instanceof Date) type = "date";
      else if (typeof value === "boolean") type = "boolean";
      
      info[col] = { name: col, type };
    }
    
    return info;
  }, [data]);

  // 필터링 및 정렬된 데이터
  const filteredAndSortedData = useMemo(() => {
    let result = [...data];

    // 검색 필터
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      result = result.filter((row) =>
        Object.values(row).some(
          (val) =>
            val !== null &&
            val !== undefined &&
            String(val).toLowerCase().includes(term)
        )
      );
    }

    // 고급 필터
    if (filters.length > 0) {
      result = result.filter((row) => {
        return filters.every((filter) => {
          const value = row[filter.column];
          if (value === null || value === undefined) return false;

          switch (filter.operator) {
            case "contains":
              return String(value)
                .toLowerCase()
                .includes(String(filter.value).toLowerCase());
            case "equals":
              return String(value) === String(filter.value);
            case ">":
              return Number(value) > Number(filter.value);
            case "<":
              return Number(value) < Number(filter.value);
            case ">=":
              return Number(value) >= Number(filter.value);
            case "<=":
              return Number(value) <= Number(filter.value);
            case "between":
              return (
                Number(value) >= Number(filter.value) &&
                Number(value) <= Number(filter.value2)
              );
            default:
              return true;
          }
        });
      });
    }

    // 정렬
    if (sortConfig.column) {
      result.sort((a, b) => {
        const aVal = a[sortConfig.column || ""];
        const bVal = b[sortConfig.column || ""];

        if (aVal === null || aVal === undefined) return 1;
        if (bVal === null || bVal === undefined) return -1;

        if (typeof aVal === "number" && typeof bVal === "number") {
          return sortConfig.direction === "asc" ? aVal - bVal : bVal - aVal;
        }

        const aStr = String(aVal).toLowerCase();
        const bStr = String(bVal).toLowerCase();

        if (sortConfig.direction === "asc") {
          return aStr.localeCompare(bStr);
        } else {
          return bStr.localeCompare(aStr);
        }
      });
    }

    return result;
  }, [data, searchTerm, sortConfig, filters]);

  // 행 선택 처리
  const handleRowSelect = (index: number) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedRows(newSelected);
    if (onRowSelect) {
      onRowSelect(
        Array.from(newSelected).map((idx) => filteredAndSortedData[idx])
      );
    }
  };

  // 전체 선택
  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedRows(new Set());
      setSelectAll(false);
    } else {
      const allIndices = new Set(
        filteredAndSortedData.map((_, idx) => idx)
      );
      setSelectedRows(allIndices);
      setSelectAll(true);
      if (onRowSelect) {
        onRowSelect(filteredAndSortedData);
      }
    }
  };

  // 필터 추가
  const addFilter = (column: string) => {
    setFilters([
      ...filters,
      { column, operator: "contains", value: "" },
    ]);
  };

  // 필터 제거
  const removeFilter = (index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
  };

  // 필터 업데이트
  const updateFilter = (
    index: number,
    updates: Partial<FilterRule>
  ) => {
    const newFilters = [...filters];
    newFilters[index] = { ...newFilters[index], ...updates };
    setFilters(newFilters);
  };

  // 컬럼 표시/숨기기
  const toggleColumnVisibility = (col: string) => {
    const newVisibleCols = new Set(visibleColumns);
    if (newVisibleCols.has(col)) {
      newVisibleCols.delete(col);
    } else {
      newVisibleCols.add(col);
    }
    setVisibleColumns(newVisibleCols);
  };

  // CSV 내보내기
  const exportToCSV = () => {
    const cols = Array.from(visibleColumns);
    const csv = [
      cols.join(","),
      ...filteredAndSortedData.map((row) =>
        cols
          .map((col) => {
            const val = row[col];
            if (val === null || val === undefined) return "";
            const str = String(val);
            if (str.includes(",") || str.includes('"') || str.includes("\n")) {
              return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
          })
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `data_export_${new Date().getTime()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // JSON 내보내기
  const exportToJSON = () => {
    const cols = Array.from(visibleColumns);
    const json = filteredAndSortedData.map((row) => {
      const obj: any = {};
      cols.forEach((col) => {
        obj[col] = row[col];
      });
      return obj;
    });

    const blob = new Blob([JSON.stringify(json, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `data_export_${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const visibleColumnsList = Array.from(visibleColumns);

  return (
    <div className="flex flex-col h-full gap-4 bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* 헤더 */}
      <div className="p-6 border-b border-gray-700 bg-gray-900">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-cyan-300">{title}</h2>
          <div className="text-sm text-gray-400">
            📊 {filteredAndSortedData.length} / {data.length} 행
            {selectedRows.size > 0 && ` | ✓ 선택됨: ${selectedRows.size}`}
          </div>
        </div>

        {/* 도구 모음 */}
        <div className="flex flex-wrap gap-2 mb-4">
          {showSearch && (
            <input
              type="text"
              placeholder="🔍 전체 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1 min-w-64 px-4 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-cyan-400 focus:outline-none"
            />
          )}
          
          {showFilter && (
            <button
              onClick={() => addFilter(visibleColumnsList[0] || "")}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              ⊕ 필터 추가
            </button>
          )}

          {showExport && (
            <>
              <button
                onClick={exportToCSV}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              >
                📥 CSV 내보내기
              </button>
              <button
                onClick={exportToJSON}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              >
                📥 JSON 내보내기
              </button>
            </>
          )}

          {/* 컬럼 표시/숨기기 */}
          <div className="relative group">
            <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors">
              👁️ 컬럼 ({visibleColumns.size})
            </button>
            <div className="hidden group-hover:block absolute right-0 top-full mt-1 bg-gray-900 border border-gray-700 rounded-lg shadow-lg p-3 z-50 min-w-48 max-h-96 overflow-y-auto">
              {Object.keys(columnInfo).map((col) => (
                <label
                  key={col}
                  className="flex items-center gap-2 p-2 hover:bg-gray-800 rounded cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={visibleColumns.has(col)}
                    onChange={() => toggleColumnVisibility(col)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-300">{col}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* 필터 표시 */}
        {filters.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {filters.map((filter, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 bg-blue-900 px-3 py-1 rounded-full text-sm"
              >
                <select
                  value={filter.column}
                  onChange={(e) =>
                    updateFilter(idx, { column: e.target.value })
                  }
                  className="bg-transparent text-white text-xs font-medium focus:outline-none"
                >
                  {visibleColumnsList.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                <select
                  value={filter.operator}
                  onChange={(e) =>
                    updateFilter(idx, {
                      operator: e.target.value as FilterRule["operator"],
                    })
                  }
                  className="bg-transparent text-white text-xs font-medium focus:outline-none"
                >
                  <option value="contains">포함</option>
                  <option value="equals">같음</option>
                  <option value=">">크다</option>
                  <option value="<">작다</option>
                  <option value=">=">=</option>
                  <option value="<=">&lt;=</option>
                  <option value="between">범위</option>
                </select>
                <input
                  type="text"
                  value={filter.value}
                  onChange={(e) =>
                    updateFilter(idx, { value: e.target.value })
                  }
                  placeholder="값"
                  className="bg-blue-800 text-white text-xs px-2 py-1 rounded w-24 focus:outline-none"
                />
                {filter.operator === "between" && (
                  <input
                    type="text"
                    value={filter.value2 || ""}
                    onChange={(e) =>
                      updateFilter(idx, { value2: e.target.value })
                    }
                    placeholder="~"
                    className="bg-blue-800 text-white text-xs px-2 py-1 rounded w-24 focus:outline-none"
                  />
                )}
                <button
                  onClick={() => removeFilter(idx)}
                  className="text-red-400 hover:text-red-300 font-bold text-lg leading-none"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 데이터 그리드 */}
      <div
        ref={gridRef}
        className="flex-1 overflow-auto"
        onScroll={(e) => {
          const target = e.target as HTMLDivElement;
          const newStartIndex = Math.floor(target.scrollTop / ROW_HEIGHT);
          setStartIndex(Math.max(0, newStartIndex - VIRTUAL_SCROLL_BUFFER));
          setScrollPosition({ x: target.scrollLeft, y: target.scrollTop });
        }}
      >
        {filteredAndSortedData.length > 0 ? (
          <table className="w-full border-collapse bg-gray-800">
            <thead className="sticky top-0 bg-gray-900 border-b-2 border-gray-600 z-10">
              <tr>
                {/* 체크박스 열 */}
                <th className="w-10 px-3 py-3 text-left">
                  <input
                    type="checkbox"
                    checked={selectAll && filteredAndSortedData.length > 0}
                    onChange={handleSelectAll}
                    className="w-4 h-4 cursor-pointer"
                  />
                </th>
                {/* 행 번호 */}
                <th className="w-12 px-3 py-3 text-left text-gray-400 text-xs font-medium bg-gray-800">#</th>
                {/* 데이터 컬럼 */}
                {visibleColumnsList.map((col) => (
                  <th
                    key={col}
                    onClick={() => {
                      if (showSort) {
                        setSortConfig({
                          column: col,
                          direction:
                            sortConfig.column === col &&
                            sortConfig.direction === "asc"
                              ? "desc"
                              : "asc",
                        });
                      }
                    }}
                    className="px-4 py-3 text-left font-semibold text-cyan-300 cursor-pointer hover:bg-gray-800 whitespace-nowrap group relative bg-gray-900"
                  >
                    <div className="flex items-center gap-1">
                      {col}
                      {sortConfig.column === col && (
                        <span className="text-xs">
                          {sortConfig.direction === "asc" ? "▲" : "▼"}
                        </span>
                      )}
                      <span className="hidden group-hover:inline text-gray-500 text-xs">
                        (클릭 정렬)
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* 위쪽 스페이서 */}
              {startIndex > 0 && (
                <tr>
                  <td colSpan={visibleColumnsList.length + 2} style={{ height: `${startIndex * ROW_HEIGHT}px` }} />
                </tr>
              )}
              
              {/* 렌더링할 행들 (최대 50행) */}
              {filteredAndSortedData
                .slice(startIndex, startIndex + 50 + VIRTUAL_SCROLL_BUFFER)
                .map((row, displayIdx) => {
                  const rowIdx = startIndex + displayIdx;
                  return (
                    <tr
                      key={rowIdx}
                      className={`border-b border-gray-700 hover:bg-gray-700 transition-colors ${
                        selectedRows.has(rowIdx) ? "bg-blue-900" : ""
                      }`}
                    >
                      {/* 체크박스 */}
                      <td className="w-10 px-3 py-2">
                        <input
                          type="checkbox"
                          checked={selectedRows.has(rowIdx)}
                          onChange={() => handleRowSelect(rowIdx)}
                          className="w-4 h-4 cursor-pointer"
                        />
                      </td>
                      {/* 행 번호 */}
                      <td className="w-12 px-3 py-2 text-xs text-gray-500">
                        {rowIdx + 1}
                      </td>
                      {/* 데이터 셀 */}
                      {visibleColumnsList.map((col) => {
                        const value = row[col];
                        const displayValue =
                          value === null || value === undefined
                            ? "∅"
                            : typeof value === "boolean"
                            ? value
                              ? "✓"
                              : "✗"
                            : typeof value === "object"
                            ? JSON.stringify(value).substring(0, 50)
                            : String(value).substring(0, 200);

                        return (
                          <td
                            key={`${rowIdx}-${col}`}
                            className="px-4 py-2 text-sm text-gray-300 font-mono"
                            title={String(value)}
                          >
                            <div className="max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">
                              {displayValue}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              
              {/* 아래쪽 스페이서 */}
              {startIndex + 50 + VIRTUAL_SCROLL_BUFFER < filteredAndSortedData.length && (
                <tr>
                  <td colSpan={visibleColumnsList.length + 2} style={{ height: `${(filteredAndSortedData.length - (startIndex + 50 + VIRTUAL_SCROLL_BUFFER)) * ROW_HEIGHT}px` }} />
                </tr>
              )}
            </tbody>
          </table>
        ) : (
          <div className="flex items-center justify-center h-96 text-gray-400">
            <div className="text-center">
              <p className="text-lg mb-2">📭 표시할 데이터가 없습니다.</p>
              <p className="text-sm text-gray-500">필터 조건을 확인해주세요.</p>
            </div>
          </div>
        )}
      </div>

      {/* 푸터 */}
      <div className="p-4 border-t border-gray-700 bg-gray-900 text-sm text-gray-400 flex justify-between items-center">
        <div>
          {selectedRows.size > 0 && (
            <button
              onClick={() => {
                setSelectedRows(new Set());
                setSelectAll(false);
              }}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
            >
              선택 취소
            </button>
          )}
        </div>
        <div>
          메모리 사용: 약 {(JSON.stringify(filteredAndSortedData).length / 1024).toFixed(2)} KB
        </div>
      </div>
    </div>
  );
};

