/**
 * 디바이스 선택 컴포넌트
 * GPU/CPU를 감지하고 사용자가 선택할 수 있는 UI를 제공합니다.
 */

import React, { useEffect, useState } from 'react';
import { useDeviceStore, DeviceType } from '../stores/deviceStore';

interface DeviceSelectorProps {
  onDeviceSelected?: (device: DeviceType) => void;
  className?: string;
}

const DeviceSelector: React.FC<DeviceSelectorProps> = ({ 
  onDeviceSelected,
  className = ''
}) => {
  const {
    availableDevices,
    selectedDevice,
    currentDevice,
    memoryInfo,
    isLoading,
    error,
    fetchAvailableDevices,
    selectDevice,
    autoSelectDevice,
    clearCache,
    resetError,
  } = useDeviceStore();

  const [showMemoryDetails, setShowMemoryDetails] = useState(false);

  // 컴포넌트 마운트 시 디바이스 감지
  useEffect(() => {
    fetchAvailableDevices();
    autoSelectDevice();
  }, []);

  // 디바이스 선택 핸들러
  const handleSelectDevice = async (deviceType: DeviceType) => {
    const success = await selectDevice(deviceType);
    if (success) {
      onDeviceSelected?.(deviceType);
    }
  };

  // 캐시 정리 핸들러
  const handleClearCache = async () => {
    await clearCache();
  };

  // 디바이스 아이콘 반환
  const getDeviceIcon = (type: string): string => {
    switch (type) {
      case 'mps':
        return '🍎';
      case 'cuda':
        return '⚡';
      case 'cpu':
        return '💾';
      default:
        return '🖥️';
    }
  };

  // 메모리 사용량 포맷팅
  const formatMemory = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    return `${(bytes).toFixed(2)} GB`;
  };

  // 메모리 사용량 퍼센트 계산
  const getMemoryPercent = (): number => {
    if (!memoryInfo?.allocated || !memoryInfo?.total) return 0;
    return (memoryInfo.allocated / memoryInfo.total) * 100;
  };

  return (
    <div className={`device-selector p-6 bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg shadow-lg ${className}`}>
      {/* 헤더 */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          🎯 컴퓨팅 디바이스 선택
        </h2>
        <p className="text-slate-600 text-sm mt-1">
          학습 및 추론에 사용할 GPU/CPU를 선택하세요
        </p>
      </div>

      {/* 에러 메시지 */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 rounded-lg text-red-700 flex justify-between items-center">
          <span className="text-sm">⚠️ {error}</span>
          <button
            onClick={resetError}
            className="text-red-700 hover:text-red-900 font-bold"
          >
            ✕
          </button>
        </div>
      )}

      {/* 로딩 상태 */}
      {isLoading && (
        <div className="mb-4 p-3 bg-blue-100 border border-blue-400 rounded-lg text-blue-700">
          <span className="text-sm">⏳ 처리 중...</span>
        </div>
      )}

      {/* 디바이스 선택 그리드 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {availableDevices.length > 0 ? (
          availableDevices.map((device) => (
            <button
              key={device.type}
              onClick={() => handleSelectDevice(device.type as DeviceType)}
              disabled={!device.is_available || isLoading}
              className={`
                p-4 rounded-lg border-2 transition-all duration-200
                ${selectedDevice === device.type
                  ? 'border-blue-500 bg-blue-50 shadow-lg'
                  : 'border-slate-200 bg-white hover:border-slate-300'
                }
                ${!device.is_available
                  ? 'opacity-50 cursor-not-allowed'
                  : 'cursor-pointer hover:shadow-md'
                }
                ${isLoading ? 'opacity-75' : ''}
              `}
            >
              <div className="text-3xl mb-2">{getDeviceIcon(device.type)}</div>
              <div className="text-left">
                <h3 className="font-bold text-slate-900 text-sm capitalize">
                  {device.type}
                </h3>
                <p className="text-xs text-slate-600 mt-1">{device.name}</p>
                
                {/* 메모리 정보 (CUDA만 표시) */}
                {device.memory_total && (
                  <p className="text-xs text-slate-500 mt-2">
                    💾 {formatMemory(device.memory_total)}
                  </p>
                )}
                
                {/* 상태 배지 */}
                <div className="mt-2">
                  {device.is_available ? (
                    <span className="inline-block px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      ✓ 사용 가능
                    </span>
                  ) : (
                    <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      ✗ 사용 불가
                    </span>
                  )}
                </div>
              </div>
            </button>
          ))
        ) : (
          <div className="col-span-full text-center py-8 text-slate-500">
            <p>🔍 디바이스를 감지하는 중...</p>
          </div>
        )}
      </div>

      {/* 현재 선택 상태 */}
      {selectedDevice && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-green-900">
                ✅ 선택된 디바이스
              </p>
              <p className="text-lg font-bold text-green-700 mt-1">
                {getDeviceIcon(selectedDevice)} {currentDevice}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 메모리 정보 */}
      {memoryInfo && memoryInfo.allocated !== null && (
        <div className="mb-6">
          <button
            onClick={() => setShowMemoryDetails(!showMemoryDetails)}
            className="w-full flex items-center justify-between p-4 bg-slate-200 hover:bg-slate-300 rounded-lg transition-colors duration-200"
          >
            <span className="font-semibold text-slate-700">
              📊 메모리 정보
            </span>
            <span className="text-slate-600">
              {showMemoryDetails ? '▼' : '▶'}
            </span>
          </button>

          {showMemoryDetails && (
            <div className="mt-3 p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3">
              {/* 메모리 바 */}
              {memoryInfo.total && (
                <>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-600">사용 중:</span>
                    <span className="font-semibold text-slate-900">
                      {formatMemory(memoryInfo.allocated)} / {formatMemory(memoryInfo.total)}
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${getMemoryPercent()}%` }}
                    />
                  </div>
                  <div className="text-xs text-slate-500 text-center">
                    {getMemoryPercent().toFixed(1)}% 사용 중
                  </div>
                </>
              )}

              {/* 상세 정보 */}
              <div className="space-y-2 pt-2 border-t border-slate-200">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">디바이스:</span>
                  <span className="font-mono text-slate-900">{memoryInfo.device}</span>
                </div>
                {memoryInfo.reserved && (
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">예약됨:</span>
                    <span className="font-mono text-slate-900">
                      {formatMemory(memoryInfo.reserved)}
                    </span>
                  </div>
                )}
                {memoryInfo.available && (
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">사용 가능:</span>
                    <span className="font-mono text-green-700 font-semibold">
                      {formatMemory(memoryInfo.available)}
                    </span>
                  </div>
                )}
              </div>

              {/* 캐시 정리 버튼 */}
              <button
                onClick={handleClearCache}
                disabled={isLoading}
                className="w-full mt-3 px-4 py-2 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-lg font-semibold transition-colors duration-200"
              >
                🧹 캐시 메모리 정리
              </button>
            </div>
          )}
        </div>
      )}

      {/* 정보 박스 */}
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-900">
        <p className="font-semibold mb-2">ℹ️ 디바이스 선택 가이드</p>
        <ul className="text-xs space-y-1 text-blue-800">
          <li>🍎 <strong>MPS</strong>: Apple Silicon (M1/M2/M3) - 가장 빠름</li>
          <li>⚡ <strong>CUDA</strong>: NVIDIA GPU - 강력한 성능</li>
          <li>💾 <strong>CPU</strong>: 일반 프로세서 - 느리지만 호환성 우수</li>
        </ul>
      </div>
    </div>
  );
};

export default DeviceSelector;


